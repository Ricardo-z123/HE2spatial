import argparse
import os
import json
import math
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.stage2_window_dataset import HER2STStage2WindowDataset
from models.contrastive_model import ContrastiveModel
from utils.misc import AvgMeter, get_lr, set_seed


# Window Stage 2 (Survey §4): attention runs within KNN windows; contrastive loss pools across windows.
# Loads the frozen Stage1 encoders; only the two ProjectionHeads train.


def load_stage1_weights(model, spot_ckpt_path, image_ckpt_path):
    """
    Load Stage 1 spot/image encoders into a ContrastiveModel.
    Key remap (Stage 1 ckpt → ContrastiveModel):
      spot ckpt:   'masked_encoder.*'  → model.spot_encoder.*
      image ckpt:  'image_encoder.*'   → model.image_cnn.*
                   'masked_encoder.*'  → model.image_encoder.*
    strict=False so unrelated keys (e.g. Stage 1 decoder_*) and the still-randomly-initialised
    ProjectionHeads don't trip the loader.
    """
    spot_ckpt = torch.load(spot_ckpt_path, map_location='cpu')      # Stage1 spot best.pt
    image_ckpt = torch.load(image_ckpt_path, map_location='cpu')    # Stage1 image best.pt

## Stage 1 spot ckpt key :
# spot_ckpt = {
#     'epoch':           9,
#     'model_state':     {... weight dict ...},   # full weights: frozen image_encoder + trainable masked_encoder
#     'optimizer_state': {...},
#     'loss':            0.21,
#     'cfg':             {...},
# }

# spot_ckpt['model_state'] = {
#     'masked_encoder.pos_embed.x_embed.weight':      <tensor>,   # ← key
#     'masked_encoder.encoder_blocks.0.attn.xxx':     <tensor>,
#     'masked_encoder.decoder_blocks.0.xxx':          <tensor>,
#     ...
# }

    spot_state = {k[len('masked_encoder.'):]: v                  ## extract all masked_encoder.* weights from spot ckpt dict, and delete Prefix.
                  for k, v in spot_ckpt['model_state'].items()   ## obtain new ckpt dict : spot_state.
                  if k.startswith('masked_encoder.')}
    # load Stage1 spot weights into spot_encoder; missing/unexpected = keys not filled / keys ignored (sanity check)
    missing, unexpected = model.spot_encoder.load_state_dict(spot_state, strict=False)
    print(f"[load Stage 1 spot encoder]  missing={len(missing)}, unexpected={len(unexpected)}")

    cnn_state = {k[len('image_encoder.'):]: v
                 for k, v in image_ckpt['model_state'].items()
                 if k.startswith('image_encoder.')}
    missing, unexpected = model.image_cnn.load_state_dict(cnn_state, strict=False)
    print(f"[load Stage 1 image CNN]     missing={len(missing)}, unexpected={len(unexpected)}")

    img_enc_state = {k[len('masked_encoder.'):]: v
                     for k, v in image_ckpt['model_state'].items()
                     if k.startswith('masked_encoder.')}
    missing, unexpected = model.image_encoder.load_state_dict(img_enc_state, strict=False)
    print(f"[load Stage 1 image encoder] missing={len(missing)}, unexpected={len(unexpected)}")


def freeze_encoders(model):
    """Freeze the three Stage 1 encoders. Only ProjectionHeads + temperature stay trainable."""
    for p in model.spot_encoder.parameters():   # frozen: Stage1 spot MaskedEncoder
        p.requires_grad = False
    for p in model.image_cnn.parameters():      # frozen: Virchow2 backbone
        p.requires_grad = False
    for p in model.image_encoder.parameters():  # frozen: Stage1 image MaskedEncoder
        p.requires_grad = False
    # left trainable: spot_projection + image_projection (+ logit_scale)


def build_optimizer(model, cfg):
    """Adam over trainable params only (frozen encoders excluded by requires_grad=False)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def build_dataloader(cfg, train=True):
    """Window DataLoader: each item = one KNN window (M = window_size**2 spots); batch_size = #windows (B).

    All windows have exactly k spots, so default collate stacks them into [B, M, ...] cleanly.
    """
    dataset = HER2STStage2WindowDataset(
        train=train, fold=cfg['fold'],
        window_size=cfg.get('window_size', 10),
        window_stride=cfg.get('window_stride', 2),
    )
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],      # number of WINDOWS per batch (B)
        shuffle=train,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False,
    )


def build_model(cfg, device):
    return ContrastiveModel(cfg).to(device)


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_meter = AvgMeter(name='train_loss')
    spots_meter = AvgMeter(name='spots_loss')
    images_meter = AvgMeter(name='images_loss')
    for batch in loader:  # each batch = B windows, shape [B, M, ...]
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ('image', 'expression', 'position', 'spot_id')}   # spot_id: duplicate-spot label
        loss, spots_loss, images_loss = model(batch)
        optimizer.zero_grad()
        loss.backward()      # grads reach ONLY the two ProjectionHeads (+logit_scale); encoders frozen
        optimizer.step()
        if model.normalize_clip:
            model.logit_scale.data.clamp_(0, math.log(100))
        count = batch['image'].shape[0] * batch['image'].shape[1]   # B*M spots (AvgMeter bookkeeping)
        loss_meter.update(loss.item(), count)
        spots_meter.update(spots_loss.item(), count)
        images_meter.update(images_loss.item(), count)
    return loss_meter.avg, spots_meter.avg, images_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_save_dir = cfg['save_dir']

    for fold in range(6):    # A4, A5, A6, B1, B2, B3
        cfg['fold'] = fold
        save_dir = os.path.join(base_save_dir, f'fold{fold}')
        os.makedirs(save_dir, exist_ok=True)

        spot_ckpt = os.path.join(cfg['spot_ckpt_dir'], f'fold{fold}', 'best.pt')
        image_ckpt = os.path.join(cfg['image_ckpt_dir'], f'fold{fold}', 'best.pt')

        train_loader = build_dataloader(cfg, train=True)
        model = build_model(cfg, device)
        load_stage1_weights(model, spot_ckpt, image_ckpt)   # load frozen Stage1 spot/image encoders
        freeze_encoders(model)                              # only ProjectionHeads train
        optimizer = build_optimizer(model, cfg)

        print(f"\n=== Window Stage2 | fold={fold} | epochs={cfg['epochs']} | windows/batch={cfg['batch_size']} ===")
        print(f"train slides: {list(train_loader.dataset.id2name.values())}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_loss = float('inf')
        best_ckpt = None          # best weights kept in RAM; written to disk once after the loop
        loss_history = []
        for epoch in range(cfg['epochs']):
            avg_loss, avg_spots, avg_images = train_one_epoch(model, train_loader, optimizer, device, epoch)
            lr = get_lr(optimizer)
            if epoch % cfg['log_every'] == 0:
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | "
                      f"spots={avg_spots:.4f} | images={avg_images:.4f} | lr={lr:.2e}")
            if avg_loss < best_loss:  # best selected on TRAIN loss (no val set)
                best_loss = avg_loss
                # .cpu().clone() is essential: state_dict() returns live refs that later epochs mutate
                best_ckpt = {'epoch': epoch,
                             'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                             'loss': avg_loss, 'cfg': cfg}
            loss_history.append({'epoch': epoch, 'loss': avg_loss, 'spots_loss': avg_spots,
                                 'images_loss': avg_images, 'lr': lr})
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f, indent=2)
        if best_ckpt is not None:
            torch.save(best_ckpt, os.path.join(save_dir, 'best.pt'))   # the ONLY disk write per fold
        print(f"=== Fold {fold} done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()

## Flow details

# L1-15     imports + 文件说明
# L18-64    ① load_stage1_weights   ← import stage1 ckpt into stage2 model
# L67-75    ② freeze_encoders       ← Frozen encoder
# L78-81    ③ build_optimizer       ← Optimze only on unfreezeed params (ProjectionHeads + logit_scale).
# L84-100   ④ build_dataloader
# L103-104  ⑤ build_model
# L107-125  ⑥ train_one_epoch       ← train loop
# L128-180  ⑦ main                  ← 6 fold train + save ckpt.