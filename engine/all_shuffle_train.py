import argparse
import os
import json
import math
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from utils.misc import AvgMeter, get_lr, set_seed


# All-Shuffle Stage 2: spots are shuffled across ALL slides (baseline mclSTExp style),

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


def build_dataloader(cfg, train=True):
    """All-shuffle DataLoader: cross-slide random batching, baseline-style.

    Each batch holds cfg['batch_size'] spots drawn uniformly at random across
    all train slides — directly opposes stage2_train.py's within-slide sampler.
    """
    dataset = HER2STStage2Dataset(train=train, fold=cfg['fold'])
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        # WARNING: plain shuffle => one batch mixes spots from DIFFERENT slides (cross-slide),
        #          unlike stage2_train.py's SlideBatchSampler which keeps each batch within one slide.
        shuffle=train,         # shuffle during training, sequential at inference
        num_workers=0,
        pin_memory=False,
    )


def build_model(cfg, device):
    return ContrastiveModel(cfg).to(device)


def build_optimizer(model, cfg):
    """Adam over trainable params only (frozen encoders excluded by requires_grad=False)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Run one epoch over all batches; return averages of total / spots / images loss."""
    model.train()
    loss_meter = AvgMeter(name='train_loss')
    spots_meter = AvgMeter(name='spots_loss')
    images_meter = AvgMeter(name='images_loss')
    for batch in loader:  # each batch = N spots randomly mixed across ALL train slides (shuffle)
        # batch keys: image [N,3,224,224] H&E patch | expression [N,785] HVG | position [N,2] (x,y) coords
        # only the three model inputs need GPU; slide_id / center stay on CPU
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ('image', 'expression', 'position')}
        # contrastive loss: symmetric CLIP CE on the [N,N] spot<->image cosine matrix (diagonal = positive pairs)
        loss, spots_loss, images_loss = model(batch)  # loss scalar; spots/images_loss detached (logging only)
        optimizer.zero_grad()  # clear grads
        loss.backward()        # backprop -- grads reach ONLY the two ProjectionHeads (+logit_scale); encoders frozen
        optimizer.step()       # update trainable params (projection heads only)
        if model.normalize_clip:                              # CLIP-style: cap logit_scale after each step (CLIP training clamps to [0, ln100])
            model.logit_scale.data.clamp_(0, math.log(100))
        count = batch['image'].size(0)  # spots per batch
        loss_meter.update(loss.item(), count)
        spots_meter.update(spots_loss.item(), count)
        images_meter.update(images_loss.item(), count)
    return loss_meter.avg, spots_meter.avg, images_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    args = parser.parse_args()

    with open(args.config) as f:   # read yaml -> config dict
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])       # set.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_save_dir = cfg['save_dir']    # save dir.

    for fold in range(6):    # A4, A5, A6, B1, B2, B3
        cfg['fold'] = fold
        save_dir = os.path.join(base_save_dir, f'fold{fold}')    # save dir.
        os.makedirs(save_dir, exist_ok=True)

        spot_ckpt = os.path.join(cfg['spot_ckpt_dir'], f'fold{fold}', 'best.pt')     #   path to Stage1 spot best.pt (this fold)
        image_ckpt = os.path.join(cfg['image_ckpt_dir'], f'fold{fold}', 'best.pt')   #   path to Stage 1 image best.pt (this fold)

        train_loader = build_dataloader(cfg, train=True)     # DataLoader ; all-shuffle cross slides.
        model = build_model(cfg, device)            # ContrastiveModel
        load_stage1_weights(model, spot_ckpt, image_ckpt)  # load pretrained Stage1 spot/image encoder weights
        freeze_encoders(model)                             # freeze encoders -> only projection heads train
        optimizer = build_optimizer(model, cfg)

        print(f"\n=== All-Shuffle | fold={fold} | epochs={cfg['epochs']} ===")
        print(f"train slides: {list(train_loader.dataset.id2name.values())}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_loss = float('inf')
        loss_history = []

        for epoch in range(cfg['epochs']):
            avg_loss, avg_spots, avg_images = train_one_epoch(model, train_loader, optimizer, device, epoch)
                     # (spot_loss + image_loss) / 2 ; gene->image CE ; image->gene CE
            lr = get_lr(optimizer)
            if epoch % cfg['log_every'] == 0:
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | "
                      f"spots={avg_spots:.4f} | images={avg_images:.4f} | lr={lr:.2e}")

            # only save best.pt 
            if avg_loss < best_loss:  # best selected on TRAIN loss (no val set)
                best_loss = avg_loss
                
                ckpt = {'epoch': epoch, 'model_state': model.state_dict(),    # model_state includes the frozen encoders 
                        'optimizer_state': optimizer.state_dict(), 'loss': avg_loss, 'cfg': cfg}
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))  # overwrite -> keep only best.pt

            loss_history.append({'epoch': epoch, 'loss': avg_loss, 'spots_loss': avg_spots,
                                 'images_loss': avg_images, 'lr': lr})
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f, indent=2)

        print(f"=== Fold {fold} done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
