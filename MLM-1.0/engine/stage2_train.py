import argparse
import os
import json
import math
import random
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from utils.misc import AvgMeter, get_lr, set_seed


class SlideBatchSampler(Sampler):
    """
    Each batch contains spots from exactly one slide.
    Slide order and within-slide spot order are reshuffled every epoch.
    Incomplete tail batches per slide are kept (math.ceil) — matches experimental code.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.slide_indices = []
        prev = 0
        for end in dataset.cumlen:
            self.slide_indices.append(list(range(prev, end)))  # global indices belonging to this slide
            prev = end

    def __iter__(self):
        all_batches = []
        for indices in self.slide_indices:
            idx = indices.copy()
            if self.shuffle:
                random.shuffle(idx)
            for i in range(0, len(idx), self.batch_size):
                all_batches.append(idx[i:i + self.batch_size])  # contiguous within-slide chunk
        if self.shuffle:
            random.shuffle(all_batches)  # shuffle slide order across epoch
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(idx) / self.batch_size) for idx in self.slide_indices)


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
    spot_ckpt = torch.load(spot_ckpt_path, map_location='cpu')
    image_ckpt = torch.load(image_ckpt_path, map_location='cpu')

    spot_state = {k[len('masked_encoder.'):]: v
                  for k, v in spot_ckpt['model_state'].items()
                  if k.startswith('masked_encoder.')}
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
    for p in model.spot_encoder.parameters():
        p.requires_grad = False
    for p in model.image_cnn.parameters():
        p.requires_grad = False
    for p in model.image_encoder.parameters():
        p.requires_grad = False


def build_dataloader(cfg, train=True):
    """Per-slide-batch DataLoader: SlideBatchSampler keeps each batch inside one tissue."""
    dataset = HER2STStage2Dataset(train=train, fold=cfg['fold'])
    sampler = SlideBatchSampler(dataset, batch_size=cfg['batch_size'], shuffle=train)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)


def build_model(cfg, device):
    return ContrastiveModel(cfg).to(device)


def build_optimizer(model, cfg):
    """Adam over trainable params only (frozen encoders excluded by requires_grad=False)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Run one epoch over all (slide, batch) pairs; return averages of total / spots / images loss."""
    model.train()
    loss_meter = AvgMeter(name='train_loss')
    spots_meter = AvgMeter(name='spots_loss')
    images_meter = AvgMeter(name='images_loss')
    for batch in loader:
        # only the three model inputs need GPU; slide_id / center stay on CPU
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ('image', 'expression', 'position')}
        loss, spots_loss, images_loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch['image'].size(0)  # spots per batch
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

    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    train_loader = build_dataloader(cfg, train=True)
    model = build_model(cfg, device)
    load_stage1_weights(model, cfg['spot_ckpt'], cfg['image_ckpt'])
    freeze_encoders(model)
    optimizer = build_optimizer(model, cfg)

    print(f"=== Stage 2 contrastive | fold={cfg['fold']} | epochs={cfg['epochs']} ===")
    print(f"train slides: {list(train_loader.dataset.id2name.values())}")
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(cfg['epochs']):
        avg_loss, avg_spots, avg_images = train_one_epoch(model, train_loader, optimizer, device, epoch)
        lr = get_lr(optimizer)
        if epoch % cfg['log_every'] == 0:
            print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | "
                  f"spots={avg_spots:.4f} | images={avg_images:.4f} | lr={lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'cfg': cfg,
            }, os.path.join(save_dir, 'best.pt'))

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_loss,
            'cfg': cfg,
        }, os.path.join(save_dir, 'latest.pt'))

        loss_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'spots_loss': avg_spots,
            'images_loss': avg_images,
            'lr': lr,
        })
        with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)

    print(f"=== Done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
