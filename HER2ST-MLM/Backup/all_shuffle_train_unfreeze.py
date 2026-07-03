import argparse
import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from engine.stage2_train import load_stage1_weights  # freeze_encoders intentionally NOT imported
from utils.misc import AvgMeter, get_lr, set_seed


# =============================================================================
# Unfreeze variant of all_shuffle_train.py:
#   - Stage 1 weights loaded as INITIALIZATION (not frozen features)
#   - Only Virchow2 CNN stays frozen (631M params, too large to fine-tune on a
#     single GPU at batch_size=128).
#   - spot_encoder transformer + image_encoder transformer + PosEmbed +
#     2 ProjectionHeads + temperature are all trainable.
#   - Recommended lr: 1e-5 (5x smaller than projection-only setting) so the
#     Stage 1 pretrained features are gently fine-tuned, not destroyed.
# =============================================================================


def partial_freeze(model):
    """Freeze only Virchow2 CNN backbone. Everything else (spot_encoder transformer,
    image_encoder transformer, PosEmbeds, ProjectionHeads, temperature) stays
    trainable. Stage 2 fine-tunes the Stage 1 pretrained features end-to-end."""
    for p in model.image_cnn.parameters():
        p.requires_grad = False


def build_dataloader(cfg, train=True):
    """All-shuffle DataLoader: cross-slide random batching, same as all_shuffle_train.py."""
    dataset = HER2STStage2Dataset(train=train, fold=cfg['fold'])
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=train,
        num_workers=0,
        pin_memory=False,
    )


def build_model(cfg, device):
    return ContrastiveModel(cfg).to(device)


def build_optimizer(model, cfg):
    """Adam over trainable params (Virchow2 CNN excluded by partial_freeze)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def train_one_epoch(model, loader, optimizer, device, epoch):
    """One epoch over all batches; return averages of total / spots / images loss."""
    model.train()
    loss_meter = AvgMeter(name='train_loss')
    spots_meter = AvgMeter(name='spots_loss')
    images_meter = AvgMeter(name='images_loss')
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ('image', 'expression', 'position')}
        loss, spots_loss, images_loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch['image'].size(0)
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

    for fold in range(6):
        cfg['fold'] = fold
        save_dir = os.path.join(base_save_dir, f'fold{fold}')
        os.makedirs(save_dir, exist_ok=True)

        spot_ckpt = os.path.join(cfg['spot_ckpt_dir'], f'fold{fold}', 'best.pt')
        image_ckpt = os.path.join(cfg['image_ckpt_dir'], f'fold{fold}', 'best.pt')

        train_loader = build_dataloader(cfg, train=True)
        model = build_model(cfg, device)
        load_stage1_weights(model, spot_ckpt, image_ckpt)
        partial_freeze(model)  # only Virchow2 frozen; transformers + projections trainable
        optimizer = build_optimizer(model, cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== All-Shuffle UNFREEZE | fold={fold} | epochs={cfg['epochs']} ===")
        print(f"train slides: {list(train_loader.dataset.id2name.values())}")
        print(f"trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")

        best_loss = float('inf')
        loss_history = []

        for epoch in range(cfg['epochs']):
            avg_loss, avg_spots, avg_images = train_one_epoch(model, train_loader, optimizer, device, epoch)
            lr = get_lr(optimizer)
            if epoch % cfg['log_every'] == 0:
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | "
                      f"spots={avg_spots:.4f} | images={avg_images:.4f} | lr={lr:.2e}")

            ckpt = {'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(), 'loss': avg_loss, 'cfg': cfg}
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
            torch.save(ckpt, os.path.join(save_dir, 'latest.pt'))

            loss_history.append({'epoch': epoch, 'loss': avg_loss, 'spots_loss': avg_spots,
                                 'images_loss': avg_images, 'lr': lr})
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f, indent=2)

        print(f"=== Fold {fold} done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
