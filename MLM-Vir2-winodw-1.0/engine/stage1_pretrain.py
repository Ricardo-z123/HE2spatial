import argparse
import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.stage1_dataset import HER2STStage1Dataset
from models.masked_encoder import MaskedEncoder
from models.modules import ImageEncoder, ImageEncoder_Resnet, ImageEncoder_Virchow2
from utils.misc import AvgMeter, get_lr, set_seed


class Stage1Model(nn.Module):
    """
    Spot branch:  forward gene expression directly into MaskedEncoder.
    Image branch: H&E patches → frozen image backbone → features → MaskedEncoder.
    """
    def __init__(self, cfg):
        super().__init__()
        self.branch = cfg['branch']
        if self.branch == 'image':
            if cfg['encoder_name'] == 'densenet121':
                self.image_encoder = ImageEncoder()
            elif cfg['encoder_name'] == 'resnet50':
                self.image_encoder = ImageEncoder_Resnet()
            elif cfg['encoder_name'] == 'virchow2':
                self.image_encoder = ImageEncoder_Virchow2()
            else:
                raise ValueError(f"Unknown encoder_name: {cfg['encoder_name']}")
        self.masked_encoder = MaskedEncoder(
            in_dim=cfg['in_dim'],
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

    def forward(self, batch):
        """
        Input batch (DataLoader batch_size=1):
          spot:  {'gene': [1, M, 785], 'coords': [1, M, 2], 'slide_id': [str]}
          image: {'patches': [1, M, 3, 224, 224], 'coords': [1, M, 2], 'slide_id': [str]}
          M = spots in a whole-slide sample or valid local window sample.
        Output: (loss, pred, mask) from MaskedEncoder
        """
        coords = batch['coords']  # [1, M, 2]
        if self.branch == 'spot':
            x = batch['gene']  # [1, M, 785]
        else:
            patches = batch['patches'][0]  # drop batch dim → [M, 3, 224, 224] for image backbone
            with torch.no_grad():  # image_encoder is frozen, no grad needed
                features = self.image_encoder(patches)  # [M, in_dim] e.g. 1024 DenseNet, 2048 ResNet, 1280 Virchow2
            x = features.unsqueeze(0)  # [1, M, in_dim] for MaskedEncoder
        return self.masked_encoder(x, coords)
        # scalar loss | [1, M, in_dim] reconstruction | [1, M] 0/1 mask


def build_dataloader(cfg, train=True):
    """Stage 1 DataLoader; batch_size=1 supports variable slide/window sequence length."""
    dataset = HER2STStage1Dataset(
        train=train,
        fold=cfg['fold'],
        branch=cfg['branch'],
        use_windows=cfg.get('use_windows', False),
        window_size=cfg.get('window_size', 10),
        window_stride=cfg.get('window_stride', 2),
        window_min_spots=cfg.get('window_min_spots', 80),
    )
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=train,
        num_workers=0,  # image branch slides ~360MB of patches each, multi-worker would OOM
        pin_memory=False,
    )


def build_model(cfg, device):
    return Stage1Model(cfg).to(device)


def build_optimizer(model, cfg):
    """Adam over trainable params only (frozen image_encoder excluded by requires_grad=False)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Run one epoch over all Stage 1 samples, return average loss across samples."""
    model.train()
    loss_meter = AvgMeter(name='train_loss')
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, _, _ = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), count=1)
    return loss_meter.avg


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

        train_loader = build_dataloader(cfg, train=True)
        model = build_model(cfg, device)
        optimizer = build_optimizer(model, cfg)

        print(f"\n=== Stage 1 {cfg['branch']} | fold={fold} | epochs={cfg['epochs']} ===")
        print(f"train slides: {train_loader.dataset.names}")
        print(f"train samples: {len(train_loader.dataset)}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_loss = float('inf')
        loss_history = []

        for epoch in range(cfg['epochs']):
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            lr = get_lr(optimizer)
            if epoch % cfg['log_every'] == 0:
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | lr={lr:.2e}")

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

            loss_history.append({'epoch': epoch, 'loss': avg_loss, 'lr': lr})
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f, indent=2)

        print(f"=== Fold {fold} done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()
