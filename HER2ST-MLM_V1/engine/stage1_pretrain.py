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
    Image branch: H&E patches → frozen image encoder Virchow2 → features → MaskedEncoder.
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
                self.image_encoder = ImageEncoder_Virchow2(image_agg=cfg['image_agg'])
            else:
                raise ValueError(f"Unknown encoder_name: {cfg['encoder_name']}")
        self.masked_encoder = MaskedEncoder(
            dim=cfg['dim'],
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

    def forward(self, batch):
        """
        Input batch (DataLoader batch_size=64, window mode M=100):
          spot:  {'gene': [B, M, 785], 'coords': [B, M, 2], 'slide_id': [str]}     ; B = 64, M = 100
          image: {'patches': [B, M, 3, 224, 224], 'coords': [B, M, 2], 'slide_id': [str]}
          B = 64 windows per batch,  M = 100 spots per window.
        Output: (loss, pred, mask) from MaskedEncoder
        """
        # batch keys: spot {'gene':[B,M,785], 'coords':[B,M,2], 'slide_id':list}
        #             image {'patches':[B,M,3,224,224], 'coords':[B,M,2], 'slide_id':list}
        coords = batch['coords']  # [B, M, 2] integer (x,y) spot grid coords
        if self.branch == 'spot':
            x = batch['gene']     # [B, M, 785] spot branch: gene expression IS the feature (no backbone)
        else:
            patches = batch['patches']     # [B, M, 3, 224, 224]
            B, M = patches.shape[0], patches.shape[1]
            patches = patches.reshape(B * M, *patches.shape[2:])     # [64*100, 3, 224, 224] ; [B*M, 3, 224, 224] flatten to a plain patch batch
            feats = []  
            with torch.no_grad():                   # Virchow2 backbone is frozen -> no grad
                for s in range(0, patches.shape[0], 256):                     #  chunk by 256 patches to bound GPU memory
                    feats.append(self.image_encoder(patches[s:s + 256]))      
            # run 256 patches through frozen Virchow2 -> [<=256, 1280]; append to feats
            # after the loop: feats = list of 6400 / 256 = 25 chunks, each [<=256, 1280]; concatenated next into [6400, 1280]

            features = torch.cat(feats, dim=0)       # [B*M, dim] ; [6400, 1280] all patch features
            x = features.reshape(B, M, -1)           # [B, M, dim=1280] ; [64, 100, 1280] — all windows kept 
        # feed feature x + coords into the MAE (this is the only trainable part)
        return self.masked_encoder(x, coords)
        # returns: scalar loss | pred [B, M, dim] reconstruction | mask [B, M] 0=keep/1=masked

## spot branch :
# gene:     [100, 785]   lib-norm+log1p HVG 
# coords:   [100, 2]     spot grid 
# slide_id: 'A5'  

## Image branch : 
# patches:  [100, 3, 224, 224]  
# coords:   [100, 2]            
# slide_id: 'A5'               

def build_dataloader(cfg, train=True):
    """Stage 1 DataLoader; batch_size=64 ."""
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
        num_workers=0, 
        pin_memory=False,
    )


def build_model(cfg, device):
    return Stage1Model(cfg).to(device)  # build model on CPU, then move to GPU?


def build_optimizer(model, cfg):    # optimize trainable parameters
    """Adam over trainable params only (frozen image_encoder excluded by requires_grad=False)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])  # Stage1 lr = 1e-4, weight_decay = 1e-3


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Run one epoch over all Stage 1 samples, return average loss across samples."""
    model.train()   # 
    loss_meter = AvgMeter(name='train_loss')   # tracks running average.

    for batch in loader:        # each batch = 64 window samples (shuffled), default collate stacks them
                                # batch = {'gene' : [64, 100, 785], 'coords' : [64, 100, 2], 'slide_id' : [64 str]}
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}  # move tensors to GPU; keep slide_id (str) on CPU
        loss, _, _ = model(batch)       # MAE forward; keep scalar loss, drop pred/mask
        optimizer.zero_grad()           
        loss.backward()                 # backprop -> grads only on trainable params (Virchow2 frozen)
        optimizer.step()                # Adam updates masked_encoder weights only (image_encoder untouched)
        loss_meter.update(loss.item(), count=1)  # count=1: every batch weighted equally
    return loss_meter.avg  # mean loss over all batches this epoch


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config') # path to the config file
    args = parser.parse_args()     

    with open(args.config) as f:
        cfg = yaml.safe_load(f)   # load config

    set_seed(cfg['seed'])      # fixed seed = 42.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_save_dir = cfg['save_dir']  # save path for best.pt

    for fold in range(6):     # 6-fold : A4, A5, A6, B1, B2, B3
        cfg['fold'] = fold
        save_dir = os.path.join(base_save_dir, f'fold{fold}') 
        os.makedirs(save_dir, exist_ok=True)

        train_loader = build_dataloader(cfg, train=True)  # read data in Dataset
        model = build_model(cfg, device)
        optimizer = build_optimizer(model, cfg)

        print(f"\n=== Stage 1 {cfg['branch']} | fold={fold} | epochs={cfg['epochs']} ===")
        print(f"train slides: {train_loader.dataset.names}")
        print(f"train samples: {len(train_loader.dataset)}")
        print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_loss = float('inf')
        loss_history = []

        for epoch in range(cfg['epochs']):
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)  # 
            lr = get_lr(optimizer)   # Stage1 lr= 1e-4
            if epoch % cfg['log_every'] == 0:    # log_every=1 → every epoch
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | lr={lr:.2e}")
            
            # save best.pt
            if avg_loss < best_loss:      # overwrite best.pt only when loss hits a new low
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),  # full weights: frozen image_encoder + trainable masked_encoder
                    'optimizer_state': optimizer.state_dict(),
                    'loss': avg_loss,
                    'cfg': cfg,  # whole hyperparam dict saved into ckpt
                }, os.path.join(save_dir, 'best.pt'))

            torch.save({  # latest.pt overwritten every epoch (same 5 keys as best.pt)
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

# ============================================================================
## Example Flow
# consts: B=64 windows | M=100 spots/window | dim=785 | mask 10% (keep 90, mask 10)
# ============================================================================
#
#  yaml ──► cfg dict ──► 6 folds, each trained from scratch, each saves its own weights
#                              │
#   ┌──────────────────────────┘  each fold gets 3 things:
#   │   loader : feeds data (64 windows at a time)
#   │   model  : Stage1Model (only masked_encoder is trainable)
#   │   optim  : Adam
#   │
#   └─► train 10 epochs; each epoch runs through every batch:
#
#        For ONE batch ↓↓↓
#
#        loader yields ──► gene  [64, 100, 785]   # 64 windows, 100 spots each, 785 genes/spot
#                          coords[64, 100, 2]     # (x,y) grid coord per spot
#                            │
#        model(batch) ───────┤  branch='spot' -> just take gene expression as the feature x
#                            │  (image branch: patches -> Virchow2 -> [64,100,1280], then x)
#                            ▼
#        masked_encoder(x, coords)  ── MAE does 3 things ──
#            target = copy of x                  [64,100,785]  # reconstruction goal (the answer)
#            (1) drop 10 spots, encoder sees only the kept 90   [64, 90,785]
#            (2) decoder fills the 10 gaps back -> full 100      [64,100,785]  # pred (the guess)
#            (3) loss = compare ONLY the 10 masked spots: (pred-target)^2  -> one number
#                            │
#                            ▼  -> loss (scalar)
#        zero_grad -> backward -> step    # clear grads -> compute grads -> update masked_encoder
#
#        (64 windows per batch, one update per batch; all batches done = 1 epoch -> avg_loss)
#
#   end of each epoch:
#       avg_loss hits a new low -> save best.pt    (lowest-loss version)
#       always                  -> save latest.pt  (last version, overwritten each epoch)
#       append one line to loss_history.json
#
# ============================================================================
#  output: one folder per fold  runs/stage1_spot_window/fold{0..5}/
#          best.pt / latest.pt / loss_history.json
#  used by: Stage2 loads masked_encoder from best.pt and uses its encode() (no masking)
#           to extract features for contrastive learning
# ============================================================================