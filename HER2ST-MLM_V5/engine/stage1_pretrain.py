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
            # Re-seed so the backbone CHOICE cannot shift the RNG stream: timm.create_model randomly
            # initialises Virchow2's 632M params before loading pretrained weights. Without this the
            # MaskedEncoder below -- the only thing Stage1 trains -- gets a different init depending
            # on the backbone, i.e. a different experiment.
            torch.manual_seed(cfg['seed'])
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
        Input batch (DataLoader batch_size=8 windows, window mode M=100):
          spot:  {'gene': [B, M, 785], 'coords': [B, M, 2], 'slide_id': [str]}     ; B = 8, M = 100
          image: {'patches': [B, M, 3, 224, 224], 'coords': [B, M, 2], 'slide_id': [str]}
          B = 8 windows per batch,  M = 100 spots per window (cfg batch_size=8).
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
            patches = patches.reshape(B * M, *patches.shape[2:])     # [8*100, 3, 224, 224] ; [B*M, 3, 224, 224] flatten to a plain patch batch
            feats = []  
            with torch.no_grad():                   # Virchow2 backbone is frozen -> no grad
                for s in range(0, patches.shape[0], 256):                     #  chunk by 256 patches to bound GPU memory
                    feats.append(self.image_encoder(patches[s:s + 256]))      
            # run <=256 patches through frozen Virchow2 -> [<=256, dim]; append to feats
            # after the loop: feats = ceil(800/256) = 4 chunks (256,256,256,32), each [<=256, 2560]
            # with image_agg='concat'; concatenated next into [800, 2560]

            features = torch.cat(feats, dim=0)       # [B*M, dim] ; [800, 2560] all patch features
            x = features.reshape(B, M, -1)           # [B, M, dim=2560] ; [8, 100, 2560] - all windows kept
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
    """Stage 1 DataLoader; batch_size = cfg['batch_size'] = 8 windows."""
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
        num_workers=cfg.get('num_workers', 4),   # infra: parallel CPU patch-crop (was 0); dataloading only, 不改模型/算法/超参
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

    for batch in loader:        # each batch = 8 window samples (shuffled), default collate stacks them
                                # batch = {'gene' : [8, 100, 785], 'coords' : [8, 100, 2], 'slide_id' : [8 str]}
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
        best_ckpt = None          # best weights kept in RAM; written to disk once after the loop
        loss_history = []

        for epoch in range(cfg['epochs']):
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)  # 
            lr = get_lr(optimizer)   # Stage1 lr= 1e-4
            if epoch % cfg['log_every'] == 0:    # log_every=1 → every epoch
                print(f"epoch={epoch:3d} | train_loss={avg_loss:.4f} | lr={lr:.2e}")
            
            # stash best in RAM (judgement unchanged); disk write happens once, after the loop
            if avg_loss < best_loss:      # overwrite best_ckpt only when loss hits a new low
                best_loss = avg_loss
                best_ckpt = {
                    'epoch': epoch,
                    # .clone() is essential: without it later epochs mutate these tensors in place
                    'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'loss': avg_loss,
                    'cfg': cfg,  # whole hyperparam dict saved into ckpt
                }


            loss_history.append({'epoch': epoch, 'loss': avg_loss, 'lr': lr})
            with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f, indent=2)

        if best_ckpt is not None:
            torch.save(best_ckpt, os.path.join(save_dir, 'best.pt'))   # the ONLY disk write per fold
        print(f"=== Fold {fold} done. Best loss = {best_loss:.4f} ===")


if __name__ == "__main__":
    main()

# ============================================================================
## Example Flow
# consts: B=8 windows | M=100 spots/window | spot dim=785 / image dim=2560 | mask_ratio 0.5 (keep 50, mask 50)
# ============================================================================
#
#  yaml ──► cfg dict ──► 6 folds (A4,A5,A6,B1,B2,B3), each trained from scratch, own weights
#                              │
#   ┌──────────────────────────┘  each fold gets 3 things:
#   │   loader : feeds data (8 windows at a time)
#   │   model  : Stage1Model (only masked_encoder is trainable)
#   │   optim  : Adam
#   │
#   └─► train 10 epochs; each epoch runs through every batch:
#
#        For ONE batch ↓↓↓
#
#        loader yields ──► gene  [8, 100, 785]    # 8 windows, 100 spots each, 785 genes/spot
#                          coords[8, 100, 2]      # (x,y) grid coord per spot
#                            │
#        model(batch) ───────┤  branch='spot' -> just take gene expression as the feature x
#                            │  (image branch: patches -> Virchow2 -> [8,100,2560], then x)
#                            ▼
#        masked_encoder(x, coords)  ── MAE does 3 things ──
#            target = copy of x                  [8,100,785]   # reconstruction goal (the answer)
#            (1) drop 50 spots, encoder sees only the kept 50   [8, 50,785]
#            (2) decoder fills the 50 gaps back -> full 100     [8,100,785]   # pred (the guess)
#            (3) loss = compare ONLY the 50 masked spots: (pred-target)^2  -> one number
#                            │
#                            ▼  -> loss (scalar)
#        zero_grad -> backward -> step    # clear grads -> compute grads -> update masked_encoder
#
#        (8 windows per batch, one update per batch; all batches done = 1 epoch -> avg_loss)
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