import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.masked_encoder import MaskedEncoder
from models.modules import ImageEncoder, ImageEncoder_Resnet, ImageEncoder_Virchow2, ProjectionHead


class ContrastiveModel(nn.Module):
    """
    Stage 2 CLIP-style contrastive model.
    Mirrors baseline mclSTExp_Attention (Baseline_mclSTExp_code/model.py:201-247),
    except both encoders use Stage 1 MaskedEncoder (loaded + frozen by engine, not here)
    and the image branch has an extra Transformer on top of the CNN backbone.
    Only the two ProjectionHeads (engine sets requires_grad).
    """
    def __init__(self, cfg):
        super().__init__()
        # spot encoder: Stage1 MaskedEncoder reused, in=785 gene-dim; frozen by engine, .encode() only (no mask/decoder)
        self.spot_encoder = MaskedEncoder(          ## 
            dim=cfg['spot_dim'],
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

        if cfg['encoder_name'] == 'densenet121':
            self.image_cnn = ImageEncoder()         
        elif cfg['encoder_name'] == 'resnet50':
            self.image_cnn = ImageEncoder_Resnet()  
        elif cfg['encoder_name'] == 'virchow2':
            self.image_cnn = ImageEncoder_Virchow2(image_agg=cfg['image_agg'])  
            # [N,3,224,224] -> [N,1280] or [N,2560](concat);  frozen by engine
        else:
            raise ValueError(f"Unknown encoder_name: {cfg['encoder_name']}")
        
        ## image_encoder is the masked_encoder for image branch.
        self.image_encoder = MaskedEncoder(        ## 
            dim=cfg['image_dim'], 
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

        # 
        _proj_kw = dict(
            projection_dim=cfg['projection_dim'],            # 256
            dropout=cfg.get('proj_dropout', 0.0),            # 0
            hidden_dim=cfg.get('proj_hidden_dim', None),     # 1024
            n_extra_layers=cfg.get('proj_extra_layers', 0),  # 1
        )
        self.image_projection = ProjectionHead(embedding_dim=cfg['image_dim'], **_proj_kw)   # 1280 -> 256 shared embedding; TRAINABLE
        self.spot_projection = ProjectionHead(embedding_dim=cfg['spot_dim'], **_proj_kw)     # 785  -> 256 shared embedding; TRAINABLE

        self.temperature = cfg['temperature']

        # learnable logit scale over L2-normalized embeddings,
        # init = log(1/0.07) ≈ 2.66, the standard CLIP temperature.
        self.normalize_clip = bool(cfg.get('normalize_clip', False))
        if self.normalize_clip:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))  ## CLIP-style scale parameter. 

    def forward(self, batch):
        """
        Input batch (all-shuffle: B spots randomly mixed across slides; B = batch_size):
          'image':      [B, 3, 224, 224]      H&E patch per spot
          'expression': [B, 785]              lib-norm + log1p HVG per spot
          'position':   [B, 2]                integer (x, y) grid coords
        Output:
          loss            symmetric CLIP loss (averaged over spots/images)
          spots_loss      detached, gene→image cross-entropy (logging only)
          images_loss     detached, image→gene cross-entropy (logging only)
        """

        coords = batch["position"].long().unsqueeze(dim=0)        # [1, B, 2]

        # ===== Image path =====
        image_features = self.image_cnn(batch["image"])           # [B, image_dim]
        image_features = image_features.unsqueeze(dim=0)          # [1, B, image_dim], add slide-batch dim for transformer
        image_features = self.image_encoder.encode(image_features, coords)  # [1, B, image_dim]  # Here image_encoder is masked_encoder for image。
        image_features = image_features.squeeze(dim=0)            # [B, image_dim]
        image_embeddings = self.image_projection(image_features)  # [B, 256], baseline L228

        # ===== Spot path =====
        spot_feature = batch["expression"]                        # [B, 785], 
        spot_features = spot_feature.unsqueeze(dim=0)             # [1, B, 785]
        spot_embeddings = self.spot_encoder.encode(spot_features, coords)  # [1, B, 785], replaces baseline L238 attn-block stack
        spot_embeddings = self.spot_projection(spot_embeddings)   # [1, B, 256]
        spot_embeddings = spot_embeddings.squeeze(dim=0)          # [B, 256]

        # ===== Contrastive loss =====
        if self.normalize_clip:
            # CLIP logits : L2-normalize -> cosine x learnable logit_scale.
            spot_n  = spot_embeddings  / spot_embeddings.norm(dim=1, keepdim=True)  
            image_n = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  
            logit_scale = self.logit_scale.exp()                                     
            cos_smi = logit_scale * (spot_n @ image_n.T)                             
        else:
            # Baseline : raw dot / temperature.
            cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature   # [N,N] sim matrix; diagonal = matched (spot_i, image_i)

        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1], device=cos_smi.device)  # [N,N] identity: target = each spot matches its own image
        spots_loss = F.cross_entropy(cos_smi, label)          # gene->image CE: each spot picks its matching image
        images_loss = F.cross_entropy(cos_smi.T, label.T)     # image->gene CE: each image picks its matching spot
        loss = (images_loss + spots_loss) / 2.0                 # symmetric CLIP loss: average of both directions, scalar
        return loss.mean(), spots_loss.detach(), images_loss.detach()

## Example Flow  ——  ContrastiveModel.forward (Stage2 CLIP, all-shuffle)
## B spots randomly mixed across slides; image_dim=1280 (Virchow2), spot_dim=785, proj=256
#
#  INPUT batch:
#    image      [B, 3, 224, 224]   # one H&E patch per spot
#    expression [B, 785]           # lib-norm+log1p HVG per spot
#    position   [B, 2]             # (x,y) grid coords
#
#  coords = position.long().unsqueeze(0)              [B,2] -> [1,B,2]   # shared by both towers
#
#  ── Image ───────────────────────────────────────────────
#    image_cnn (Virchow2, frozen)        [B,3,224,224] -> [B,1280]
#    unsqueeze(0)                         [B,1280]      -> [1,B,1280]
#    image_encoder.encode(.,coords)       [1,B,1280]    -> [1,B,1280]   # +PE, self-attn, NO mask/decoder
#    squeeze(0)                           [1,B,1280]    -> [B,1280]
#    image_projection (TRAINABLE)         [B,1280]      -> [B,256]
#
#  ── Spot ────────────────────────────────────────────────
#    expression                           [B,785]
#    unsqueeze(0)                         [B,785]       -> [1,B,785]
#    spot_encoder.encode(.,coords)        [1,B,785]     -> [1,B,785]    # +PE, self-attn, NO mask/decoder
#    spot_projection (TRAINABLE)          [1,B,785]     -> [1,B,256]
#    squeeze(0)                           [1,B,256]     -> [B,256]
#
#  ── Contrastive loss ──────────────────────────────────────────
#    cos_smi = scale * (spot_n @ image_n.T)            [B,256]@[256,B] -> [B,B]
#              (normalize_clip: L2-norm -> cosine x learnable logit_scale)
#    label   = torch.eye(B)                                  # diagonal = matched (spot_i, image_i)
#    loss    = ( CE(cos_smi, label) + CE(cos_smi.T, label.T) ) / 2     # symmetric CLIP
#
#  OUTPUT: loss (scalar), spots_loss, images_loss (detached, logging only)
#  trained params: image_projection + spot_projection (+ logit_scale); all 3 encoders frozen