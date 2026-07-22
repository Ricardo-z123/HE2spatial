import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.masked_encoder import MaskedEncoder
from models.modules import ImageEncoder, ImageEncoder_Resnet, ImageEncoder_Virchow2, ProjectionHead


class ContrastiveModel(nn.Module):
    """
    Stage 2 CLIP-style contrastive model, WINDOW mode.
    Mirrors baseline mclSTExp_Attention (Baseline_mclSTExp_code/model.py:201-247),
    except both encoders use Stage 1 MaskedEncoder (loaded + frozen by engine, not here)
    and the image branch has an extra Transformer on top of the CNN backbone.
    Only the two ProjectionHeads train (engine sets requires_grad).

    Input is [B, M, ...] (B windows x M spatial-neighbour spots), NOT a flat [B, ...] spot batch.
    Attention runs WITHIN each window (over the M dim; B is a true parallel batch, windows do NOT
    attend to each other), so every spot's embedding is contaminated only by its own slide's
    spatial neighbours -- the input distribution the frozen Stage1 encoder was trained on.
    The contrastive loss then pools all B*M spots (negatives still span windows / slides):
    Survey §4 -- "produce per-window, consume mixed".

    Three cfg switches shape the contrastive loss (5 valid combinations):
      normalize_clip      true  -> logits = logit_scale.exp() * cosine(L2-normalised)
                          false -> logits = raw dot / temperature   (baseline / BLEEP)
      duplicate_spot_mode 'none'     -> label = torch.eye
                          'positive' -> every slot of the same spot is a positive, row-normalised.
                                        A strict generalisation: with no duplicates the row is
                                        1/1 = 1, element-identical to eye.
      soft_labels         true  -> BLEEP-style targets from the embeddings' own similarity.
                                  REQUIRES normalize_clip=false and duplicate_spot_mode='none'
                                  (asserted below), so it contributes exactly 1 combination.
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
        
        torch.manual_seed(cfg['seed'])

        ## image_encoder is the masked_encoder for image branch.
        self.image_encoder = MaskedEncoder(        ## 
            dim=cfg['image_dim'],           # 2560 (Virchow2 concat) 
            enc_depth=cfg['enc_depth'],     # 4
            dec_depth=cfg['dec_depth'],     # 2
            num_heads=cfg['num_heads'],     # 8
            dim_head=cfg['dim_head'],       # 64
            mask_ratio=cfg['mask_ratio'],   # now fixed 0.5, originally 0.1. hyperparam.
        )

        # 
        _proj_kw = dict(
            projection_dim=cfg['projection_dim'],            # 256
            dropout=cfg.get('proj_dropout', 0.0),            # 0
            hidden_dim=cfg.get('proj_hidden_dim', None),     # 512, hidden dim = 512 is now the best option, may vary after following experiments.
            n_extra_layers=cfg.get('proj_extra_layers', 0),  # 1
        )
        self.image_projection = ProjectionHead(embedding_dim=cfg['image_dim'], **_proj_kw)   # 2560 -> 256 shared embedding; TRAINABLE
        self.spot_projection = ProjectionHead(embedding_dim=cfg['spot_dim'], **_proj_kw)     # 785  -> 256 shared embedding; TRAINABLE

        self.temperature = cfg['temperature']     # 1.0

        # learnable logit scale over L2-normalized embeddings,
        # init = log(1/0.07) ≈ 2.66, the standard CLIP temperature.
        self.normalize_clip = bool(cfg.get('normalize_clip', False))
        if self.normalize_clip:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))  ## CLIP-style scale parameter.

        # How to label the SAME spot appearing at several slots of the flattened [B*M] batch:
        #   'none'     -> torch.eye, i.e. those slots are negatives of each other (original behaviour)
        #   'positive' -> all slots of one spot are positives, row-normalised (see forward)

        #   Example : 3 slots, slot0 & slot2 are the SAME spot A   ->  spot_id = [0, 1, 0]
        #
        #   'none' = torch.eye(3)              'positive', row-normalised
        #          A0    B1    A2                        A0    B1    A2
        #   A0 |  1.0   0.0   0.0               A0 |    0.5   0.0   0.5
        #   B1 |  0.0   1.0   0.0               B1 |    0.0   1.0   0.0   <- no duplicate: identical
        #   A2 |  0.0   0.0   1.0               A2 |    0.5   0.0   0.5
        #   A0 vs A2 = negatives  ✗             A0 & A2 share the weight            
        self.duplicate_spot_mode = str(cfg.get('duplicate_spot_mode', 'none'))
        assert self.duplicate_spot_mode in ('none', 'positive'), \
            "duplicate_spot_mode must be 'none' or 'positive', got %s" % self.duplicate_spot_mode
        

        # BLEEP-style contrastive loss: 
        self.soft_labels = bool(cfg.get('soft_labels', False))
        if self.soft_labels:
            assert not self.normalize_clip, \
                "soft_labels requires normalize_clip=false (BLEEP uses raw dot / temperature)"
            assert self.duplicate_spot_mode == 'none', \
                "soft_labels builds its own targets; set duplicate_spot_mode=none"

    def forward(self, batch):
        """
        Input batch (window mode; B windows, M = window_size**2 spots each; real run B=8, M=100):
          'image':      [B, M, 3, 224, 224]   H&E patch per spot
          'expression': [B, M, 785]           lib-norm + log1p HVG per spot
          'position':   [B, M, 2]             (x, y) grid coords per spot
          'spot_id':    [B, M]                globally unique spot id (only read when
                                              duplicate_spot_mode == 'positive')
        Output:
          loss            symmetric CLIP loss (averaged over spots/images)
          spots_loss      detached, gene→image cross-entropy 
          images_loss     detached, image→gene cross-entropy 
        """
        coords = batch["position"].long()                # [B, M, 2]
        image = batch["image"]                           # [B, M, 3, 224, 224]
        expression = batch["expression"]                 # [B, M, 785]
        B, M = image.shape[0], image.shape[1]

        # ===== image path (attention within each window, over the M dim) =====
        img = image.reshape(B * M, *image.shape[2:])                  # [B*M, 3, 224, 224]
        # Chunk by 256, matching Stage1's Virchow2 batch (stage1_pretrain.py). cuBLAS switches kernel
        # above ~256, so feeding all B*M=800 at once returns features that differ from Stage1's by
        # ~2.6e-2 -- the SAME spot would then carry one identity in Stage1 and another in Stage2.
        # Virchow2 is per-patch independent, so chunking is mathematically equivalent to one pass:
        # only the fp16 rounding moves, and it now agrees with Stage1.
        feats = []
        for s in range(0, img.shape[0], 256):   # for s in range(0, 800, 256)
            feats.append(self.image_cnn(img[s:s + 256]))              # [<=256, image_dim] per chunk
        img_feat = torch.cat(feats, dim=0).reshape(B, M, -1)          # [B, M, image_dim]  (Virchow2, frozen)
        img_feat = self.image_encoder.encode(img_feat, coords)       # [B, M, image_dim]  +PE, within-window self-attn
        image_embeddings = self.image_projection(img_feat.reshape(B * M, -1))   # [B*M, 256] ; [800, 256]

        # ===== spot path =====
        spot_feat = self.spot_encoder.encode(expression, coords)     # [B, M, 785] -> [B, M, 785]
        spot_embeddings = self.spot_projection(spot_feat.reshape(B * M, -1))    # [B*M, 256] ; [800, 256]

        # ===== contrastive loss: N = B*M = 800 spots pooled across all windows =====
        if self.normalize_clip:
            # CLIP logits : L2-normalize -> cosine x learnable logit_scale.
            spot_n = spot_embeddings / spot_embeddings.norm(dim=1, keepdim=True)
            image_n = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            cos_smi = logit_scale * (spot_n @ image_n.T)             # [B*M, B*M]
        else:
            # Baseline : raw dot / temperature.
            cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature

        N = cos_smi.shape[0]  # N = 800
        if self.soft_labels:
            # BLEEP-style targets: softmax over the AVERAGED self-similarities, i.e. the batch tells
            # the model how similar the pairs really are instead of asserting a hard 1-of-N identity.
            # Raw dot products here (NOT the L2-normalised ones) -- that is why normalize_clip must
            # be false, so logits and targets live on the same scale.
            images_similarity = image_embeddings @ image_embeddings.T   # [N, N] image-image dot
            spots_similarity = spot_embeddings @ spot_embeddings.T      # [N, N] spot-spot dot
            label = F.softmax(
                (images_similarity + spots_similarity) / 2 / self.temperature, dim=-1
            )                                                           # [N, N] soft targets, rows sum to 1
        elif self.duplicate_spot_mode == 'positive':
            # Windows overlap heavily (stride 2 -> a spot sits in ~25 of them, measured on A4), so after
            # the flatten ONE spot owns several slots. torch.eye would make those slots negatives of each
            # other, i.e. push a spot away from its own image. Treat them as positives instead.

            sid = batch['spot_id'].reshape(N).to(cos_smi.device)  # reshape [B*M] = 800 spot_id, and move to GPU.
            label = (sid[:, None] == sid[None, :]).float()           # [N, N] 1 where same spot
            # Row-normalise: without it a spot occupying k slots would carry k times the loss weight
            label = label / label.sum(dim=1, keepdim=True)
            
        else:
            label = torch.eye(N, N, device=cos_smi.devipce)          # positive = each spot matches its own image
        spots_loss = F.cross_entroy(cos_smi, label)          # gene->image CE: each spot picks its matching image
        images_loss = F.cross_entropy(cos_smi.T, label.T)     # image->gene CE: each image picks its matching spot
        loss = (images_loss + spots_loss) / 2.0                 # symmetric CLIP loss: average of both directions, scalar
        return loss.mean(), spots_loss.detach(), images_loss.detach()


# L11-36    class + docstring（含 3 个开关说明）
# ────────── __init__ ──────────
# L37-47    ① spot_encoder      MaskedEncoder, dim=785
# L49-64    ② image_cnn         Virchow2 
# L66-74    ③ image_encoder     MaskedEncoder, dim=2560
# L76-84    ④ 两个 ProjectionHead   ← 唯一可训练的部分
# L86-109   ⑤ temperature / logit_scale / duplicate_spot_mode / soft_labels
# ────────── forward ──────────
# L111-135  ① 取 batch + 图像路径
# L136-140  ② spot 路径
# L142-152  ③ 相似度矩阵
# L154-175  ④ label + loss    ← ⭐ 最难

## Example Flow  ——  ContrastiveModel.forward (Stage2 CLIP, WINDOW mode)
## B=8 windows x M=100 spatial-neighbour spots = 800 spots; image_dim=2560 (Virchow2 concat),
## spot_dim=785, proj=256, duplicate_spot_mode='positive'
#
#  INPUT batch:
#    image      [8, 100, 3, 224, 224]   # one H&E patch per spot
#    expression [8, 100, 785]           # lib-norm+log1p HVG per spot
#    position   [8, 100, 2]             # (x,y) grid coords
#    spot_id    [8, 100]                # global spot id (slide_idx*100000 + row)
#
#  coords = position.long()                           [8,100,2]   # shared by both towers, NO unsqueeze
#
#  ── Image ───────────────────────────────────────────────
#    reshape                              [8,100,3,224,224] -> [800,3,224,224]
#    image_cnn (Virchow2, frozen)         chunked by 256 -> 4 chunks -> [800,2560]
#    reshape                              [800,2560]    -> [8,100,2560]
#    image_encoder.encode(.,coords)       [8,100,2560]  -> [8,100,2560]  # +PE, attn WITHIN each window
#    reshape + image_projection           [800,2560]    -> [800,256]     # TRAINABLE
#
#  ── Spot ────────────────────────────────────────────────
#    spot_encoder.encode(expr,coords)     [8,100,785]   -> [8,100,785]   # +PE, attn WITHIN each window
#    reshape + spot_projection            [800,785]     -> [800,256]     # TRAINABLE
#
#  ── Contrastive loss (windows are POOLED here; negatives span windows/slides) ──
#    cos_smi = scale * (spot_n @ image_n.T)            [800,256]@[256,800] -> [800,800]
#              (normalize_clip=true -> L2-norm -> cosine x learnable logit_scale;
#               the else-branch temperature is never used with the shipped config)
#    label:  soft_labels=true -> softmax((img@img.T + spot@spot.T)/2/temperature)   [800,800]
#            'positive'       -> (spot_id[:,None]==spot_id[None,:]).float(), row-normalised
#                                (windows overlap -> ONE spot occupies ~several of the 800 slots)
#            'none'           -> torch.eye(800)
#    loss    = ( CE(cos_smi, label) + CE(cos_smi.T, label.T) ) / 2     # symmetric CLIP
#
#  OUTPUT: loss (scalar), spots_loss, images_loss (detached, logging only)
#  trained params: image_projection + spot_projection (+ logit_scale); all 3 encoders frozen
#
#  NOTE evel.py builds this same class but never calls forward -- it drives .image_cnn /
#  .image_encoder.encode / .spot_encoder.encode / the two projections directly.