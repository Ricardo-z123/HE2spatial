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
        self.spot_encoder = MaskedEncoder(
            in_dim=cfg['spot_in_dim'],
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

        if cfg['encoder_name'] == 'densenet121':
            self.image_cnn = ImageEncoder()         # baseline L209 — outputs [N, 1024]
        elif cfg['encoder_name'] == 'resnet50':
            self.image_cnn = ImageEncoder_Resnet()  # baseline L207 — outputs [N, 2048]
        elif cfg['encoder_name'] == 'virchow2':
            self.image_cnn = ImageEncoder_Virchow2()  # outputs [N, 1280]
        else:
            raise ValueError(f"Unknown encoder_name: {cfg['encoder_name']}")
        self.image_encoder = MaskedEncoder(
            in_dim=cfg['image_in_dim'],
            enc_depth=cfg['enc_depth'],
            dec_depth=cfg['dec_depth'],
            num_heads=cfg['num_heads'],
            dim_head=cfg['dim_head'],
            mask_ratio=cfg['mask_ratio'],
        )

        self.image_projection = ProjectionHead(
            embedding_dim=cfg['image_in_dim'],
            projection_dim=cfg['projection_dim'],
        )
        self.spot_projection = ProjectionHead(
            embedding_dim=cfg['spot_in_dim'],
            projection_dim=cfg['projection_dim'],
        )

        self.temperature = cfg['temperature']

    def forward(self, batch):
        """
        Input batch (single-slide batch from SlideBatchSampler, all N spots same tissue):
          'image':      [N, 3, 224, 224]   FloatTensor   H&E patch per spot
          'expression': [N, 785]           FloatTensor   lib-norm + log1p HVG per spot
          'position':   [N, 2]             FloatTensor   integer (x, y) grid coords
        Output:
          loss         scalar tensor   symmetric CLIP loss (averaged over spots/images)
          spots_loss   scalar tensor   detached, gene→image cross-entropy (logging only)
          images_loss  scalar tensor   detached, image→gene cross-entropy (logging only)
        """
        # ===== Image path =====
        image_features = self.image_cnn(batch["image"])           # [N, image_in_dim], baseline L226
        image_features = image_features.unsqueeze(dim=0)          # [1, N, image_in_dim], add slide-batch dim for transformer
        coords = batch["position"].long().unsqueeze(dim=0)        # [1, N, 2]
        image_features = self.image_encoder.encode(image_features, coords)  # [1, N, image_in_dim]
        image_features = image_features.squeeze(dim=0)            # [N, image_in_dim]
        image_embeddings = self.image_projection(image_features)  # [N, 256], baseline L228

        # ===== Spot path =====
        spot_feature = batch["expression"]                        # [N, 785], baseline L227
        spot_features = spot_feature.unsqueeze(dim=0)             # [1, N, 785], baseline L236
        spot_embeddings = self.spot_encoder.encode(spot_features, coords)  # [1, N, 785], replaces baseline L238 attn-block stack
        spot_embeddings = self.spot_projection(spot_embeddings)   # [1, N, 256], baseline L239
        spot_embeddings = spot_embeddings.squeeze(dim=0)          # [N, 256], baseline L240

        # ===== Contrastive loss (verbatim from baseline L242-247) =====
        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1], device=cos_smi.device)
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean(), spots_loss.detach(), images_loss.detach()
