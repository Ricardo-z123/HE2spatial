import os

# Virchow2 weights already live in /root/autodl-tmp/huggingface/hub. Force offline mode so timm
# never round-trips to huggingface.co when building the model -- otherwise every fold (and, in raw
# mode, every dataloader worker) stalls on a network check.
# MUST be set BEFORE the timm import below: huggingface_hub reads this env var at import time.
# If you ever need to download new weights, comment this line out.
os.environ["HF_HUB_OFFLINE"] = "1"

import timm
import torch
from torch import nn, einsum
from einops import rearrange
import torchvision.models as models
import torch.nn.functional as F
from timm.layers import SwiGLUPacked


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Standard Transformer FFN: Linear → GELU → Dropout → Linear → Dropout.
    Input:  x [N, L, D]   e.g. [1, 300, 785] slide spot features
    Output: x [N, L, D]   same shape, channel-mixed per spot
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention over the spot dimension (spots within a slide attend to each other).
    Input:  x [N, L, D]   e.g. [1, 300, 785] slide spot features
    Output: x [N, L, D]   same shape, each spot mixed with other spots on the same slide
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 64*8 = 512
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5      # 1/sqrt(64)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # Linear(785 -> 1536)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # Linear(512 -> 785)
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # b=N batch, n=L spots, h=heads(8); x is [N, L, D]
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # x[N,L,D] -> [N,L,1536] -> q,k,v each [N,L,inner_dim=512]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split heads -> each [N,heads,L,dim_head=64]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # logits = q@kᵀ*scale [N,heads,L,L] attention scores (spot_i vs spot_j)
        attn = self.attend(dots)  # softmax over keys (last dim j) -> [N,heads,L,L], each row sums to 1
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # attn@v weighted values -> [N,heads,L,dim_head=64]
        out = rearrange(out, 'b h n d -> b n (h d)')  # merge heads -> [N,L,inner_dim=512]
        return self.to_out(out)  # Linear 512->D -> [N,L,D] (or Identity if project_out=False)


class attn_block(nn.Module):
    """
    Pre-norm Transformer block: attention + FFN, both with residual.
    Input:  x [N, L, D]   e.g. [1, 300, 785] slide spot sequence
    Output: x [N, L, D]   same shape, refined spot representations
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x  # [N,L,D] -> LN+attention, residual -> [N,L,D]
        x = self.ff(x) + x  # [N,L,D] -> LN+FFN, residual -> [N,L,D]
        return x  # [N,L,D] shape unchanged through the block


class ImageEncoder(nn.Module):
    """
    Frozen DenseNet121 backbone.
    Used for offline pre-extraction of spot patch features (image branch input).
    Input:  x [B, 3, 224, 224]   e.g. [300, 3, 224, 224] H&E patches of spots in one slide
    Output: x [B, 1024]          e.g. [300, 1024] DenseNet features per spot
    """
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # global pool over spatial dims of patch
        x = x.view(x.size(0), -1)  # flatten to per-spot 1024-d feature
        return x
        # [B, 1024] frozen DenseNet feature per spot


class ImageEncoder_Resnet(nn.Module):
    """
    Frozen ResNet50 backbone. Alternative image backbone.
    Input:  x [B, 3, 224, 224]   e.g. [300, 3, 224, 224] H&E patches of spots in one slide
    Output: x [B, 2048]          e.g. [300, 2048] ResNet features per spot
    """
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # global pool over spatial dims of patch
        x = x.view(x.size(0), -1)  # flatten to per-spot 2048-d feature
        return x
        # [B, 2048] frozen ResNet50 feature per spot


class ImageEncoder_Virchow2(nn.Module):
    """
    Frozen Virchow2 backbone
    Turns each H&E spot patch into a fixed feature vector; weights are NEVER trained.

    Input:  x [B, 3, 224, 224]   B H&E spot patches (RGB), after ToTensor(), values in [0, 1].
                                 B = number of individual spot patches in this batch
                                 (NOT windows / slides; the window structure is flattened upstream).
    Output: x [B, D]             one D-d feature per spot. image_agg selects how the 261
                                 Virchow2 tokens collapse:
                                 'patch_mean'  = mean of the 256 patch tokens   -> D=1280;
                                 'class_token' = the CLS token (token 0)         -> D=1280;
                                 'concat'      = cat(CLS, patch-mean)     -> D=2560.

    NOTE: downstream dim MUST match D — concat -> set image_dim/dim=2560; else 1280.
    """
    def __init__(self, cache_dir="/root/autodl-tmp/huggingface/hub", image_agg='patch_mean'):
        super().__init__()
        self.image_agg = image_agg               # 'patch_mean'(原始) | 'class_token'(E47: Virchow2 CLS token)
        self.model = timm.create_model(          # load the pretrained Virchow2 ViT from the HuggingFace hub
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,              # Virchow2-specific MLP + activation, required for correct init
            act_layer=torch.nn.SiLU,
            cache_dir=cache_dir,                 # local dir to cache the downloaded weights
        )
        self.model.eval()                        # eval mode: disable dropout, use running stats

        for p in self.model.parameters():
            p.requires_grad = False              # FREEZE: no gradients, optimizer never updates it (fixed extractor)

        # ImageNet mean/std stored as non-learnable buffers: they auto-move with .to(device)/.half()
        # and are saved in state_dict. Needed here because the upstream dataset does ToTensor only (no Normalize).
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),  # per-RGB-channel mean, shape [1,3,1,1] to broadcast over [B,3,H,W]
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),  # per-RGB-channel std, shape [1,3,1,1]
        )

    def forward(self, x):                        # x [B, 3, 224, 224] H&E spot patches, values in [0, 1]
        self.model.eval()                        # re-assert eval every call (guard against an external .train())
        x = (x - self.mean) / self.std           # ImageNet normalize per channel -> [B,3,224,224], values now ~[-2.2, 2.7]

        with torch.inference_mode(), torch.autocast(   # no-grad + mixed precision: faster & lighter (model is frozen)
            device_type=x.device.type,
            dtype=torch.float16,                 # run Virchow2 in fp16 ...
            enabled=x.is_cuda,                   # ... but only on CUDA; on CPU stay fp32
        ):
            output = self.model(x)               # ViT forward: [B,3,224,224] -> [B, 261, 1280]
                                                 # 261 tokens = 1 class token + 4 register tokens + 256 patch tokens

        if self.image_agg == 'class_token':      # Virchow2 CLS token
            x = output[:, 0]                     # [B, 1280] token 0 (class token)
        elif self.image_agg == 'concat':         # cat(CLS, patch-mean)
            x = torch.cat([output[:, 0], output[:, 5:].mean(1)], dim=-1)  # [B, 2560]
        else:                                    # 'patch_mean' (default)
            patch_tokens = output[:, 5:]         # keep only the 256 patch tokens -> [B, 256, 1280]
                                                 # (drop token 0 = class, tokens 1-4 = register tokens)
            x = patch_tokens.mean(1)             # mean-pool over the 256 tokens (dim 1) -> [B, 1280] per-spot feature
        return x      # [B, 1280] (patch_mean/class_token) 或 [B, 2560] (concat) frozen Virchow2 feature


class ProjectionHead(nn.Module):
    """
    Stage-2 contrastive MLP head: Linear → GELU → Linear → Dropout → residual → LayerNorm.
    Maps encoder output to a shared 256-d contrastive space (used by both spot and image branches).
    Input:  x [B, embedding_dim]    e.g. [128, 785] spot encoder output, or [128, 256] image encoder output
    Output: x [B, projection_dim]   e.g. [128, 256] embedding in shared contrastive space
    """
    def __init__(self, embedding_dim, projection_dim, dropout=0., hidden_dim=None, n_extra_layers=0):
        # Defaults (hidden_dim=None -> projection_dim, n_extra_layers=0, dropout=0) reproduce the
        # ORIGINAL head exactly. E27 widens (hidden_dim>proj) / deepens (n_extra_layers>0); E28 sets dropout.
        super().__init__()
        hidden_dim = hidden_dim or projection_dim
        self.projection = nn.Linear(embedding_dim, projection_dim)  # residual base in proj space
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, hidden_dim)            # proj -> hidden (orig: proj->proj)
        self.extra = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_extra_layers)])
        # map hidden back to proj only when needed (keeps orig graph when hidden==proj & no extra)
        self.out = nn.Linear(hidden_dim, projection_dim) if (hidden_dim != projection_dim or n_extra_layers > 0) else None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)   # Linear(785→256)
        h = self.gelu(projected)         # GELU [B, 256]
        h = self.fc(h)                # Linear(256→1024)
        for lin in self.extra:       # 1 extra layer, GELU→Linear(1024→1024) 
            h = lin(self.gelu(h))
        if self.out is not None:
            h = self.out(self.gelu(h))
        h = self.dropout(h)
        x = h + projected
        x = self.layer_norm(x)
        return x
        # [B, projection_dim] e.g. [128, 256] L2-normalize-ready contrastive embedding
