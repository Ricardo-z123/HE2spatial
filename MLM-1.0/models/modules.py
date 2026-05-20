import timm
import torch
from torch import nn, einsum
from einops import rearrange
import torchvision.models as models
import torch.nn.functional as F


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
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # split last dim into Q, K, V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split heads per spot
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # spot-to-spot attention scores
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # weighted spot aggregation
        out = rearrange(out, 'b h n d -> b n (h d)')  # merge heads back per spot
        return self.to_out(out)


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
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


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


class ProjectionHead(nn.Module):
    """
    Stage-2 contrastive MLP head: Linear → GELU → Linear → Dropout → residual → LayerNorm.
    Maps encoder output to a shared 256-d contrastive space (used by both spot and image branches).
    Input:  x [B, embedding_dim]    e.g. [128, 785] spot encoder output, or [128, 256] image encoder output
    Output: x [B, projection_dim]   e.g. [128, 256] embedding in shared contrastive space
    """
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
        # [B, projection_dim] e.g. [128, 256] L2-normalize-ready contrastive embedding
