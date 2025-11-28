import torch
from torch import nn, einsum
import timm
from einops import rearrange
import config as CFG
from timm.models import load_checkpoint


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name,
        #     pretrained=False,         #
        #     num_classes=0,
        #     global_pool="avg",
        #     checkpoint_path='pytorch_model.bin'  # local weight files
        # )

        self.model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        load_checkpoint(self.model, 'pytorch_model.bin', strict=False)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


# class ImageEncoder_resnet50(nn.Module):
#     """
#     Encode images to a fixed size vector
#     """
#
#     def __init__(
#             self, model_name='resnet50', pretrained=CFG.pretrained, trainable=CFG.trainable
#     ):
#         super().__init__()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=0, global_pool="avg"
#         )
#         for p in self.model.parameters():
#             p.requires_grad = trainable
#
#     def forward(self, x):
#         return self.model(x)


class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name='resnet101', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name='resnet152', pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name="vit_base_patch32_224", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name="vit_base_patch32_224_clip_laion2b", pretrained=True, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name="vit_large_patch32_224_in21k", pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',


# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())

#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]

## ============================== Transformer block and basic position encoding ============================== ##

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Feed forward network with GELU activiation
    """

    def __init__(self, dim, hidden_dim, dropout=CFG.dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=CFG.dropout):
        super().__init__()
        inner_dim = dim_head * heads
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
        """
        Args:
             x: (batch_size, seq_len, dim) or (batch_size, dim)
        Returns:
            output: same shape as input
        """
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, dim)
            squeeze_output = True

        b, n, _, h = *x.shape, self.heads
        # generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # calculate attention scores
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if squeeze_output:
            out = out.squeeze(1)

        return out


class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=CFG.dropout):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class SpotEncoder_v1(nn.Module):
    """
    SpotEncoder_v1
    """

    def __init__(self, spot_dim, max_position=1024, num_layer=1,
                 heads=8, dim_head=64, mlp_dim=None, dropout=0.):
        super().__init__()
        self.spot_dim = spot_dim
        self.x_embed = nn.Embedding(max_position, spot_dim)
        self.y_embed = nn.Embedding(max_position, spot_dim)

        self.transformer = nn.Sequential(
            *[attn_block(spot_dim,
                         heads=heads,
                         dim_head=dim_head,
                         mlp_dim=mlp_dim or spot_dim,
                         dropout=dropout) for _ in range(num_layer)]
        )

    def forward(self, spot_features, positions):
        """
        Args:
            spot_features: (B, num_spots, spot_dim)
            positions: (B, num_spots, 2)  (x, y) for each spot
        Returns:
            spot_embeddings: (B, num_spots, spot_dim)
        """
        # B, N, D = spot_features.shape
        device = spot_features.device

        # 位置 embedding
        x = positions[:, 0].long().to(device)
        y = positions[:, 1].long().to(device)
        centers_x = self.x_embed(x)  # (B, N, spot_dim)
        centers_y = self.y_embed(y)  # (B, N, spot_dim)

        # 将位置 embedding 与原始特征相加
        spot_features = spot_features + centers_x + centers_y

        # Transformer 输入
        spot_embeddings = self.transformer(spot_features)  # (B, N, spot_dim)
        return spot_embeddings


import torch
import torch.nn as nn
from einops import rearrange


class SpotEncoder_RoPE(nn.Module):
    """
    SpotEncoder RoPE
    """

    def __init__(self, spot_dim, num_layer=1, heads=8, dim_head=64, mlp_dim=None, dropout=0.):
        super().__init__()
        self.spot_dim = spot_dim
        self.inner_dim = spot_dim
        self.heads = heads
        self.dim_head = dim_head

        self.transformer = nn.Sequential(
            *[attn_block(spot_dim, heads=heads, dim_head=dim_head,
                         mlp_dim=mlp_dim or spot_dim, dropout=dropout)
              for _ in range(num_layer)]
        )

    def make_rope(self, batch_size, seq_len, outputs):
        """
        Rotary positional embedding
        outputs: (B, seq_len, inner_dim*2)
        返回 qw, kw: (B, seq_len, inner_dim)
        """
        qw = outputs
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim, device=outputs.device)

        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)

        qw = qw * cos_pos + qw2 * sin_pos

        return qw

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, spot_features):
        """
        spot_features: (B, seq_len, spot_dim*2)
        return: (B, seq_len, spot_dim)
        """
        # 将 RoPE 融合到 spot_features
        spot_features = self.make_rope(spot_features.size(0), spot_features.size(1),
                                       spot_features)  # (B, seq_len, spot_dim)

        spot_embeddings = self.transformer(spot_features)
        return spot_embeddings


## ========================================== End of Transformer block ========================================== ##

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
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