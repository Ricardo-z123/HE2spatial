import torch
from torch import nn

from models.modules import attn_block
from models.pos_embed import PosEmbed


class MaskedEncoder(nn.Module):
    """
    MAE-style masked encoder for one slide. Spot/image branches share this class;
    only in_dim differs:
        spot branch:  in_dim=785  (gene expression)
        image branch: in_dim=1024 (DenseNet features)
    enc_dim = dec_dim = in_dim (no input/output projection for now).
    """
    def __init__(self, in_dim, enc_depth, dec_depth, num_heads,
                 dim_head=64, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.pos_embed = PosEmbed(in_dim)   ## Position embedding
        self.encoder_blocks = nn.ModuleList([
            attn_block(in_dim, heads=num_heads, dim_head=dim_head, mlp_dim=in_dim)
            for _ in range(enc_depth)
        ])
        self.encoder_norm = nn.LayerNorm(in_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)  # learnable placeholder for masked spots
        self.decoder_blocks = nn.ModuleList([
            attn_block(in_dim, heads=num_heads, dim_head=dim_head, mlp_dim=in_dim)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(in_dim)
        self.decoder_pred = nn.Linear(in_dim, in_dim)  # reconstruction head back to spot feature space

    def random_masking(self, x):
        """
        Per-slide random masking by per-sample shuffling (argsort random noise).
        Mask ratio is taken from self.mask_ratio (set in __init__).
        Input:
          x [N, L, D]   e.g. [1, 300, 785] slide spot sequence (post pos_embed)
        Output:
          x_masked    [N, len_keep, D]   e.g. [1, 150, 785] kept spots in shuffled order
          mask        [N, L]             e.g. [1, 300] 0=keep, 1=mask (original spot order)
          ids_restore [N, L]             e.g. [1, 300] restore shuffled→original spot order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small=keep, large=remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # pull kept spots into shuffled order

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # unshuffle mask back to original spot order

        return x_masked, mask, ids_restore
        # [1,len_keep,D] kept spots(shuffled) | [1,L] 0=keep,1=mask | [1,L] restore shuffled→original

    def forward_encoder(self, x, coords):
        """
        Input:
          x      [N, L, in_dim]   e.g. [1, 300, 785] spot features on one slide
          coords [N, L, 2]        e.g. [1, 300, 2]   integer (x, y) coords per spot
        Output:
          x_kept_encoded [N, len_keep, in_dim]   e.g. [1, 150, 785] encoder output, kept spots only (shuffled order)
          mask           [N, L]                  e.g. [1, 300] 0=keep, 1=masked (original spot order)
          ids_restore    [N, L]                  e.g. [1, 300] indices to restore shuffled→original
        """
        x = x + self.pos_embed(coords)  # add learnable spatial PE per spot
        x_kept, mask, ids_restore = self.random_masking(x)  # drop 50% of spots, keep the rest in shuffled order
        for blk in self.encoder_blocks:
            x_kept = blk(x_kept)
        x_kept = self.encoder_norm(x_kept)
        return x_kept, mask, ids_restore
        # [1,len_keep,D] kept-spot encodings (shuffled) | [1,L] 0/1 mask | [1,L] restore index

    def encode(self, x, coords):
        """
        No-masking encoder pass used in Stage 2 contrastive training and inference.
        Adds learnable spatial PE, runs encoder transformer blocks + norm; no decoder.
        Input:
          x      [N, L, in_dim]   e.g. [1, 128, 785] N spots from one slide (Stage 2 SlideBatchSampler)
          coords [N, L, 2]        e.g. [1, 128, 2]   integer (x, y) coords per spot
        Output:
          x [N, L, in_dim]   e.g. [1, 128, 785] encoder features for every spot, original order
        """
        x = x + self.pos_embed(coords)  # add learnable spatial PE per spot
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x
        # [1,L,D] no-mask encoder output, original spot order

    def forward_decoder(self, x, ids_restore, coords):
        """
        Input:
          x           [N, len_keep, in_dim]   e.g. [1, 150, 785] encoder output for kept spots (shuffled)
          ids_restore [N, L]                  e.g. [1, 300] restore index from forward_encoder
          coords      [N, L, 2]               e.g. [1, 300, 2] integer (x, y) coords per spot, original order
        Output:
          pred [N, L, in_dim]   e.g. [1, 300, 785] reconstructed spot features in original slide order
        """
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)  # one mask_token per dropped spot
        x = torch.cat([x, mask_tokens], dim=1)  # append placeholders so length matches full slide L
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle back to original spot order
        x = x + self.pos_embed(coords)  # add spatial PE so mask_tokens know their position
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        return pred
        # [1,L,D] reconstructed spot features in original slide order

    def forward_loss(self, target, pred, mask):
        """
        Input:
          target [N, L, in_dim]   e.g. [1, 300, 785] original spot features (reconstruction target)
          pred   [N, L, in_dim]   e.g. [1, 300, 785] decoder output
          mask   [N, L]           e.g. [1, 300] 0=keep, 1=masked
        Output:
          loss   scalar tensor    MSE averaged over masked spots only
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-spot mean over feature dim → [N, L]
        loss = (loss * mask).sum() / mask.sum()  # average MSE over masked spots only
        return loss
        # scalar mask-weighted MSE on masked spots

    def forward(self, x, coords):
        """
        Input:
          x      [N, L, in_dim]   e.g. [1, 300, 785]
          coords [N, L, 2]        e.g. [1, 300, 2]
        Output:
          loss   scalar tensor
          pred   [N, L, in_dim]   e.g. [1, 300, 785]
          mask   [N, L]           e.g. [1, 300] 0=keep, 1=masked
        """
        target = x.clone()  # snapshot before forward_encoder adds pos_embed
        x_kept_encoded, mask, ids_restore = self.forward_encoder(x, coords)
        pred = self.forward_decoder(x_kept_encoded, ids_restore, coords)
        loss = self.forward_loss(target, pred, mask)
        return loss, pred, mask
        # scalar MAE loss | [1,L,D] reconstructed spots | [1,L] 0/1 mask
