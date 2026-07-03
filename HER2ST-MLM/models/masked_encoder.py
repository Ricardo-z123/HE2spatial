import torch
from torch import nn

from models.modules import attn_block
from models.pos_embed import PosEmbed


class MaskedEncoder(nn.Module):
    """
    MAE-style masked autoencoder over the spots of ONE window (or slide).

    Idea: randomly DROP a fraction (mask_ratio) of the spots, let the ENCODER see only the
    KEPT spots, then let a small DECODER insert placeholders ('mask_token') for the dropped
    spots and RECONSTRUCT every spot's features. The loss is computed ONLY on the dropped
    (masked) spots -> the encoder is forced to learn spatial context (predict a missing spot
    from its neighbours + coordinates).

    Spot and image branches share this class; only dim differs:
        spot branch:  dim = 785   (gene-expression vector per spot)
        image branch: dim = 1280  (Virchow2 feature per spot); also 1024/2048 for DenseNet/ResNet
    enc_dim = dec_dim = dim  (no encoder->decoder projection; "MAE-lite").

    Shapes below use the ACTUAL training run (window mode):
        B = 64 windows per batch,  L = 100 spots per window,  dim = 785,
        mask_ratio = 0.1  ->  len_keep = 90 kept,  10 masked.
    """

    ## Frozen pos_embed, encoder_nlocks, and encoder_norm.
    def __init__(self, dim, enc_depth, dec_depth, num_heads,
                 dim_head=64, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio                          # fraction of spots to drop (0.1)
        self.pos_embed = PosEmbed(dim)   ## Position embedding   # learnable (x,y) spatial PE, output dim = dim

        self.encoder_blocks = nn.ModuleList([                 # ENCODER = enc_depth Transformer blocks (real: 4)
            attn_block(dim, heads=num_heads, dim_head=dim_head, mlp_dim=dim)
            for _ in range(enc_depth)
        ])
        self.encoder_norm = nn.LayerNorm(dim)              # LayerNorm on encoder output

        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # init ONE learnable placeholder vector, reused for every masked spot
        self.decoder_blocks = nn.ModuleList([                 # DECODER = dec_depth Transformer blocks (real: 2)
            attn_block(dim, heads=num_heads, dim_head=dim_head, mlp_dim=dim)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(dim)              # LayerNorm on decoder output
        self.decoder_pred = nn.Linear(dim, dim)  # reconstruction head: maps decoder output back to spot feature space (dim -> dim)

    def random_masking(self, x):
        """
        Randomly choose which spots to keep, per sample, via argsort of random noise (the MAE trick).
        No real "masking matrix": we SHUFFLE spots randomly, keep the first len_keep, drop the rest.

        Input:
          x [B, L, D]   e.g. [64, 100, 785]   spot sequence (already + pos_embed)
        Output:
          x_masked    [B, len_keep, D]   e.g. [64, 90, 785]   the KEPT spots, in SHUFFLED order
          mask_label  [B, L]             e.g. [64, 100]        0 = kept, 1 = masked, in ORIGINAL spot order
          ids_restore [B, L]             e.g. [64, 100]        inverse permutation: used later to undo the shuffle
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))  # spots to keep, e.g. L=100,ratio=0.1 -> 90 keep / 10 masked

        noise = torch.rand(B, L, device=x.device)  # one random number in [0,1) per spot, per sample -> [B, L]

        ids_shuffle = torch.argsort(noise, dim=1)  # shape: [64, 100] sort noise: gives a RANDOM permutation of spot ids 0..L-1 (small noise first)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # shape : [64, 100] argsort of a permutation = its INVERSE (maps shuffled position -> original position)

        ids_keep = ids_shuffle[:, :len_keep]       # [B, 90] keep first len_keep 90 ids_shuffle positions as x_kept (random order）
        # ids_keep [B,90] --unsqueeze(-1)--> [B,90,1] --repeat(1,1,785)--> [B,90,785]
        # x[64,100,785] --> x_masked [64,90,785].
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # gather those kept spots' features -> [B, 90, 785] (random order)

        mask_label = torch.ones([B, L], device=x.device)  # [64, 100], build the 0/1 label: start all 1 (= masked)
        mask_label[:, :len_keep] = 0         # first len_keep 90 positions are kept -> 0 (NOTE: still in shuffled order here)
        mask_label = torch.gather(mask_label, dim=1, index=ids_restore)  # unshuffle the label back to ORIGINAL spot order -> [B, L]

        return x_masked, mask_label, ids_restore  # [64, 90, 785], [64, 100], [64, 100], [64, 100].
    
    ## Example:
    # (L=4, mask_ratio=0.5 -> len_keep=2, one window):
    # x        = [spot0, spot1, spot2, spot3]              # 4 spots, original order
    # noise    = [0.7,   0.2,   0.9,   0.4 ]               # one random number per spot
    # original positon [0, 1, 2, 3]

    # ids_shuffle = argsort(noise) = [1, 3, 0, 2]  correspond spot : [0.2, 0.4, 0.7, 0.9]    # spot ids sorted by noise (small first) = a random order
    # ids_restore = argsort(ids_shuffle) = [2, 0, 3, 1] !! argsort for [1, 3, 0, 2] 
    #  e.g. [0.7->2, 0.2->0, 0.9->3, 0.4->1]  # inverse: original spot i sits at shuffled pos ids_restore[i]
    #
    # # --- pick which to keep: first len_keep of the shuffle ---
    # ids_keep = ids_shuffle[:2] = [1, 3]        # keep spot1, spot3 ; drop spot0, spot2
    # x_masked = x[[1, 3]] = [spot1, spot3]      # gathered kept features, SHUFFLED order  -> [2, D]
    #
    # # --- build mask_label ---
    # ones                 = [1, 1, 1, 1]                  # start all masked
    # mask label : set first 2 to 0     = [0, 0, 1, 1]                  # kept=0, masked=1, but in SHUFFLED order
    # gather(mask_label, ids_restore=[2, 0, 3, 1]) -> [1, 0, 1, 0]   # original mask label.
    #   e.g. out[0]=label[2]=1, out[1]=label[0]=0, out[2]=label[3]=1, out[3]=label[1]=0
    #                                                      #   spot0=masked, spot1=kept, spot2=masked, spot3=kept
    #
    # # Returns:
    # #   x_masked    = [spot1, spot3]   (kept features, shuffled)
    # #   mask_label  = [1, 0, 1, 0]     (original order, 1=masked)
    # #   ids_restore = [2, 0, 3, 1]     (to unshuffle later in the decoder)

    def forward_encoder(self, x, coords):
        """
        Add spatial PE, drop spots (random_masking), then encode ONLY the kept spots.
        (Encoder runs on 90 spots, not 100 -> this is MAE's compute saving.)

        Input:
          x      [B, L, dim]   e.g. [64, 100, 785]   spot features
          coords [B, L, 2]        e.g. [64, 100, 2]     integer (x, y) grid coords per spot
        Output:
          x_kept (encoded) [B, len_keep, dim]   e.g. [64, 90, 785]   encoder output for kept spots (shuffled order)
          mask_label       [B, L]                  e.g. [64, 100]       0=keep, 1=masked (original order)
          ids_restore      [B, L]                  e.g. [64, 100]       to restore shuffled -> original later
        """
        x = x + self.pos_embed(coords)  # add learnable spatial PE per spot -> [B, L, dim]
        x_kept, mask_label, ids_restore = self.random_masking(x)  # drop spots -> x_kept [B,90,dim]!!; mask_label/ids_restore [B,100]
        for blk in self.encoder_blocks:
            x_kept = blk(x_kept)        # Transformer block, shape unchanged -> [B, 90, dim]
        x_kept = self.encoder_norm(x_kept)  # final LayerNorm -> x_kept [B, 90, dim]
        return x_kept, mask_label, ids_restore

    ## def encode used for satge2.
    def encode(self, x, coords):
        """
        Used by Stage 2 contrastive training & inference.
        Adds PE, runs the SAME encoder blocks + norm; never touches mask / decoder.

        Input:
          x      [B, L, dim]   e.g. [1, 128, 785]   B spots from one slide (Stage2 unsqueezes to a 1-slide batch)
          coords [B, L, 2]        e.g. [1, 128, 2]     integer (x, y) coords per spot
        Output:
          x [B, L, dim]   encoder features for EVERY spot, original order (no spots dropped)
        """
        x = x + self.pos_embed(coords)  # add spatial PE -> [B, L, dim]
        for blk in self.encoder_blocks:
            x = blk(x)                  # encoder Transformer blocks, shape unchanged
        x = self.encoder_norm(x)        # final LayerNorm
        return x

    def forward_decoder(self, x, ids_restore, coords):
        """
        Re-insert placeholders for the dropped spots, restore original order, then decode + reconstruct.

        Input:
          x     [B, len_keep, dim]   e.g. [64, 90, 785]   encoder output for kept spots (SHUFFLED order)
          ids_restore [B, L]                  e.g. [64, 100]       restore index from forward_encoder
          coords      [B, L, 2]               e.g. [64, 100, 2]    original-order spot coords
        Output:
          pred [B, L, dim]   e.g. [64, 100, 785]   reconstructed features for ALL spots, original order
        """
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)  # one mask_token per dropped spot -> [64, 10, 785]
        x = torch.cat([x, mask_tokens], dim=1)  # append placeholders so length is full L again -> [B, 100, dim] (still shuffled order)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle: put every spot (kept + mask_token) back to ORIGINAL order -> [B, 100, dim]
        x = x + self.pos_embed(coords)  # add spatial PE again so mask_tokens know WHERE they are -> [B, 100, dim]
        for blk in self.decoder_blocks:
            x = blk(x)                  # decoder Transformer blocks, shape unchanged
        x = self.decoder_norm(x)        # final LayerNorm
        pred = self.decoder_pred(x)  # [B, L, dim] reconstructed spot features, full slide order
        return pred  # [B, 100, 785] total 100 spots pred gene expression.

    def forward_loss(self, target, pred, mask_label):
        """
        Reconstruction loss = MSE, but ONLY on the masked (dropped) spots.

        Input:
          target     [B, L, dim]   e.g. [64, 100, 785]   ORIGINAL spot features (the reconstruction goal)
          pred       [B, L, dim]   e.g. [64, 100, 785]   decoder output
          mask_label [B, L]           e.g. [64, 100]        0=keep, 1=masked
        Output:
          loss scalar   mean MSE over masked spots only
        """
        loss = (pred - target) ** 2  # [B, L, dim] squared error per gene/feature
        loss = loss.mean(dim=-1)  # per-spot mean over feature dim -> [B, L], [64, 100], mask_label [64, 100]
        loss = (loss * mask_label).sum() / mask_label.sum().clamp(min=1.0)  # keep only masked spots (mask_label=1), average over them -> scalar
        return loss

    def forward(self, x, coords):
        """
        Full MAE step: encode kept spots -> decode/reconstruct all -> loss on masked spots only.

        Input:
          x      [B, L, dim]   e.g. [64, 100, 785]   raw spot features (gene or image)
          coords [B, L, 2]        e.g. [64, 100, 2]     integer (x, y) coords per spot
        Output:
          loss       scalar tensor                      mean MSE over masked spots
          pred       [B, L, dim]   e.g. [64, 100, 785]   reconstruction for all spots
          mask_label [B, L]           e.g. [64, 100]        0=keep, 1=masked
        """
        target = x.clone()  # snapshot the RAW features as the reconstruction target, BEFORE any pos_embed is added
        x_kept_encoded, mask_label, ids_restore = self.forward_encoder(x, coords)  # encode kept spots only -> [B, 90, dim]
        pred = self.forward_decoder(x_kept_encoded, ids_restore, coords)     # fill mask_token + decode -> pred [B, 100, dim]
        loss = self.forward_loss(target, pred, mask_label)  # MSE on MASKED spots only -> scalar
        return loss, pred, mask_label

## Example for encoder + decoder:
    # ## Flow (forward = encoder + decoder + loss):
    #
    #  forward(x [64,100,785], coords [64,100,2])
    #    target = x.clone()                                  # [64,100,785] raw features, BEFORE pos_embed (reconstruction target)
    #    │
    #    ├─ forward_encoder:
    #    │    x + pos_embed                                  # [64,100,785]
    #    │    random_masking  -> x_kept [64,90,785] (shuffled, drop 10)
    #    │                       mask_label [64,100], ids_restore [64,100]
    #    │    4x encoder block + norm                        # x_kept [64,90,785]  (still shuffle order) (only 90 go through encoder)
    #    │
    #    ├─ forward_decoder(x_kept, ids_restore, coords):
    #    │    mask_tokens = mask_token.repeat                # [64,10,785] (10 placeholders)
    #    │    cat(x_kept, mask_tokens)                       # [64,100,785] (shuffled: 90 real + 10 placeholder)
    #    │    gather(ids_restore)                            # [64,100,785] (back to ORIGINAL order)
    #    │    + pos_embed                                    # [64,100,785]
    #    │    2x decoder block + norm + Linear(decoder_pred) # pred [64,100,785] (reconstruct ALL spots)
    #    │
    #    └─ forward_loss(target, pred, mask_label):
    #         (pred-target)^2 -> mean over genes -> * mask_label (keep masked only) -> mean
    #                                                         # loss = scalar (MSE on the 10 masked spots)
    #
    #    return loss, pred, mask_label