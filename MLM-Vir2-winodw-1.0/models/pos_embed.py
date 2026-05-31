from torch import nn


class PosEmbed(nn.Module):
    """
    Learnable spatial positional encoding for spots on a slide.
    Two independent nn.Embedding tables for the x- and y-axis, summed at each spot.

    Each spot has integer (x, y) coordinates on the slide grid. HER2ST selected
    slides use small x/y values, so vocab=64 covers the current coordinate range
    while avoiding huge mostly-unused embedding tables.

    Args:
        dim: embedding dimension. Spot branch uses 785 (gene-expression dim);
             image branch uses 1024 (DenseNet feature dim).

    Note:
        Two branches must instantiate this module separately — coords are shared but
        the embedding semantics (gene vs. image feature space) differ, so weights are
        not shared across branches.
    """
    def __init__(self, dim):
        super().__init__()
        self.x_embed = nn.Embedding(64, dim)
        self.y_embed = nn.Embedding(64, dim)

    def forward(self, coords):
        """
        Input:  coords [N, L, 2] long tensor.   e.g. [1, 300, 2]
                  N = 1 slide per forward, L = 300–600 spots, last dim = (x, y) integer coords
        Output: pe     [N, L, dim] float.       e.g. [1, 300, 785] (spot) or [1, 300, 1024] (image)
                  per-spot positional encoding to be added to spot/image features
        """
        x = coords[..., 0].long()  # per-spot x coord on slide
        y = coords[..., 1].long()  # per-spot y coord on slide
        return self.x_embed(x) + self.y_embed(y)
        # [N, L, dim] e.g. [1, 300, 785] learnable spatial PE per spot, summed x+y embeddings
