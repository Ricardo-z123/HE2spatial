import random
import numpy as np
import torch
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score as ari_score


class AvgMeter:
    """Track running average of a metric (e.g. loss). Reset at epoch start."""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    """Return current learning rate from optimizer.param_groups[0]."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seed(seed):
    """Set random seeds for random / numpy / torch / torch.cuda for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_R(data1, data2, dim=1, func=pearsonr):
    """
    Compute Pearson correlation between two AnnData objects per gene/spot.
    Used at inference time to evaluate gene expression prediction accuracy.

    Input:
      data1, data2  AnnData with .X attribute, e.g. shape [N_spot, 785]
                    data1 = predicted, data2 = ground truth
      dim           1 = per-gene Pearson (correlate across spots), 0 = per-spot Pearson
      func          correlation function, default pearsonr
    Output:
      r1  [G] or [N] correlation values
      p1  [G] or [N] p-values
    """
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])  # one gene across all spots
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])  # one spot across all genes
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1
    # [G or N] Pearson r per gene/spot | [G or N] p-values


def cluster(adata, label):
    """
    K-means clustering on PCA embeddings. Used to evaluate whether the learned
    representation captures pathology labels (invasive cancer, breast glands, etc.).
    Note: baseline's version was missing sc / KMeans / sklearn metric imports — added at file top.

    Input:
      adata  AnnData object with .X feature matrix per spot
      label  pathology labels per spot (np.array of str); 'undetermined' rows are ignored
    Output:
      p     predicted cluster labels per spot (str)
      ari   Adjusted Rand Index (rounded 3 decimals)
      nmi   Normalized Mutual Information (rounded 3 decimals)
    """
    idx = label != 'undetermined'  # drop spots without a known pathology label
    tmp = adata[idx]
    l = label[idx]
    sc.pp.pca(tmp, n_comps=9)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p = kmeans.labels_.astype(str)
    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p  # write cluster ids back to original-order array, undetermined rows keep sentinel
    adata.obs['kmeans'] = lbl
    nmi = normalized_mutual_info_score(l, p)
    return p, round(ari_score(p, l), 3), round(nmi, 3)
    # [N_labeled] cluster ids | float ARI | float NMI
