import argparse
import os
import yaml
import anndata
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from datasets.stage2_window_dataset import HER2STStage2WindowDataset
from models.contrastive_model import ContrastiveModel
from utils.misc import get_R

## Build up 6 slide test_loader.
def build_loaders_inference(cfg=None, eval_window_size=None):
    """Load the held-out test slide of each of the 6 folds (A4..B3); concat into one DataLoader."""
    cfg = cfg or {}
    # window_size comes from the SAME config the training used -> train/eval can never drift apart.
    # --eval_window_size is an explicit opt-out for sensitivity probes only.
    win = eval_window_size or cfg.get('window_size', 10)  # default = 10.
    datasets = []
    for i in range(6):
        dataset = HER2STStage2WindowDataset(train=False, fold=i,  # i-fold test slide ; fold0→A4, ..., fold5→B3
                                            window_size=win,
                                            anchor_all=True)      # one window per spot, centred on it, stride = 1
        print(dataset.id2name[0])  # print test slide name [A4..B3] for each fold. 
        datasets.append(dataset)   # List; concat all 6 slides into one dataset
        # datasets = [
        #   HER2STStage2WindowDataset(train=False, fold=0, anchor_all=True),   # A4  len = 343
        #   ... ...
        # ]   len(datasets) = 6, one per fold's HELD-OUT slide (train=False -> names = [that slide])
        #
        # anchor_all=True => every spot is an anchor => len(ds) == n_spots of that slide,
        # and window j is spot j in meta row order. Total spots across the 6 = 1898.
        #
        # Each dataset[j] returns ONE window (M = window_size**2 = 100 spots, all from that slide):
        #   {'image':      [100, 3, 224, 224],   # H&E patches (eval: ToTensor only, no aug)
        #    'expression': [100, 785],
        #    'position':   [100, 2],
        #    'anchor_pos': int,                  # which of the 100 rows is spot j itself  <- eval reads this
        #    'spot_id':    [100]}                # unused at eval time

    concat_dataset = torch.utils.data.ConcatDataset(datasets)  ## 6-slide
    test_loader = DataLoader(concat_dataset, batch_size=8, shuffle=False, num_workers=0)
    print("Finished building loaders")
    return test_loader, datasets


def get_embeddings(model, test_loader, device):
    """Window-mode encoding: attention runs inside each spot's own KNN window (same shape the
    model was trained on), and only the ANCHOR's output is kept.

    Input:
      test_loader   yields batches of B windows, each M = window_size**2 spots.
                    anchor_all=True + shuffle=False -> window i is spot i, order = A4..B3 meta order,
                    which is what main() relies on when splitting by datasize.
    Output:
      image_embeddings [total_spots, 256]   retrieval QUERY pool (one image emb per spot)
      spot_embeddings  [total_spots, 256]   retrieval KEY   pool (one expression emb per spot)

    A spot appears in many windows (neighbouring windows overlap heavily), but its embedding is
    taken ONLY from the window it anchors -- so it is unique by construction, nothing is averaged.
    """
    model.eval()                     # eval mode: disable dropout, freeze BN stats
    model = model.to(device)

    image_embeddings, spot_embeddings = [], []
    with torch.no_grad():            # inference only: no autograd graph, saves memory
        for batch in tqdm(test_loader):
            img = batch["image"].to(device)                    # [B, M, 3, 224, 224] H&E patches
            gene = batch["expression"].to(device)              # [B, M, 785]
            coords = batch["position"].long().to(device)       # [B, M, 2]  long for PosEmbed lookup
            apos = batch["anchor_pos"].to(device)              # [B] anchor row inside each window
            B, M = gene.shape[0], gene.shape[1]
            widx = torch.arange(B, device=device)     # B=8 → tensor([0,1,2,3,4,5,6,7])

            # ===== image branch: (Virchow2 ->) MAE Transformer encoder -> projection =====
            # chunk 256 = Stage1/Stage2's Virchow2 batch; identical maths, but keeps the fp16
            # rounding aligned with training (cuBLAS switches kernel above ~256).

            im = img.reshape(B * M, *img.shape[2:])  # [8,100,3,224,224] ─reshape →  [800,3,224,224]

            feats = [model.image_cnn(im[s:s + 256]) for s in range(0, im.shape[0], 256)]  
            img_feat = torch.cat(feats, dim=0).reshape(B, M, -1)      ## Virchow2 → [800,2560] → reshape → [8, 100, 2560] 
            img_feat = model.image_encoder.encode(img_feat, coords)          # +PE, within-window attn

            img_emb = model.image_projection(img_feat.reshape(B * M, -1)).reshape(B, M, -1)
            ## [8,100,2560] → [800,2560] → proj → [800,256] → [8,100,256]
            
            image_embeddings.append(img_emb[widx, apos])     # [8,100,256] --[widx,apos]--> [8,256] anchors only

            # ===== spot branch: gene expression -> MAE Transformer encoder -> projection =====
            spot_feat = model.spot_encoder.encode(gene, coords)                    # [B, M, 785]
            spot_emb = model.spot_projection(spot_feat.reshape(B * M, -1)).reshape(B, M, -1)
            spot_embeddings.append(spot_emb[widx, apos])                          # [B, 256] anchors only 

    # concat all batches along dim 0; order preserved (shuffle=False) so main() can split by datasize
    return torch.cat(image_embeddings), torch.cat(spot_embeddings)
            # tensor [total_spots,256] ； tensor [total_spots,256] 


# Retrieval aggregation methods.
#   simple / average / bleep_weighted  -> BLEEP_inference.ipynb cell 5 ("simple"/"average"/"weighted_average")
#   mcl_weighted                       -> mclSTExp evel-1-32.py:185-189 (1/L1^2)
RETRIEVAL_METHODS = ('simple', 'average', 'bleep_weighted', 'mcl_weighted')

def find_matches(spot_embeddings, query_embeddings, top_k=200):
    """For each query, return the indices of its top_k nearest spots (cosine similarity).

    Input:
      spot_embeddings  [N_train, 256]  5-slide train-set expression embeddings (retrieval KEYS)
      query_embeddings [N_test, 256]   held-out slide image embeddings (retrieval QUERIES)
    Output:
      indices [N_test, top_k] int    row ids of the nearest keys, ranked nearest-first
    """
    spot_embeddings = torch.tensor(spot_embeddings)     # 5-slide train set gene expressions [N_train, 256]
    query_embeddings = torch.tensor(query_embeddings)   # 1-slide test set image embeddings [N_test, 256]

    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)  # L2-normalize
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)    # L2-normalize

    dot_similarity = query_embeddings @ spot_embeddings.T     # cosine similarity matrix
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)   ## for each row (test spot), choose top_k.
    return indices.cpu().numpy()
    # [N_test, top_k] most similar spot indices


def aggregate(method, i, idx, spot_key, image_query, expression_key):
    """Turn one test spot's retrieved neighbours into a predicted expression vector.

    Input:
      method         one of RETRIEVAL_METHODS
      i              row index of this test spot
      idx            [top_k] retrieved neighbour ids, ranked nearest-first (idx[0] = closest)
      spot_key       [N_train, 256] / image_query [N_test, 256]  UNNORMALISED embeddings
      expression_key [N_train, 785] neighbours' real expression -- what we borrow as prediction
    Output:
      [785] predicted expression for test spot i
    """
    if method == 'simple':
        # BLEEP 'simple': the single closest neighbour, no averaging.
        # top_k=1
        return expression_key[idx[0], :]

    if method == 'average':
        # BLEEP 'average': unweighted mean over the top_k.
        return np.average(expression_key[idx, :], axis=0)

    if method == 'bleep_weighted':
        # BLEEP 'weighted_average':
        #     a = np.sum((spot_key[indices[i,0],:] - image_query[i,:])**2)        
        #     weights = np.exp(-(np.sum((spot_key[indices[i,:],:] - image_query[i,:])**2, axis=1) - a + 1))
        # SQUARED L2 on the UNNORMALISED embeddings, exponentially decayed, shifted by the minimum
        d = np.sum((spot_key[idx, :] - image_query[i, :]) ** 2, axis=1)   # [top_k] squared L2 distance
        w = np.exp(-(d - d[0] + 1))                                       # idx[0] is the nearest -> d[0] = min
        return np.average(expression_key[idx, :], axis=0, weights=w)

    # mcl_weighted: mclSTExp's 1/L1^2 on the raw (unnormalised) embeddings (evel-1-32.py:185-189).
    a = np.linalg.norm(spot_key[idx, :] - image_query[i, :], axis=1, ord=1)
    w = np.reciprocal(a ** 2)
    w = (w / np.sum(w)).flatten()
    return np.average(expression_key[idx, :], axis=0, weights=w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Stage 2 yaml config')  # require a stage2 config yaml file.
    parser.add_argument('--ckpt_dir', type=str, default=None,                             # model weight ckpt file.
                        help='dir containing fold{i}/best.pt (default: cfg["save_dir"])')
    parser.add_argument('--top_k', type=int, default=50,
                        help='neighbours retrieved per test spot. BLEEP uses 50 for average/'
                             'weighted_average and 1 for simple; simple here ignores it (rank 0 only)')
    parser.add_argument('--weight_mode', type=str, default=None,
                        choices=list(RETRIEVAL_METHODS),
                        help='retrieval aggregation; overrides cfg["retrieval_method"]. '
                             'default: cfg value, else mcl_weighted')
    parser.add_argument('--tag', type=str, default='sweep')
    parser.add_argument('--eval_window_size', type=int, default=None,
                        help='override eval window size (sensitivity probe only; '
                             'default None = use the training config, keeping train/eval aligned)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

########
    method = args.weight_mode or cfg.get('retrieval_method', 'mcl_weighted')
    assert method in RETRIEVAL_METHODS, 'unknown retrieval method: %s' % method
    print('[retrieval] method=%s  top_k=%d%s'
          % (method, args.top_k, '   (simple uses rank 0 only)' if method == 'simple' else ''))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']   # 6 折 slide;names[0]='A4'
    ckpt_base = args.ckpt_dir if args.ckpt_dir else cfg['save_dir']   
    pdim = cfg['projection_dim']    # projection dim = 256 

    print("Loading data once...")
    test_loader, datasets = build_loaders_inference(cfg, args.eval_window_size)   # build 6 slide test set.
    datasize = [len(ds) for ds in datasets]    # spot number in each slide ; datasize = [325, 331, 309, 295, 300, 302]

    # ground-truth (and retrieval-neighbour) expressions: lib-norm + log1p HVG, per slide
    spot_expressions = [
        np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")   # real gene expression in 6 slides. [785, N]  
        for name in names
    ]
    gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))  # 785 gene list.

    heg_pcc_list, hvg_pcc_list, mse_list, mae_list = [], [], [], []
    all_gene_pcc = []  # 每折的全 785-gene PCC(NaN→0),用于跨折统计 top-20 易预测基因

    # ===== for fold in range(6): evaluate each fold (leave-one-slide-out) =====
    #   1. load this fold's trained model            (fold{fold}/best.pt)
    #   2. get_embeddings: encode all 6 slides -> image_emb & spot_emb [total_spots, 256]
    #   3. split embeddings back per-slide by datasize (shuffle=False keeps order)
    #   4. pick query / key:
    #        query   = test slide's IMAGE embeddings      [256, N_test]   (retrieve WITH image)
    #        key pool = other 5 slides' SPOT(expression) embeddings [256, N_train]  (search AGAINST expression)
    #        also grab: test slide's real expression (ground truth) + train slides' real expression (to borrow as prediction)
    #   5. defensive transpose: align shapes to [N, pdim] / [N, 785]
    #   6. find_matches: retrieve top_k nearest train spots per test spot (cosine)
    #   7. predict: weighted (1/L1^2) average of neighbours' REAL expression   <- core
    #   8. metrics: HEG-PCC / HVG-PCC / MSE / MAE
    #   9. append the 4 metrics to their lists
    # ===== after loop: average over 6 folds + print =====

    for fold in range(6):
        print(f"\n{'=' * 50}")
        print(f"Processing fold {fold} ({names[fold]})")
        print(f"{'=' * 50}")

        # load this fold's trained model
        model = ContrastiveModel(cfg)    # build stage2 Contrastive model.
        ckpt = torch.load(os.path.join(ckpt_base, f'fold{fold}', 'best.pt'), map_location='cpu')   ## read ckpt weight.
        assert ckpt.get('cfg', {}).get('fold', fold) == fold, "checkpoint fold mismatch"
        model_state = ckpt['model_state']
        model.load_state_dict(model_state, strict=True)    ## load in ckpt weight.

        # encode all 6 slides, then split embeddings back per slide
        img_emb_all, spot_emb_all = get_embeddings(model, test_loader, device)   # extract all 6 slides' image and spot embeddings.
        img_emb_all = img_emb_all.cpu().numpy()      # torch → numpy  [1898, 256]
        spot_emb_all = spot_emb_all.cpu().numpy()    # torch → numpy  [1898, 256]

        
        image_embeddings, spot_embeddings = [], []
        for i in range(len(datasize)):               # datasize = [325, 331, 309, 295, 300, 302] ; len(datasize) = 6.
            ## int i slide in img_emb_all start row and end row. 
            s, e = sum(datasize[:i]), sum(datasize[:i + 1])  #  i=0: s=0,e=325 ; i=1: s=325,e=656 ; ...
            image_embeddings.append(img_emb_all[s:e].T)     # # [N_i, 256] -> [256, N_i] 
            spot_embeddings.append(spot_emb_all[s:e].T)

# image_embeddings = [
#     img_emb of A4,   # [256, 343]   index [0] = fold0 的测试 slide
#     img_emb of A5,   # [256, 332]   index [1] = fold1
#     img_emb of A6,   # [256, 360]
#     img_emb of B1,   # [256, 295]
#     img_emb of B2,   # [256, 270]
#     img_emb of B3,   # [256, 298]   index [5] = fold5
# ]
# # list, len=6; each element numpy [256, N_i] (col=spot)

# spot_embeddings = [
#     spot_emb of A4,  # [256, 343]
#     spot_emb of A5,  # [256, 334]
#     ...
# ]

        # query = held-out test slide's image embeddings; key pool = the other 5 slides' spot embeddings
        image_query = image_embeddings[fold]       # test slide, image embedding
        expression_gt = spot_expressions[fold]     # test slide real gene expression ； ground truth.
        spot_key = np.concatenate(spot_embeddings[:fold] + spot_embeddings[fold + 1:], axis=1)   # concat rest 5 slide gene expression embedding
        expression_key = np.concatenate(spot_expressions[:fold] + spot_expressions[fold + 1:], axis=1)  # concat rest 5 slide real gene expression

        if image_query.shape[1] != pdim:   # check whether the shape is [N, 256]
            image_query = image_query.T
        if expression_gt.shape[0] != image_query.shape[0]: # test slide ground truth expression shape check: [785, N] -> [N, 785]
            expression_gt = expression_gt.T
        if spot_key.shape[1] != pdim:      # check whether the shape is [N, 256]
            spot_key = spot_key.T
        if expression_key.shape[0] != spot_key.shape[0]: # check whether the shape is [N, 785]
            expression_key = expression_key.T

        # retrieval prediction: aggregate the top_k neighbours' REAL expression (method picks how)
        indices = find_matches(spot_key, image_query, top_k=args.top_k)   # [N_test, top_k] neighbour ids
        pred = np.zeros((indices.shape[0], expression_key.shape[1]))   # init all 0 matrix for pred result. [N_test, 785]

        for i in range(indices.shape[0]):
            pred[i, :] = aggregate(method, i, indices[i, :],
                                   spot_key, image_query, expression_key)

        true = expression_gt                       # [N_test, 785] ground-truth expression of test slide
        adata_true = anndata.AnnData(true)         # wrap pred/true into AnnData so we can index genes by name
        adata_pred = anndata.AnnData(pred)         # [N_test, 785]  pred gene expression
        adata_true.var_names = gene_list           # attach 785 gene names to columns
        adata_pred.var_names = gene_list

        # HEG PCC: 50 highest-expressed genes; HVG PCC: all 785 genes (per-gene Pearson over spots)
        gene_mean = np.mean(adata_true.X, axis=0)                          # [785] mean expression per gene over all test spots
        top50_names = adata_true.var_names[np.argsort(gene_mean)[::-1][:50]]  # names of the 50 highest-expressed genes (HEG set)
        heg_pcc, _ = get_R(adata_pred[:, top50_names], adata_true[:, top50_names])  # [50] per-gene Pearson r on HEG genes
        hvg_pcc, _ = get_R(adata_pred, adata_true)                         # [785] per-gene Pearson r on all genes
        all_gene_pcc.append(np.nan_to_num(hvg_pcc, nan=0.0))              # 存全 785(NaN→0),供跨折 top-20 基因统计
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]                              # drop NaN (gene constant on test slide -> r undefined)

        mse = mean_squared_error(true, pred)       # scalar, over all spots x 785 genes
        mae = mean_absolute_error(true, pred)      # scalar, over all spots x 785 genes

        heg_pcc_list.append(np.mean(heg_pcc))      # this fold's HEG-PCC = mean over the 50 genes
        hvg_pcc_list.append(np.mean(hvg_pcc))      # this fold's HVG-PCC = mean over all (non-NaN) genes
        mse_list.append(mse)                       # collect this fold's MSE
        mae_list.append(mae)                       # collect this fold's MAE
        print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, "
              f"HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Per-fold table, repeated here so a single copy-paste of the tail carries every number
    # (the per-fold lines printed during the loop get buried under tqdm output).
    n_fold = len(heg_pcc_list)
    print(f"\n{'=' * 60}")
    print(f"[{args.tag}] top_k={args.top_k} retrieval_method={method}")
    print(f"Per-fold results ({n_fold} folds):")
    print(f"{'=' * 60}")
    print(f"{'fold':<6}{'slide':<8}{'HEG':>9}{'HVG':>9}{'MSE':>9}{'MAE':>9}")
    for i in range(n_fold):
        print(f"{i:<6}{names[i]:<8}{heg_pcc_list[i]:>9.4f}{hvg_pcc_list[i]:>9.4f}"
              f"{mse_list[i]:>9.4f}{mae_list[i]:>9.4f}")
    print(f"{'avg':<6}{'-':<8}{np.mean(heg_pcc_list):>9.4f}{np.mean(hvg_pcc_list):>9.4f}"
          f"{np.mean(mse_list):>9.4f}{np.mean(mae_list):>9.4f}")

    print(f"\n{'=' * 60}")
    print(f"[{args.tag}] top_k={args.top_k} retrieval_method={method}")
    print(f"Final Results ({n_fold}-fold average):")
    print(f"{'=' * 60}")
    print(f"avg HEG PCC: {np.mean(heg_pcc_list):.4f}")
    print(f"avg HVG PCC: {np.mean(hvg_pcc_list):.4f}")
    print(f"avg MSE:     {np.mean(mse_list):.4f}")
    print(f"avg MAE:     {np.mean(mae_list):.4f}")

    # ===== top-20 易预测基因:按跨 6 折平均的 per-gene PCC 排序 =====
    mean_gene_pcc = np.mean(np.array(all_gene_pcc), axis=0)   # [785] 每个基因 6 折平均 PCC(与 gene_list 对齐)
    top20_idx = np.argsort(mean_gene_pcc)[::-1][:20]          # PCC 最高的 20 个基因
    print(f"\n{'=' * 60}")
    print("Top 20 genes by average PCC (6-fold mean):")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6}{'Gene':<18}{'Avg PCC':<10}")
    for rank, i in enumerate(top20_idx, 1):
        print(f"{rank:<6}{str(gene_list[i]):<18}{mean_gene_pcc[i]:.4f}")


if __name__ == "__main__":
    main()

## Example Flow  ——  evel.py (Stage2 retrieval-based evaluation, 6-fold leave-one-slide-out)
## No prediction network: model only generates embeddings; expression = borrowed from retrieved neighbours (BLEEP-style)
## consts: image_dim=2560 (Virchow2 concat=CLS+patch-mean), spot_dim=785, proj=256,
##         M=window_size**2=100 spots per window; top_k default 200, overridable via --top_k
#
#  ── setup (once, shared by all folds) ─────────────────────────────
#    build_loaders_inference()   -> test_loader: 6 slides concat, anchor_all=True, batch=8 WINDOWS
#                                   (= 800 spots/batch), shuffle=False  ← order must stay meta order
#                                   datasets (list of 6), datasize=[N_A4,...,N_B3]
#    anchor_all=True => window i IS spot i => len(dataset) == n_spots, order == meta order
#    spot_expressions  = [np.load(preprocessed_matrix.npy) for 6 slides]   # each [785, N] REAL expression
#
#  ── for fold in range(6):  test slide = names[fold], train = other 5 ──
#
#    1. load model:  ContrastiveModel(cfg) + load fold{fold}/best.pt (strict=True); assert ckpt.fold == fold
#
#    2. get_embeddings(model, test_loader):   # same towers as training, stop at the projections
#         image tower: [B,M,3,224,224] -> image_cnn(Virchow2, chunk 256) -> [B,M,2560]
#                      -> image_encoder.encode(+PE, attn WITHIN window) -> image_projection -> [B,M,256]
#         spot  tower: [B,M,785] -> spot_encoder.encode -> spot_projection -> [B,M,256]
#         ★ ANCHOR ONLY:  emb[widx, apos]  [B,M,256] -> [B,256]
#            a spot appears in ~25 windows, but its embedding is taken ONLY from the window it
#            anchors -> unique by construction, nothing averaged; the other 99 rows are discarded
#         -> img_emb_all [total_spots, 256], spot_emb_all [total_spots, 256]  (order A4..B3)
#
#    3. split back per slide by datasize (shuffle=False makes this exact):
#         image_embeddings / spot_embeddings = list of 6, each [256, N_i]
#
#    4. pick query / key pools:
#         image_query    = image_embeddings[fold]          test slide IMAGE emb   -> [N_test, 256]  (QUERY)
#         expression_gt  = spot_expressions[fold]          test slide REAL expr   -> [N_test, 785]  (ground truth)
#         spot_key       = concat other 5 slides' spot emb -> [N_train, 256]      (KEY pool)
#         expression_key = concat other 5 slides' real expr-> [N_train, 785]      (borrow pool)
#         (defensive .T so all are row = spot)
#
#    5. retrieval:  indices = find_matches(spot_key, image_query, top_k=args.top_k)
#         L2-normalize both -> cosine [N_test, N_train] -> topk -> indices [N_test, k]
#
#    6. predict (per test spot i):  aggregate(method, ...) -> [785]   # borrows neighbours' REAL expr
#         simple           BLEEP  expression_key[idx[0]]                       (rank 0 only)
#         average          BLEEP  unweighted mean over top_k
#         bleep_weighted   BLEEP  w = exp(-(sqL2 - min_sqL2 + 1)) on RAW emb
#         mcl_weighted     mcl    w ∝ 1 / L1^2 on RAW emb                      (default)
#         -> pred [N_test, 785]
#
#    7. metrics (per fold):     # per-gene Pearson: predicted expr vs true expr, over all spots
#         HEG-PCC = mean Pearson over the 50 highest-expressed genes
#         HVG-PCC = mean Pearson over all 785 genes (drop NaN = constant genes)
#         MSE / MAE = over all spots x 785 genes
#
#  ── after 6 folds: average HEG-PCC / HVG-PCC / MSE / MAE across folds -> final result ──
#  KEY IDEA: image_query retrieves against spot_key (cross-modal), then borrows expression_key (real expr) as prediction.