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

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from utils.misc import get_R

## Build up 6 slide test_loader.
def build_loaders_inference():
    """Load the held-out test slide of each of the 6 folds; concat into one DataLoader."""
    datasets = []
    for i in range(6):
        dataset = HER2STStage2Dataset(train=False, fold=i)  # i-fold test slide ; fold0→A4, fold1→A5, ..., fold5→B3
        print(dataset.id2name[0])  # print test slide name
        datasets.append(dataset)   # List; concat all 6 slides into one dataset

    concat_dataset = torch.utils.data.ConcatDataset(datasets)  ## 6-slide 
    test_loader = DataLoader(concat_dataset, batch_size=32, shuffle=False, num_workers=0)
    print("Finished building loaders")
    return test_loader, datasets


def get_embeddings(model, test_loader, device):
    """
    Input:
      test_loader   yields batches of N spots (N=32, shuffle=False so order = A4..B3, needed for per-slide split)
    Output:
      image_embeddings [total_spots, 256]   retrieval QUERY pool (one image emb per spot, all 6 slides)
      spot_embeddings  [total_spots, 256]   retrieval KEY   pool (one expression emb per spot, all 6 slides)
    """
    model.eval()                     # eval mode: disable dropout, freeze BN stats
    model = model.to(device)

    image_embeddings, spot_embeddings = [], []
    with torch.no_grad():            # inference only: no autograd graph, saves memory
        for batch in tqdm(test_loader):
            img = batch["image"].to(device)                             # [N, 3, 224, 224] H&E patch per spot
            gene = batch["expression"].to(device)                       # [N, 785] lib-norm+log1p HVG per spot
            coords = batch["position"].long().to(device).unsqueeze(0)   # [1, N, 2] (x,y) coords; long for PosEmbed lookup; shared by both proj head.

            # ===== image branch: Virchow2 backbone -> MAE Transformer encoder -> projection =====
            img_feat = model.image_cnn(img).unsqueeze(0)                # [N,3,224,224] -> [N,1280] -> [1, N, 1280] 
            img_feat = model.image_encoder.encode(img_feat, coords).squeeze(0)  # [1,N,1280] -> [1,N,1280] (+PE, spot-to-spot self-attn) -> [N, 1280]
            image_embeddings.append(model.image_projection(img_feat))   # [N, 1280] -> [N, 256] image embedding (retrieval QUERY)

            # ===== spot branch: gene expression -> MAE Transformer encoder -> projection =====
            spot_feat = model.spot_encoder.encode(gene.unsqueeze(0), coords).squeeze(0)  # [N,785] -> [1,N,785] encode -> [N, 785]
            spot_embeddings.append(model.spot_projection(spot_feat))    # [N, 785] -> [N, 256] spot embedding (retrieval KEY)

    # concat all batches along dim 0; order preserved (shuffle=False) so main() can split by datasize
    return torch.cat(image_embeddings), torch.cat(spot_embeddings)      
            # tensor [total_spots,256] ； tensor [total_spots,256] 


def find_matches(spot_embeddings, query_embeddings, top_k=200):
    """For each query, return the indices of its top_k most similar spots (cosine similarity)."""
    spot_embeddings = torch.tensor(spot_embeddings)     # 5-slide train set gene expressions [N_train, 256]
    query_embeddings = torch.tensor(query_embeddings)   # 1-slide test set image embeddings [N_test, 256]

    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)  # L2-normalize
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)    # L2-normalize

    dot_similarity = query_embeddings @ spot_embeddings.T     # cosine similarity matrix
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)   ## for each row (test spot), choose top_k = 200. 
    return indices.cpu().numpy()
    # return [N_test, 200]  most similar spot indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Stage 2 yaml config')  # require a stage2 config yaml file.
    parser.add_argument('--ckpt_dir', type=str, default=None,                             # model weight ckpt file.
                        help='dir containing fold{i}/best.pt (default: cfg["save_dir"])')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']                     # train and test slide names. names[0]='A4'
    ckpt_base = args.ckpt_dir if args.ckpt_dir else cfg['save_dir']   
    pdim = cfg['projection_dim']    # projection dim = 256 

    print("Loading data once...")
    test_loader, datasets = build_loaders_inference()   # build 6 slide test set.
    datasize = [len(ds) for ds in datasets]             # spot number in each slide ; datasize = [325, 331, 309, 295, 300, 302]

    # ground-truth (and retrieval-neighbour) expressions: lib-norm + log1p HVG, per slide
    spot_expressions = [
        np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")   # real gene expression in 6 slides. [785, N]  
        for name in names
    ]
    gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))  # 785 gene list.

    heg_pcc_list, hvg_pcc_list, mse_list, mae_list = [], [], [], []

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
        model.load_state_dict(ckpt['model_state'], strict=True)    ## load in ckpt weight.

        # encode all 6 slides, then split embeddings back per slide
        img_emb_all, spot_emb_all = get_embeddings(model, test_loader, device)   # extract all 6 slides' image and spot embeddings.
        img_emb_all = img_emb_all.cpu().numpy()      # torch → numpy
        spot_emb_all = spot_emb_all.cpu().numpy()    # torch → numpy

        
        image_embeddings, spot_embeddings = [], []
        for i in range(len(datasize)):               # datasize = [325, 331, 309, 295, 300, 302] ; len(datasize) = 6.
            ## int i slide in img_emb_all start row and end row. 
            s, e = sum(datasize[:i]), sum(datasize[:i + 1])  #  i=0: s=0,e=325 ; i=1: s=325,e=656 ; ...
            image_embeddings.append(img_emb_all[s:e].T)     # # [N_i, 256] -> [256, N_i] 
            spot_embeddings.append(spot_emb_all[s:e].T)

# image_embeddings = [
#     img_emb of A4,   # [256, 325]   index [0] = fold0 slide
#     img_emb of A5,   # [256, 331]   index [1] = fold1 slide
#     img_emb of A6,   # [256, 309]
#     img_emb of B1,   # [256, 295]
#     img_emb of B2,   # [256, 300]
#     img_emb of B3,   # [256, 302]   index [5] = fold5 slide
# ]
# # list, len=6; each element numpy [256, N_i] (col=spot)

# spot_embeddings = [
#     spot_emb of A4,  # [256, 325]
#     spot_emb of A5,  # [256, 331]
#     ...
# ]

        # query = held-out test slide's image embeddings; key pool = the other 5 slides' spot embeddings
        image_query = image_embeddings[fold]       # test slide, image embedding
        expression_gt = spot_expressions[fold]     # test slide real gene expression
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

        # retrieval prediction: weighted (1 / L1^2) average of the top_k neighbours' real expression
        indices = find_matches(spot_key, image_query, top_k=200)  # [N_test, 200] indices. 
        pred = np.zeros((indices.shape[0], expression_key.shape[1]))   # init all 0 matrix for pred result. [N_test, 785]

        for i in range(indices.shape[0]):      # i from 0 to N_test-1.
            # a: [200] L1 distance in emb space from test spot i's query (img embedding) 
            # to each of its 200 retrieved neighbours (spot embedding) (smaller = more similar)
            a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)  # [200] L1 distance
            weights = np.reciprocal(a ** 2)             # 1 / a², a is smaller,  1 / a² is bigger, weight is bigger.
            weights = (weights / np.sum(weights)).flatten()  # wights normalize

            #weighted average of neighbours' real expression.
            pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

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
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]                              # drop NaN (gene constant on test slide -> r undefined)

        mse = mean_squared_error(true, pred)       # scalar, over all spots x 785 genes
        mae = mean_absolute_error(true, pred)      # scalar, over all spots x 785 genes

        heg_pcc_list.append(np.mean(heg_pcc))      # this fold's HEG-PCC = mean over the 50 genes
        hvg_pcc_list.append(np.mean(hvg_pcc))      # this fold's HVG-PCC = mean over all (non-NaN) genes
        mse_list.append(mse)                       # collect this fold's MSE
        mae_list.append(mae)                       # collect this fold's MAE
        print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, "
              f"HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    print(f"\n{'=' * 60}")
    print("Final Results (6-fold average):")
    print(f"{'=' * 60}")
    print(f"avg HEG PCC: {np.mean(heg_pcc_list):.4f}")
    print(f"avg HVG PCC: {np.mean(hvg_pcc_list):.4f}")
    print(f"avg MSE:     {np.mean(mse_list):.4f}")
    print(f"avg MAE:     {np.mean(mae_list):.4f}")


if __name__ == "__main__":
    main()

## Example Flow  ——  evel.py (Stage2 retrieval-based evaluation, 6-fold leave-one-slide-out)
## No prediction network: model only generates embeddings; expression = borrowed from retrieved neighbours (BLEEP-style)
## consts: image_dim=1280 (Virchow2), spot_dim=785, proj=256, top_k=200; N_test/N_train = spots in test/train slides
#
#  ── setup (once, shared by all folds) ─────────────────────────────
#    build_loaders_inference()   -> test_loader (6 slides concat, batch=32, shuffle=False)
#                                   datasets (list of 6), datasize=[N_A4,...,N_B3]
#    spot_expressions  = [np.load(preprocessed_matrix.npy) for 6 slides]   # each [785, N] REAL expression
#
#  ── for fold in range(6):  test slide = names[fold], train = other 5 ──
#
#    1. load model:  ContrastiveModel(cfg) + load fold{fold}/best.pt (strict=True); assert ckpt.fold == fold
#
#    2. get_embeddings(model, test_loader):        # same two projection head forward as training, but stop at embeddings
#         image tower: img -> image_cnn (Virchow2) -> image_encoder.encode (MLM encoder) -> image_projection
#         spot  tower: gene ->  spot_encoder.encode (MLM encoder) -> spot_projection
#         -> img_emb_all [total_spots, 256],  spot_emb_all [total_spots, 256]   (order A4..B3, shuffle=False)
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
#    5. retrieval:  indices = find_matches(spot_key, image_query, top_k=200)   # [N_test, 200]
#         per test spot: L2-normalize -> cosine sim [N_test, N_train] -> top-200 nearest train spots (indices)
#
#    6. predict (for each test spot i):     # model does NOT decode expression; it borrows neighbours' real expr
#         a       = L1 dist (emb) from query i to its 200 neighbours          [200]
#         weights = (1 / a^2) normalized                                      [200]  (closer -> larger weight)
#         pred[i] = weighted average of the 200 neighbours' REAL expression   [785]
#         -> pred [N_test, 785]
#
#    7. metrics (per fold):     # per-gene Pearson: predicted expr vs true expr , over allspots
#         HEG-PCC = mean Pearson over the 50 highest-expressed genes
#         HVG-PCC = mean Pearson over all 785 genes (drop NaN = constant genes)
#         MSE / MAE = over all spots x 785 genes
#
#  ── after 6 folds: average HEG-PCC / HVG-PCC / MSE / MAE across folds -> final result ──
#  KEY IDEA: image_query retrieves against spot_key (cross-modal), then borrows expression_key (real expr) as prediction.