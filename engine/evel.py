import argparse
import os
import json
import yaml
import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# mclSTExp baseline targets (same 6-fold leave-one-slide-out), for side-by-side logging.
MCLSTEXP_TARGET = {
    'per_fold': {
        'A4': {'heg': 0.1863, 'hvg': 0.0895, 'mse': 0.6310, 'mae': 0.6368},
        'A5': {'heg': 0.1667, 'hvg': 0.0898, 'mse': 0.6414, 'mae': 0.6411},
        'A6': {'heg': 0.2568, 'hvg': 0.0927, 'mse': 0.6286, 'mae': 0.6355},
        'B1': {'heg': 0.5908, 'hvg': 0.4360, 'mse': 0.5335, 'mae': 0.4881},
        'B2': {'heg': 0.6346, 'hvg': 0.4790, 'mse': 0.5548, 'mae': 0.5027},
        'B3': {'heg': 0.5630, 'hvg': 0.4162, 'mse': 0.5204, 'mae': 0.5083},
    },
    'average': {'heg': 0.3997, 'hvg': 0.2672, 'mse': 0.5850, 'mae': 0.5688},
}

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from utils.misc import get_R


# =============================================================================
# Cross-slide inference (baseline-style) — paired with engine/all_shuffle_train.py.
# vs engine/inference.py: this file uses default DataLoader (cross-slide batches),
# matching all_shuffle_train.py's training-time batch strategy. 
# =============================================================================


def build_loaders_inference():
    """加载 6 个 fold 的 test slide（每个 fold 的 test = 1 张 slide），合并成单 DataLoader。
    跟 baseline evel-1-32.py L15-27 一致：默认 sampler + shuffle=False，
    """
    datasets = []
    for i in range(6):
        dataset = HER2STStage2Dataset(train=False, fold=i)
        print(dataset.id2name[0])
        datasets.append(dataset)

    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(concat_dataset, batch_size=32, shuffle=False, num_workers=0)
    print("Finished building loaders")
    return test_loader, datasets


def get_embeddings(model, test_loader, device):
    """ ContrastiveModel 
      - model.image_cnn -> model.image_encoder.encode -> model.image_projection
      - model.spot_encoder.encode (PE built into MaskedEncoder) -> model.spot_projection
    Output:
      image_emb     [total_spots, 256]
      spot_emb      [total_spots, 256]
    Ground-truth expressions are loaded from preprocessed_matrix.npy in main(), matching baseline.
    """
    model.eval()
    model = model.to(device)

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            img = batch["image"].to(device)                                # [N, 3, 224, 224]
            gene = batch["expression"].to(device)                          # [N, 785]
            coords = batch["position"].long().to(device).unsqueeze(0)      # [1, N, 2]

            # Image branch: CNN -> Transformer encode -> projection
            cnn_feat = model.image_cnn(img)                                # [N, 1024]
            img_feat = cnn_feat.unsqueeze(0)                               # [1, N, 1024]
            img_feat = model.image_encoder.encode(img_feat, coords)        # [1, N, 1024]
            img_feat = img_feat.squeeze(0)                                 # [N, 1024]
            image_embeddings = model.image_projection(img_feat)            # [N, 256] image embedding (becomes retrieval QUERY)
            test_image_embeddings.append(image_embeddings)

            # Spot branch: gene -> MaskedEncoder.encode -> projection
            gene_seq = gene.unsqueeze(dim=0)                               # [1, N, 785]
            spot_features = model.spot_encoder.encode(gene_seq, coords)    # [1, N, 785]
            spot_features = spot_features.squeeze(0)                       # [N, 785]
            spot_embedding = model.spot_projection(spot_features)          # [N, 256] spot embedding (becomes retrieval KEY)
            spot_embeddings.append(spot_embedding)

    # img_emb [total_spots, 256] (query pool), spot_emb [total_spots, 256] (key pool); order = A4..B3 (shuffle=False)
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def get_embeddings_per_slide(model, datasets, device, cnn_chunk=64):
    model.eval()
    model = model.to(device)
    img_out, spot_out = [], []
    with torch.no_grad():
        for ds in datasets:                      # each ds = one slide (test split of a fold)
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            batch = next(iter(loader))
            img = batch["image"].to(device)                              # [n, 3, 224, 224]
            gene = batch["expression"].to(device)                        # [n, 785]
            coords = batch["position"].long().to(device).unsqueeze(0)    # [1, n, 2]

            feats = []
            for s in range(0, img.shape[0], cnn_chunk):
                feats.append(model.image_cnn(img[s:s + cnn_chunk]))      # Virchow2 in chunks
            cnn_feat = torch.cat(feats, dim=0)                           # [n, 1280]
            img_feat = model.image_encoder.encode(cnn_feat.unsqueeze(0), coords).squeeze(0)  # full-slide attn
            img_out.append(model.image_projection(img_feat))            # [n, 256]

            spot_feat = model.spot_encoder.encode(gene.unsqueeze(0), coords).squeeze(0)
            spot_out.append(model.spot_projection(spot_feat))           # [n, 256]
    return torch.cat(img_out), torch.cat(spot_out)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    """ Find most similar top_k spots """
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)  # L2-normalize -> dot product = cosine
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T  # [N_test, N_train] cosine sim of every test img vs every train spot
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)  # per test spot, indices of its top_k most similar train spots
    return indices.cpu().numpy()  # [N_test, top_k] indices into the train key/expression pool


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Stage 2 yaml config')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='override cfg["save_dir"]: dir containing fold{i}/best.pt to evaluate')
    parser.add_argument('--tag', type=str, default=None,
                        help='name for the saved metrics files (default: basename of ckpt dir)')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='where to write <tag>_metrics.json / <tag>_metrics.md')
    parser.add_argument('--encode_mode', type=str, default='batch', choices=['batch', 'per_slide'],
                        help="B3/D4: 'batch'=original 32-spot batches (default, =EXP-1); "
                             "'per_slide'=encode each slide's spots as one sequence (no cross-batch contamination)")
    parser.add_argument('--weight_mode', type=str, default='raw', choices=['raw', 'normed', 'cossoft'],
                        help="'raw'=1/L1^2 on raw emb (=EXP-1); 'normed'=1/L1^2 on L2-normalized emb; "
                             "'cossoft'=softmax(cosine/temp) over top-k (sharper, anti-oversmoothing)")
    parser.add_argument('--top_k', type=int, default=200,
                        help="retrieval neighbours to aggregate (default 200=EXP-1; 50=BLEEP, less over-smoothing)")
    parser.add_argument('--cossoft_temp', type=float, default=0.05,
                        help="temperature for weight_mode=cossoft")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']

    ckpt_base = args.ckpt_dir if args.ckpt_dir else cfg['save_dir']
    tag = args.tag if args.tag else os.path.basename(ckpt_base.rstrip('/'))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Evaluating checkpoints in: {ckpt_base}  (tag={tag})")

    # ===== 加载数据一次（所有 fold 共用）=====
    print("Loading data once...")
    test_loader, datasets = build_loaders_inference()
    datasize = [len(ds) for ds in datasets]

    spot_expressions_per_slide = [
        np.load(f"/root/autodl-tmp/HER2ST-MLM/data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")
        for name in names
    ]
    gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))

    heg_pcc_list = []
    hvg_pcc_list = []
    mse_list = []
    mae_list = []
    per_fold_records = []  # structured per-fold metrics for JSON/MD export

    for fold in range(6):
        print(f"\n{'='*60}")
        print(f"Processing fold {fold} ({names[fold]})")
        print(f"{'='*60}")

        # ===== 加载该 fold 的模型 =====
        ckpt_path = os.path.join(ckpt_base, f'fold{fold}', 'best.pt')
        model = ContrastiveModel(cfg)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_fold = ckpt.get('cfg', {}).get('fold', None)
        assert ckpt_fold == fold, (
            f"ckpt was trained on fold={ckpt_fold}, but expected fold={fold}"
        )
        model.load_state_dict(ckpt['model_state'], strict=True)
        print(f"Loaded ckpt from {ckpt_path} (epoch {ckpt['epoch']}, train_loss={ckpt['loss']:.4f})")

        # ===== 提取所有 6 张 slide 的 embeddings =====
        if args.encode_mode == 'per_slide':
            img_embeddings_all, spot_embeddings_all = get_embeddings_per_slide(model, datasets, device)
        else:
            img_embeddings_all, spot_embeddings_all = get_embeddings(model, test_loader, device)
        img_embeddings_all = img_embeddings_all.cpu().numpy()
        spot_embeddings_all = spot_embeddings_all.cpu().numpy()

        # ===== 切分 per-slide embeddings =====
        image_embeddings_per_slide = []
        spot_embeddings_per_slide = []
        for i in range(len(datasize)):
            s, e = sum(datasize[:i]), sum(datasize[:i + 1])
            image_embeddings_per_slide.append(img_embeddings_all[s:e].T)
            spot_embeddings_per_slide.append(spot_embeddings_all[s:e].T)

        # ===== 评估该 fold =====
        image_query = image_embeddings_per_slide[fold]      # QUERY: held-out test slide's image emb [256, N_test]
        expression_gt = spot_expressions_per_slide[fold]    # ground-truth expression of test slide [785, N_test]
        spot_embeddings_rest = spot_embeddings_per_slide[:fold] + spot_embeddings_per_slide[fold + 1:]   # other 5 slides' spot emb
        spot_expressions_rest = spot_expressions_per_slide[:fold] + spot_expressions_per_slide[fold + 1:] # other 5 slides' true expr

        spot_key = np.concatenate(spot_embeddings_rest, axis=1)        # KEY pool: train spot emb [256, N_train]
        expression_key = np.concatenate(spot_expressions_rest, axis=1) # neighbour true-expression pool [785, N_train]

        pdim = cfg['projection_dim']  # was hardcoded 256; use actual proj dim so E11 (512) etc. work
        if image_query.shape[1] != pdim:
            image_query = image_query.T
        if expression_gt.shape[0] != image_query.shape[0]:
            expression_gt = expression_gt.T
        if spot_key.shape[1] != pdim:
            spot_key = spot_key.T
        if expression_key.shape[0] != spot_key.shape[0]:
            expression_key = expression_key.T

        indices = find_matches(spot_key, image_query, top_k=args.top_k)  # [N_test, top_k] nearest train spots per test spot

        # weighting over the retrieved top-k. raw=1/L1^2 on raw emb (EXP-1);
        # normed=1/L1^2 on L2-normalized emb; cossoft=softmax(cosine/temp) (sharper).
        skn = spot_key / (np.linalg.norm(spot_key, axis=1, keepdims=True) + 1e-8)
        iqn = image_query / (np.linalg.norm(image_query, axis=1, keepdims=True) + 1e-8)

        # BLEEP-style retrieval prediction: pred = weighted avg of neighbours' REAL expression (NOT a regression/decode)
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))  # pred expr [N_test, 785]
        for i in range(indices.shape[0]):                    # for each test spot, weight then average its top_k neighbours
            if args.weight_mode == 'cossoft':
                cos = skn[indices[i, :], :] @ iqn[i, :]          # [k] cosine sims (normalized)
                z = cos / args.cossoft_temp
                z = z - z.max()
                weights = np.exp(z)
                weights = (weights / weights.sum()).flatten()
            else:
                sk_w, iq_w = (skn, iqn) if args.weight_mode == 'normed' else (spot_key, image_query)
                a = np.linalg.norm(sk_w[indices[i, :], :] - iq_w[i, :], axis=1, ord=1)  # [k] L1 dist query->each neighbour
                weights = np.reciprocal(a ** 2)              # closer neighbour -> larger weight (1 / L1^2)
                weights = (weights / np.sum(weights)).flatten()
            # weighted avg of the k neighbours' real expression -> predicted expression [785] for test spot i
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

        true = expression_gt                      # [N_test, 785] ground-truth expression
        pred = matched_spot_expression_pred       # [N_test, 785] retrieval-aggregated prediction

        adata_ture = anndata.AnnData(true)
        adata_pred = anndata.AnnData(pred)
        adata_pred.var_names = gene_list
        adata_ture.var_names = gene_list

        gene_mean_expression = np.mean(adata_ture.X, axis=0)              # [785] mean expression per gene over test spots
        top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]  # 50 highest-expressed genes (HEG set)
        top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
        top_50_genes_expression = adata_ture[:, top_50_genes_names]       # true subset [N_test, 50]
        top_50_genes_pred = adata_pred[:, top_50_genes_names]             # pred subset [N_test, 50]

        # HEG-PCC: per-gene Pearson r (over spots), on the 50 highest-expressed genes; get_R returns r per gene
        heg_pcc, _ = get_R(top_50_genes_pred, top_50_genes_expression)    # [50]
        hvg_pcc, _ = get_R(adata_pred, adata_ture)                        # HVG-PCC: per-gene Pearson r over all 785 genes [785]
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]                             # drop NaN (gene constant on test slide -> undefined r)

        mse = mean_squared_error(true, pred)      # scalar, over all spots & all 785 genes
        mae = mean_absolute_error(true, pred)     # scalar, over all spots & all 785 genes

        heg_pcc_list.append(np.mean(heg_pcc))   # this fold's HEG-PCC = mean over the 50 genes
        hvg_pcc_list.append(np.mean(hvg_pcc))   # this fold's HVG-PCC = mean over all (non-NaN) genes
        mse_list.append(mse)
        mae_list.append(mae)

        per_fold_records.append({
            'fold': fold,
            'slide': names[fold],
            'heg': float(np.mean(heg_pcc)),
            'hvg': float(np.mean(hvg_pcc)),
            'mse': float(mse),
            'mae': float(mae),
        })

        print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # ===== 最终平均结果 =====
    avg = {
        'heg': float(np.mean(heg_pcc_list)),
        'hvg': float(np.mean(hvg_pcc_list)),
        'mse': float(np.mean(mse_list)),
        'mae': float(np.mean(mae_list)),
    }
    print(f"\n{'='*60}")
    print("Final Results (6-fold average):")
    print(f"{'='*60}")
    print(f"avg HEG PCC: {avg['heg']:.4f}")
    print(f"avg HVG PCC: {avg['hvg']:.4f}")
    print(f"avg MSE:     {avg['mse']:.4f}")
    print(f"avg MAE:     {avg['mae']:.4f}")

    # ===== 保存指标到 JSON + Markdown(保留上面的 print)=====
    result = {
        'tag': tag,
        'ckpt_dir': ckpt_base,
        'config': args.config,
        'encode_mode': args.encode_mode,
        'weight_mode': args.weight_mode,
        'top_k': args.top_k,
        'per_fold': per_fold_records,
        'average': avg,
        'mclstexp_target': MCLSTEXP_TARGET,
    }
    json_path = os.path.join(args.out_dir, f'{tag}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    md_path = os.path.join(args.out_dir, f'{tag}_metrics.md')
    tgt_pf = MCLSTEXP_TARGET['per_fold']
    tgt_avg = MCLSTEXP_TARGET['average']
    lines = []
    lines.append(f"# Eval metrics — `{tag}`")
    lines.append("")
    lines.append(f"- checkpoints: `{ckpt_base}`")
    lines.append(f"- config: `{args.config}`")
    lines.append("")
    lines.append("| fold | slide | HEG | HVG | MSE | MAE | "
                 "mcl HEG | mcl HVG | mcl MSE | mcl MAE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in per_fold_records:
        t = tgt_pf[r['slide']]
        lines.append(
            f"| {r['fold']} | {r['slide']} | {r['heg']:.4f} | {r['hvg']:.4f} | "
            f"{r['mse']:.4f} | {r['mae']:.4f} | {t['heg']:.4f} | {t['hvg']:.4f} | "
            f"{t['mse']:.4f} | {t['mae']:.4f} |")
    lines.append(
        f"| **avg** | — | **{avg['heg']:.4f}** | **{avg['hvg']:.4f}** | "
        f"**{avg['mse']:.4f}** | **{avg['mae']:.4f}** | {tgt_avg['heg']:.4f} | "
        f"{tgt_avg['hvg']:.4f} | {tgt_avg['mse']:.4f} | {tgt_avg['mae']:.4f} |")
    lines.append("")
    with open(md_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved metrics -> {json_path}")
    print(f"Saved metrics -> {md_path}")


if __name__ == "__main__":
    main()
