import argparse
import os
import yaml
import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from datasets.stage2_dataset import HER2STStage2Dataset
from models.contrastive_model import ContrastiveModel
from utils.misc import get_R


# =============================================================================
# Cross-slide inference (baseline-style) — paired with engine/all_shuffle_train.py.
# vs engine/inference.py: this file uses default DataLoader (cross-slide batches),
# matching all_shuffle_train.py's training-time batch strategy. inference.py uses
# SlideBatchSampler. Train/inference batching must match — switching strategies
# would feed the encoders a context distribution they didn't train on.
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
    """Mirrors baseline L30-55, with ContrastiveModel interface adaptation:
      - baseline:  model.image_encoder(...) -> model.image_projection
      - ours:      model.image_cnn -> model.image_encoder.encode -> model.image_projection
      - baseline:  manual x_embed/y_embed PE + nn.Sequential spot_encoder loop
      - ours:      model.spot_encoder.encode (PE built into MaskedEncoder) -> model.spot_projection
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
            image_embeddings = model.image_projection(img_feat)            # [N, 256]
            test_image_embeddings.append(image_embeddings)

            # Spot branch: gene -> MaskedEncoder.encode -> projection
            gene_seq = gene.unsqueeze(dim=0)                               # [1, N, 785]
            spot_features = model.spot_encoder.encode(gene_seq, coords)    # [1, N, 785]
            spot_features = spot_features.squeeze(0)                       # [N, 785]
            spot_embedding = model.spot_projection(spot_features)          # [N, 256]
            spot_embeddings.append(spot_embedding)

    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    """找到最相似的 top_k 个 spots（baseline evel-1-32.py L58-67 verbatim）"""
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    return indices.cpu().numpy()


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Stage 2 yaml config')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']

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

    for fold in range(6):
        print(f"\n{'='*60}")
        print(f"Processing fold {fold} ({names[fold]})")
        print(f"{'='*60}")

        # ===== 加载该 fold 的模型 =====
        ckpt_path = os.path.join(cfg['save_dir'], f'fold{fold}', 'best.pt')
        model = ContrastiveModel(cfg)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_fold = ckpt.get('cfg', {}).get('fold', None)
        assert ckpt_fold == fold, (
            f"ckpt was trained on fold={ckpt_fold}, but expected fold={fold}"
        )
        model.load_state_dict(ckpt['model_state'], strict=True)
        print(f"Loaded ckpt from {ckpt_path} (epoch {ckpt['epoch']}, train_loss={ckpt['loss']:.4f})")

        # ===== 提取所有 6 张 slide 的 embeddings =====
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
        image_query = image_embeddings_per_slide[fold]
        expression_gt = spot_expressions_per_slide[fold]
        spot_embeddings_rest = spot_embeddings_per_slide[:fold] + spot_embeddings_per_slide[fold + 1:]
        spot_expressions_rest = spot_expressions_per_slide[:fold] + spot_expressions_per_slide[fold + 1:]

        spot_key = np.concatenate(spot_embeddings_rest, axis=1)
        expression_key = np.concatenate(spot_expressions_rest, axis=1)

        if image_query.shape[1] != 256:
            image_query = image_query.T
        if expression_gt.shape[0] != image_query.shape[0]:
            expression_gt = expression_gt.T
        if spot_key.shape[1] != 256:
            spot_key = spot_key.T
        if expression_key.shape[0] != spot_key.shape[0]:
            expression_key = expression_key.T

        indices = find_matches(spot_key, image_query, top_k=200)

        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
            reciprocal_of_square_a = np.reciprocal(a ** 2)
            weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
            weights = weights.flatten()
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

        true = expression_gt
        pred = matched_spot_expression_pred

        adata_ture = anndata.AnnData(true)
        adata_pred = anndata.AnnData(pred)
        adata_pred.var_names = gene_list
        adata_ture.var_names = gene_list

        gene_mean_expression = np.mean(adata_ture.X, axis=0)
        top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
        top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
        top_50_genes_expression = adata_ture[:, top_50_genes_names]
        top_50_genes_pred = adata_pred[:, top_50_genes_names]

        heg_pcc, _ = get_R(top_50_genes_pred, top_50_genes_expression)
        hvg_pcc, _ = get_R(adata_pred, adata_ture)
        hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)

        heg_pcc_list.append(np.mean(heg_pcc))
        hvg_pcc_list.append(np.mean(hvg_pcc))
        mse_list.append(mse)
        mae_list.append(mae)

        print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # ===== 最终平均结果 =====
    print(f"\n{'='*60}")
    print("Final Results (6-fold average):")
    print(f"{'='*60}")
    print(f"avg HEG PCC: {np.mean(heg_pcc_list):.4f}")
    print(f"avg HVG PCC: {np.mean(hvg_pcc_list):.4f}")
    print(f"avg MSE:     {np.mean(mse_list):.4f}")
    print(f"avg MAE:     {np.mean(mae_list):.4f}")


if __name__ == "__main__":
    main()
