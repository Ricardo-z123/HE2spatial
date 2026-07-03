import argparse
import math
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
from engine.stage2_train import load_stage1_weights
from utils.misc import get_R


class SlideBatchSampler(torch.utils.data.Sampler):
    """推理专用：按 slide 分批，batch 内全部来自同一 slide，不打乱顺序。

    Each input dataset is expected to be single-slide
    (HER2STStage2Dataset(train=False, fold=i) yields exactly 1 slide via te_names),
    so wrapping the list-of-datasets and chunking each slide separately gives
    slide-aware batching automatically — same trick as experimental L16-33.
    """
    def __init__(self, datasets, batch_size):
        self.batch_size = batch_size
        self.slide_indices = []
        prev = 0
        for ds in datasets:
            n = len(ds)
            self.slide_indices.append(list(range(prev, prev + n)))
            prev += n

    def __iter__(self):
        for indices in self.slide_indices:
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(idx) / self.batch_size) for idx in self.slide_indices)


def build_loaders_inference():
    """只调用一次，加载全部 6 个 fold 的 test 切片（每个=1 张 slide），合并成单 DataLoader。

    All datasets use train=False so transforms reduce to ToTensor() — no augmentation
    on inference reference embeddings. Mirrors experimental L36-49.
    """
    datasets = []
    for i in range(6):
        dataset = HER2STStage2Dataset(train=False, fold=i)
        print(dataset.id2name[0])
        datasets.append(dataset)

    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    sampler = SlideBatchSampler(datasets, batch_size=32)
    test_loader = DataLoader(concat_dataset, batch_sampler=sampler, num_workers=0)

    print("Finished building loaders")
    return test_loader, datasets


def get_embeddings(model, test_loader):
    """用已加载的 test_loader 提取 image / spot embeddings + ground-truth gene expression。

    Mirrors experimental L52-78. Only difference: encoder接口 changes from baseline's
    monolithic image_encoder / nn.Sequential spot_encoder to ContrastiveModel's
    three-stage layout (CNN -> Transformer encode -> projection).
    Output:
      image_embs   [total_spots, 256]
      spot_embs    [total_spots, 256]
      gene_truths  [total_spots, 785]   used as Pearson R target
    """
    model.eval()
    model = model.to('cuda')

    test_image_embeddings = []
    spot_embeddings_list = []
    gene_truths = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            img = batch["image"].cuda()                                   # [N, 3, 224, 224]
            gene = batch["expression"].cuda()                             # [N, 785]
            coords = batch["position"].long().cuda().unsqueeze(0)         # [1, N, 2]

            # Image branch: CNN -> Transformer encode -> projection
            cnn_feat = model.image_cnn(img)                               # [N, 1024]
            img_feat = cnn_feat.unsqueeze(0)                              # [1, N, 1024]
            img_feat = model.image_encoder.encode(img_feat, coords)       # [1, N, 1024]
            img_feat = img_feat.squeeze(0)                                # [N, 1024]
            image_embeddings = model.image_projection(img_feat)           # [N, 256]
            test_image_embeddings.append(image_embeddings)

            # Spot branch: gene -> MaskedEncoder.encode -> projection
            gene_seq = gene.unsqueeze(dim=0)                              # [1, N, 785]
            spot_features = model.spot_encoder.encode(gene_seq, coords)   # [1, N, 785]
            spot_features = spot_features.squeeze(0)                      # [N, 785]
            spot_embedding = model.spot_projection(spot_features)         # [N, 256]
            spot_embeddings_list.append(spot_embedding)

            gene_truths.append(gene)                                      # [N, 785]

    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings_list), torch.cat(gene_truths)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    """找到最相似的 top_k 个 spots（实验代码 L81-90 verbatim）"""
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
    parser.add_argument('--stage2_ckpt', type=str, required=True, help='Stage 2 best.pt')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']  # 6 slides

    # ========== 只加载一次数据 ==========
    print("Loading data once...")
    test_loader, datasets = build_loaders_inference()
    datasize = [len(ds) for ds in datasets]

    # ========== 初始化模型 ==========
    model = ContrastiveModel(cfg)

    # ========== 加载模型权重 ==========
    # Stage 1 encoders first (defensive: in case stage2 ckpt is partial),
    # then Stage 2 ckpt overwrites projection heads + transformer fine-tunes.
    load_stage1_weights(model, cfg['spot_ckpt'], cfg['image_ckpt'])
    stage2_ckpt = torch.load(args.stage2_ckpt, map_location='cpu')
    model.load_state_dict(stage2_ckpt['model_state'], strict=False)
    print(f"Loaded Stage 2 ckpt from epoch {stage2_ckpt['epoch']}, train_loss={stage2_ckpt['loss']:.4f}")

    # ========== 提取 embeddings (一次性提取所有 6 张 slide 的) ==========
    img_embeddings_all, spot_embeddings_all, gene_truths_all = get_embeddings(model, test_loader)
    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    gene_truths_all = gene_truths_all.cpu().numpy()
    print("img_embeddings_all.shape", img_embeddings_all.shape)
    print("spot_embeddings_all.shape", spot_embeddings_all.shape)

    # ========== 切分到 per-slide（保持 [256, n_i] / [785, n_i] 列向布局，对齐实验代码 L155-156）==========
    image_embeddings_per_slide = []
    spot_embeddings_per_slide = []
    spot_expressions_per_slide = []
    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings_per_slide.append(img_embeddings_all[index_start:index_end].T)
        spot_embeddings_per_slide.append(spot_embeddings_all[index_start:index_end].T)
        spot_expressions_per_slide.append(gene_truths_all[index_start:index_end].T)

    # ==================== 评估部分 ====================

    # 加载基因列表
    gene_list_path = './data/her_hvg_cut_1000.npy'
    gene_list = list(np.load(gene_list_path, allow_pickle=True))

    # 第一轮只评估 fold=cfg['fold']（实验代码是 6-fold loop，留作 TODO）
    fold = cfg['fold']

    image_query = image_embeddings_per_slide[fold]
    expression_gt = spot_expressions_per_slide[fold]
    spot_embeddings = spot_embeddings_per_slide[:fold] + spot_embeddings_per_slide[fold + 1:]
    spot_expressions_rest = spot_expressions_per_slide[:fold] + spot_expressions_per_slide[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    # 形状校正（实验代码 L194-201 verbatim）
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

    # 计算 HEG (top 50 highly expressed genes) PCC
    gene_mean_expression = np.mean(adata_ture.X, axis=0)
    top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
    top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
    top_50_genes_expression = adata_ture[:, top_50_genes_names]
    top_50_genes_pred = adata_pred[:, top_50_genes_names]

    heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)

    # 计算 HVG (all 785 genes) PCC
    hvg_pcc, hvg_p = get_R(adata_pred, adata_ture)

    hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)

    # ========== 输出当前 fold 的完整指标 ==========
    print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, "
          f"HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    # TODO: 6-fold loop + top-20 genes ranking after fold=0 verified


if __name__ == "__main__":
    main()
