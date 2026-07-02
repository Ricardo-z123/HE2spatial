import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention
from dataset import HERDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
from train import generate_args


def build_loaders_inference():
    """只调用一次，加载全部 32 个切片"""
    datasets = []
    for i in range(6):
        dataset = HERDataset(train=False, fold=i)
        print(dataset.id2name[0])
        datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Finished building loaders")
    return test_loader


def get_embeddings(model, test_loader):
    """用已加载的 test_loader 提取 embeddings"""
    model.eval()
    model = model.to('cuda')

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()     ### 输入1
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y

            spot_features = spot_feature.unsqueeze(dim=0)
            spot_embedding = model.spot_encoder(spot_features)
            spot_embedding = model.spot_projection(spot_embedding).squeeze(dim=0)
            spot_embeddings.append(spot_embedding)

    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    """找到最相似的 top_k 个 spots"""
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    return indices.cpu().numpy()


# ==================== 主程序 ====================

SAVE_EMBEDDINGS = True

names = os.listdir("/root/autodl-tmp/Her2st/data/ST-cnts")
names.sort()
names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']

datasize = [np.load(f"/root/autodl-tmp/mclSTExp/data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy").shape[1] 
            for name in names]

# ========== 只加载一次数据 ==========
print("Loading data once...")
test_loader = build_loaders_inference()

# ========== 初始化模型 ==========
args = generate_args()
model = mclSTExp_Attention(
    encoder_name=args.encoder_name,
    spot_dim=args.dim,
    temperature=args.temperature,
    image_dim=args.image_embedding_dim,
    projection_dim=args.projection_dim,
    heads_num=args.heads_num,
    heads_dim=args.heads_dim,
    head_layers=args.heads_layers,
    dropout=args.dropout
)

if SAVE_EMBEDDINGS:
    for fold in range(6):
        print(f"\n{'='*50}")
        print(f"Processing fold {fold}: {names[fold]}")
        print(f"{'='*50}")

        # ========== 加载模型权重 ==========
        model_path = f"/root/autodl-tmp/mclSTExp/model_result/her2st/{names[fold]}/best_{fold}.pt"
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')
            new_key = new_key.replace('well', 'spot')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        print(f"Loaded model: {model_path}")
        
        # ========== 提取 embeddings ==========
        img_embeddings_all, spot_embeddings_all = get_embeddings(model, test_loader)
        img_embeddings_all = img_embeddings_all.cpu().numpy()
        spot_embeddings_all = spot_embeddings_all.cpu().numpy()
        print("img_embeddings_all.shape", img_embeddings_all.shape)
        print("spot_embeddings_all.shape", spot_embeddings_all.shape)
        
        # ========== 保存 embeddings ==========
        save_path = f"/root/autodl-tmp/mclSTExp/embedding_result/her2st_result/embeddings_{fold}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(datasize)):
            index_start = sum(datasize[:i])
            index_end = sum(datasize[:i + 1])
            image_embeddings = img_embeddings_all[index_start:index_end]
            spot_embeddings = spot_embeddings_all[index_start:index_end]
            np.save(save_path + f"img_embeddings_{i + 1}.npy", image_embeddings.T)
            np.save(save_path + f"spot_embeddings_{i + 1}.npy", spot_embeddings.T)
        
        print(f"Saved embeddings to {save_path}")


# ==================== 评估部分 ====================
## 输入2
spot_expressions = [np.load(f"/root/autodl-tmp/mclSTExp/data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")
                    for name in names]   

# 加载基因列表
gene_list_path = "/root/autodl-tmp/mclSTExp/her_hvg_cut_1000.npy"
gene_list = list(np.load(gene_list_path, allow_pickle=True))

hvg_pcc_list = []
heg_pcc_list = []
mse_list = []
mae_list = []

# 用于存储所有 fold 的每个基因的 PCC
all_gene_pcc = []  # 每个元素是一个 fold 的 785 个基因的 PCC

for fold in range(6):
    save_path = f"/root/autodl-tmp/mclSTExp/embedding_result/her2st_result/embeddings_{fold}/"
    spot_embeddings = [np.load(save_path + f"spot_embeddings_{i + 1}.npy") for i in range(6)]
    image_embeddings = np.load(save_path + f"img_embeddings_{fold + 1}.npy")

    image_query = image_embeddings
    expression_gt = spot_expressions[fold]
    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    pred_save_path = f"/root/autodl-tmp/mclSTExp/her2st_pred_att/{names[fold]}/"
    os.makedirs(pred_save_path, exist_ok=True)
    
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
    
    # 保存每个基因的 PCC（用于最后计算 top 20）
    hvg_pcc_clean = hvg_pcc.copy()
    hvg_pcc_clean[np.isnan(hvg_pcc_clean)] = 0  # 将 NaN 替换为 0
    all_gene_pcc.append(hvg_pcc_clean)
    
    hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

    heg_pcc_list.append(np.mean(heg_pcc))
    hvg_pcc_list.append(np.mean(hvg_pcc))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true, pred)
    mse_list.append(mse)
    mae = mean_absolute_error(true, pred)
    mae_list.append(mae)
    
    # ========== 输出每个 fold 的完整指标 ==========
    print(f"Fold {fold} ({names[fold]}) - HEG PCC: {np.mean(heg_pcc):.4f}, HVG PCC: {np.mean(hvg_pcc):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")


# ==================== 计算每个基因在所有 fold 上的平均 PCC ====================

all_gene_pcc = np.array(all_gene_pcc)  # [17, 785]
mean_gene_pcc = np.mean(all_gene_pcc, axis=0)  # [785]

# 找出 top 20 基因
top_20_indices = np.argsort(mean_gene_pcc)[::-1][:20]
top_20_genes = [gene_list[i] for i in top_20_indices]
top_20_pcc = mean_gene_pcc[top_20_indices]


# ==================== 输出最终结果 ====================

print(f"\n{'='*60}")
print("Final Results:")
print(f"{'='*60}")
print(f"avg HEG PCC: {np.mean(heg_pcc_list):.4f}")
print(f"avg HVG PCC: {np.mean(hvg_pcc_list):.4f}")
print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")

print(f"\n{'='*60}")
print("Top 20 Genes by Average PCC:")
print(f"{'='*60}")
print(f"{'Rank':<6}{'Gene':<15}{'Avg PCC':<10}")
print(f"{'-'*31}")
for rank, (gene, pcc) in enumerate(zip(top_20_genes, top_20_pcc), 1):
    print(f"{rank:<6}{gene:<15}{pcc:.4f}")

# 保存 top 20 基因到文件
top20_save_path = "/root/autodl-tmp/mclSTExp/her2st_pred_att/top20_genes.txt"
with open(top20_save_path, 'w') as f:
    f.write("Rank\tGene\tAvg_PCC\n")
    for rank, (gene, pcc) in enumerate(zip(top_20_genes, top_20_pcc), 1):
        f.write(f"{rank}\t{gene}\t{pcc:.4f}\n")
print(f"\nTop 20 genes saved to: {top20_save_path}")