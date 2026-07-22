#!/usr/bin/env python
# coding: utf-8

import numpy as np

import scanpy as sc
import pickle
import pandas as pd
import warnings
import os

import scprep as scp
warnings.filterwarnings("ignore")


# only need to run once to save hvg_matrix.npy
# filter expression matrices to only include HVGs shared across all datasets

def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)


def her2_hvg_selection_and_pooling(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)

    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]  # [spots, K] keep only genes shared across all slides
        print(adata.shape)
        # Preprocess the data
        sc.pp.normalize_total(adata)  # per-spot library-size normalize (row = spot here)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)  # flag top-1000 variable genes

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    hvg_union = hvg_bools[0]
    hvg_intersection = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
        print(sum(hvg_intersection), sum(hvg_bools[i]))
        hvg_intersection = hvg_intersection & hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())
    print("Number of HVGs (intersection): ", hvg_intersection.sum())

    with open('/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/her2_hvgs_intersection.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/her2_hvgs_union.pickle', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add all the HVGs

    gene_list_path = "/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))  ## 785 gene list -> fixed list[785] that truly decides the 785 dim

    hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())  # [spots,785].X -> .T -> [785, spots] dense ndarray
    return filtered_exp_mtxs  # list[6] of [785, spots] raw count, only 785 genes kept




names = os.listdir("/root/autodl-tmp/Her2st/data/ST-cnts")
names.sort()
names = ['A4', 'A5', 'A6', 'B1', 'B2', 'B3']   # 6 折:A4,A5,A6,B1,B2,B3(2 病人×3 张)
print(names)
print(len(names))
#
adata_list = [sc.AnnData(pd.read_csv(f"/root/autodl-tmp/Her2st/data/ST-cnts/{name}.tsv",
                                     sep='\t', index_col=0)) for name in names]  ## raw count data. each adata.X [spots, ~15638]
#
filtered_mtx = her2_hvg_selection_and_pooling(adata_list)  # list[6] of [785, spots]
#
for i in range(len(filtered_mtx)):
    pathset = f"/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/filtered_expression_matrices/her2/{names[i]}"
    if not (os.path.exists(pathset)):
        os.makedirs(pathset)

    np.save(f"/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/filtered_expression_matrices/her2/{names[i]}/hvg_matrix_plusmarkers.npy", filtered_mtx[i])  # save [785, spots] raw count (un-normalized)


def her2_pool_gene_list(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)

    hvg_bools = []

    gene_list_path = "/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())  # [spots,785] -> .T -> [785, spots] dense ndarray
    return filtered_exp_mtxs  # list[6] of [785, spots] raw count


adata_list = [sc.AnnData(pd.read_csv(f"/root/autodl-tmp/Her2st/data/ST-cnts/{name}.tsv",
                                     sep='\t', index_col=0)) for name in names]  # re-read 6 tsv (adata was subset in-place above)
filtered_mtx = her2_pool_gene_list(adata_list)  # [[785, spots], [785, spots], ...]


preprocessed_mtx = []
for i, mtx in enumerate(filtered_mtx):  # 6 slide，each [785, spots]. each slide raw count, filtered to 785 HVGs.
    # library_size_normalize scales each ROW to equal total; row=gene here -> normalizes per GENE, not per spot (orientation quirk)
    # log_transformed_expression = scp.transform.log(scp.normalize.library_size_normalize(mtx))  # [785, spots] lib-norm then log

    ## 归一化方向修改：# per-spot 归一化:先转成 [spots, genes] 让 scprep 按 spot(行)归一,再转回 [785, spots] 保存
    mtx_spot = mtx.T 
    normed   = scp.transform.log(scp.normalize.library_size_normalize(mtx_spot)) 
    log_transformed_expression = normed.T 

    preprocessed_mtx.append(log_transformed_expression)

    pathset = f"/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/preprocessed_expression_matrices/her2st/{names[i]}"
    if not os.path.exists(pathset):
        os.makedirs(pathset)
    np.save(f"/root/autodl-tmp/HE2spatial/HER2ST-MLM_V5/data/preprocessed_expression_matrices/her2st/{names[i]}/preprocessed_matrix.npy",
            log_transformed_expression)   # save [785, spots] (axes = [gene, spot]); the one product downstream actually reads
    print(f"her_data_preprocessed_mtx[{i}]:", log_transformed_expression.shape)

    # [785, spots]  normlaize + log_transform gene expression. 
    # Read in by stage1_dataset.py