#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import os
import sys
sys.path.append('/home/chenjn/biock')

import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import episcanpy as esp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack
from biock import random_string
from biock.genomics.single_cell import tfidf_transform
import utils
from biock import HUMAN_CHROMS_NO_Y_MT
import logging
logger = logging.getLogger(__name__)

LIBRARY_SIZE_KEY = "__library_size__"
RAWCOUNT_KEY = "__raw_count__"
ATAC_LIB_SIZE = 1000
RNA_LIB_SIZE = 2000
LIB_SCALE = 1000

def get_adata_stats(adata: AnnData) -> Dict[str, Any]:
    stats = {
        "shape": adata.shape,
        "X.data(min/mean/max)": (adata.X.data.min(), adata.X.data.mean(), adata.X.data.max()),
        "density": round(np.sum(adata.X.data > 0) / adata.shape[0] / adata.shape[1], 4)
    }
    return stats


def load_adata(
        h5ad: Union[str, List[str]], 
        count_key: str, # batch_key: str, label_key: str,
        log1p: bool, binarize: bool, tfidf: bool, 
        min_cells: int=None, max_cells: int=None, 
        min_genes: int=None, max_genes: int=None, 
        # clip_high: float=0
    ) -> AnnData:
    """
    clip_high: remove outliers with extremely high values
    keep_counts: keep raw values
    """
    assert not (log1p and binarize), "log1p and binarize should not be used simutanously"
    # assert clip_high < 0.5, "clip ratio should be below 0.5 (50%)"

    if type(h5ad) is str:
        h5ad = [h5ad]
    adata = list()

    for fn in h5ad:
        adata.append(sc.read_h5ad(fn))
        if not issparse(adata[-1].X):
            adata[-1].X = csr_matrix(adata[-1].X)
        logger.info("loaded {}: {}".format(fn, get_adata_stats(adata[-1])))
        assert np.array_equal(adata[0].var.index, adata[-1].var.index), "only anndata with identical var could be concaticated"
        if 'batch' not in adata[-1].obs:
            adata[-1].obs["batch"] = fn
    adata = ad.concat(adata, axis=0)
    adata.obs["lib_size"] = adata.X.sum(axis=1).A1 # save unfiltered counts
    
    # save count/label/batch key in AnnData.uns
    # adata.uns["label_key"] = label_key
    # adata.uns["batch_key"] = batch_key
    # import pdb;pdb.set_trace()

    if count_key is not None and count_key != "X":
        adata.uns["count_key"] = count_key
        adata.X = adata.layers[count_key].copy()
    else:
        count_key = "counts"
        adata.uns["count_key"] = "counts"
        adata.layers[count_key] = adata.X.copy()
    if type(adata.X) is not csr_matrix:
        adata.X = csr_matrix(adata.X)
    raw_shape = adata.shape

    ## filtering
    if min_cells is not None:
        if min_cells < 1:
            min_cells = int(round(min_cells * adata.shape[0]))
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_cells is not None:
        if max_cells < 1:
            max_cells = int(round(max_cells * adata.shape[0]))
        sc.pp.filter_genes(adata, max_cells=max_cells)
    logger.info("  filtering gene: {}->{}".format(raw_shape, adata.shape))

    if min_genes is not None:
        if min_genes < 1:
            min_genes = int(round(min_genes * adata.shape[1]))
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        if max_genes < 1:
            max_genes = int(round(max_genes * adata.shape[1]))
        sc.pp.filter_cells(adata, max_genes=max_genes)
    logger.info("  filtering cell: {}->{}".format(raw_shape, adata.shape))
    logger.info("  stats after filtering: {}".format(get_adata_stats(adata)))

    if log1p:
        if not check_counts(adata.X):
            raise ValueError("Data in {} has been normalized and cannot be log-transformed".format(h5ad))
        # sc.pp.normalize_total(adata, target_sum=1E6)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        logger.info("  total normalized and log-transformed, stats: {}".format(get_adata_stats(adata)))
    elif binarize:
        adata.X.data = (adata.X.data > 0).astype(np.float32)
        logger.info("  binarization transformation, stats: {}".format(get_adata_stats(adata)))

    if tfidf:
        if adata.X.data.max() > 1:
            logger.warning("X in adata has not been binarized!")
        adata.X = tfidf_transform(adata.X, norm=None)
        logger.info("- using 'None' norm in TfidfTransformer")
    logger.info("finished")

    # half float max
    hf_max = np.finfo(np.float16).max
    if adata.X.max() > hf_max:
        logger.warning("values in X exceeding {} were set to {}".format(hf_max, hf_max))
        adata.X.data = np.minimum(hf_max, adata.X.data)
    if adata.layers[count_key].max() > hf_max:
        logger.warning("counts exceeding {} were set to {}".format(hf_max, hf_max))
        adata.layers[count_key].data = np.minimum(hf_max, adata.layers[count_key].data)

    if "CellType" in adata.obs.columns and "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"]=adata.obs["CellType"]

    return adata

class PairedModeDataset(Dataset):
    def __init__(self,
                 atac: AnnData,
                 mrna: AnnData,
                 batch_key: str,
                 select_peak: Literal["var", "flanking"],
                 gene_info: Dict[str, List[str]],
                 n_top_genes=None,
                 n_top_peaks=None,
                 flanking: int=100000,
                 aug_func: str=None,
                 aug_num: int=None,
                 with_fea: str=None,
                 **kwargs):
        super(PairedModeDataset, self).__init__()
        # logger.info("- {} info:{}".format(__class__.__name__,))
        self.aug_func = aug_func
        self.aug_num = aug_num

        if select_peak == "var":
            assert n_top_peaks is not None, "n_top_peaks is required to select peak by `var`"

        # if "chrom" not in mrna.var.columns:
        #     assert gene_info is not None, "gene_info is required to obtain tss information"
        #     logger.info("  add gene info ...")
        #     mrna = utils.add_gene_info(mrna, gene_info=gene_info)
        # if "chrom" not in atac.var.columns:
        #     logger.info("  add chrom info ...")
        #     atac.var["chrom"] = [c.split(':')[0] for c in atac.var.index]
        
        # ## remove non-chromosome gene/peak
        # keep_gene = np.arange(mrna.shape[1])[np.isin(mrna.var["chrom"], list(HUMAN_CHROMS_NO_Y_MT))]
        # if len(keep_gene) < mrna.shape[1]:
        #     mrna = mrna[:, keep_gene]
        #     logger.warning("- removing {} genes on chrY/chrM/contigs".format(mrna.shape[1] - len(keep_gene)))
        # keep_peak = np.arange(atac.shape[1])[np.isin(atac.var["chrom"], list(HUMAN_CHROMS_NO_Y_MT))]
        # if len(keep_peak) < mrna.shape[1]:
        #     atac = atac[:, keep_peak]
        #     logger.warning("- removing {} peaks on chrY/chrM/contigs".format(atac.shape[1] - len(keep_peak)))
        
        if not np.array_equal(atac.obs.index, mrna.obs.index):
            common_index = sorted(list(
                set(atac.obs.index).intersection(set(mrna.obs.index))
            ))
            atac = atac[common_index, :]
            mrna = mrna[common_index, :]
            logger.warning("Conflicting index has been fixed")
        
        ## select features: mRNA: high-variable; ATAC: neighboring 100kbp
        raw_mrna_shape, raw_atac_shape = mrna.shape, atac.shape
        print(mrna.shape, atac.shape)
        sc.pp.highly_variable_genes(mrna, flavor="cell_ranger", check_values=True, batch_key=batch_key, subset=True, n_top_genes=n_top_genes)
        logger.info("mRNA: {} -> {}".format(raw_mrna_shape, mrna.shape))

        if select_peak == "flanking":
            peaks_kept = utils.select_neighbor_peaks(mrna.var["tss"], atac.var.index, flanking=flanking)
            atac = atac[:, peaks_kept]
        else:
            logger.warning("- experimtal feature to select peaks by var")
            atac = esp.pp.select_var_feature(atac, nb_features=n_top_peaks, show=False, copy=True)
        logger.info("ATAC: {} -> {}".format(raw_atac_shape, atac.shape))

        self.n_cells = atac.shape[0]
        self.n_genes = mrna.shape[1]
        self.n_peaks = atac.shape[1]

        batches = np.unique(atac.obs[batch_key])
        self.n_batches = len(batches)

        batch2id, onehot  = dict(), list()
        for b, n in enumerate(batches):
            batch2id[n] = b
            ar = [0 for _ in batches]
            ar[b] = 1
            onehot.append(np.array(ar))
        onehot = np.array(onehot)
        assert onehot.shape[1] == len(batches)

        self.batches = csr_matrix(np.array([
            onehot[batch2id[n]] for n in atac.obs[batch_key]
        ], dtype=np.float32))

        self.a_libsize = atac.obs["lib_size"].to_numpy() / LIB_SCALE
        self.atac_obs, self.atac_var = atac.obs.copy(), atac.var.copy()
        del atac.obs, atac.var
        self.atac_count = atac.layers["counts"].copy()
        del atac.layers["counts"]
        self.atac_X = atac.X.copy()
        del atac.X, atac

        self.m_libsize = mrna.obs["lib_size"].to_numpy() / LIB_SCALE
        self.mrna_obs, self.mrna_var = mrna.obs.copy(), mrna.var.copy()
        del mrna.obs, mrna.var
        self.mrna_count = mrna.layers["counts"].copy()
        del mrna.layers["counts"]
        self.mrna_X = mrna.X.copy()
        del mrna.X, mrna

        if self.aug_func is None:
            self.atac_aug = None
            self.m_aug = None
        elif self.aug_func == "clear":
            if self.aug_num != 1:
                logger.warning("self.aug_num was set to 1 for `clear' augmentation")
                self.aug_num = 1
            self.apply_clear_aug()
        else:
            raise NotImplementedError(self.aug_func)

        self.with_fea = with_fea
        if with_fea:
            self.features = np.load('./output/{}/mu_f.npy'.format(with_fea))

    def __getitem__(self, index):
        a_x = self.atac_X[index].toarray().squeeze()
        m_x = self.mrna_X[index].toarray().squeeze()

        batch = self.batches[index].toarray().squeeze()
        # if self.require_counts:
        a_libsize = self.a_libsize[index]
        m_libsize = self.m_libsize[index]
        a_counts = self.atac_count[index].toarray().squeeze()
        m_counts = self.mrna_count[index].toarray().squeeze()

        if self.with_fea:
            feature = self.features[index]

        # logger.info('now sampling', index)

        if self.with_fea:
            return a_x, a_counts, a_libsize, m_x, m_counts, m_libsize, batch, index, feature
        else:
            return a_x, a_counts, a_libsize, m_x, m_counts, m_libsize, batch, index

    def apply_aug_v2(self, 
            inputs, 
            aug_num: int=10,
            mask_ratio: float=0.2, mask_prob: float=0.5, 
            gaussian_ratio: float=0.8, gaussian_prob: float=0.5,
            swap_ratio: float=0.2, swap_prob: float=0.5,
            replace_ratio: float=0.1, replace_prob: float=0.2
            ):
        a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, index = map(list, zip(*inputs))
        batch_size = len(index)
        a_aug, m_aug = list(), list()
        for i in range(batch_size):
            seed = np.random.randint(0, 5)
            sample_a_aug, sample_m_aug = list(), list()
            for _ in range(aug_num):
                a_, m_ = a_x[i].copy(), m_x[i].copy()
                if seed < 2: # atac only
                    a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * mask_ratio), p=mask_prob)
                    a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * swap_ratio), p=swap_prob)
                    a_ = self.random_replace(a_, ratio=replace_ratio, p=replace_prob, feature="ATAC")
                elif seed < 4: ## mrna only
                    m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * mask_ratio), p=mask_prob)
                    m_ = self.random_gaussian(m_, ratio=gaussian_ratio, p=gaussian_prob)
                    m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * swap_ratio), p=swap_prob)
                    m_ = self.random_replace(m_, ratio=replace_ratio, p=replace_prob, feature="mRNA")
                else:
                    a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * mask_ratio / 2), p=mask_prob)
                    a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * swap_ratio), p=swap_prob)
                    a_ = self.random_replace(a_, ratio=replace_ratio, p=replace_prob, feature="ATAC")

                    m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * mask_ratio / 2), p=mask_prob)
                    m_ = self.random_gaussian(m_, ratio=gaussian_ratio, p=gaussian_prob)
                    m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * swap_ratio), p=swap_prob)
                    m_ = self.random_replace(m_, ratio=replace_ratio, p=replace_prob, feature="mRNA")
                sample_a_aug.append(a_)
                sample_m_aug.append(m_)
            a_aug.append(np.stack(sample_a_aug))
            m_aug.append(np.stack(sample_m_aug))
        a_x = torch.tensor(np.stack(a_x))
        a_counts = torch.tensor(np.stack(a_counts))
        a_lib = torch.tensor(np.stack(a_lib))
        m_x = torch.tensor(np.stack(m_x))
        m_counts = torch.tensor(np.stack(m_counts))
        m_lib = torch.tensor(np.stack(m_lib))
        batch = torch.tensor(np.stack(batch))
        a_aug = torch.tensor(np.stack(a_aug))
        m_aug = torch.tensor(np.stack(m_aug))
        # return torch.tensor(np.stack(a_x)), torch.tensor(np.stack(a_counts)), torch.tensor(np.stack)
        return a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, a_aug, m_aug
 

    def collate_fn(self, inputs):
        if self.with_fea:
            a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, index, fea = map(list, zip(*inputs))
        else:
            a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, index = map(list, zip(*inputs))
        batch_size = len(index)
        a_aug, m_aug = list(), list()
        for i in range(batch_size):
            seed = np.random.randint(0, 5)
            a_, m_ = a_x[i].copy(), m_x[i].copy()
            # if seed < 2: # atac only
            #     a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * 0.2), p=0.5)
            #     a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * 0.2), p=0.5)
            #     a_ = self.random_replace(a_, ratio=0.1, p=0.2, feature="ATAC")
            # elif seed < 4: ## mrna only
            #     m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * 0.2), p=0.5)
            #     m_ = self.random_gaussian(m_, ratio=0.3, p=0.5)
            #     m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * 0.2), p=0.5)
            #     m_ = self.random_replace(m_, ratio=0.1, p=0.2, feature="mRNA")
            # else:
            #     a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * 0.1), p=0.5)
            #     a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * 0.1), p=0.5)
            #     a_ = self.random_replace(a_, ratio=0.1, p=0.2, feature="ATAC")
            #
            #     m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * 0.1), p=0.5)
            #     m_ = self.random_gaussian(m_, ratio=0.15, p=0.5)
            #     m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * 0.1), p=0.5)
            #     m_ = self.random_replace(m_, ratio=0.1, p=0.2, feature="mRNA")
            if seed < 2: # atac only
                a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * 0.2), p=0.3)
                a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * 0.2), p=0.3)
                a_ = self.random_replace(a_, ratio=0.1, p=0.2, feature="ATAC")
            elif seed < 4: ## mrna only
                m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * 0.2), p=0.3)
                m_ = self.random_gaussian(m_, ratio=0.3, p=0.3)
                m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * 0.2), p=0.3)
                m_ = self.random_replace(m_, ratio=0.1, p=0.2, feature="mRNA")
            else:
                a_ = self.random_dropout(a_, ratio=max(0.01, np.random.rand() * 0.1), p=0.3)
                a_ = self.random_swap(a_, ratio=max(0.01, np.random.rand() * 0.1), p=0.3)
                a_ = self.random_replace(a_, ratio=0.1, p=0.2, feature="ATAC")

                m_ = self.random_dropout(m_, ratio=max(0.01, np.random.rand() * 0.1), p=0.3)
                m_ = self.random_gaussian(m_, ratio=0.15, p=0.3)
                m_ = self.random_swap(m_, ratio=max(0.01, np.random.rand() * 0.1), p=0.3)
                m_ = self.random_replace(m_, ratio=0.1, p=0.2, feature="mRNA")
            a_aug.append(a_)
            m_aug.append(m_)
        a_x = torch.tensor(np.stack(a_x))
        a_counts = torch.tensor(np.stack(a_counts))
        a_lib = torch.tensor(np.stack(a_lib))
        m_x = torch.tensor(np.stack(m_x))
        m_counts = torch.tensor(np.stack(m_counts))
        m_lib = torch.tensor(np.stack(m_lib))
        batch = torch.tensor(np.stack(batch))
        a_aug = torch.tensor(np.stack(a_aug))
        m_aug = torch.tensor(np.stack(m_aug))

        if self.with_fea:
            fea = torch.tensor(np.stack(fea))
        # return torch.tensor(np.stack(a_x)), torch.tensor(np.stack(a_counts)), torch.tensor(np.stack)

        if self.with_fea:
            return a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, a_aug, m_aug, fea
        else:
            return a_x, a_counts, a_lib, m_x, m_counts, m_lib, batch, a_aug, m_aug
    
    def __len__(self):
        return self.atac_X.shape[0]
    
    def __str__(self):
        return "{}".format(self.__class__.__name__) + \
            "(ATAC: {} (obs.columns: {}, var.columns: {})".format(self.atac_X.shape, self.atac_obs.columns, self.atac_var.columns) + \
            ", mRNA: {} (obs.columns: {}, var.columns: {}))".format(self.mrna_X.shape, self.mrna_obs.columns, self.mrna_var.columns)
    
    def apply_clear_aug(self):
        r"""
        augmentation like clear
        """
        atac_aug, m_aug = list(), list()
        for c in tqdm(range(self.n_cells), total=self.n_cells, desc="CLEAR augmentation"):
            a, m = self.atac_X[c].toarray().squeeze(), self.mrna_X[c].toarray().squeeze()
            a = self.random_dropout(a, ratio=0.2, p=0.5)
            m = self.random_dropout(m, ratio=0.2, p=0.5)
            ## Gaussian
            m = self.random_gaussian(m, ratio=0.8, p=0.5)
            ## shuffle
            a = self.random_swap(a, ratio=0.2, p=0.5)
            m = self.random_swap(m, ratio=0.2, p=0.5)
            ## replace
            a = self.random_replace(a, ratio=0.25, feature="ATAC", p=0.5)
            m = self.random_replace(m, ratio=0.25, feature="mRNA", p=0.5)
            atac_aug.append(csr_matrix(a))
            m_aug.append(csr_matrix(m))
        self.atac_aug = vstack(atac_aug)
        self.m_aug = vstack(m_aug)

    def augmentation_v0(self, inputs):
        r"""
        add dropout 0.2
        """
        assert self.aug_num > 0, "self.aug_num = {}".format(self.aug_num)
        a_x, a_counts, a_libsize, m_x, m_counts, m_libsize, batch = map(list, zip(*inputs))
        n = len(a_x)
        # a_x = a_x.repeat_interleave(self.aug_num, dim=0)
        atac_aug, m_aug = list(), list()
        aug_idx = list()
        for i in range(n):
            a, m = a_x[i], m_x[i]
            atac_aug.append(a)
            m_aug.append(m)
            aug_idx.append(i)
            for aug in range(self.aug_num - 1):
                aug_idx.append(i)
                mod = aug % 3
                if mod == 0:
                    mask = (torch.rand_like(a) > 0.2).float()
                    atac_aug.append(a * mask)
                    m_aug.append(m)
                elif mod == 1:
                    atac_aug.append(a)
                    mask = (torch.rand_like(m) > 0.2).float()
                    m_aug.append(m * mask)
                elif mod == 2:
                    mask = (torch.rand_like(a) > 0.1).float()
                    atac_aug.append(a * mask)
                    mask = (torch.rand_like(m) > 0.1).float()
                    m_aug.append(m * mask)
        atac_aug = torch.stack(atac_aug)
        m_aug = torch.stack(m_aug)
        a_x = torch.stack(a_x).repeat_interleave(self.aug_num, dim=0)
        a_counts = torch.stack(a_counts).repeat_interleave(self.aug_num, dim=0)
        a_libsize = torch.stack(a_libsize).repeat_interleave(self.aug_num, dim=0)
        m_counts = torch.stack(m_counts).repeat_interleave(self.aug_num, dim=0)
        m_x = torch.stack(m_x).repeat_interleave(self.aug_num, dim=0)
        m_libsize = torch.stack(m_libsize).repeat_interleave(self.aug_num, dim=0)
        batch = torch.stack(batch).repeat_interleave(self.aug_num, dim=0)
        return a_x, atac_aug, a_counts, a_libsize, m_x, m_aug, m_counts, m_libsize, batch, torch.tensor(aug_idx)

    
    def random_dropout(self, x: np.ndarray, ratio: float, p=0.5):
        if ratio == 0 or p == 0 or (p < 1 and np.random.rand() > p):
            return x
        x = x * (np.random.rand(*x.shape) > ratio).astype(np.float32)
        return x
    
    def random_gaussian(self, x: np.ndarray, ratio: float=1, p: float=0.5, mean=None, std=None):
        if ratio == 0 or p == 0 or (p < 1 and np.random.rand() > p):
            return x
        zero = np.where(x == 0)[0]
        noise = np.random.rand(*x.shape)
        if mean is not None:
            noise = noise * std + mean
        noise[zero] = 0
        if ratio < 1:
            inds = np.random.choice(np.arange(x.shape[0]), int(ratio * x.shape[0]))
            x[inds] += noise[inds]
        return np.maximum(0, x * noise)
    
    def random_swap(self, x: np.ndarray, ratio: float, p: float=0.5):
        if ratio == 0 or p == 0 or (p < 1 and np.random.rand() > p):
            return x
        inds = np.random.choice(np.arange(x.shape[0]), int(ratio * x.shape[0]))
        x[inds] = x[np.random.permutation(inds)]
        return x
    
    def random_replace(self, x: np.ndarray, ratio: float, feature: str, p: float=0.5):
        if ratio == 0 or p == 0 or (p < 1 and np.random.rand() > p):
            return x
        target = np.random.randint(0, high=self.n_cells)
        if feature == "ATAC":
            t = self.atac_X[target].toarray().squeeze()
        else:
            t = self.mrna_X[target].toarray().squeeze()
        inds = np.random.choice(np.arange(x.shape[0]), int(ratio * x.shape[0]))
        x[inds] = t[inds]
        return x


class scATACData():
    pass

class scRNAData():
    pass

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--seed', type=int, default=2020)
    return p

def check_counts(inputs: Union[AnnData, csr_matrix, np.ndarray, Tensor], layer_key=None):
    ## check raw counts (integers)
    if type(inputs) is Tensor:
        inputs = inputs.cpu().numpy()
    if isinstance(inputs, AnnData):
        if layer_key is None:
            is_count = np.abs(inputs.X.data % 1).max() == 0
        else:
            is_count = np.abs(inputs.layers[layer_key].data % 1).max() == 0
    elif isinstance(inputs, csr_matrix):
        is_count = np.abs(inputs.data % 1).max() == 0
    else:
        is_count = np.abs(inputs % 1).max() == 0
    return is_count

def load_demo() -> PairedModeDataset:
    logger.setLevel(logging.WARNING)
    atac = load_adata(
        "/home/chenken/data/local/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.ATAC.batch-s1d1.h5ad",
        count_key="counts", 
        log1p=False, binarize=True, tfidf=False
    )
    mrna = load_adata(
        "/home/chenken/data/local/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.mRNA.batch-s1d1.h5ad",
        count_key="counts",
        log1p=True, binarize=False, tfidf=False
    )
    dataset = PairedModeDataset(
        atac, 
        mrna, 
        batch_key="batch",
        gene_info="/home/chenken/db/gencode/annotations/gencode.v32.chr_patch_hapl_scaff.tss.bed", 
        select_peak="var",
        flanking=None,
        n_top_genes=2000,
        n_top_peaks=10000,
    )
    return dataset

if __name__ == "__main__":
    args = get_args().parse_args()

    
