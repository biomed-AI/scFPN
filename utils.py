import json
import scanpy as sc
from typing import List, Dict, Tuple, Union
import numpy as np
from collections import OrderedDict
import anndata as ad
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from anndata import AnnData
from biock.logger import make_logger
from biock import HUMAN_CHROMS_NO_Y_MT, HUMAN_CHROMS_NO_MT, split_chrom_start_end, random_string
from biock.genomics import ensembl_remove_version
from collections import defaultdict
import pybedtools
from pybedtools import BedTool, Interval
from torch import Tensor

logger = make_logger(title="", level="DEBUG")


## gene & peaks
def load_gene_info(tss_bed, prefix=None) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    return: gene2chrom, gene2tss, gene_id2name, gene_name2id
    """
    all_loaded = True if prefix is not None else False
    if prefix is not None:
        try:
            gene2chrom = json.load(open("{}.gene2chrom.json".format(prefix)))
            gene2tss = json.load(open("{}.gene2tss.json".format(prefix)))
            gene_id2name = json.load(open("{}.gene_id2name.json".format(prefix)))
            gene_name2id = json.load(open("{}.gene_name2id.json".format(prefix)))
            logger.info("loading gene info cache")
        except FileNotFoundError:
            all_loaded = False
    
    if not all_loaded:
        gene2tss, gene2chrom = defaultdict(set), defaultdict(set)
        gene_name2id, gene_id2name = defaultdict(set), defaultdict(set)
        non_chrom_contigs = set()
        with open(tss_bed) as infile:
            for l in infile:
                chrom, _, tss, name, _, strand = l.strip().split('\t')
                non_chrom_contigs.add(chrom)
                # if chrom not in HUMAN_CHROMS_NO_MT:
                #     continue
                gene_id, gene_name, gene_type, tx_id, tss, strand = name.split('|')
                gene_id = ensembl_remove_version(gene_id)
                tss = int(tss)
                gene2tss[gene_id].add((chrom, tss))
                gene2tss[gene_name].add((chrom, tss))
                gene2chrom[gene_id].add(chrom)
                gene2chrom[gene_name].add(chrom)

                gene_name2id[gene_name].add(gene_id)
                gene_id2name[gene_id].add(gene_name)
        non_chrom_contigs = non_chrom_contigs.difference(HUMAN_CHROMS_NO_MT)

        gene2chrom = dict(gene2chrom)
        gene2tss = dict(gene2tss)
        gene_name2id = dict(gene_name2id)
        gene_id2name = dict(gene_id2name)

        for g, chroms in gene2chrom.items():
            if len(chroms) > 1 and 'chrY' in chroms:
                chroms.remove('chrY')
            if len(chroms.difference(non_chrom_contigs)) > 0:
                chroms = chroms.difference(non_chrom_contigs)
            tss = list()
            for c, t in gene2tss[g]:
                if c in chroms:
                    tss.append([c, t])
            gene2tss[g] = tss
            gene2chrom[g] = list(chroms)

        gene_name2id = {g: list(v) for g, v in gene_name2id.items()}
        gene_id2name = {g: list(v) for g, v in gene_id2name.items()}

        if prefix is not None:
            json.dump(gene2chrom, open("{}.gene2chrom.json".format(prefix), 'w'), indent=4) 
            json.dump(gene2tss, open("{}.gene2tss.json".format(prefix), 'w'), indent=4)
            json.dump(gene_id2name, open("{}.gene_id2name.json".format(prefix), 'w'), indent=4)
            json.dump(gene_name2id, open("{}.gene_name2id.json".format(prefix), 'w'), indent=4)
        logger.info("load gene info")
    return gene2chrom, gene2tss, gene_id2name, gene_name2id


def add_gene_info(adata: AnnData, gene_info: str, gene_key: bool=None, drop_ambiguous: bool=True, drop_unknown: bool=True) -> AnnData:
    ## gene_info: biock tss bed format
    logger.info("loading gene info from {}".format(gene_info))
    gene2chrom, gene2tss, gene_id2name, gene_name2id = load_gene_info(gene_info, prefix=gene_info.replace(".tss.bed", ''))
    logger.info("done")
    genes = adata.var.index if gene_key is None else adata.var[gene_key]

    # gene_id, gene_name = list(), list()
    chrom, tss = list(), list()
    keep_inds = list()
    unknown, ambiguous = set(), set()
    for i, g in enumerate(genes):
        if g not in gene2chrom:
            if drop_unknown:
                unknown.add(g)
                continue
            else:
                raise NotImplementedError("do not drop unknown is not supported")
        elif len(gene2chrom[g]) > 1:
            if drop_ambiguous:
                ambiguous.add(g)
                continue
            else:
                raise NotImplementedError("do not drop ambiguous is not supported")
        else:
            keep_inds.append(i)
            chrom.append(gene2chrom[g][0])
            tss.append(';'.join(["{}:{}".format(c, t) for c, t in gene2tss[g]]))
            # gene_id.append(';'.join(gene_name2id[g]))
    if len(unknown) > 0:
        logger.warning("unknown genes({}): {}".format(len(unknown), unknown))
    if len(ambiguous) > 0:
        logger.warning("ambiguous genes({}): {}".format(len(ambiguous), ambiguous))
    logger.info("{} genes removed because of missing info".format(adata.shape[1] - len(keep_inds)))
    adata = adata[:, keep_inds]
    adata.var["chrom"] = chrom
    adata.var["tss"] = tss
    logger.info("chrom and tss added to data")

    return adata


def select_neighbor_peaks(tss_list: List[str], peak_list: List[str], flanking:int=100000) -> List[str]:
    ## tss_list: ["chr1:100;chr2:200", "chr3:300", ...]
    ## peak_list: ["chr1:100-200", "chr2:100-200", ...] (var index)
    logger.info("tss_list/peak_list: {}".format((len(tss_list), len(peak_list))))
    tss_bed = list()
    for r in tss_list:
        if len(r) == 0:
            continue
        for c in r.split(';'):
            c, t = c.split(':')
            tss_bed.append(Interval(c, int(t) - 1, int(t)))

    peak_bed = list()
    for p in peak_list:
        chrom, start, end = split_chrom_start_end(p)
        peak_bed.append(Interval(chrom, int(start), int(end), p))
    tmp_fn = "/tmp/pybedtools.{}.bed".format(random_string(n=10))
    logger.info("{}".format((len(peak_bed), len(tss_bed))))
    # with open('/tmp/tss.bed', 'w') as out:
    #     for x in tss_bed:
    #         out.write("{}\t{}\t{}\n".format(*x))
    BedTool(peak_bed).window(BedTool(tss_bed), w=flanking, u=True).moveto(tmp_fn)    
    peak_subset = list()
    with open(tmp_fn) as infile:
        for l in infile:
            name = l.strip().split('\t')[3]
            peak_subset.append(name)
    pybedtools.cleanup()
    return peak_subset


## single cell processing
def check_counts(inputs: Union[AnnData, csr_matrix, np.ndarray, Tensor], layer_key=None):
    ## check raw counts (integers)
    if type(inputs) is csc_matrix:
        inputs = csr_matrix(inputs)
    assert type(inputs) in {AnnData, csr_matrix, np.ndarray, Tensor}, "{} was given, while AnnData/csr_matrix/np.ndarray/Tensor is required".format(type(inputs))
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



def filter_cells(adata: AnnData, min_genes: int=None, max_genes: int=None) -> None:
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)


def filter_genes(adata: AnnData, min_cells: int=None, max_cells: int=None) -> None:
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_cells is not None:
        sc.pp.filter_genes(adata, max_cells=max_cells)


def tfidf(adata: AnnData):
    ## TODO: 
    raise NotImplementedError


def warn_kwargs(kwargs):
    if len(kwargs) > 0:
        logger.warning("unused args: {}".format(kwargs))


def recover_grouped_index(raw_ar):
    ordered_groups = np.unique(raw_ar)
    group2index = OrderedDict()
    recover_mapping = dict()
    accm_index = 0
    for g in ordered_groups:
        group2index[g] = np.where(raw_ar == g)[0]
        for i, raw_ind in enumerate(group2index[g]):
            recover_mapping[raw_ind] = i + accm_index
        accm_index += len(group2index[g])
    recover_index = [recover_mapping[i] for i in range(len(raw_ar))]
    return recover_index


def kl_weight(it, delay=10, maximum: float=1):
    if it < delay:
        w = 0
    else:
        w = 1 / (1 + np.exp(10 - it/5))
    return min(w, maximum)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    start = -stop
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = max(0, v)
            v += step
            i += 1
    return L

def find_resolution(features, n_clusters, random=666, var=2):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.
    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """

    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    adata_ = AnnData(features)
    sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X")

    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions) / 2
        adata = sc.tl.louvain(adata_, resolution=current_res, copy=True)
        labels = adata.obs['louvain']
        obtained_clusters = len(set(labels))

        if n_clusters - obtained_clusters > var:
            resolutions[0] = current_res
        elif obtained_clusters - n_clusters > var:
            resolutions[1] = current_res

        iteration += 1

    return current_res

def find_res_label(features, n_clusters, random=666, var=2):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.
    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """

    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    adata_ = AnnData(features)
    sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X")

    while abs(obtained_clusters - n_clusters) > var and iteration < 1000:
        current_res = sum(resolutions) / 2
        # print(current_res)
        adata = sc.tl.louvain(adata_, resolution=current_res, copy=True)
        labels = adata.obs['louvain']
        obtained_clusters = len(set(labels))

        if  n_clusters - obtained_clusters > var:
            resolutions[0] = current_res
        elif obtained_clusters - n_clusters > var:
            resolutions[1] = current_res
        else:
            return adata.obs['louvain']

        iteration += 1

        if iteration == 1000:
            print("Hard!!!!")
            return adata.obs['louvain']
        

# def find_res_label(features, n_clusters, random=666, var=2):
#     """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
#     Arguments:
#     ------------------------------------------------------------------
#     - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
#     - n_clusters: `int`, Number of clusters.
#     - random: `int`, The random seed.
#     Returns:
#     ------------------------------------------------------------------
#     - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
#     """

#     obtained_clusters = -1
#     iteration = 0
#     resolutions = [0., 1000.]

#     adata_ = AnnData(features)
#     sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X")

#     while obtained_clusters - n_clusters != var and iteration < 1000:
#         current_res = sum(resolutions) / 2
#         # print(current_res)
#         adata = sc.tl.louvain(adata_, resolution=current_res, copy=True)
#         labels = adata.obs['louvain']
#         obtained_clusters = len(set(labels))

#         if obtained_clusters - n_clusters < var:
#             resolutions[0] = current_res
#         elif obtained_clusters - n_clusters > var:
#             resolutions[1] = current_res
#         else:
#             return adata.obs['louvain']

#         iteration += 1

#         if iteration == 1000:
#             print("Hard!!!!")
#             return adata.obs['louvain']

if __name__ == "__main__":

    raw_rna = sc.read_h5ad("../data/GSE126074_SNAREseq_CellMixture/GSE126074_CellMixture.RNA.raw.h5ad")
    raw_atac = sc.read_h5ad("../data/GSE126074_SNAREseq_CellMixture/GSE126074_CellMixture.ATAC.raw.h5ad")
