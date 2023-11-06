#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

#Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos

import sys

sys.path.append('/home/chenjn/biock')

import argparse, os, sys, time, gzip
from ast import arg
import shutil
import json
import numpy as np
import time
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from biock import make_directory, backup_file
from biock.logger import make_logger
from biock.gpu import select_device
from biock.pytorch import model_summary, set_seed
import dataloaders
from com_vae_v5 import CombineVAE
# import basic_vae, vae_poe_mse, 

TO_SAVE = [
    __file__,
    dataloaders,
    CombineVAE
]

def _get_value(dict_: Dict, key: str, default: Any) -> Any:
    if key in dict_:
        default = dict_[key]
    return default

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-a', "--atac", help="ATAC", required=True)
    p.add_argument('-r', "--rna", help="RNA", required=True)
    p.add_argument('--count-key', default="counts")
    p.add_argument('--batch-key', default="batch")
    p.add_argument('-n', '--model-name', default="CombineVAE", type=str)
    p.add_argument("--enc-dims", type=int, nargs='+', default=None)
    p.add_argument("--dec-dims", type=int, nargs='+', default=None)
    p.add_argument('-z', "--z-dim", required=True, type=int)
    p.add_argument("--combine", choices=("poe", "concat", "mean"), required=True)
    p.add_argument("--peak-loss", choices=("bce", "nb", "zinb"), default="bce")
    p.add_argument("--gene-loss", choices=("mse", "nb", "zinb"), default="mse")
    p.add_argument("--com-loss", choices=("en", "nb", "zinb"), default="en")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--apply-gan", action="store_true")
    p.add_argument("--apply-contrastive", action="store_true")
    p.add_argument('--max-gene', type=int, default=2000)
    p.add_argument('--max-peak', type=int, default=10000)
    p.add_argument('-b', "--batch-size", default=128, type=int)
    p.add_argument('-lr', "--learning-rate", type=float, default=1E-3)
    p.add_argument('-w', "--weight-decay", type=float, default=1E-6)
    p.add_argument('-o', "--outdir", required=True, help="output folder")
    p.add_argument("--ind-epoch", default=100, type=int)
    p.add_argument("--com-epoch", default=20, type=int)
    p.add_argument("--gene-info", default=None, help="gene tss")
    p.add_argument("--patience", default=20, type=int)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('-t', '--num-workers', default=0, type=int)
    p.add_argument('--cycle-kl', action="store_true")
    p.add_argument("--aug-num", type=int, default=10, help="augmentation number")
    p.add_argument("--device", type=str, default=None, help="GPU(None for auto, number for GPU ID) or CPU")
    p.add_argument('--resume', required=False, help="continue", type=str)
    p.add_argument('--pretrain', required=False, help="checkpoint of trained model", type=str)
    p.add_argument("--debug", action="store_true", help="debug mode")
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument("--dname", type=str, default='default')
    p.add_argument("--ae", action="store_true")
    return p

def header(**kwargs):
    s = list()
    s.append("\n##{}".format(time.asctime()))
    s.append("##pwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(sys.argv)))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    set_seed(args.seed)

    # save dataset_name for tensorboard
    args.dname = args.atac

    if args.atac == "default":
        args.atac = "/home/chenken/data/local/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.ATAC.batch-s1d1.h5ad"
        args.rna = "/home/chenken/data/local/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.mRNA.batch-s1d1.h5ad"
    elif args.atac == "neurips":
        args.atac = "./data/neurips/ATAC.h5ad"
        args.rna = "./data/neurips/mRNA.h5ad"
    elif args.atac == "share-seq-mouse-brain":
        args.atac = "./data/share-mb/ATAC.h5ad"
        args.rna = "./data/share-mb/mRNA.h5ad"
    elif args.atac == "snare-mixture":
        args.atac = "./data/snare/ATAC.h5ad"
        args.rna = "./data/snare/mRNA.h5ad"
    # newly adding

    elif args.atac == "sci-CAR":
        args.atac = "./data/sci-CAR/ATAC.h5ad"
        args.rna = "./data/sci-CAR/mRNA.h5ad"
    elif args.atac == "10x-pbmc10k":
        args.atac = "./data/pbmc/ATAC.h5ad"
        args.rna = "./data/pbmc/mRNA.h5ad"

    # abnormal datasets
    elif args.atac == "hcaner":
        args.atac = "/bigdat1/user/dailab/new_data/hcaner/atac_dataset_origin.h5ad"
        args.rna = "/bigdat1/user/dailab/new_data/hcaner/rna_dataset_origin.h5ad"
    elif args.atac == "pair-hcl":
        args.atac = "/bigdat1/user/dailab/new_data/pair_hcl/atac_dataset_origin.h5ad"
        args.rna = "/bigdat1/user/dailab/new_data/pair_hcl/rna_dataset_origin.h5ad"

    # add in 2023/2/17
    elif args.atac == "smage":
        args.atac = "./data/smage/mRNA_raw.h5ad"
        args.rna = "./data/smage/ATAC_raw.h5ad"

    outdir = make_directory(args.outdir)
    savedir = make_directory(os.path.join(args.outdir, args.dname))
    src_dir = make_directory(os.path.join(args.outdir, 'src'))

    logger = make_logger(title="", filename="{}/train.log".format(outdir), level="DEBUG" if args.debug else "INFO", show_line=True, filemode='a' if args.resume else 'w')

    for m in TO_SAVE:
        dst = backup_file(m, src_dir, readonly=True)
        logger.info("- backup: {}  =>  {}".format(m, dst))


    logger.info(header(args=args))
    
    atac = dataloaders.load_adata(
        args.atac,
        count_key=args.count_key,
        log1p=False, 
        binarize=True, 
        tfidf=False
    )
    mrna = dataloaders.load_adata(
        args.rna,
        count_key=args.count_key,
        log1p=True, 
        binarize=False, 
        tfidf=False
    )
    dataset = dataloaders.PairedModeDataset(
        atac, 
        mrna, 
        batch_key=args.batch_key,
        select_peak="var",
        n_top_genes=args.max_gene,
        n_top_peaks=args.max_peak,
        gene_info=args.gene_info
    )

    gpu_id, device = select_device(args.device)
    logger.info("#device: {}({})".format(device, gpu_id))

    ## setup model
    model_config = {
        "args": args,
        "n_peak": dataset.n_peaks,
        "n_gene": dataset.n_genes,
        "n_batch": dataset.n_batches,
        "n_kind": len(set(dataset.mrna_obs['cell_type'])),
        "enc_dims": args.enc_dims,
        "dec_dims": args.dec_dims,
        "z_dim": args.z_dim,
        "combine": args.combine,
        "peak_loss": args.peak_loss,
        "gene_loss": args.gene_loss,
        "com_loss": args.com_loss,
        "dropout": args.dropout,
        "apply_gan": args.apply_gan,
        "apply_contrastive": args.apply_contrastive,
    }

    VAE = CombineVAE(args,
                     n_gene=dataset.n_genes,
                     n_peak=dataset.n_peaks,
                     n_kind=len(set(dataset.mrna_obs['cell_type'])),
                     enc_dims=args.enc_dims,
                     dec_dims=args.dec_dims,
                     z_dim=args.z_dim,
                     combine=args.combine,
                     peak_loss=args.peak_loss,
                     gene_loss=args.gene_loss,
                     com_loss=args.com_loss,
                     dropout=args.dropout,
                     apply_gan=args.apply_gan,
                     apply_contrastive=args.apply_contrastive,
                     n_batch=dataset.n_batches,
                     device=device
                     )
    print(next(VAE.parameters()).device)

    VAE = VAE.to(device)

    print(next(VAE.parameters()).device)

    saved_ckpt = "{}/{}_5/checkpoint_{}.pt".format(outdir, args.dname, args.pretrain)

    t0 = time.time()
    c0 = time.perf_counter()
    p0 = time.process_time()

    ## train model or load trained model
    if args.pretrain is None:
        set_seed(args.seed)
        # best_results, saved_ckpt = \
        VAE.pretrain(dataset,
                     test_ratio=args.test_ratio,
                     batch_size=args.batch_size,
                     outdir=outdir,
                     lr=args.learning_rate,
                     ind_epoch=args.ind_epoch,
                     com_epoch=args.com_epoch,
                     patience=args.patience,
                     num_workers=args.num_workers,
                     resume=args.resume,
                     # cycle_kl=args.cycle_kl,
                     cycle_kl=True,
                     aug_num=args.aug_num,
                     seed=args.seed,
                     ds=args.dname
                    )
    else:
        print('loading...')
        VAE.load_state_dict(torch.load(saved_ckpt)["model_state"])
    
    t1 = time.time()
    c1 = time.perf_counter()
    p1 = time.process_time()

    spend1 = t1 - t0
    spend2 = c1 - c0
    spend3 = p1 - p0

    VAE.main_train(  dataset,
                     test_ratio=args.test_ratio,
                     batch_size=1024,
                     outdir=outdir,
                     lr=args.learning_rate,
                     ind_epoch=args.ind_epoch,
                     com_epoch=args.com_epoch,
                     patience=args.patience,
                     num_workers=args.num_workers,
                     resume=args.resume,
                     aug_num=args.aug_num,
                     seed=args.seed,
                     ds=args.dname
                   )
    
    t2 = time.time()
    c2 = time.perf_counter()
    p2 = time.process_time()

    spend4 = t2 - t0
    spend5 = c2 - c0
    spend6 = p2 - p0

    print("time()方法用时：{}s".format(spend1))
    print("perf_counter()用时：{}s".format(spend2))
    print("process_time()用时：{}s".format(spend3))

    print("time()方法用时：{}s".format(spend4))
    print("perf_counter()用时：{}s".format(spend5))
    print("process_time()用时：{}s".format(spend6))

    # VAE.contrastive_train(dataset,
    #                test_ratio=args.test_ratio,
    #                batch_size=1024,
    #                outdir=outdir,
    #                lr=args.learning_rate,
    #                ind_epoch=args.ind_epoch,
    #                com_epoch=args.com_epoch,
    #                patience=args.patience,
    #                num_workers=args.num_workers,
    #                resume=args.resume,
    #                aug_num=args.aug_num,
    #                seed=args.seed,
    #                ds=args.dname
    #                )

    VAE.get_output(
        dataset,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers - 1, 0),
        outdir=outdir,
        ds=args.dname
    )

    print("Finish!")


