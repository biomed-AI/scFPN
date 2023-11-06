# align before decode with dec embedding

import math
import os
import sys
sys.path.append('/home/chenjn/biock')
import time

from typing import Any, Dict, Iterable, List, Literal, Tuple, Union, Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from anndata._core.views import ArrayView
from numpy import ndarray
from scipy import sparse
import scipy.io as sio
from sklearn.metrics import (adjusted_rand_score, mean_squared_error,
                             normalized_mutual_info_score)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from biock import make_logger, make_directory
from biock.genomics import counts_per_million, counts_per_thousand
from biock.ml_tools import EarlyStopping, roc_pr_auc_scores_2d, roc_auc_score_2d
from biock.pytorch import set_seed
from biock.stats import pearsonr2d
from dataloaders import PairedModeDataset
from utils import kl_weight, warn_kwargs, frange_cycle_linear
from biock.pytorch import kl_divergence
from biock.pytorch import build_cnn1d, build_mlp
from _negative_binomial import NegativeBinomial, ZeroInflatedNegativeBinomial
from vae import MLPEncoder, ExptDecoder, NBDecoder, ZINBDecoder, ProductOfExperts
from _negative_binomial import NegativeBinomial
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import MiniBatchKMeans, KMeans

import logging
logger = logging.getLogger(__name__)


class CombineVAE(nn.Module):
    def __init__(self,
                 args,
                 n_gene: int,
                 n_peak: int,
                 n_kind: int,
                 enc_dims: List[int],
                 dec_dims: List[int],
                 z_dim: int,
                 n_batch: int,
                 combine: Literal["poe", "mean", "concat"],
                 peak_loss: Literal["bce", "nb", "zinb"],
                 gene_loss: Literal["mse", "nb", "zinb"],
                 com_loss: Literal["en", "nb", "zinb"],
                 dropout: float = 0.1,
                 apply_gan: bool = False,
                 apply_contrastive: bool = False,
                 device: Any = None,
                 **kwargs
                 ):
        super(CombineVAE, self).__init__()

        self.combine = combine
        self.apply_gan = apply_gan
        self.apply_contrastive = apply_contrastive
        self.n_peak = n_peak
        self.n_gene = n_gene
        self.n_kind = n_kind
        self.n_batch = n_batch
        self.device = device

        self.p_dis = list()
        self.update_p_epoch = 10

        if enc_dims is None:
            gene_enc_dims = [max(256, 2 * int(np.sqrt(n_gene))), max(128, int(np.sqrt(n_gene)))]
            peak_enc_dims = [max(256, 2 * int(np.sqrt(n_peak))), max(128, int(np.sqrt(n_peak)))]
            gene_dec_dims = [gene_enc_dims[-1]]
            peak_dec_dims = [peak_enc_dims[-1]]
        else:
            gene_enc_dims, peak_enc_dims = enc_dims, enc_dims
            gene_dec_dims, peak_dec_dims = dec_dims, dec_dims

        self.a_encoder = MLPEncoder(x_dim=n_peak, h_dim=peak_enc_dims, z_dim=z_dim, bn=True, dropout=dropout)
        self.m_encoder = MLPEncoder(x_dim=n_gene, h_dim=gene_enc_dims, z_dim=z_dim, bn=True, dropout=dropout)

        self.gene_loss, self.peak_loss, self.com_loss = gene_loss, peak_loss, com_loss

        if self.peak_loss == "bce":
            self.a_decoder = ExptDecoder(  # bernoulli
                x_dim=n_peak, h_dim=peak_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout,
                output_activation=nn.Sigmoid()
            )
        elif self.peak_loss == "nb":
            self.a_decoder = NBDecoder(
                x_dim=n_peak, h_dim=peak_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout
            )
        elif self.peak_loss == "zinb":
            self.a_decoder = ZINBDecoder(
                x_dim=n_peak, h_dim=peak_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout
            )

        if self.gene_loss == "mse":
            self.m_decoder = ExptDecoder(
                x_dim=n_gene, h_dim=gene_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout,
                output_activation=nn.Softplus()
            )
        elif self.gene_loss == "nb":
            self.m_decoder = NBDecoder(
                x_dim=n_gene, h_dim=gene_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout
            )
        elif self.gene_loss == "zinb":
            self.m_decoder = NBDecoder(
                x_dim=n_gene, h_dim=gene_dec_dims, z_dim=z_dim * 2 + n_batch + 1,
                bn=True, dropout=dropout
            )

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(max(256, 2 * int(np.sqrt(n_gene))) + max(256, 2 * int(np.sqrt(n_peak))), max(128, int(np.sqrt(n_gene))) + max(128, int(np.sqrt(n_peak)))))
        self.fc.append(nn.Linear(max(128, int(np.sqrt(n_gene))) + max(128, int(np.sqrt(n_peak))), 64))
        self.fc.append(nn.Linear(64, 64))

        self.fc_fix = nn.ModuleList()
        self.fc_fix.append(nn.Sigmoid())
        self.fc_fix.append(nn.Sigmoid())
        self.fc_fix.append(nn.Sigmoid())

        self.z_dim = z_dim
        self.poe = ProductOfExperts()

        # self.clusterCenter = nn.Parameter(torch.zeros(self.n_kind, z_dim * 2))
        self.clusterCenter = nn.Parameter(torch.Tensor(self.n_kind, z_dim * 2))
        torch.nn.init.xavier_normal_(self.clusterCenter.data)

        self.alpha = 1.0

        # save for tensorboard
        str_name = self.get_parameter_str(args)
        today = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        log_write_dir = "./log/" + str_name + '-in-' + today
        if not os.path.exists(log_write_dir):
            os.mkdir(log_write_dir)
        self.writer = SummaryWriter(log_write_dir)

        self.mu_cache = None

        self.use_cache = True

        self.z_mask = None

    def get_parameter_str(self, args):
        # return some parameters that make senses in the task to form the name of a logger
        str_name = ""
        non_sense = ['atac','rna','count_key', 'tss', 'batch_key', 'enc_dims', 'dec_dims', 'seed',
                     'debug', 'pretain', 'resume', 'device', 'cycle_kl',"t","test_ratio","patience","gene_info","o",
                     "ind_epoch", "com_epoch","dropout","max_gene","max_peak","weight_decay","outdir","num_workers",
                     "aug_num","pretrain"
                     ]
        for arg in vars(args):
            print(arg, getattr(args, arg))
            if arg in non_sense:
                continue
            str_name += str(arg) + "-" + str(getattr(args, arg))

        return str_name

    def encode_x2z(self, x: Tensor, mode: str) -> Tuple:
        """
        return: mu, logvar, z
        """
        if mode == 'a':
            (z, mu, logvar), hid = self.a_encoder.forward(x)
        elif mode == 'm':
            (z, mu, logvar), hid = self.m_encoder.forward(x)

        z = self.reparametrize(mu, logvar)

        return mu, logvar, z, hid

    def forward(self,
            atac: Tensor,
            a_libsize: Tensor,
            mrna: Tensor,
            m_libsize: Tensor,
            batch: Tensor) -> Tuple[Any, Tensor, Tensor, Tensor, Any, Tensor, Tensor, Tensor]:
        """
        Return
        -------
        a_out : Tensor or Tensor tuple of distributions ()
        m_out : Tensor or Tensor tuple of distributions ()
        mu : Tensor
        logvar : Tensor
        z : Tensor
        """

        if batch.ndim == 1:
            batch = batch.unsqueeze(1)
        if a_libsize.ndim == 1:
            a_libsize = a_libsize.unsqueeze(1)
        if m_libsize.ndim == 1:
            m_libsize = m_libsize.unsqueeze(1)

        a_mu, a_logvar, a_z, a_hid = self.encode_x2z(atac, 'a')
        m_mu, m_logvar, m_z, m_hid = self.encode_x2z(mrna, 'm')

        z_cat_emb = torch.cat((a_z, m_z), dim=1)

        a_z_emb = torch.cat((z_cat_emb, batch, a_libsize), dim=1)
        a_out, _ = self.a_decoder(a_z_emb)

        m_z_emb = torch.cat((z_cat_emb, batch, m_libsize), dim=1)
        m_out, _ = self.m_decoder(m_z_emb)

        return a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z

    def spp_forward(self,
            atac: Tensor,
            a_libsize: Tensor,
            mrna: Tensor,
            m_libsize: Tensor,
            batch: Tensor) -> Tuple[Any, Tensor, Tensor, Any, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return
        -------
        a_out : Tensor or Tensor tuple of distributions ()
        m_out : Tensor or Tensor tuple of distributions ()
        mu : Tensor
        logvar : Tensor
        z : Tensor
        """

        if batch.ndim == 1:
            batch = batch.unsqueeze(1)
        if a_libsize.ndim == 1:
            a_libsize = a_libsize.unsqueeze(1)
        if m_libsize.ndim == 1:
            m_libsize = m_libsize.unsqueeze(1)

        a_mu, a_logvar, a_z, a_hid = self.encode_x2z(atac, 'a')
        m_mu, m_logvar, m_z, m_hid = self.encode_x2z(mrna, 'm')

        mu_cat = torch.cat((a_mu.data, m_mu.data), dim=1)
        tmp = None

        for i in range(len(a_hid)):
            # print(a_hid[i].shape, m_hid[i].shape)
            now = torch.cat((a_hid[i].data, m_hid[i].data), dim=1)
            if tmp is not None:
                # print(tmp.shape, now.shape)
                tmp = tmp + now
            else:
                tmp = now
            tmp = self.fc_fix[i](self.fc[i](tmp))

        mu_cat = mu_cat + tmp

        z_cat_emb = torch.cat((a_z, m_z), dim=1)
        # z_cat_emb = mu_cat

        a_z_emb = torch.cat((z_cat_emb, batch, a_libsize), dim=1)
        a_out, _ = self.a_decoder(a_z_emb)

        m_z_emb = torch.cat((z_cat_emb, batch, m_libsize), dim=1)
        m_out, _ = self.m_decoder(m_z_emb)

        return a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat

    def updateClusterCenter(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of num_classes x num_features
        """
        tmp = cc.copy()
        self.clusterCenter.data = torch.from_numpy(tmp)

    def getTDistribution(self, x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        xe = x.unsqueeze(1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(xe ** 2, 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        res = q / torch.sum(q, dim=1, keepdim=True)

        return res

    def getDistance(self, x, clusterCenter, alpha=1.0):
        """
        it should minimize the distince to
         """
        xe = torch.unsqueeze(x, 1).cuda() - clusterCenter.cuda()
        # need to sum up all the point to the same center - axis 1
        d = torch.sum(torch.mul(xe, xe), 2)

        return d

    def get_center_labels(self, features, resolution=3.0):
        '''
        resolution: Value of the resolution parameter, use a value above
              (below) 1.0 if you want to obtain a larger (smaller) number
              of communities.
        '''

        print("\nInitializing cluster centroids using the louvain method.")

        adata0 = AnnData(features)
        sc.pp.neighbors(adata0, n_neighbors=15, use_rep="X")
        # adata0 = sc.tl.louvain(adata0, resolution=resolution, random_state=0, copy=True)
        adata0 = sc.tl.louvain(adata0, random_state=0, copy=True)
        y_pred = adata0.obs['louvain']
        y_pred = np.asarray(y_pred, dtype=int)

        features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, adata0.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)

        init_centroid = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = init_centroid.shape[0]

        # print("\n " + str(n_clusters) + " micro-clusters detected. \n")
        return init_centroid, y_pred

    def find_resolution(self, features, n_clusters, random=666):
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
            obtained_clusters = len(np.unique(labels))

            if obtained_clusters < n_clusters:
                resolutions[0] = current_res
            else:
                resolutions[1] = current_res

            iteration = iteration + 1

        return current_res

    def clustering(self, dataset):
        loader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=False)
        device = self.device
        latent = list()
        n_cluster = len(set(dataset.mrna_obs["cell_type"]))

        self.eval()
        for a_x, _, a_size, m_x, _, m_size, batch, _ in loader:
            a_x, a_size = a_x.float().to(device), a_size.float().to(device)
            m_x, m_size = m_x.float().to(device), m_size.float().to(device)
            batch = batch.float().to(device)

            a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat = self.spp_forward(a_x, a_size, m_x, m_size, batch)
            latent.append(torch.cat((a_mu.data, m_mu.data), dim=1).detach().cpu().numpy())

        latent = np.concatenate(latent, axis=0)

        resolution = self.find_resolution(latent, n_cluster)

        self.cluster_centers, _ = self.get_center_labels(latent, resolution=resolution)
        print(self.cluster_centers.shape)
        self.updateClusterCenter(self.cluster_centers)

        self.mu_cache = latent

    def get_mid(self, x):
        return  x, self.getTDistribution(x, self.clusterCenter), self.getDistance(x, self.clusterCenter), F.softmax(x, dim=1)

    @staticmethod
    def target_distribution(q):
        weight = (q ** 2) / q.sum(0)
        # print('q',q)
        return (weight.t() / weight.sum(1)).t()

    @staticmethod
    def loss_function(p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    @staticmethod
    def distincePerClusterCenter(dist):
        totalDist = torch.sum((torch.min(dist, dim=-1)).values)

        return totalDist

    # @staticmethod
    # def distincePerClusterCenter(dist):
    #     totalDist = torch.sum(torch.sum(dist, dim=0) / (torch.max(dist) * dist.size(1)))
    #     return totalDist

    def pretrain(self,
            dataset: PairedModeDataset,
            test_ratio: float,
            batch_size: int,
            outdir: str,
            lr: float,
            ind_epoch: int = 200,
            com_epoch: int = 500,
            patience: int = 10,
            num_workers: int = 8,
            resume: str = None,
            seed: int = 2020,
            cycle_kl: bool = False,
            ds: str = None,
            **kwargs
            ):

        # basic configuration
        set_seed(seed)
        outdir = make_directory(outdir)
        device = self.get_device()

        train_inds, test_inds = train_test_split(range(len(dataset)), test_size=test_ratio, stratify=dataset.atac_obs["batch"])

        train_loader = DataLoader(
            Subset(dataset, indices=train_inds),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers, drop_last=True,
            collate_fn=dataset.collate_fn
        )

        tracker = EarlyStopping(
            eval_keys=["a_auc", "m_pcc"],
            score_keys=["a_auc", "m_pcc"],
            loss_keys=[], n_delay=5, weight=None, patience=patience
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1E-8)

        eval_sample = np.random.choice(len(test_inds), min(1000, len(test_inds)))
        saved_checkpoint = "{}/{}_5/checkpoint_{}.pt".format(outdir, ds, ind_epoch - 1)

        # load pretrained model
        if resume is not None:
            d = torch.load(resume, map_location=device)
            start = d['epoch'] + 1
            self.load_state_dict(d["model_state_dict"])
            optimizer.load_state_dict(d["optimizer_state_dict"])
            logger.info("- continue training (start epoch: {})".format(start))
        else:
            start = 0

        self.results_store = [0.0, 0.0, 0.0, 0.0, 0.0]
        i = start
        while i < ind_epoch:
            self.train()
            pbar = tqdm(train_loader, total=len(train_loader), desc="Epoch{}".format(i))
            epoch_loss, epoch_a_loss, epoch_m_loss, epoch_c_loss, epoch_kl_loss, epoch_pre_loss, epoch_s_loss = 0, 0, 0, 0, 0, 0, 0

            if cycle_kl:
                total_iter = len(train_loader) + 1
                if i < 5:
                    kl_list = np.zeros(total_iter)
                elif i < 10:
                    kl_list = frange_cycle_linear(n_iter=total_iter, stop=0.0001)
                elif i < 20:
                    kl_list = frange_cycle_linear(n_iter=total_iter, stop=0.0005)
                elif i < 30:
                    kl_list = frange_cycle_linear(n_iter=total_iter, stop=0.001)
                else:
                    kl_list = frange_cycle_linear(n_iter=total_iter, stop=0.005)

            for it, (a_x, a_counts, a_size, m_x, m_counts, m_size, batch, a_aug, m_aug) in enumerate(pbar):
                kl_w = kl_list[it]
                batch = batch.float().to(device)
                a_x, a_size = a_x.float().to(device), a_size.float().to(device)
                m_x, m_size = m_x.float().to(device), m_size.float().to(device)

                a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z = self.forward(a_x, a_size, m_x, m_size, batch)

                # positive pair
                a_aug, m_aug = a_aug.float().to(device), m_aug.float().to(device)
                a_aug_out, _, _, _, m_aug_out, _, _, _ = self.forward(a_aug, a_size, m_aug, m_size, batch)  # aug

                a_counts = a_counts.to(device)
                ## atac loss
                if self.peak_loss == "bce":
                    a_loss = F.binary_cross_entropy(a_out, a_x) + F.binary_cross_entropy(a_aug_out, a_x)
                elif self.peak_loss == "nb":
                    a_mu, a_theta = a_out
                    a_aug_mu, a_aug_theta = a_aug_out
                    a_loss = -(NegativeBinomial(mu=a_mu, theta=a_theta).log_prob(a_counts) +
                               NegativeBinomial(mu=a_aug_mu, theta=a_aug_theta).log_prob(a_counts))
                    a_loss = torch.mean(a_loss)
                elif self.peak_loss == "zinb":
                    a_mu, a_theta, a_zi = a_out
                    a_aug_mu, a_aug_theta, a_aug_zi = a_aug_out
                    a_loss = -(ZeroInflatedNegativeBinomial(mu=a_mu, theta=a_theta, zi_logits=a_zi).log_prob(a_counts) +
                               ZeroInflatedNegativeBinomial(mu=a_aug_mu, theta=a_aug_theta, zi_logits=a_aug_zi).log_prob(a_counts))
                    a_loss = torch.mean(a_loss)

                m_counts = m_counts.to(device)
                ## mrna loss
                if self.gene_loss == "mse":
                    m_loss = F.mse_loss(m_out, m_x) + F.mse_loss(m_aug_out, m_x)
                elif self.gene_loss == "nb":
                    m_mu, m_theta = m_out
                    m_aug_mu, m_aug_theta = m_aug_out
                    m_loss = -(NegativeBinomial(mu=m_mu, theta=m_theta).log_prob(m_counts) +
                               NegativeBinomial(mu=m_aug_mu, theta=m_aug_theta).log_prob(m_counts))
                    m_loss = torch.mean(m_loss)
                elif self.gene_loss == "zinb":
                    m_mu, m_theta, m_zi = m_out
                    m_aug_mu, m_aug_theta, m_aug_zi = m_aug_out
                    m_loss = -(ZeroInflatedNegativeBinomial(mu=m_mu, theta=m_theta, zi_logits=m_zi).log_prob(m_counts) +
                               ZeroInflatedNegativeBinomial(mu=m_aug_mu, theta=m_aug_theta, zi_logits=m_aug_zi).log_prob(m_counts))
                    m_loss = torch.mean(m_loss)

                if self.combine == "poe":
                    mu, logvar = self.prior_expert((1, batch_size, self.z_dim))
                    mu = torch.cat((mu, a_mu.unsqueeze(0), m_mu.unsqueeze(0)), dim=0)
                    logvar = torch.cat((logvar, a_logvar.unsqueeze(0), m_logvar.unsqueeze(0)), dim=0)
                    mu, logvar = self.poe.forward(mu, logvar)
                elif self.combine == "mean":
                    mu, logvar = self.prior_expert((1, batch_size, self.z_dim))
                    mu = torch.cat((mu, a_mu.unsqueeze(0), m_mu.unsqueeze(0)), dim=0)
                    logvar = torch.cat((logvar, a_logvar.unsqueeze(0), m_logvar.unsqueeze(0)), dim=0)
                    mu, logvar = self.poe.forward(mu, logvar)
                elif self.combine == "concat":
                    mu = torch.cat((a_mu, m_mu), dim=1)
                    logvar = torch.cat((a_logvar, m_logvar), dim=1)

                # cross loss
                if self.z_mask == None:
                    ref = torch.rand(m_z.shape)
                    self.z_mask = torch.tensor(ref > 0.95).to(m_z.device)
                c_loss = F.mse_loss(m_z.masked_fill(self.z_mask, 0.), a_z.masked_fill(self.z_mask, 0.))

                kl_loss = kl_divergence(mu, logvar)
                m_value = m_loss.item()

                loss = (m_value / a_loss.item()) * a_loss + m_loss + kl_w * kl_loss #  + (m_value / c_loss.item()) * 0.1 * c_loss

                epoch_loss += loss.item()
                epoch_a_loss += a_loss.item()
                epoch_m_loss += m_loss.item()
                epoch_kl_loss += kl_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix_str("loss(a/m/s/kl)/lr={:.3f}({:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.2g})/{:.2g}".format(epoch_loss / (it + 1), epoch_a_loss / (it + 1), epoch_m_loss / (it + 1), epoch_s_loss / (it + 1), epoch_kl_loss / (it + 1), kl_w, optimizer.param_groups[0]['lr']))

            # eval
            a_x, a_recon, m_x, m_recon = self.predict_prob(
                Subset(dataset, indices=test_inds),
                batch_size=batch_size,
                num_workers=num_workers
            )
            m_pcc = np.mean(pearsonr2d(m_x, m_recon))
            a_auc = np.mean(roc_auc_score_2d(a_x[eval_sample, :], a_recon[eval_sample, :]))
            tracker.update(
                i,
                a_auc=a_auc,
                m_pcc=m_pcc,
            )

            logger.info("Validation({}): ATAC-AUC/RNA-PCC={:.4g}/{:.4g}".format(i, a_auc, m_pcc))

            result = self.print_clustering_metric(dataset, outdir, i, ds)
            if i >= ind_epoch - 20:
                self.results_store = [self.results_store[i] + result[i] for i in range(len(result))]

            if not os.path.exists('./output/{}_5'.format(ds)):
                os.mkdir('./output/{}_5'.format(ds))
            torch.save(
                {
                    "model_state": self.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "epoch": i
                }, "{}/{}_5/checkpoint_{}.pt".format(outdir, ds, i)
            )
            logger.info("model saved\n")

            i += 1


    def main_train(self,
                   dataset: PairedModeDataset,
                    test_ratio: float,
                    batch_size: int,
                    outdir: str,
                    lr: float,
                    ind_epoch: int = 200,
                    com_epoch: int = 500,
                    patience: int = 10,
                    num_workers: int = 8,
                    resume: str = None,
                    seed: int = 2020,
                    cycle_kl: bool = False,
                    ds: str = None,
                    **kwargs
                    ):

        # basic configuration
        set_seed(seed)
        outdir = make_directory(outdir)
        device = self.get_device()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers, drop_last=True,
            collate_fn=dataset.collate_fn
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1E-8)

        # loss_func = nn.CrossEntropyLoss()
        loss_func = nn.KLDivLoss(size_average=False)

        self.results_store = [0.0, 0.0, 0.0, 0.0, 0.0]
        i = 0
        while i < com_epoch:
            self.train()

            if i == 0:
                self.clustering(dataset)
            else:
                pbar = tqdm(train_loader, total=len(train_loader), desc="Epoch{}".format(i))
                epoch_loss, epoch_a_loss, epoch_m_loss, epoch_c_loss, epoch_kl_loss, epoch_pre_loss, epoch_s_loss = 0, 0, 0, 0, 0, 0, 0
                epoch_d = 0.0

                for it, (a_x, a_counts, a_size, m_x, m_counts, m_size, batch, a_aug, m_aug) in enumerate(pbar):
                    batch = batch.float().to(device)
                    a_x, a_size = a_x.float().to(device), a_size.float().to(device)
                    m_x, m_size = m_x.float().to(device), m_size.float().to(device)

                    a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat = self.spp_forward(a_x, a_size, m_x, m_size, batch)

                    # if i < 50:
                    if self.use_cache:
                        mu_one = torch.tensor(self.mu_cache[it * batch_size: (it + 1) * batch_size]).to(device)
                        p_loss = F.mse_loss(mu_one, mu_cat)
                    else:
                        mu = torch.cat((a_mu, m_mu), dim=1)
                        p_loss = F.mse_loss(mu.detach(), mu_cat)

                    ## spp loss
                    pred, q, dist, clssfied = self.get_mid(mu_cat)
                    # p = self.target_distribution(q).detach()
                    if i % self.update_p_epoch == 1:
                        self.p_dis[it * batch_size : (it + 1) * batch_size] = self.target_distribution(q.data).tolist()
                    p = torch.tensor(self.p_dis[it * batch_size : (it + 1) * batch_size]).to(device)

                    s_loss = loss_func(q, p) / q.shape[0]

                    lambd = 0.01
                    loss = lambd * s_loss + p_loss

                    epoch_loss += loss.item()
                    epoch_s_loss += s_loss.item()
                    epoch_m_loss += p_loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix_str("loss(p/s/kl)/lr={:.3f}({:.3f}/{:.3f}/{:.3f}/{:.2g}".format(epoch_loss / (it + 1), epoch_m_loss / (it + 1), epoch_s_loss / (it + 1), epoch_kl_loss / (it + 1), optimizer.param_groups[0]['lr']))

            # eval
            result = self.print_clustering_metric(dataset, outdir, i + 100, ds)
            if i >= ind_epoch - 20:
                self.results_store = [self.results_store[i] + result[i] for i in range(len(result))]

            if not os.path.exists('./output/{}_5'.format(ds)):
                os.mkdir('./output/{}_5'.format(ds))
            torch.save(
                {
                    "model_state": self.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": 0.0,

                    "epoch": i
                }, "{}/{}_5/checkpoint_{}.pt".format(outdir, ds, i + 100)
            )
            logger.info("model saved\n")

            i += 1


    def contrastive_train(self,
                   dataset: PairedModeDataset,
                    test_ratio: float,
                    batch_size: int,
                    outdir: str,
                    lr: float,
                    ind_epoch: int = 200,
                    com_epoch: int = 500,
                    patience: int = 10,
                    num_workers: int = 8,
                    resume: str = None,
                    seed: int = 2020,
                    cycle_kl: bool = False,
                    ds: str = None,
                    **kwargs
                    ):

        # basic configuration
        set_seed(seed)
        outdir = make_directory(outdir)
        device = self.get_device()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers, drop_last=True,
            collate_fn=dataset.collate_fn
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1E-8)
        criterion = Loss(batch_size, 0.5, 1.0, device).to(device)

        self.results_store = [0.0, 0.0, 0.0, 0.0, 0.0]
        i = 0
        while i < com_epoch:
            self.train()
            pbar = tqdm(train_loader, total=len(train_loader), desc="Epoch{}".format(i))
            epoch_loss, epoch_a_loss, epoch_m_loss, epoch_c_loss, epoch_kl_loss, epoch_pre_loss, epoch_s_loss = 0, 0, 0, 0, 0, 0, 0

            for it, (a_x, a_counts, a_size, m_x, m_counts, m_size, batch, a_aug, m_aug) in enumerate(pbar):
                batch = batch.float().to(device)
                a_x, a_size = a_x.float().to(device), a_size.float().to(device)
                m_x, m_size = m_x.float().to(device), m_size.float().to(device)

                a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat = self.spp_forward(a_x, a_size, m_x, m_size, batch)

                ## con loss
                con_loss = criterion.forward_feature(a_z, m_z)

                loss = con_loss
                # loss = s_loss + p_loss
                # loss = p_loss

                epoch_loss += loss.item()
                epoch_c_loss += con_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix_str("loss(c)/lr={:.3f}({:.3f})/{:.2g}".format(epoch_loss / (it + 1), epoch_c_loss / (it + 1), optimizer.param_groups[0]['lr']))


    def print_clustering_metric(self, dataset, outdir, epoch, ds):
        ldata, a_ldata, m_ldata, gd_ldata, ca_ldata = self.evaluate(
            dataset,
            batch_size=256,
            num_workers=1,
            # use_rep='mu',
            outdir=outdir, cluster_method=["leiden", "louvain"],
            save_name=".final.{}.pdf".format(1),
            cluster_key="cell_type", color_keys=["cell_type"],
            epoch=epoch,
            ds=ds
        )

        ## cluster with concat feature
        cell_types = ldata.obs["cell_type"]
        # cell_types = [s.replace(' ', '') for s in ldata.obs["cell_type"]]
        louvain_labels = ldata.obs["louvain"]
        leiden_labels = ldata.obs["leiden"]

        logger.info('class number\t{}\t{}\t{}'.format(len(set(cell_types)), len(set(louvain_labels)), len(set(leiden_labels))))

        louvain_ari = adjusted_rand_score(cell_types, louvain_labels)
        louvain_nmi = normalized_mutual_info_score(cell_types, louvain_labels)
        leiden_ari = adjusted_rand_score(cell_types, leiden_labels)
        leiden_nmi = normalized_mutual_info_score(cell_types, leiden_labels)

        logger.info("louvain(ARI/NMI):\t{:.4f}\t{:.4f}".format(louvain_ari, louvain_nmi))
        logger.info("leiden(ARI/NMI):\t{:.4f}\t{:.4f}".format(leiden_ari, leiden_nmi))

        self.writer.add_scalar('con/louvain(ARI)', louvain_ari, epoch)
        self.writer.add_scalar('con/louvain(NMI)', louvain_nmi, epoch)

        self.writer.add_scalar('con/leiden(ARI)', leiden_ari, epoch)
        self.writer.add_scalar('con/leiden(NMI)', leiden_nmi, epoch)
        self.writer.flush()

        ## cluster with only atac feature
        a_cell_types = a_ldata.obs["cell_type"]
        a_louvain_labels = a_ldata.obs["louvain"]
        a_leiden_labels = a_ldata.obs["leiden"]

        logger.info('class number\t{}\t{}\t{}'.format(len(set(a_cell_types)), len(set(a_louvain_labels)), len(set(a_leiden_labels))))

        a_louvain_ari = adjusted_rand_score(a_cell_types, a_louvain_labels)
        a_louvain_nmi = normalized_mutual_info_score(a_cell_types, a_louvain_labels)
        a_leiden_ari = adjusted_rand_score(a_cell_types, a_leiden_labels)
        a_leiden_nmi = normalized_mutual_info_score(a_cell_types, a_leiden_labels)

        logger.info("louvain(ARI/NMI):\t{:.4f}\t{:.4f}".format(a_louvain_ari, a_louvain_nmi))
        logger.info("leiden(ARI/NMI):\t{:.4f}\t{:.4f}".format(a_leiden_ari, a_leiden_nmi))

        self.writer.add_scalar('atac/louvain(ARI)', a_louvain_ari, epoch)
        self.writer.add_scalar('atac/louvain(NMI)', a_louvain_nmi, epoch)
        self.writer.add_scalar('atac/leiden(ARI)', a_leiden_ari, epoch)
        self.writer.add_scalar('atac/leiden(NMI)', a_leiden_nmi, epoch)
        self.writer.flush()

        ## cluster with only mrna feature
        m_cell_types = m_ldata.obs["cell_type"]
        m_louvain_labels = m_ldata.obs["louvain"]
        m_leiden_labels = m_ldata.obs["leiden"]

        logger.info('class number\t{}\t{}\t{}'.format(len(set(m_cell_types)), len(set(m_louvain_labels)), len(set(m_leiden_labels))))

        m_louvain_ari = adjusted_rand_score(m_cell_types, m_louvain_labels)
        m_louvain_nmi = normalized_mutual_info_score(m_cell_types, m_louvain_labels)
        m_leiden_ari = adjusted_rand_score(m_cell_types, m_leiden_labels)
        m_leiden_nmi = normalized_mutual_info_score(m_cell_types, m_leiden_labels)


        logger.info("louvain(ARI/NMI):\t{:.4f}\t{:.4f}".format(m_louvain_ari, m_louvain_nmi))
        logger.info("leiden(ARI/NMI):\t{:.4f}\t{:.4f}".format(m_leiden_ari, m_leiden_nmi))

        self.writer.add_scalar('mrna/louvain(ARI)', m_louvain_ari, epoch)
        self.writer.add_scalar('mrna/louvain(NMI)', m_louvain_nmi, epoch)

        self.writer.add_scalar('mrna/leiden(ARI)', m_leiden_ari, epoch)
        self.writer.add_scalar('mrna/leiden(NMI)', m_leiden_nmi, epoch)
        self.writer.flush()

        ## cluster with combined feature
        gd_cell_types = gd_ldata.obs["cell_type"]
        gd_louvain_labels = gd_ldata.obs["louvain"]
        gd_leiden_labels = gd_ldata.obs["leiden"]

        logger.info('class number\t{}\t{}\t{}'.format(len(set(gd_cell_types)), len(set(gd_louvain_labels)), len(set(gd_leiden_labels))))

        gd_louvain_ari = adjusted_rand_score(gd_cell_types, gd_louvain_labels)
        gd_louvain_nmi = normalized_mutual_info_score(gd_cell_types, gd_louvain_labels)
        gd_leiden_ari = adjusted_rand_score(gd_cell_types, gd_leiden_labels)
        gd_leiden_nmi = normalized_mutual_info_score(gd_cell_types, gd_leiden_labels)

        logger.info("louvain(ARI/NMI):\t{:.4f}\t{:.4f}".format(gd_louvain_ari, gd_louvain_nmi))
        logger.info("leiden(ARI/NMI):\t{:.4f}\t{:.4f}".format(gd_leiden_ari, gd_leiden_nmi))

        self.writer.add_scalar('ga/louvain(ARI)', gd_louvain_ari, epoch)
        self.writer.add_scalar('ga/louvain(NMI)', gd_louvain_nmi, epoch)

        self.writer.add_scalar('ga/leiden(ARI)', gd_leiden_ari, epoch)
        self.writer.add_scalar('ga/leiden(NMI)', gd_leiden_nmi, epoch)
        self.writer.flush()

        ## cluster with cache feature
        ca_cell_types = ca_ldata.obs["cell_type"]
        ca_louvain_labels = ca_ldata.obs["louvain"]
        ca_leiden_labels = ca_ldata.obs["leiden"]

        logger.info('class number\t{}\t{}\t{}'.format(len(set(ca_cell_types)), len(set(ca_louvain_labels)),
                                                      len(set(ca_leiden_labels))))

        ca_louvain_ari = adjusted_rand_score(ca_cell_types, ca_louvain_labels)
        ca_louvain_nmi = normalized_mutual_info_score(ca_cell_types, ca_louvain_labels)
        ca_leiden_ari = adjusted_rand_score(ca_cell_types, ca_leiden_labels)
        ca_leiden_nmi = normalized_mutual_info_score(ca_cell_types, ca_leiden_labels)

        logger.info("louvain(ARI/NMI):\t{:.4f}\t{:.4f}".format(ca_louvain_ari, ca_louvain_nmi))
        logger.info("leiden(ARI/NMI):\t{:.4f}\t{:.4f}".format(ca_leiden_ari, ca_leiden_nmi))

        self.writer.add_scalar('ca/louvain(ARI)', ca_louvain_ari, epoch)
        self.writer.add_scalar('ca/louvain(NMI)', ca_louvain_nmi, epoch)

        self.writer.add_scalar('ca/leiden(ARI)', ca_leiden_ari, epoch)
        self.writer.add_scalar('ca/leiden(NMI)', ca_leiden_nmi, epoch)
        self.writer.flush()

        if ca_louvain_ari > louvain_ari:
            self.use_cache = True
        else:
            self.use_cache = False
        print(self.use_cache)

        # if ca_louvain_ari < louvain_ari:
        #     self.mu_cache = ca_ldata.X
        # self.use_cache = True
        # print(self.use_cache)

        return [leiden_ari, m_leiden_ari, a_leiden_ari, gd_leiden_ari]


    @torch.no_grad()
    def evaluate(self, dataset: PairedModeDataset, batch_size: int, num_workers: int, cluster_key: str,
                 cluster_method: Union[str, List[str]] = "louvain", color_keys: List[str] = 'cell_type', use_rep="mu",
                 outdir: str = None, save_name: str = None, ex = False, ds = None, epoch = 0, **kwargs):
        warn_kwargs(kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        device = self.device
        latent = list()
        a_latent = list()
        m_latent = list()
        sp_latent = list()

        self.eval()
        for a_x, _, a_size, m_x, _, m_size, batch, _ in loader:
            a_x, a_size = a_x.float().to(device), a_size.float().to(device)
            m_x, m_size = m_x.float().to(device), m_size.float().to(device)
            batch = batch.float().to(device)

            a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat = self.spp_forward(a_x, a_size, m_x, m_size, batch)

            z = torch.cat((a_z, m_z), dim=1)
            mu = torch.cat((a_mu, m_mu), dim=1)

            if use_rep == "z":
                latent.append(z.detach().cpu().numpy())
                a_latent.append(a_z.detach().cpu().numpy())
                m_latent.append(m_z.detach().cpu().numpy())
            else:
                latent.append(mu.detach().cpu().numpy())
                a_latent.append(a_mu.detach().cpu().numpy())
                m_latent.append(m_mu.detach().cpu().numpy())
                sp_latent.append(mu_cat.detach().cpu().numpy())

        latent = np.concatenate(latent, axis=0)
        a_latent = np.concatenate(a_latent, axis=0)
        m_latent = np.concatenate(m_latent, axis=0)
        sp_latent = np.concatenate(sp_latent, axis=0)
        if self.mu_cache is not None:
            ca_latent = self.mu_cache
        else:
            ca_latent = latent

        if epoch >= 90:
            pd.DataFrame(data=latent, index=dataset.mrna_obs).to_csv(os.path.join(outdir, ds, 'co_embedding_' + str(epoch) +'.csv'))
            print(os.path.join(outdir, ds, 'co_embedding_' + str(epoch) +'.csv'))
            pd.DataFrame(data=a_latent, index=dataset.mrna_obs).to_csv(os.path.join(outdir, ds, 'atac_embedding_' + str(epoch) +'.csv'))
            pd.DataFrame(data=m_latent, index=dataset.mrna_obs).to_csv(os.path.join(outdir, ds, 'mrna_embedding_' + str(epoch) +'.csv'))
            pd.DataFrame(data=sp_latent, index=dataset.mrna_obs).to_csv(os.path.join(outdir, ds, 'spp_embedding_' + str(epoch) +'.csv'))

        resolution = self.find_resolution(latent, n_clusters=self.n_kind)
        a_resolution = self.find_resolution(a_latent, n_clusters=self.n_kind)
        m_resolution = self.find_resolution(m_latent, n_clusters=self.n_kind)
        sp_resolution = self.find_resolution(sp_latent, n_clusters=self.n_kind)
        ca_resolution = self.find_resolution(ca_latent, n_clusters=self.n_kind)

        ldata = AnnData(X=latent, obs=dataset.mrna_obs)
        a_ldata = AnnData(X=a_latent, obs=dataset.mrna_obs)
        m_ldata = AnnData(X=m_latent, obs=dataset.mrna_obs)
        sp_ldata = AnnData(X=sp_latent, obs=dataset.mrna_obs)
        ca_ldata = AnnData(X=ca_latent, obs=dataset.mrna_obs)

        sc.pp.neighbors(ldata, use_rep='X')
        sc.pp.neighbors(a_ldata, use_rep='X')
        sc.pp.neighbors(m_ldata, use_rep='X')
        sc.pp.neighbors(sp_ldata, use_rep='X')
        sc.pp.neighbors(ca_ldata, use_rep='X')

        cluster_method = [cluster_method] if type(cluster_method) is str else cluster_method
        for m in cluster_method:
            if hasattr(sc.tl, m):
                if m not in ldata.obs:
                    # getattr(sc.tl, m)(ldata)
                    # getattr(sc.tl, m)(a_ldata)
                    # getattr(sc.tl, m)(m_ldata)
                    # getattr(sc.tl, m)(sp_ldata)
                    # getattr(sc.tl, m)(ca_ldata)
                    getattr(sc.tl, m)(ldata, resolution=resolution)
                    getattr(sc.tl, m)(a_ldata, resolution=a_resolution)
                    getattr(sc.tl, m)(m_ldata, resolution=m_resolution)
                    getattr(sc.tl, m)(sp_ldata, resolution=sp_resolution)
                    getattr(sc.tl, m)(ca_ldata, resolution=ca_resolution)
            else:
                logger.error("undefined clustering method: {}".format(m))


        return ldata, a_ldata, m_ldata, sp_ldata, ca_ldata

    def get_output(self, dataset: PairedModeDataset, batch_size: int, num_workers: int,
                   outdir: str = None, ds = None, **kwargs):
        warn_kwargs(kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        device = self.device
        a_latent = list()
        m_latent = list()

        self.eval()
        for a_x, _, a_size, m_x, _, m_size, batch, _ in loader:
            a_x, a_size = a_x.float().to(device), a_size.float().to(device)
            m_x, m_size = m_x.float().to(device), m_size.float().to(device)
            batch = batch.float().to(device)

            a_out, a_mu, a_logvar, a_z, m_out, m_mu, m_logvar, m_z, mu_cat = self.spp_forward(a_x, a_size, m_x, m_size, batch)

            a_latent.append(a_out.detach().cpu().numpy())
            m_latent.append(m_out.detach().cpu().numpy())

        a_latent = np.concatenate(a_latent, axis=0)
        m_latent = np.concatenate(m_latent, axis=0)

        # a_mtx = sparse.csr_matrix(a_latent)
        # m_mtx = sparse.csr_matrix(m_latent)

        a_adata = anndata.AnnData(a_latent, obs=dataset.atac_obs, var=dataset.atac_var)
        m_adata = anndata.AnnData(m_latent, obs=dataset.mrna_obs, var=dataset.mrna_var)

        m_adata.write_h5ad(os.path.join(outdir, ds, 'mrna_out.h5ad'))
        a_adata.write_h5ad(os.path.join(outdir, ds, 'atac_out.h5ad'))

        # sio.mmwrite(os.path.join(outdir, ds, 'atac_out.mtx'), a_mtx.T)
        # sio.mmwrite(os.path.join(outdir, ds, 'mrna_out.mtx'), m_mtx.T)
        print('save in ', os.path.join(outdir, ds))


    def get_device(self):
        return next(self.parameters()).device

    def prior_expert(self, size: Tuple[int, int, int]) -> Tuple[Tensor, Tensor]:
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).

        @param size: integer
                    dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                        cast CUDA on variables
        """
        device = self.get_device()
        mu = torch.autograd.Variable(torch.zeros(size, device=device))
        logvar = torch.autograd.Variable(torch.log(torch.ones(size, device=device)))
        # mu, logvar = mu.to(device), logvar.to(device)
        return mu, logvar

    def predict_prob(self, dataset: PairedModeDataset, batch_size: int, num_workers: int, **kwargs) \
            -> List[Union[List[Any], ndarray]]:
        """
        Return
        -------
        a_x
        a_recon
        m_x
        m_recon
        """

        device = self.get_device()
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        results = dict()
        res_keys = ["a_x", "a_recon", "m_x", "m_recon"]
        for k in res_keys:
            results[k] = list()

        with torch.no_grad():
            self.eval()
            for a_x, _, a_size, m_x, _, m_size, batch, _ in tqdm(loader, total=len(loader), desc="predicting"):
                a_x, a_size = a_x.float().to(device), a_size.float().to(device)
                m_x, m_size = m_x.float().to(device), m_size.float().to(device)
                batch = batch.float().to(device)

                a_out, _, _, _, m_out, _, _, _ =  self.forward(a_x, a_size, m_x, m_size, batch)

                if self.peak_loss == "bce":
                    a_recon = a_out
                elif self.peak_loss == "nb":
                    a_recon = NegativeBinomial(mu=a_out[0], theta=a_out[1]).sample()
                    a_recon = torch.log1p(a_recon)
                elif self.peak_loss == "zinb":
                    a_recon = ZeroInflatedNegativeBinomial(mu=a_out[0], theta=a_out[1], logits=a_out[2]).sample()
                    a_recon = torch.log1p(a_recon)

                if self.gene_loss == "mse":
                    m_recon = m_out
                elif self.gene_loss == "nb":
                    m_recon = NegativeBinomial(mu=m_out[0], theta=m_out[1]).sample()
                    m_recon = torch.log1p(m_recon)
                elif self.gene_loss == "zinb":
                    m_recon = ZeroInflatedNegativeBinomial(mu=m_out[0], theta=m_out[1], logits=m_out[2]).sample()
                    m_recon = torch.log1p(m_recon)

                results["a_x"].append(a_x.detach().cpu().numpy())  # (B, D)
                results["a_recon"].append(a_recon.detach().cpu().numpy())
                results["m_x"].append(m_x.detach().cpu().numpy())
                results["m_recon"].append(m_recon.detach().cpu().numpy())

        for k in results:
            results[k] = np.concatenate(results[k], axis=0)

        return [results[k] for k in res_keys]

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
        # std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)
        return z

