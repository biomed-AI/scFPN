#!/usr/bin/env python3

import argparse
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch.nn.functional as F
from tqdm import tqdm
# from biock.pytorch import PerformerEncoder, PerformerEncoderLayer, kl_divergence
from torch.utils.data import DataLoader
from biock.logger import make_logger

logger = make_logger(level="DEBUG")


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


# def elbo(recon_x, x, z_params, binary=True):
#     """
#     elbo = likelihood - kl_divergence
#     L = -elbo
#
#     Params:
#         recon_x:
#         x:
#     """
#     mu, logvar = z_params
#     kld = kl_divergence(mu, logvar)
#     if binary:
#         likelihood = -binary_cross_entropy(recon_x, x)
#     else:
#         # likelihood = -F.mse_loss(recon_x, x)
#         likelihood = -F.smooth_l1_loss(recon_x, x, beta=10)
#         # likelihood = - ((recon_x - x) * x / x.size(1) * (recon_x - x)).sum(dim=1)
#     return torch.sum(likelihood), torch.sum(kld)
#     # return likelihood, kld

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        return self.reparametrize(mu, log_var), mu, log_var

def build_mlp(layers, nhead: int=1, activation=nn.ReLU(), bn=True, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        if nhead == 1:
            net.append(nn.Linear(layers[i - 1], layers[i]))
        else:
            net.append(MultiHeadLinear(layers[i - 1], layers[i], nhead=nhead))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    # return nn.Sequential(*net)
    return nn.ModuleList(net)

class MLPEncoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(MLPEncoder, self).__init__()
        self.hidden = build_mlp([x_dim] + h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim] + h_dim)[-1], z_dim)

    def forward(self, x: Tensor) -> Tuple:
        r"""
        Args:
            x: input tensor

        Return:
            z:
            mu: 
            logvar:
        """
        out = list()
        for layer in self.hidden:
            # print(x.device)
            tmp = layer(x)
            if (tmp.shape[1] != x.shape[1]):
                out.append(tmp)
            x = tmp
        x = self.sample(x)
        out.append(x[0])
        return x, out ## -> z, mu, log_var


# class AttentionEncoder(nn.Module):
#     def __init__(self, \
#             d_model: int, \
#             nhead: int, num_layers :int, \
#             dropout: float, attention: str="vanilla",
#             **kwargs):
#         """
#         Transformer-based VAE Encoder
#
#         Dealing with DNA sequence embedding
#         seq_dim {int}:
#         attention: {'performer', 'linear', 'vanilla'}
#         """
#         super(AttentionEncoder, self).__init__()
#         if attention == 'performer':
#             enc_layer = PerformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=2 * d_model,
#                 dim_head=d_model // nhead,
#                 dropout=dropout
#             )
#             self.attention = PerformerEncoder(
#                 enc_layer,
#                 num_layers=num_layers
#             )
#         elif attention == "vanilla":
#             # input: (S, N, E): sequence length, batch size, embedding dimension
#             enc_layer = nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=2 * d_model,
#                 dropout=dropout
#             )
#             self.attention = nn.TransformerEncoder(
#                 enc_layer,
#                 num_layers=num_layers
#             )
#
#     def forward(self, x: Tensor, seq: Tensor=None) -> Tensor:
#         ## x: (B, N) ; seq: (N, E)
#         if seq is not None:
#             seq = seq.unsqueeze(0).repeat_interleave(x.size(0), 0) #
#             x = torch.cat((x.unsqueeze(2), seq), dim=2)
#             del seq
#         x = self.attention.forward(x.transpose(0, 1)).transpose(0, 1)
#         return x

class NBDecoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(NBDecoder, self).__init__()

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.mu = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            nn.Softplus()
        ) # mu >= 0
        self.theta = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            nn.Softplus()
        ) # theta > 0

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple]:
        x = self.hidden(x)
        return self.mu(x), self.theta(x)


class ZINBDecoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        """
        """
        super(ZINBDecoder, self).__init__()

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.mu = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            nn.Softplus()
        )
        self.theta = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            nn.Softplus()
        )
        self.zi_logits = nn.Linear([z_dim, *h_dim][-1], x_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        mu : >= 0
        theta : >= 0
        zi_logits : real number
        """
        for layer in self.hidden:
            x = layer(x)
        return self.mu(x), self.theta(x), self.zi_logits(x)
 
 

class MSENBDecoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0, output_activation=nn.Softplus()):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(MSENBDecoder, self).__init__()

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)

        self.recon = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            output_activation
        )
        self.mu = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            output_activation
        )
        self.theta = nn.Sequential(
            nn.Linear([z_dim, *h_dim][-1], x_dim),
            output_activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        return self.recon(x), self.mu(x), self.theta(x) + 1E-8
 

class ExptDecoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0, output_activation=nn.Sigmoid()):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(ExptDecoder, self).__init__()

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
#         self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)

        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tuple:
        out = list()
        out.append(x)
        for layer in self.hidden:
            tmp = layer(x)
            if(tmp.shape[1] != x.shape[1]):
                out.append(tmp)
            x = tmp
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x)), out
        else:
            return self.reconstruction(x), out
        
class MultiHeadLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, nhead: int, bias: bool=True, device=None, dtype=None) -> None:
        super(MultiHeadLinear, self).__init__()
        assert nhead > 1
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.weight = nn.Parameter(torch.empty(nhead, in_features, out_features)) # (D, H, H')
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, nhead))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ## input: (B, H) or (B, H, D)
        # return F.linear(input, self.weight, self.bias)
        assert len(input.size()) <= 3
        if len(input.size()) == 3:
            assert input.size(2) == self.weight.size(0), "dimension should be same in MultiHeadLinear, while input: {}, weight: {}".format(input.size(), self.weight.size())
            input = input.transpose(0, 1).transpose(0, 2)
        input = torch.matmul(input, self.weight).transpose(0, 1).transpose(1, 2)
        if self.bias is not None:
            input += self.bias
        return input

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, nhead={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.nhead
        )


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
    
    def loss_function(self, atac, mrna, a_recon, m_recon, mu, logvar):
        atac_recon_loss = F.binary_cross_entropy(a_recon, atac)
        mrna_recon_loss = F.mse_loss(m_recon, mrna)
        kl_loss = kl_divergence(mu, logvar)
        return atac_recon_loss, mrna_recon_loss, kl_loss
    
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
        mu     = torch.autograd.Variable(torch.zeros(size, device=device))
        logvar = torch.autograd.Variable(torch.log(torch.ones(size, device=device)))
        # mu, logvar = mu.to(device), logvar.to(device)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)
        return z
    
    def fit(self, ):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    def predict(self, ):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()
    p.add_argument('--seed', type=int, default=2020)
    return p


# if __name__ == "__main__":
#     p = get_args()
#     args = p.parse_args()
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#
#     attn = AttentionEncoder(
#         d_model=32, nhead=4, dim_head=8,
#         num_layers=3, dropout=0.1, attention="vanilla"
#     ).eval()
#
#     seq = torch.rand(100, 31)
#     x = np.random.rand(3, 100)
#     x_rev = torch.as_tensor(x[::-1, ].copy(), dtype=torch.float)
#     x = torch.as_tensor(x, dtype=torch.float)
#
#     out = attn.forward(x, seq).detach().numpy()
#     print(out.shape)
#     out1 = attn.forward(x_rev, seq).detach().numpy()
#     out1 = out1[::-1, ]
#     print(out1.shape)
#     print(np.abs(out - out1).max())

