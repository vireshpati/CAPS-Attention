# From https://github.com/jackyue1994/OLinear

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

################
# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

from utils.tools import forward_fill


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        # x [b,l,n]
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                # print(type(mask))
                x = x.masked_fill(mask, 0)  # in case other values are filled
                self.mean = (torch.sum(x, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                # self.mean could be nan or inf
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            self.stdev = (torch.sqrt(torch.sum((x - self.mean) ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        # x should be zero, if the values are masked
        if mask is not None:
            # forward fill
            # x, mask2 = forward_fill(x, mask)
            # x = x.masked_fill(mask2, 0)

            # mean imputation
            x = x.masked_fill(mask, 0)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
################

from layers.Transformer_EncDec import Encoder_ori, LinearEncoder

class Encoder_ori(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, one_output=False, CKA_flag=False):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag
        if self.CKA_flag:
            print('CKA is enabled...')

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, nvars, D]
        attns = []
        X0 = None  # to make Pycharm happy
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0:
                    X0 = x

                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    CudaCKA1 = CudaCKA(device=x.device)
                    cka_value = CudaCKA1.linear_CKA(X0.flatten(0, 1)[:1000], x.flatten(0, 1)[:1000])
                    print(f'CKA: \t{cka_value:.3f}')

        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns


from layers.SelfAttention_Family import AttentionLayer, EnhancedAttention

import sys


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        self.Q_chan_indep = configs.Q_chan_indep

        q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        if not os.path.isfile(q_mat_dir):
            q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
        assert os.path.isfile(q_mat_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_mat.ndim == 3 if self.Q_chan_indep else self.Q_mat.ndim == 2)
        assert (self.Q_mat.shape[0] == self.enc_in if self.Q_chan_indep else self.Q_mat.shape[0] == self.seq_len)

        q_out_mat_dir = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if not os.path.isfile(q_out_mat_dir):
            q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)
        assert os.path.isfile(q_out_mat_dir)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_out_mat.ndim == 3 if self.Q_chan_indep else self.Q_out_mat.ndim == 2)
        assert (self.Q_out_mat.shape[0] == self.enc_in if self.Q_chan_indep else
                self.Q_out_mat.shape[0] == self.pred_len)

        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation, token_num=self.enc_in,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size)
        )

        # learnable delta
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # orthogonal transformation
        # [B, N, D, T]
        if self.Q_chan_indep:
            x_trans = torch.einsum('bndt,ntv->bndv', x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = torch.einsum('bndt,tv->bndv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
            # added on 25/1/30
            # x_trans = F.gelu(x_trans)
            # [B, N, D, T]
        assert x_trans.shape[-1] == self.seq_len

        # ########## transformer ####
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)

        # [B, N, D, tau]; orthogonal transformation
        if self.Q_chan_indep:
            x = torch.einsum('bndt,ntv->bndv', x_trans, self.Q_out_mat)
        else:
            x = torch.einsum('bndt,tv->bndv', x_trans, self.Q_out_mat) + self.delta2
            # added on 25/1/30
            # x = F.gelu(x)

        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out