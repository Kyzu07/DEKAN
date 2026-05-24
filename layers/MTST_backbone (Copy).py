__all__ = ['MTST_backbone']


# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from einops import rearrange, reduce, repeat, einsum
from yacs.config import CfgNode as CN
from layers.rel_pe import RelativeSinPE, RelativeFreqPE
from models.KANS.hahn import HahnPolynomials
# from models.KANS.scratchwavkan import NaiveWaveletKANLayer
from layers.fkan import *



class KanMixer(nn.Module):
    def __init__(self, dim, len):
        super().__init__()
        self.intrapatch_kan = HahnPolynomials(dim, dim, 3, 1, 1, 7)
        self.interpatch_kan = HahnPolynomials(len, len, 3, 1, 1, 7)
        self.weights_intra = nn.Parameter(torch.randn(len, dim))
        self.weights_inter = nn.Parameter(torch.randn(len, dim))
        nn.init.ones_(self.weights_intra)
        nn.init.ones_(self.weights_inter)


    def forward(self, x):
        ab = self.intrapatch_kan(x) + x
        res_ab = x
        ab = ab.permute(0,2,1)
        ab = self.interpatch_kan(ab)
        ab = ab.permute(0,2,1)
        ab = ab + res_ab
        ba = x.permute(0,2,1)
        ba = self.interpatch_kan(ba)
        ba = ba.permute(0,2,1) + x
        res_ba = ba
        ba = self.intrapatch_kan(ba)
        ba = ba + res_ba
        return ab + ba
        # return self.weights_intra * ab + self.weights_inter * ba

    

class MTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=1, n_branches:int=3, d_model=128,
                 d_ff:int=256, norm:str='BatchNorm', xxx_dropout:float=0., dropout:float=0., act:str="gelu",
                 padding_var:Optional[int]=None,  pre_norm:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 cfg=CN(),
                 **kwargs
                 ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        if isinstance(patch_len, str):
            patch_len= patch_len.split(',')
            patch_len= [int(i) for i in patch_len]
        if isinstance(stride, str):
            stride = stride.split(',')
            stride = [int(i) for i in stride]

        patch_num = [int((context_window - patch_len[j]) / stride[j] + 1) for j in range(n_branches)]
        if padding_patch == 'end':
            patch_num =[p_n + 1 for p_n in patch_num]

        # Backbone
        self.backbone = TSTiEncoder(learn_pe, pe, c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, padding_patch=padding_patch,
                                n_layers=n_layers, n_branches=n_branches, d_model=d_model, d_ff=d_ff,
                                xxx_dropout=xxx_dropout, dropout=dropout, act=act, padding_var=padding_var, pre_norm=pre_norm,
                               cfg=cfg, **kwargs)

        # Head
        self.head_nf = [d_model * p_n for p_n in patch_num] # to be modified
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.head_dropout = head_dropout
        self.individual = individual
        self.target_window = target_window
        self.n_layers = n_layers
        self.n_branches = n_branches


        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.heads = nn.ModuleList()
            for i in range(self.n_branches):
                self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # do patching in each layer
        # ------- Encoder ------
        z, xxx = self.backbone(z)                                                      # z: [bs x nvars x seq_len]

        draw_list = [self.heads[i](z[i]) for i in range(len(z))]  # 3 branches of the last layer to diff linear layer
        z = torch.stack(draw_list, dim=-1).sum(dim=-1, keepdim = False)

        # denorm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            draw_list = [self.revin_layer(z_.permute(0,2,1), 'denorm') for z_ in draw_list]
            # draw_list = [z_.permute(0,2,1) for z_ in draw_list]
            z = z.permute(0,2,1)
        return z, draw_list, xxx

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            # x = self.head(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self,learn_pe, pe, c_in, patch_num, patch_len, stride, max_seq_len=1024, padding_patch='end',
                 n_layers=1, n_branches=3, d_model=128,
                 d_ff=256, norm='BatchNorm', xxx_dropout=0., dropout=0., act="gelu", store_xxx=False,
                padding_var=None, pre_norm=False,
                 **kwargs):

        super().__init__()

        cfg = kwargs.get('cfg', CN())

        self.n_branches = n_branches
        self.n_layers = n_layers
        self.seq_len = cfg.get('seq_len', 336)
        self.res_xxx = cfg.get('res_xxx', False)
        self.individual = cfg.get('individual', False)
        self.n_vars = cfg.get('c_in', 7)
        self.head_dropout = cfg.get('head_dropout', 0)
        self.head_nf = d_model * np.sum(patch_num)


        # Encoder
        self.encoder = nn.ModuleList([
        nn.ModuleList([TSTEncoder(learn_pe = learn_pe, pe = pe, q_len=patch_num[j], patch_len=patch_len[j], stride=stride[j], padding_patch=padding_patch,
                                    d_model=d_model, d_ff=d_ff, norm=norm, xxx_dropout=xxx_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, n_layers=n_layers, n_branches=n_branches, cfg=cfg) for j in range(n_branches)])
            for i in range(n_layers)])

        self.bottle_neck = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.seq_len , self.head_dropout)

    def forward(self, x) -> Tensor:
        # x: [bs x nvars x seq_len]
        scores = [None for j in range(self.n_branches)]
        input = x
        for i in range(self.n_layers):
            output_ls = []
            xxx_ls = []
            for j in range(self.n_branches):
                if self.res_xxx:
                    output, xxx, scores[j] = self.encoder[i][j](input, scores=scores[j])
                else:
                    output = self.encoder[i][j](input)
                output_ls.append(output.flatten(2))           # output = [bs x nvars x patch_num x d_model]
                # xxx_ls += xxx

            # except the last layer, repreject it to seq_len
            if i < self.n_layers-1:
                # for outs in output_ls:
                    # print('shape, ', outs.shape)
                # k = input()
                input =  torch.cat(output_ls, dim = -1)  # input = [bs x nvars x patch_num*d_model*3]
                input = self.bottle_neck(input)    # input = [bs x nvars x seq_len]

        return output_ls, xxx_ls #return only last layer of branches output, len(z) = n_branches


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, learn_pe, pe, q_len, patch_len, stride, padding_patch, d_model, d_ff=None,
                        norm='BatchNorm', xxx_dropout=0., dropout=0., activation='gelu', n_layers=1, pre_norm=False,
                        n_branches=3, cfg=CN()
                 ):
        super().__init__()

        self.n_layers = n_layers
        self.n_branch = n_branches
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            q_len += 1 # q_len is patch number

        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        # TST encoding
        self.layer = TSTEncoderLayer(q_len, d_model=d_model, patch_num= q_len-1, d_ff=d_ff, norm=norm,
                                            xxx_dropout=xxx_dropout, dropout=dropout,
                                            activation=activation,
                                            pre_norm=pre_norm, cfg=cfg,
                                            )



    def forward(self, x:Tensor):
        """
        input: [bs * nvars x patch_num x d_model]
        output: [bs * nvars x patch_num x d_model]
        """
        xxx_bias=None

        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        x = x.reshape(x.size(0), x.size(1), -1, self.patch_len)

        # x: [bs x nvars x patch_num x patch_len]
        n_vars = x.shape[1]
        # Linear Projection
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u)  # u: [bs * nvars x patch_num x d_model]

        output = self.layer(u)
        output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
        return output  # List[Tensor( bs x nvars x patch_num x d_model)]


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, patch_num, d_ff=256,
                 norm='BatchNorm', xxx_dropout=0, dropout=0., bias=True, activation="gelu",
                 pre_norm=False, cfg=CN(),
                 ):
        super().__init__()

        self.pre_norm = pre_norm

        self.kan = KanMixer(d_model, patch_num)
        # Add & Norm
        self.dropout_xxx = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_xxx = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_xxx = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = cfg.pre_norm


    def forward(self, src:Tensor) -> Tensor:
        # src : [10272, 85, 128] = [bs x nvar, patch_num, d_model]
        res = src
        if self.pre_norm:
            src = self.norm_xxx(src)

        # Add & Norm
        src = self.kan(src)
        src = res + self.dropout_xxx(src) # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_xxx(src)

        res = src
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        ## Position-wise Feed-Forward
        src = self.ff(src)
        ## Add & Norm
        src = res + self.dropout_ffn(src) # Add: residual connection with residual dropout
        if not self.pre_norm: # default pre_norm = False
            src = self.norm_ffn(src)


        return src


