# --- START OF FILE MTST_backbone.py ---

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
# from layers.fkan import *
# from models.KANS.mlpkan import MLPKAN
# from models.KANS.hermiite import *
# from models.KANS.hahnkanpp import *
from models.KANS.krawt import *
# from models.KANS.lucas import *
# from models.KANS.mono import *
# from models.KANS.cheby import *
from models.KANS.eff_kan import *
import math

#global script
degree = 3
hahn_alpha = 1
hahn_beta = 1
hahn_N = 7
time_proj_rank = 64
fusion_head_mlp_drop = 0
# trend_head_kernel = 1
# trend_head_bias = True


mpst_Harmonics = 2
mpst_init_trend_gate=0.2               # start conservative; model can increase if helpful
mpst_init_season_gate=0.2
mpst_use_learnable_trend_weights=True  # soft-mix across kernels with a learned softmax
mpst_pad_mode='reflect'
ar_head_kernel = 25

#weather
# mpst_period_list=(144, 288)
# mpst_kernel_sizes=(73,145,289)

#ETTm1, ETTm2
# mpst_period_list=(96,192)
# mpst_kernel_sizes=(49,97,193)

#electricity, Traffic, ETTh1, ETTh2
mpst_kernel_sizes = (25, 73, 169)
mpst_period_list = (24, 168)

krawt_q = 0.6
krawt_N = 255



class KanMixer(nn.Module):
    """
    IMPROVEMENT 1: KanMixer with Adaptive Gating.
    """
    def __init__(self, dim, len):
        super().__init__()
        # self.intrapatch_kan = MLPKAN(dim, dim, 'xavier')
        # self.interpatch_kan = MLPKAN(len, len, 'xavier')
        # self.intrapatch_kan = HahnPolynomials(dim, dim, degree, hahn_alpha, hahn_beta, hahn_N)
        # self.interpatch_kan = HahnPolynomials(len, len, degree, hahn_alpha, hahn_beta, hahn_N)
        self.intrapatch_kan = KrawtchoukPolynomials(dim, dim, degree, krawt_q, krawt_N)
        self.interpatch_kan = KrawtchoukPolynomials(len, len, degree, krawt_q, krawt_N)

        # self.intrapatch_kan = Monomials(dim, dim, degree)
        # self.interpatch_kan = Monomials(len, len, degree)

        # self.intrapatch_kan = nn.Linear(dim, dim)
        # self.interpatch_kan = nn.Linear(len, len)

        # self.intrapatch_kan = KANLinear(dim, dim, degree)
        # self.interpatch_kan = KANLinear(len, len, degree)

        # self.intrapatch_kan = ChebyshevPolynomials(dim, dim, degree)
        # self.interpatch_kan = ChebyshevPolynomials(len, len, degree)


        # self.intrapatch_kan = HermiteKAN(dim, dim)
        # self.interpatch_kan = HermiteKAN(len, len)
        # self.intrapatch_kan = HahnKANPP(dim,dim, degree=8, alpha=0, beta=0, N=255,
        #              ln_basis=True, basis_drop=0.1, use_skip=True, input_map='sigmoid')
        # self.interpatch_kan = HahnKANPP(len, len, degree=8, alpha=0, beta=0, N=255,
        #              ln_basis=True, basis_drop=0.1, use_skip=True, input_map='sigmoid')
        self.gating_layer = nn.Linear(dim, 2)

        self.alpha = nn.Parameter(torch.tensor(0.2))

        # depthwise conv over patch axis P (channel = D)
        # self.smooth = nn.Conv1d(
        #     in_channels=dim, out_channels=dim,
        #     kernel_size=3, padding=1, groups=dim, bias=False
        # )
        # # init as light moving average
        # with torch.no_grad():
        #     self.smooth.weight.zero_()
        #     k = torch.ones(3, dtype=self.smooth.weight.dtype, device=self.smooth.weight.device) / 3.0
        #     # weight shape: [D, 1, 3] when groups=D
        #     for c in range(dim):
        #         self.smooth.weight[c, 0, :] = k

        # self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: [BN, P, D]
        returns: [BN, P, D]
        """
        # smooth along P only, per-feature (depthwise)
        # Conv1d expects [N, C, L] -> use D as channels, P as length
        # x_td = x.transpose(1, 2)         # [BN, D, P]
        # y_td = self.smooth(x_td)         # [BN, D, P]
        # y    = y_td.transpose(1, 2)      # [BN, P, D]

        # x = x + self.beta * y            # light denoise gate

        ab = self.intrapatch_kan(x)
        ab = ab.permute(0, 2, 1)
        ab = self.interpatch_kan(ab)
        ab = ab.permute(0, 2, 1)

        ba = x.permute(0, 2, 1)
        ba = self.interpatch_kan(ba)
        ba = ba.permute(0, 2, 1)
        ba = self.intrapatch_kan(ba)

        # output = x + ba

        gating_input = reduce(x, 'b l d -> b d', 'mean')
        gates = F.softmax(self.gating_layer(gating_input), dim=-1)
        g_ab = gates[:, 0].unsqueeze(-1).unsqueeze(-1)
        g_ba = gates[:, 1].unsqueeze(-1).unsqueeze(-1)
        output = x + g_ab * ab + g_ba * ba

        # output = x + g_ba * ba

        # new gating
        # gates = F.softmax(self.gating_layer(x), dim=-1)  # [B,L,2]
        # g_ab = gates[..., 0].unsqueeze(-1)               # [B,L,1]
        # g_ba = gates[..., 1].unsqueeze(-1)               # [B,L,1]
        # output = x + self.alpha * (g_ab * ab + g_ba * ba)


        return output


class ConvMixer(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        # Depthwise separable convolution: efficient and channel-independent
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, 
                              kernel_size=kernel_size, padding='same', groups=d_model)

    def forward(self, x):
        # x shape: [bs*nvars, patch_num, d_model]
        # Permute for Conv1d: [bs*nvars, d_model, patch_num]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Permute back: [bs*nvars, patch_num, d_model]
        return x.permute(0, 2, 1)


class BranchAggregator(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x):
        pass

class AttentionFusionHeadMLP(nn.Module):
    def __init__(self, d_model: int, n_branches: int, hidden: int = None, p_drop: float = 0.0):
        super().__init__()
        self.n_branches = n_branches
        hidden = d_model if hidden is None else hidden
        feat_dim = d_model * 2  # mean + std
        self.scorer = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)
        )

    def _feat(self, b: torch.Tensor) -> torch.Tensor:
        # # b: [bs, nvars, p, d]
        # mean = torch.mean(b, dim=2)
        # std  = torch.sqrt(torch.var(b, dim=2, unbiased=False) + 1e-6)
        # return torch.cat([mean, std], dim=-1)  # [bs,nvars,2d]
        # print(b.shape)
        mean  = b.mean(2)
        std   = (b.var(2, unbiased=False) + 1e-6).sqrt()
        # slope = (b[:, :, 1:, :] - b[:, :, :-1, :]).mean(2)
        feat  = torch.cat([mean, std], dim=-1)   # set feat_dim = 3*d_model
        return feat

    def forward(self, branch_outputs_list):
        feats = [self._feat(b) for b in branch_outputs_list]       # n_br * [B,V,2d]
        feats = torch.stack(feats, dim=2)                          # [B,V,N,2d]
        B,V,N,F = feats.shape
        scores = self.scorer(feats.reshape(B*V*N, F)).reshape(B,V,N,1)
        return torch.softmax(scores.squeeze(-1), dim=2)            # [B,V,N]

class MPSTDecomp(nn.Module):
    """
    Multi-Period Seasonal–Trend decomposition for multivariate time series.

    Inputs:
      x: [B, N, L]  (batch, variables/channels, seq_len)

    Returns:
      residual: [B, N, L]
      trend:    [B, N, L]
      seasonal: [B, N, L]
      aux: dict with gates and per-kernel trend details
    """
    def __init__(
        self,
        seq_len: int,
        trend_kernel_sizes,  # ~day / ~3-day / ~week (adjust for your sampling)
        periods,                 # daily, weekly (adjust if not hourly)
        harmonics_per_period,            # 1–3 usually enough
        init_trend_gate,               # start conservative; model can increase if helpful
        init_season_gate,
        use_learnable_trend_weights,  # soft-mix across kernels with a learned softmax
        pad_mode,                # padding for MA filters
    ):
        super().__init__()
        self.L = int(seq_len)
        self.trend_kernel_sizes = tuple(int(k) for k in trend_kernel_sizes)
        self.periods = tuple(int(p) for p in periods)
        self.H = int(harmonics_per_period)
        self.pad_mode = pad_mode

        # ---- Trend: register one buffer per MA kernel (avoids concat-of-different-lengths)
        self.ma_kernel_ids = []
        for i, k in enumerate(self.trend_kernel_sizes):
            w = torch.ones(1, 1, k, dtype=torch.float32) / float(k)  # simple moving average
            self.register_buffer(f'_ma_w_{i}', w)                     # shape [1,1,k]
            self.ma_kernel_ids.append(i)
        self.Kt = len(self.ma_kernel_ids)

        # Optional learnable mixture across Kt trend candidates
        if use_learnable_trend_weights:
            self.trend_logits = nn.Parameter(torch.zeros(self.Kt))
        else:
            self.register_buffer('trend_logits', torch.zeros(self.Kt))
        self.use_learnable_trend_weights = use_learnable_trend_weights

        # ---- Seasonal: fixed Fourier basis with multiple periods & harmonics
        # Build basis B: [M, L], M = 2 * H * len(periods)  (cos & sin for each harmonic)
        if self.H > 0 and len(self.periods) > 0:
            B_list = []
            t = torch.arange(self.L, dtype=torch.float32).unsqueeze(0)  # [1, L]
            for P in self.periods:
                P = float(P)
                for h in range(1, self.H + 1):
                    ang = 2.0 * math.pi * h * t / P                     # [1, L]
                    B_list.append(torch.cos(ang))
                    B_list.append(torch.sin(ang))
            B = torch.cat(B_list, dim=0) if B_list else torch.zeros(0, self.L)  # [M, L]
            if B.numel() > 0:
                B = B / (B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)).sqrt()  # row-normalize
        else:
            B = torch.zeros(0, self.L)
        self.register_buffer('fourier_basis', B)  # [M, L]
        self.M = B.shape[0]

        # ---- Learnable gates (sigmoid) for how much to subtract
        self.trend_gate = nn.Parameter(torch.tensor(self._inv_sigmoid(init_trend_gate), dtype=torch.float32))
        self.season_gate = nn.Parameter(torch.tensor(self._inv_sigmoid(init_season_gate), dtype=torch.float32))

    @staticmethod
    def _inv_sigmoid(p):
        p = float(min(max(p, 1e-6), 1.0 - 1e-6))
        return math.log(p / (1.0 - p))

    def _depthwise_ma_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply each MA kernel separately (with appropriate padding),
        then stack the outputs along a new last dim.

        Args:
          x: [B, N, L]

        Returns:
          T: [B, N, L, Kt] — trend candidates for each kernel
        """
        B, N, L = x.shape
        x1 = x.reshape(B * N, 1, L)  # depthwise via reshape
        outs = []
        for i in self.ma_kernel_ids:
            w = getattr(self, f'_ma_w_{i}')            # [1,1,k_i]
            k_i = int(w.shape[-1])
            pad = (k_i - 1) // 2
            if pad > 0:
                xpad = F.pad(x1, (pad, pad), mode=self.pad_mode)
            else:
                xpad = x1
            yi = F.conv1d(xpad, w)                     # [B*N,1,L]
            outs.append(yi)
        T = torch.cat(outs, dim=1)                     # [B*N, Kt, L]
        T = T.transpose(1, 2).reshape(B, N, L, self.Kt)  # [B, N, L, Kt]
        return T

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, L]
        returns: residual, trend, seasonal, aux
        """
        B, N, L = x.shape
        assert L == self.L, f"MPSTDecomp: seq_len mismatch (got {L}, expected {self.L})"

        # ---- Trend candidates via MA bank
        trends = self._depthwise_ma_stack(x)  # [B, N, L, Kt]

        if self.use_learnable_trend_weights:
            w = torch.softmax(self.trend_logits, dim=0)          # [Kt]
        else:
            w = torch.full((self.Kt,), 1.0 / max(1, self.Kt), device=x.device, dtype=x.dtype)

        trend = (trends * w.view(1, 1, 1, -1)).sum(dim=-1)       # [B, N, L]

        # ---- Seasonal via Fourier projection (multi-period, multi-harmonic)
        if self.M > 0:
            # c = x @ B^T,  seasonal = c @ B
            # x: [B, N, L], B: [M, L]
            c = torch.matmul(x, self.fourier_basis.t())          # [B, N, M]
            seasonal = torch.matmul(c, self.fourier_basis)       # [B, N, L]
        else:
            seasonal = torch.zeros_like(x)

        # ---- Gates
        g_t = torch.sigmoid(self.trend_gate)
        g_s = torch.sigmoid(self.season_gate)

        # Residual after subtracting (gated) trend & seasonal
        residual = x - g_t * trend - g_s * seasonal

        aux = {
            'trend_gate': g_t.detach(),
            'season_gate': g_s.detach(),
            'trend_mix_weights': (w.detach() if isinstance(w, torch.Tensor) else w),
            'trend_candidates': trends.detach(),  # [B, N, L, Kt]
        }
        return residual, trend, seasonal, aux

# class ARResidual(nn.Module):
#     def __init__(self, in_len, out_len, n_vars, kernel=25, bias=False, init="crop_or_hold"):
#         super().__init__()
#         self.in_len, self.out_len = in_len, out_len
#         self.proj = nn.Conv1d(n_vars, n_vars, kernel_size=kernel,
#                               padding=kernel // 2, groups=n_vars, bias=False)
#         self.tproj = nn.Linear(in_len, out_len, bias=bias)  # shared across B,N
#         self._init_timeproj(self.tproj, init)

#     @torch.no_grad()
#     def _init_timeproj(self, layer: nn.Linear, mode: str):
#         W = torch.zeros(self.out_len, self.in_len)  # Linear stores weight as [T, L]
#         if mode == "crop_or_hold":
#             if self.out_len <= self.in_len:
#                 W[:, self.in_len - self.out_len:] = torch.eye(self.out_len)  # copy last T steps
#             else:
#                 W[:, :self.in_len] = torch.eye(self.in_len)
#                 W[:, self.in_len - 1:] = 0.0
#                 W[:, -1] = 1.0  # repeat last step for extra positions
#         elif mode == "linear_resample":
#             t = torch.linspace(0, self.in_len - 1, self.out_len)
#             for j in range(self.out_len):
#                 i0 = int(torch.floor(t[j]).item())
#                 i1 = min(i0 + 1, self.in_len - 1)
#                 w1 = float(t[j] - i0)
#                 W[j, i0] = 1.0 - w1
#                 W[j, i1] += w1
#         else:
#             nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)
#             return
#         layer.weight.copy_(W)
#         if layer.bias is not None:
#             layer.bias.zero_()

#     def forward(self, x):            # x: [B, L, N]
#         y = self.proj(x.transpose(1, 2))     # [B, N, L]
#         y = self.tproj(y)                    # [B, N, T], applied on last dim
#         return y.transpose(1, 2)             # [B, T, N]


class TimeProjector(nn.Module):
    """Low-rank linear map on the time axis: [B,N,L] -> [B,N,T]."""
    def __init__(self, in_len: int, out_len: int, rank: int = 64):
        super().__init__()
        r = min(rank, in_len, out_len)
        self.A = nn.Parameter(torch.randn(in_len, r) * 0.01)   # L x r
        self.B = nn.Parameter(torch.randn(out_len, r) * 0.01)  # T x r
        self.bias = nn.Parameter(torch.zeros(out_len))

    def forward(self, x):  # x: [B,N,L]
        y = x @ self.A                 # [B,N,r]
        y = y @ self.B.transpose(0,1)  # [B,N,T]
        return y + self.bias.view(1,1,-1)



# class MTST_backbone(nn.Module):
#     def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
#                  n_layers:int=1, n_branches:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
#                  d_ff:int=256, norm:str='BatchNorm', xxx_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
#                  padding_var:Optional[int]=None, xxx_mask:Optional[Tensor]=None,  pre_norm:bool=False, store_xxx:bool=False,
#                  pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
#                  pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
#                  verbose:bool=False, res_attention: bool = True,
#                  cfg=CN(),
#                  **kwargs
#                  ):

#         super().__init__()

#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

#         # Patching
#         if isinstance(patch_len, str):
#             patch_len= patch_len.split(',')
#             patch_len= [int(i) for i in patch_len]
#         if isinstance(stride, str):
#             stride = stride.split(',')
#             stride = [int(i) for i in stride]

#         patch_num = [int((context_window - patch_len[j]) / stride[j] + 1) for j in range(n_branches)]
#         if padding_patch == 'end':
#             patch_num =[p_n + 1 for p_n in patch_num]

#         # Backbone
#         self.backbone = TSTiEncoder(learn_pe, pe, c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, padding_patch=padding_patch,
#                                 n_layers=n_layers, n_branches=n_branches, d_model=d_model, d_ff=d_ff,
#                                 xxx_dropout=xxx_dropout, dropout=dropout, act=act, padding_var=padding_var, pre_norm=pre_norm,
#                                cfg=cfg, **kwargs)

#         # Head
#         self.head_nf = [d_model * p_n for p_n in patch_num] # to be modified
#         self.n_vars = c_in
#         self.pretrain_head = pretrain_head
#         self.head_type = head_type
#         self.head_dropout = head_dropout
#         self.individual = individual
#         self.target_window = target_window
#         self.n_layers = n_layers
#         self.n_branches = n_branches


#         if self.pretrain_head:
#             self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
#         elif head_type == 'flatten':
#             self.heads = nn.ModuleList()
#             for i in range(self.n_branches):
#                 self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))


#     def forward(self, z, temp):                                                                   # z: [bs x nvars x seq_len]
#         # norm
#         if self.revin:
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'norm')
#             z = z.permute(0,2,1)

#         # do patching in each layer
#         # ------- Encoder ------
#         z, xxx = self.backbone(z)                                                      # z: [bs x nvars x seq_len]

#         draw_list = [self.heads[i](z[i]) for i in range(len(z))]  # 3 branches of the last layer to diff linear layer
#         z = torch.stack(draw_list, dim=-1).sum(dim=-1, keepdim = False)

#         # denorm
#         if self.revin:
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'denorm')
#             draw_list = [self.revin_layer(z_.permute(0,2,1), 'denorm') for z_ in draw_list]
#             # draw_list = [z_.permute(0,2,1) for z_ in draw_list]
#             z = z.permute(0,2,1)
#         return z, draw_list, xxx

#     def create_pretrain_head(self, head_nf, vars, dropout):
#         return nn.Sequential(nn.Dropout(dropout),
#                     nn.Conv1d(head_nf, vars, 1)
#                     )

class SeasonalHeadFourier(nn.Module):
    """
    Projects the decomposed seasonal component onto the same Fourier basis used
    by MPSTDecomp, then continues it into the forecast horizon.

    Expects:
      seasonal_in: [B, N, L]     (seasonal from MPSTDecomp)
    Returns:
      seasonal_out: [B, N, T]    (continued seasonal forecast)
    """
    def __init__(self, decomp: MPSTDecomp, out_len: int):
        super().__init__()
        self.L = int(decomp.L)                 # context length
        self.T = int(out_len)                  # forecast length
        self.periods = decomp.periods
        self.H = decomp.H
        # reuse row-normalized basis on input window
        self.register_buffer('B_in', decomp.fourier_basis.clone())   # [M, L]
        self.M = self.B_in.shape[0]

        # build the *future* basis B_out for steps [L, L+T)
        if self.M > 0:
            t = torch.arange(self.L, self.L + self.T, dtype=torch.float32).unsqueeze(0)  # [1, T]
            B_list = []
            for P in self.periods:
                P = float(P)
                for h in range(1, self.H + 1):
                    ang = 2.0 * math.pi * h * t / P
                    B_list.append(torch.cos(ang))
                    B_list.append(torch.sin(ang))
            B_out = torch.cat(B_list, dim=0)     # [M, T]
            # row-normalize to match B_in’s convention
            B_out = B_out / (B_out.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)).sqrt()
        else:
            B_out = torch.zeros(0, self.T, dtype=torch.float32)

        self.register_buffer('B_out', B_out)     # [M, T]

        # safety gate (starts ~0 so this head cannot hurt initial accuracy)
        self.beta = nn.Parameter(torch.tensor(-2.2))   # sigmoid(-2.2) ≈ 0.10

    def forward(self, seasonal_in: torch.Tensor) -> torch.Tensor:
        # seasonal_in ≈ C @ B_in, with B_in rows normalized ⇒ C ≈ seasonal_in @ B_in^T
        if self.M == 0:
            return torch.zeros(seasonal_in.shape[0], seasonal_in.shape[1], self.T,
                               device=seasonal_in.device, dtype=seasonal_in.dtype)
        C = torch.matmul(seasonal_in, self.B_in.transpose(0, 1))         # [B, N, M]
        seasonal_out = torch.matmul(C, self.B_out)                        # [B, N, T]
        return torch.sigmoid(self.beta) * seasonal_out


class MTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=1, n_branches:int=3, d_model=128,
                 d_ff:int=256, norm:str='BatchNorm', xxx_dropout:float=0.05, dropout:float=0.05, act:str="gelu",
                 padding_var:Optional[int]=None,  pre_norm:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0.05, head_dropout = 0.05, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 cfg=CN(),
                 **kwargs
                 ):
        super().__init__()
        # print(revin, affine, subtract_last)
        # affine = True
        # subtract_last = True
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        if isinstance(patch_len, str):
            patch_len= patch_len.split(',')
            patch_len= [int(i) for i in patch_len]
        if isinstance(stride, str):
            stride = stride.split(',')
            stride = [int(i) for i in stride]

        patch_num = [int((context_window - p_len) / s + 1) for p_len, s in zip(patch_len, stride)]

        # print(dropout, xxx_dropout,fc_dropout, head_dropout)
        
        self.backbone = TSTiEncoder(learn_pe, pe, c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, padding_patch=padding_patch,
                                n_layers=n_layers, n_branches=n_branches, d_model=d_model, d_ff=d_ff,
                                xxx_dropout=xxx_dropout, dropout=dropout, act=act, padding_var=padding_var, pre_norm=pre_norm,
                               cfg=cfg, **kwargs)

        if padding_patch == 'end':
            patch_num_for_head = [p_n + 1 for p_n in patch_num]
        else:
            patch_num_for_head = patch_num
            
        self.head_nf = [d_model * p_n for p_n in patch_num_for_head]
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.head_dropout = head_dropout
        self.individual = individual
        self.target_window = target_window
        self.n_layers = n_layers
        self.n_branches = n_branches

        # in MTST_backbone.__init__(...)
        # self.fusion_head = AttentionFusionHeadMLP(d_model, n_branches, hidden=d_model, p_drop=0.0)
        # self.fusion_head = GumbelTopKRouter(
        #     d_model, n_branches,
        #     hidden=d_model, topk=2,  # try 1–2
        #     tau=0.9, hard=True, hard_eval=True, detach_probe=True
        # )

        # before:
        # weights = attention_fusion_head_mlp(branch_outputs)   # [B,V,N]
        # fused   = sum_n weights[:,:,n,None,None] * branch_outputs[n]

        # after (choose one head):
        # self.fusion_head = TokenWiseBranchAttentionMLP(d_model=d_model, n_branches=n_branches, hidden=d_model, p_drop=0.1)
        # or: head = ConcatLinearFusion(D, N, p_drop=0.1)
        # self.fusion_head = MTSTStyleFuse(D, N, n_patches=P, p_drop=0.1)
        # or: head = UniformAverageFusion()
        # or: head = GumbelTopKFusion(D, N, hidden=D, k=1, tau=1.0)

        



        # self.fusion_head = AttentionFusionHead(d_model, n_branches)
        self.fusion_head = AttentionFusionHeadMLP(d_model, n_branches, hidden=d_model, p_drop=fusion_head_mlp_drop)
        # self.fusion_head = KANRouterFusionHead(d_model, n_branches, topk=2, hidden=64, freeze_kan=True)
        # self.fusion_head = SGSavGolRouterFusionHead(
        #     d_model, n_branches,
        #     window=7, polyorder=3,   # SG settings (odd window, degree < window)
        #     topk=2,                  # keep 2 best branches per variable (optional)
        #     hidden=64,               # small scorer MLP
        #     temperature=0.9          # <1.0 sharpens selection a bit
        # )

        # self.fusion_head = DCTRouterFusionHead(
        #     d_model, n_branches,
        #     keep_k=None,           # or an int (e.g., 4). None -> keep_ratio is used
        #     keep_ratio=0.25,       # first 25% DCT coeffs (low-freq)
        #     topk=2,                # optional sparsification
        #     hidden=64,
        #     temperature=0.9
        # )

        # self.decomp = SeasonalTrendDecomp(seq_len=context_window, keep_ratio=0.25)  # try 0.25 or keep_k=24/168
        # self.trend_gate = nn.Parameter(torch.tensor(0.5))


        self.time_proj_trend = TimeProjector(in_len=context_window, out_len=target_window, rank=time_proj_rank)
        # (Optional) if you add a seasonal head: self.time_proj_season = TimeProjector(context_window, target_window, rank=32)




        self.decomp = MPSTDecomp(
            seq_len=context_window,           # your encoder input length
            trend_kernel_sizes=mpst_kernel_sizes, # ~day / ~3-day / ~week (tweak to your sampling)
            periods=mpst_period_list,                # daily & weekly cycles (adjust if not hourly)
            harmonics_per_period=mpst_Harmonics,           # 2-3 usually enough
            init_trend_gate=mpst_init_trend_gate,
            init_season_gate=mpst_init_season_gate,
            use_learnable_trend_weights = mpst_use_learnable_trend_weights,
            pad_mode = mpst_pad_mode,
        )

        self.seasonal_head = SeasonalHeadFourier(self.decomp, out_len=target_window)

        # optional: learned fusion gates (sigmoid used at call)
        self.alpha_t = nn.Parameter(torch.tensor(0.0))  # trend gate (start small)
        self.alpha_s = nn.Parameter(torch.tensor(0.0))  # seasonal gate (start small)

        # ensure you have a trend head; reuse your ARResidual or a 1x1 conv as fallback
        # Example if you already had ARResidual:
        #   self.ar_head = ARResidual(d_model=..., kernel_size=73)
        # Otherwise:
        # self.trend_head = nn.Conv1d(in_channels=self.n_vars, out_channels=self.n_vars, kernel_size=trend_head_kernel, bias=trend_head_bias)
        

        if self.pretrain_head:
            self.head = self.create_pretrain_head(sum(self.head_nf), c_in, fc_dropout)
        elif head_type == 'flatten':
            self.heads = nn.ModuleList()
            for i in range(self.n_branches):
                self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))

        self.ar_head = ARResidual(in_len=context_window, out_len=target_window, n_vars=c_in, kernel=ar_head_kernel)

    # without mpstdecomp forward
    # def forward(self, z, temp = 0.1):
    #     B, N, L = z.shape
    #     if getattr(self, 'revin', False):
    #         z_norm = z.permute(0, 2, 1)                  # [B, L, N]
    #         z_norm = self.revin_layer(z_norm, 'norm')    # [B, L, N]
    #         z_norm = z_norm.permute(0, 2, 1)             # [B, N, L]
    #     else:
    #         z_norm = z

    #     features_per_branch, aux_backbone = self.backbone(z_norm, temp)

    #     branch_preds = [self.heads[i](features_per_branch[i]) for i in range(self.n_branches)]

    #     stacked = torch.stack(branch_preds, dim=2)             # [B, N, N_br, T]

    #     fuse_w = self.fusion_head(features_per_branch)         
    #     fuse_w = torch.softmax(fuse_w, dim=-1)                 
    #     fused_residual_pred = (stacked * fuse_w.unsqueeze(-1)).sum(dim=2)  # [B, N, T]


    #     y_pred = fused_residual_pred

    #     if getattr(self, 'revin', False):
    #         y_denorm = y_pred.permute(0, 2, 1)                 # [B, T, N]
    #         y_denorm = self.revin_layer(y_denorm, 'denorm')    # [B, T, N]
    #         y_denorm = y_denorm.permute(0, 2, 1)               # [B, N, T]

    #         branch_preds_denorm = []
    #         for bp in branch_preds:
    #             bpd = self.revin_layer(bp.permute(0, 2, 1), 'denorm').permute(0, 2, 1)
    #             branch_preds_denorm.append(bpd)

    #         aux = None
    #         return y_denorm, branch_preds_denorm, aux

    #     aux = {'decomp': decomp_aux, 'backbone': aux_backbone}
    #     return y_pred, branch_preds, aux
    
    #mpst forward
    def forward(self, z, temp = 0.1):
        """
        z: [B, N, L]  (batch, variables, context_len)
        Returns: (y_pred, branch_preds, aux) with the same shapes/semantics as your original code
        """
        B, N, L = z.shape

        # ---------------------------
        # 1) RevIN (if enabled)
        # ---------------------------
        if getattr(self, 'revin', False):
            # your revin likely expects [B, L, N]
            z_norm = z.permute(0, 2, 1)                  # [B, L, N]
            z_norm = self.revin_layer(z_norm, 'norm')    # [B, L, N]
            z_norm = z_norm.permute(0, 2, 1)             # [B, N, L]
        else:
            z_norm = z

        # ---------------------------
        # 2) Seasonal–Trend Decomposition
        # ---------------------------
        # residual is what goes into the main backbone; trend/seasonal have their own small heads
        residual, trend, seasonal, decomp_aux = self.decomp(z_norm)  # each [B, N, L]

                # ---- DEBUG/INSPECT: extract first univariate series of first batch
                # ---- DEBUG/INSPECT: extract first univariate series of first batch (first 96) as NumPy + save to disk
        # if True:
        #     import os
        #     import numpy as np

        #     L_KEEP = 96

        #     # Where to save: use self.save_dir if set, otherwise default folder
        #     save_root = getattr(self, 'save_dir', 'mtst_components')
        #     os.makedirs(save_root, exist_ok=True)

        #     # Full-length (normalized) 1D tensors
        #     s_norm_full        = z_norm[0, 0].detach().clone()        # [L]
        #     trend_norm_full    = trend[0, 0].detach().clone()
        #     seasonal_norm_full = seasonal[0, 0].detach().clone()
        #     residual_norm_full = residual[0, 0].detach().clone()

        #     # Slice to first 96
        #     s_norm_96        = s_norm_full[:L_KEEP]
        #     trend_norm_96    = trend_norm_full[:L_KEEP]
        #     seasonal_norm_96 = seasonal_norm_full[:L_KEEP]
        #     residual_norm_96 = residual_norm_full[:L_KEEP]

        #     # Helper: denorm full series, then slice to 96 (keeps RevIN state consistent)
        #     def _denorm_1d_first96(series_1d_full: torch.Tensor) -> torch.Tensor:
        #         s = series_1d_full.unsqueeze(0).unsqueeze(-1)   # [1, L, 1]
        #         s = self.revin_layer(s, 'denorm')               # [1, L, 1]
        #         return s.squeeze(0).squeeze(-1)[:L_KEEP]        # [96]

        #     # Build dict of NumPy arrays (normalized)
        #     comp_np = {
        #         'z_norm':        s_norm_96.detach().cpu().numpy(),
        #         'trend_norm':    trend_norm_96.detach().cpu().numpy(),
        #         'seasonal_norm': seasonal_norm_96.detach().cpu().numpy(),
        #         'residual_norm': residual_norm_96.detach().cpu().numpy(),
        #     }

        #     # Save normalized arrays
        #     np.save(os.path.join(save_root, 'z_norm_first96.npy'),        comp_np['z_norm'])
        #     np.save(os.path.join(save_root, 'trend_norm_first96.npy'),    comp_np['trend_norm'])
        #     np.save(os.path.join(save_root, 'seasonal_norm_first96.npy'), comp_np['seasonal_norm'])
        #     np.save(os.path.join(save_root, 'residual_norm_first96.npy'), comp_np['residual_norm'])

        #     # Original-scale (if RevIN is enabled)
        #     if getattr(self, 'revin', False):
        #         comp_np.update({
        #             'z_orig':        _denorm_1d_first96(s_norm_full).detach().cpu().numpy(),
        #             'trend_orig':    _denorm_1d_first96(trend_norm_full).detach().cpu().numpy(),
        #             'seasonal_orig': _denorm_1d_first96(seasonal_norm_full).detach().cpu().numpy(),
        #             'residual_orig': _denorm_1d_first96(residual_norm_full).detach().cpu().numpy(),
        #         })
        #         # Save original-scale arrays
        #         np.save(os.path.join(save_root, 'z_orig_first96.npy'),        comp_np['z_orig'])
        #         np.save(os.path.join(save_root, 'trend_orig_first96.npy'),    comp_np['trend_orig'])
        #         np.save(os.path.join(save_root, 'seasonal_orig_first96.npy'), comp_np['seasonal_orig'])
        #         np.save(os.path.join(save_root, 'residual_orig_first96.npy'), comp_np['residual_orig'])

        #     x = input()
        #     # Keep in memory as well for quick access
        #     self.last_components = comp_np


        # Suppose x is the original input [B, L, N]
        # trend_pred, seasonal_pred are from MPSTDecomp
        # residual = x - (trend_pred + seasonal_pred)

        # compute energy ratio
        # num = torch.sum(residual ** 2).item()
        # den = torch.sum(z_norm ** 2).item()
        # residual_energy_ratio = num / (den + 1e-8)

        # print(f"[Residual Energy Ratio] {residual_energy_ratio:.4f}")


        # ---------------------------
        # 3) Backbone on residual branch
        # ---------------------------
        # Your backbone likely returns a list of per-branch features (one per patch scale).
        # Here we assume it returns: features_per_branch (list/tensor), aux_backbone
        features_per_branch, aux_backbone = self.backbone(residual, temp)


        # old aggregation starts
        # # Per-branch forecasts with your existing heads
        branch_preds = [self.heads[i](features_per_branch[i]) for i in range(self.n_branches)]
        # # Stack to [B, N, N_branches, T] or [B, N, T, N_branches] depending on your head output.
        # # Below assumes head returns [B, N, T]
        stacked = torch.stack(branch_preds, dim=2)             # [B, N, N_br, T]

        # # Fusion weights from fusion head -> [B, N, N_br]
        fuse_w = self.fusion_head(features_per_branch)         # adapt if your fusion head API differs
        fuse_w = torch.softmax(fuse_w, dim=-1)                 # ensure it’s a convex mix

        # # Weighted sum across branches → seasonal/residual forecast
        # # expand fuse_w to broadcast over T
        fused_residual_pred = (stacked * fuse_w.unsqueeze(-1)).sum(dim=2)  # [B, N, T]
        # old aggregation ends

        # ---------------------------
        # 4) Trend & Seasonal small heads
        if hasattr(self, 'ar_head') and self.ar_head.out_len <= L:
            trend_pred = self.ar_head(trend.transpose(1, 2)).transpose(1, 2)  # L->L (<=L)
        else:
            trend_pred = self.time_proj_trend(trend)  # L -> T (works when T> L)

        # If you actually use seasonal_pred:
        # seasonal_pred = self.time_proj_season(seasonal)

        # ---------------------------
        # Trend head: prefer ARResidual if you have it, otherwise 1x1 conv on time dim
        # if hasattr(self, 'ar_head'):
        #     # ar_head often expects [B, T, N] → project and permute back
        #     trend_pred = self.ar_head(trend.transpose(1, 2)).transpose(1, 2)   # [B, N, T]
        # else:
        #     # simple linear over channels per time-step
        #     trend_pred = self.trend_head(trend)                                # [B, N, L]→ if needed resize to T later
            # If your horizon T != L, add a tiny head to map [L]→[T]; otherwise keep as is.

        # Optional: seasonal head (often not needed; the backbone already models high-freq)
        # If you do want it, a very small conv or linear can help:
        if hasattr(self, 'seasonal_head'):
            seasonal_pred = self.seasonal_head(seasonal)                        # [B, N, T]
        else:
            seasonal_pred = torch.zeros_like(fused_residual_pred)

        # Learned gates (sigmoid) keep the extra branches from hurting early on
        alpha_t = torch.sigmoid(self.alpha_t) if hasattr(self, 'alpha_t') else 0.0
        alpha_s = torch.sigmoid(self.alpha_s) if hasattr(self, 'alpha_s') else 0.0

        y_pred = fused_residual_pred + alpha_t * trend_pred + alpha_s * seasonal_pred  # [B, N, T]

        # ---------------------------
        # 5) RevIN denorm (if enabled)
        # ---------------------------
        if getattr(self, 'revin', False):
            y_denorm = y_pred.permute(0, 2, 1)                 # [B, T, N]
            y_denorm = self.revin_layer(y_denorm, 'denorm')    # [B, T, N]
            y_denorm = y_denorm.permute(0, 2, 1)               # [B, N, T]

            # Optionally denorm each branch prediction if your original return includes them
            branch_preds_denorm = []
            for bp in branch_preds:
                bpd = self.revin_layer(bp.permute(0, 2, 1), 'denorm').permute(0, 2, 1)
                branch_preds_denorm.append(bpd)

            # You can pass auxiliary info out if your signature expects it
            aux = {'decomp': decomp_aux, 'backbone': aux_backbone}
            return y_denorm, branch_preds_denorm, aux

        # If RevIN is off, return raw predictions
        aux = {'decomp': decomp_aux, 'backbone': aux_backbone}
        return y_pred, branch_preds, aux


    # seasonal vs trend forward
    # def forward(self, z):
    #     if self.revin:
    #         z = z.permute(0,2,1)
    #         z = self.revin_layer(z, 'norm')
    #         z = z.permute(0,2,1)

    #     # --- NEW: seasonal–trend split along time ---
    #     seasonal, trend = self.decomp(z)          # [B, N, L] each

    #     # Feed SEASONAL into your existing backbone
    #     z_backbone, xxx = self.backbone(seasonal) # unchanged API

    #     # Forecast from branches (seasonal forecast)
    #     branch_forecasts = [self.heads[i](z_backbone[i]) for i in range(self.n_branches)]
    #     stacked = torch.stack(branch_forecasts, dim=2)          # [B, N, T, N_br]
    #     weights = self.fusion_head(z_backbone)                  # [B, N, N_br]
    #     seasonal_pred = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, T]

    #     # --- TREND head (reuse ARResidual) ---
    #     pred_trend = self.ar_head(trend.transpose(1, 2)).transpose(1, 2)  # [B, T, N] -> back to [B, N, T]

    #     z_final = seasonal_pred + self.trend_gate * pred_trend  # fused forecast

    #     if self.revin:
    #         z_final_denorm = z_final.permute(0,2,1)
    #         z_final_denorm = self.revin_layer(z_final_denorm, 'denorm')
    #         z_final_denorm = z_final_denorm.permute(0,2,1)
    #         draw_list_denorm = [self.revin_layer(f.permute(0,2,1), 'denorm').permute(0,2,1) for f in branch_forecasts]
    #         return z_final_denorm, draw_list_denorm, xxx

    #     return z_final, branch_forecasts, xxx

    # original forward
    # def forward(self, z, temp):
    #     if self.revin:
    #         z = z.permute(0,2,1)
    #         z = self.revin_layer(z, 'norm')
    #         z = z.permute(0,2,1)

    #     z_backbone, xxx = self.backbone(z)

    #     fusion_weights = self.fusion_head(z_backbone)

    #     branch_forecasts = [self.heads[i](z_backbone[i]) for i in range(self.n_branches)]

    #     stacked_forecasts = torch.stack(branch_forecasts, dim=2)
    #     weighted_forecasts = stacked_forecasts * fusion_weights.unsqueeze(-1)
    #     z_final = weighted_forecasts.sum(dim=2)

    #     # + AR residual (operate in the same (RevIN-normalized) space as z_final)
    #     # pred_ar = self.ar_head(z.transpose(1, 2))          # z is your (optionally RevIN-normalized) input [B, in_len, n_vars]
    #     # z_final = z_final + pred_ar.transpose(1,2)


    #     if self.revin:
    #         z_final_denorm = z_final.permute(0,2,1)
    #         z_final_denorm = self.revin_layer(z_final_denorm, 'denorm')
    #         z_final_denorm = z_final_denorm.permute(0,2,1)
    #         draw_list_denorm = [self.revin_layer(f.permute(0,2,1), 'denorm').permute(0,2,1) for f in branch_forecasts]
    #         return z_final_denorm, draw_list_denorm, xxx

    #     return z_final, branch_forecasts, xxx

    # def create_pretrain_head(self, head_nf, vars, dropout):
    #     return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))

class ARResidual(nn.Module):
    """
    Channel-independent AR-like residual: a depthwise 1D conv over time.
    Expects x: [B, in_len, n_vars]  ->  returns [B, out_len, n_vars]
    """
    def __init__(self, in_len, out_len, n_vars, kernel=25):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        # depthwise conv: each variable gets its own linear filter
        self.proj = nn.Conv1d(n_vars, n_vars, kernel_size=kernel,
                              padding=kernel // 2, groups=n_vars, bias=False)

    def forward(self, x):                       # x: [B, in_len, n_vars]
        y = self.proj(x.transpose(1, 2))        # [B, n_vars, in_len]
        y = y[:, :, -self.out_len:]             # keep the last out_len steps
        return y.transpose(1, 2)                # [B, out_len, n_vars]


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

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            # print('x before ', x.shape)
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
            # print('x after ', x.shape)
        return x

class TSTiEncoder(nn.Module):
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
        self.n_vars = cfg.get('c_in', 7)
        self.head_dropout = cfg.get('head_dropout', 0)

        if padding_patch == 'end':
            patch_num_for_bottleneck = [p_n + 1 for p_n in patch_num]
        else:
            patch_num_for_bottleneck = patch_num
        self.head_nf = d_model * sum(patch_num_for_bottleneck)

        self.encoder = nn.ModuleList([
            nn.ModuleList([TSTEncoder(n_vars = c_in, learn_pe=learn_pe, pe=pe, q_len=patch_num[j], patch_len=patch_len[j], stride=stride[j], padding_patch=padding_patch,
                                      d_model=d_model, d_ff=d_ff, norm=norm, xxx_dropout=xxx_dropout, dropout=dropout,
                                      pre_norm=pre_norm, activation=act, cfg=cfg) for j in range(n_branches)])
            for i in range(n_layers)])

        self.bottle_neck = nn.Sequential(
            nn.Linear(self.head_nf, self.head_nf // 2),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_nf // 2, self.seq_len)
        )
        # self.aggregator = AttentionFusionHeadMLP(d_model, n_branches)

    def forward(self, x, temp = 0.1) -> Tensor:
        input_data = x
        for i in range(self.n_layers):
            output_ls = [self.encoder[i][j](input_data, temp) for j in range(self.n_branches)]

            if i == self.n_layers - 1:
                break

            # old aggregation

            flattened_outputs = [out.flatten(start_dim=2) for out in output_ls]
            fused_output = torch.cat(flattened_outputs, dim=-1) # [bs, nvars, total_features]

            bs, nvars, total_features = fused_output.shape
            
            fused_reshaped = fused_output.reshape(bs * nvars, total_features)
            
            reprojected = self.bottle_neck(fused_reshaped) # [bs * nvars, seq_len]
            
            input_data = reprojected.view(bs, nvars, self.seq_len)

            # new aggregation

            # input_data = self.aggregator(output_ls)

            # after:
            # input_data = (reprojected.view(bs, nvars, self.seq_len) + input_data) * 0.5


        return output_ls, []



class TSTEncoder(nn.Module):
    def __init__(self, n_vars, learn_pe, pe, q_len, patch_len, stride, padding_patch, d_model, d_ff=None,
                        norm='BatchNorm', xxx_dropout=0., dropout=0., activation='gelu', pre_norm=False, cfg=CN()):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            q_len += 1

        self.W_P = nn.Linear(patch_len + 2, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer = TSTEncoderLayer(q_len, d_model=d_model, patch_num=q_len, d_ff=d_ff, norm=norm,
                                     xxx_dropout=xxx_dropout, dropout=dropout, activation=activation,
                                     pre_norm=pre_norm, cfg=cfg)


        # self.calendar = CalendarFourier(
        #     seq_len=cfg.get('seq_len', 336),
        #     steps_per_day=cfg.get('steps_per_day', 144),
        #     steps_per_week=cfg.get('steps_per_week', 1008),
        #     per_variable_gate=True,
        #     n_vars=cfg.get('c_in', 21),
        #     init_gate=0.1,   # ~0.1 after sigmoid → almost off initially
        # )
        # Instantiate the VariableMixer
        # base = VariableMixer(d_model, n_vars, q_len)
        # self.variable_mixer = VariableMixerID(d_model, n_vars)
        # base = SafeVariableMixer(d_model, n_vars)
        # self.variable_mixer = VariableMixerAdaptive(d_model, n_vars)
        # In TSTEncoder.__init__(...)
        # Choose your safest mixer (identity init). Two good options you already have:
        # base = SafeVariableMixer(d_model, n_vars, dropout=0.1, alpha_init=0.0)
        # or: base = VariableMixerLR(d_model, n_vars=c_in, rank=16, dropout=0.1, alpha=0.05)

        # self.variable_mixer = CorrAwareMixer(base_mixer=base, n_vars=n_vars, tau=0.25, ema=0.9, max_gate=0.6)


    def forward(self, x:Tensor, temp = 0.1):
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # B, N, P, _ = x.shape
        x = x.reshape(x.size(0), x.size(1), -1, self.patch_len)
        # *** NEW: Calculate and append statistics ***
        patch_mean = x.mean(dim=-1, keepdim=True)
        patch_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-6)

        # cal = self.calendar.make_feats(
        #     P=P, B=B, N=N, patch_len=self.patch_len, stride=self.stride,
        #     device=x.device, dtype=x.dtype
        # )  # [B,N,P,4]

        # x = torch.cat([x, patch_mean, patch_std, cal], dim=-1)

        x = torch.cat([x, patch_mean, patch_std], dim=-1)
        x = self.W_P(x)
        # *** NEW: Apply Variable Mixer ***
        # x = self.variable_mixer(x)
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u + self.W_pos)
        output = self.layer(u, temp)
        output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
        return output


from layers.SMamba import *
class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, patch_num, d_ff=256,
                 norm='BatchNorm', xxx_dropout=0, dropout=0., bias=True, activation="gelu",
                 pre_norm=False, cfg=CN()):
        super().__init__()
        self.pre_norm = pre_norm
        # self.kan = RouterMixerLite(d_model, patch_num)
        self.kan = KanMixer(d_model, patch_num)
        # layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=16,
        #     dim_feedforward=d_ff,
        #     dropout=dropout,
        #     batch_first=True,          # expect [B, L, D]
        #     norm_first=True,
        #     activation=activation,
        # )
        # self.kan = nn.TransformerEncoder(layer, num_layers=1)
        # layer = SMambaEncoderLayer(d_model=d_model, dim_feedforward=1024, dropout=0.1,
        #                        norm_first=True, batch_first=False,
        #                        mamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2})
        # self.kan = SMambaEncoder(layer, num_layers=1, norm=nn.LayerNorm(d_model))
        self.dropout_xxx = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_xxx = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            # print('yo dadu')
            self.norm_xxx = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src:Tensor, temp = 0.1) -> Tensor:
        res = src
        if self.pre_norm:
            # print('aisi')
            src = self.norm_xxx(src)
        src = self.kan(src)
        src = res + self.dropout_xxx(src)
        if not self.pre_norm:
            # print('aiisi na')
            src = self.norm_xxx(src)

        res = src
        if self.pre_norm:
            src = self.norm_ffn(src)
        src = self.ff(src)
        src = res + self.dropout_ffn(src)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src

# --- END OF FILE MTST_backbone.py ---