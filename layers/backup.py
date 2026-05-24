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
# from models.KANS.cheby import ChebyshevPolynomials as LucasPolynomials

# class ImprKANMixer(nn.Module):
#     """Improved KAN-based mixer for intra/inter patch mixing.

#     - Applies intra->inter and inter->intra paths (like before)
#     - Normalizes each path and applies learnable scaling
#     - Optional dropout and residual scaling to stabilize training
#     """
#     def __init__(self, dim: int, length: int, poly_order: int = 3, dropout: float = 0.0):
#         super().__init__()
#         # KAN modules (Hahn polynomials)
#         self.intrapatch_kan = HahnPolynomials(dim, dim, poly_order, 1, 1, 7)
#         self.interpatch_kan = HahnPolynomials(length, length, poly_order, 1, 1, 7)

#         # Normalizations to stabilize outputs
#         self.norm_intra = nn.LayerNorm(dim)
#         self.norm_inter = nn.LayerNorm(dim)

#         # Learnable scales to correct magnitude mismatch
#         self.scale_ab = nn.Parameter(torch.tensor(1.0))
#         self.scale_ba = nn.Parameter(torch.tensor(1.0))

#         # Gating across channels (per-dim) is often more expressive than a single scalar
#         self.gating_layer = nn.Linear(dim, 2)

#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#         # final layer norm to keep residual stable
#         self.out_norm = nn.LayerNorm(dim)

#     def forward(self, x: torch.Tensor):
#         # x: [batch, patch_num, dim]
#         # path A: intra -> inter
#         ab = self.intrapatch_kan(x)               # [b, p, d]
#         ab = ab.permute(0, 2, 1)                  # [b, d, p]
#         ab = self.interpatch_kan(ab)              # [b, d, p]
#         ab = ab.permute(0, 2, 1)                  # [b, p, d]
#         ab = self.norm_intra(ab)
#         ab = self.scale_ab * ab

#         # path B: inter -> intra
#         ba = x.permute(0, 2, 1)                   # [b, d, p]
#         ba = self.interpatch_kan(ba)              # [b, d, p]
#         ba = ba.permute(0, 2, 1)                  # [b, p, d]
#         ba = self.intrapatch_kan(ba)              # [b, p, d]
#         ba = self.norm_inter(ba)
#         ba = self.scale_ba * ba

#         # gating: use the mean over positions to compute per-channel gates
#         gating_input = reduce(x, 'b p d -> b d', 'mean')  # [b, d]
#         gates = F.softmax(self.gating_layer(gating_input), dim=-1)  # [b, 2]

#         g_ab = gates[:, 0].unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1]
#         g_ba = gates[:, 1].unsqueeze(-1).unsqueeze(-1)

#         out = x + self.dropout(g_ab * ab + g_ba * ba)
#         out = self.out_norm(out)
#         return out


# class ConvMixer(nn.Module):
#     """Depthwise separable Conv1d mixer (backwards compatible).

#     Input: [bs * nvars, patch_num, d_model]
#     """
#     def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.0):
#         super().__init__()
#         # Depthwise conv across patch positions (channels independent)
#         self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
#                               kernel_size=kernel_size, padding=(kernel_size // 2), groups=d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x: torch.Tensor):
#         # x shape: [bs*nvars, patch_num, d_model]
#         x = x.permute(0, 2, 1)  # -> [bs*nvars, d_model, patch_num]
#         x = self.conv(x)
#         x = x.permute(0, 2, 1)  # -> [bs*nvars, patch_num, d_model]
#         x = self.dropout(self.norm(x))
#         return x


# class HybridMixer(nn.Module):
#     """Hybrid mixer that fuses ImprKANMixer and ConvMixer.

#     Features:
#     - Per-channel gating (learned) that blends kan_out and conv_out
#     - Learnable scales for each path
#     - Option to use residual correction form (kan adjusts conv)
#     - Utilities to expose gate statistics for analysis
#     """
#     def __init__(self, d_model: int, patch_num: int, use_per_channel_gate: bool = True,
#                  poly_order: int = 3, kan_dropout: float = 0.0, conv_kernel: int = 3,
#                  init_gate_bias: float = -2.0, residual_kan: bool = True):
#         super().__init__()
#         self.kan_mixer = ImprKANMixer(d_model, patch_num, poly_order, dropout=kan_dropout)
#         self.conv_mixer = ConvMixer(d_model, kernel_size=conv_kernel)

#         # gating parameter: either scalar per-dim or scalar per-layer
#         if use_per_channel_gate:
#             # gate shape: [1, 1, d_model] -> broadcast over batch & positions
#             self.gate = nn.Parameter(torch.zeros(1, 1, d_model))
#         else:
#             # single scalar gate
#             self.gate = nn.Parameter(torch.zeros(1))

#         # initialize gate towards conv (sigmoid(gate) -> ~0 when negative)
#         nn.init.constant_(self.gate, init_gate_bias)

#         # learnable scales for paths
#         self.scale_kan = nn.Parameter(torch.ones(1, 1, d_model)) if use_per_channel_gate else nn.Parameter(torch.ones(1))
#         self.scale_conv = nn.Parameter(torch.ones(1, 1, d_model)) if use_per_channel_gate else nn.Parameter(torch.ones(1))

#         # final normalization and residual
#         self.out_norm = nn.LayerNorm(d_model)
#         self.residual_kan = residual_kan

#     def forward(self, x: torch.Tensor):
#         # Expect x: [bs*nvars, patch_num, d_model]
#         kan_out = self.kan_mixer(x)
#         conv_out = self.conv_mixer(x)

#         gate = torch.sigmoid(self.gate)  # either [1,1,d] or [1]

#         # apply scales
#         kan_scaled = kan_out * self.scale_kan
#         conv_scaled = conv_out * self.scale_conv

#         if self.residual_kan:
#             # Conv as base, KAN provides correction
#             fused = conv_scaled + gate * (kan_scaled - conv_scaled)
#         else:
#             fused = gate * kan_scaled + (1 - gate) * conv_scaled

#         # add residual connection to preserve input information
#         out = self.out_norm(fused + x)
#         return out

class SSMMixer(nn.Module):
    """
    Diagonal State-Space Mixer (causal), drop-in replacement for HybridMixer.

    Input:  x [B, L, D]  (B = bs*nvars, L = patch_num, D = d_model)
    Output: y [B, L, D]

    Model:
      y_t = a ⊙ y_{t-1} + b ⊙ x_t
      z_t = c ⊙ y_t + d ⊙ x_t
    Implemented as a depthwise causal convolution with kernel
      h[k] = c ⊙ (a^k) ⊙ b  for k = 0..L-1
    plus a skip (d ⊙ x). A small gated 1×1 mixing is added before/after.
    """
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # patch_num

        # Channel-mixing before/after the SSM (lightweight)
        self.in_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Gating (GLU-style) to modulate inputs
        self.gate_in  = nn.Linear(d_model, d_model)
        self.gate_out = nn.Linear(d_model, d_model)

        # SSM parameters (per feature)
        # a is constrained to (-1, 1) via tanh for stability
        self.a = nn.Parameter(torch.zeros(d_model))
        self.b = nn.Parameter(torch.randn(d_model) * 0.02)
        self.c = nn.Parameter(torch.randn(d_model) * 0.02)
        self.d = nn.Parameter(torch.ones(d_model))  # skip

        # Optional learned dropout on the SSM output (kept small)
        self.dropout = nn.Dropout(0.05)

        # Buffer to hold the reversed causal kernel for conv1d (recomputed per forward length if needed)
        self.register_buffer("_arange_cache", torch.arange(0, seq_len).float(), persistent=False)

    def _build_kernel(self, L: int, device, dtype):
        """
        Build per-channel causal kernel h of shape [D, L],
        then reverse it for conv1d (so padding=L-1 is causal).
        """
        # Ensure we have an arange of the right length on the right device/dtype
        if (self._arange_cache is None) or (self._arange_cache.numel() != L) or (self._arange_cache.device != device):
            self._arange_cache = torch.arange(0, L, device=device).float()

        k = self._arange_cache  # [L]
        a = torch.tanh(self.a).to(device=device, dtype=dtype)     # [D], stable |a|<1
        b = self.b.to(device=device, dtype=dtype)                 # [D]
        c = self.c.to(device=device, dtype=dtype)                 # [D]

        # a^k for all k (broadcast: [D, L])
        a_pows = a.unsqueeze(-1).pow(k.unsqueeze(0))              # [D, L]
        h = (c.unsqueeze(-1) * a_pows) * b.unsqueeze(-1)          # [D, L]

        # Reverse for conv1d (so earliest lag is at the end)
        h_rev = torch.flip(h, dims=[-1])                          # [D, L]
        # Convert to conv1d depthwise weights: [D(out), 1, L], groups=D
        return h_rev.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        returns same shape
        """
        B, L, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Input gating + mixing
        gate = torch.sigmoid(self.gate_in(x))     # [B, L, D]
        xin  = self.in_proj(x) * gate             # [B, L, D]

        # Depthwise causal convolution implementing the SSM
        # Prepare kernel and run conv1d per-channel
        weight = self._build_kernel(L, device=x.device, dtype=x.dtype)  # [D, 1, L]

        # x for conv1d: [B, D, L]
        xin_c = xin.transpose(1, 2)                                   # [B, D, L]
        y_ssm = F.conv1d(xin_c, weight=weight, bias=None,
                         stride=1, padding=L-1, groups=D)             # [B, D, L + L - 1]
        y_ssm = y_ssm[:, :, :L]                                       # causal trim -> [B, D, L]
        y_ssm = y_ssm.transpose(1, 2)                                  # [B, L, D]

        # Skip connection from x via parameter d
        y = y_ssm + x * self.d.unsqueeze(0).unsqueeze(0).to(x.dtype).to(x.device)

        # Output gating + mixing
        gate_out = torch.sigmoid(self.gate_out(y))
        y = self.out_proj(y) * gate_out

        # Mild dropout for regularization
        y = self.dropout(y)
        return y


class SSMMixerV2(nn.Module):
    """
    Multi-Exponential Diagonal SSM with content gating and low-rank mixing.
    Drop-in replacement for HybridMixer. Input/Output: [B, L, D].

    Kernel per channel = sum_{k=1..K} c_k * (a_k^n) * b_k,  n=0..L-1,  |a_k|<1 for stability.
    Implemented as depthwise conv1d with the causal kernel, plus gated skip and low-rank mixes.
    """
    def __init__(self, d_model: int, seq_len: int, K: int = 4, rank: int = 64, p_drop: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.K = K
        self.rank = min(rank, d_model)

        # Low-rank channel mixing (before/after scan)
        self.pre_u = nn.Linear(d_model, self.rank, bias=False)
        self.pre_v = nn.Linear(self.rank, d_model, bias=False)
        self.post_u = nn.Linear(d_model, self.rank, bias=False)
        self.post_v = nn.Linear(self.rank, d_model, bias=False)

        # Gating (content-dependent)
        self.gate_in  = nn.Linear(d_model, d_model, bias=True)
        self.gate_out = nn.Linear(d_model, d_model, bias=True)

        # Diagonal SSM parameters: K exponentials per channel
        # Parameterize a_k via tanh to keep |a_k|<1; b_k, c_k free (small init)
        self.a = nn.Parameter(torch.zeros(d_model, K))                        # [-1,1] after tanh
        self.b = nn.Parameter(torch.randn(d_model, K) * 0.02)
        self.c = nn.Parameter(torch.randn(d_model, K) * 0.02)

        # Per-channel skip
        self.d = nn.Parameter(torch.ones(d_model))

        # Residual scaling
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        # Normalization & dropout
        self.norm_in  = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(p_drop)

        # Cache arange for kernel build
        self.register_buffer("_arange_cache", torch.arange(0, seq_len).float(), persistent=False)

    def _build_kernel(self, L: int, device, dtype):
        # Ensure k = [0..L-1] on device
        if (self._arange_cache is None) or (self._arange_cache.numel() != L) or (self._arange_cache.device != device):
            self._arange_cache = torch.arange(0, L, device=device).float()
        k = self._arange_cache  # [L]

        # Shapes: a,b,c: [D,K]
        a = torch.tanh(self.a).to(device=device, dtype=dtype)        # stable
        b = self.b.to(device=device, dtype=dtype)
        c = self.c.to(device=device, dtype=dtype)

        # a^k: [D,K,L]
        # Use exp(log(|a|+eps)*k) with sign to be numerically stable near 0
        eps = 1e-5
        mag = torch.clamp(torch.abs(a), min=eps)                      # [D,K]
        sign = torch.sign(a)                                          # [-1 or 1], [D,K]
        logmag = torch.log(mag)                                       # [D,K]
        # broadcast: [D,K] x [L] -> [D,K,L]
        a_pows = torch.exp(logmag.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0)) * (sign.unsqueeze(-1) ** k)

        # h = sum_k c_k * (a_k^n) * b_k  over k
        # (D,K,L) * (D,K,1) * (D,K,1) -> (D,K,L) then sum K -> (D,L)
        h = (c.unsqueeze(-1) * a_pows) * b.unsqueeze(-1)
        h = h.sum(dim=1)                                              # [D,L]

        # Reverse for causal conv
        h_rev = torch.flip(h, dims=[-1])                              # [D,L]
        return h_rev.unsqueeze(1)                                     # [D,1,L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        assert D == self.d_model

        # Pre-norm and low-rank mix
        x0 = self.norm_in(x)
        x_m = self.pre_v(torch.relu(self.pre_u(x0)))                  # [B,L,D]

        # Content gating on input (token-wise GLU-ish)
        gin = torch.sigmoid(self.gate_in(x0))
        xin = x_m * gin                                               # [B,L,D]

        # Build kernel for this L and convolve depthwise
        weight = self._build_kernel(L, device=x.device, dtype=x.dtype)    # [D,1,L]
        xin_c = xin.transpose(1, 2)                                       # [B,D,L]
        y_ssm = F.conv1d(xin_c, weight=weight, bias=None,
                         stride=1, padding=L-1, groups=D)                 # [B,D,2L-1]
        y_ssm = y_ssm[:, :, :L].transpose(1, 2)                           # [B,L,D]

        # Skip path
        y = y_ssm + x * self.d.view(1, 1, D).to(x)

        # Post low-rank mix and output gating
        y = self.post_v(torch.relu(self.post_u(y)))
        gout = torch.sigmoid(self.gate_out(y))
        y = y * gout

        # Post-norm, dropout, and residual scale
        y = self.norm_out(y)
        y = self.dropout(y)

        # Residual connection
        return x + self.res_scale * y


class KanMixer(nn.Module):
    """
    IMPROVEMENT 1: KanMixer with Adaptive Gating.
    """
    def __init__(self, dim, len):
        super().__init__()
        # self.intrapatch_kan = LucasPolynomials(dim,dim ,3)
        # self.interpatch_kan = LucasPolynomials(len,len, 3)
        self.intrapatch_kan = HahnPolynomials(dim, dim, 3, 1, 1, 7)
        self.interpatch_kan = HahnPolynomials(len, len, 3, 1, 1, 7)
        self.gating_layer = nn.Linear(dim, 2)

    def forward(self, x):
        ab = self.intrapatch_kan(x)
        ab = ab.permute(0, 2, 1)
        ab = self.interpatch_kan(ab)
        ab = ab.permute(0, 2, 1)

        ba = x.permute(0, 2, 1)
        ba = self.interpatch_kan(ba)
        ba = ba.permute(0, 2, 1)
        ba = self.intrapatch_kan(ba)

        gating_input = reduce(x, 'b l d -> b d', 'mean')
        gates = F.softmax(self.gating_layer(gating_input), dim=-1)
        g_ab = gates[:, 0].unsqueeze(-1).unsqueeze(-1)
        g_ba = gates[:, 1].unsqueeze(-1).unsqueeze(-1)
        output = x + g_ab * ab + g_ba * ba

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

# --- Replace KanMixer with this HybridMixer in TSTEncoderLayer ---
class HybridMixer(nn.Module):
    def __init__(self, d_model, patch_num):
        super().__init__()
        self.kan_mixer = KanMixer(d_model, patch_num)
        self.conv_mixer = ConvMixer(d_model)
        # A simple learnable gate to combine the two mixer outputs
        self.gate = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        kan_out = self.kan_mixer(x)
        return kan_out
        conv_out = self.conv_mixer(x)
        return conv_out
        # Use a sigmoid gate to blend the two outputs
        g = torch.sigmoid(self.gate)
        return g * kan_out + (1 - g) * conv_out

class AttentionFusionHead(nn.Module):
    """
    IMPROVEMENT 2B: Attention module to fuse multi-scale branch outputs.
    """
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.n_branches = n_branches
        # *** EINSUM FIX ***: Define the query as a simple vector of shape [d_model].
        # The previous shape [1, 1, d_model] required unsupported '()' notation in einsum.
        self.attention_query = nn.Parameter(torch.randn(d_model))

    def forward(self, branch_outputs_list):
        # branch_outputs_list: A list of tensors, each of shape [bs, nvars, patch_num, d_model]
        pooled_outputs = [reduce(branch, 'b v p d -> b v d', 'mean') for branch in branch_outputs_list]
        stacked_pools = torch.stack(pooled_outputs, dim=2) # [bs, nvars, n_branches, d_model]

        # *** EINSUM FIX ***: Use a supported einsum pattern.
        # 'd, b v n d -> b v n' performs a dot product between the query (d) and
        # the last dimension of the stacked pools (d), resulting in the desired attention scores.
        attn_scores = einsum(self.attention_query, stacked_pools, 'd, b v n d -> b v n')
        attn_weights = F.softmax(attn_scores, dim=-1) # [bs, nvars, n_branches]
        return attn_weights


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

        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        if isinstance(patch_len, str):
            patch_len= patch_len.split(',')
            patch_len= [int(i) for i in patch_len]
        if isinstance(stride, str):
            stride = stride.split(',')
            stride = [int(i) for i in stride]

        patch_num = [int((context_window - p_len) / s + 1) for p_len, s in zip(patch_len, stride)]
        
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

        self.fusion_head = AttentionFusionHead(d_model, n_branches)

        if self.pretrain_head:
            self.head = self.create_pretrain_head(sum(self.head_nf), c_in, fc_dropout)
        elif head_type == 'flatten':
            self.heads = nn.ModuleList()
            for i in range(self.n_branches):
                self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))


    def forward(self, z):
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        z_backbone, xxx = self.backbone(z)

        fusion_weights = self.fusion_head(z_backbone)

        branch_forecasts = [self.heads[i](z_backbone[i]) for i in range(self.n_branches)]

        stacked_forecasts = torch.stack(branch_forecasts, dim=2)
        weighted_forecasts = stacked_forecasts * fusion_weights.unsqueeze(-1)
        z_final = weighted_forecasts.sum(dim=2)

        if self.revin:
            z_final_denorm = z_final.permute(0,2,1)
            z_final_denorm = self.revin_layer(z_final_denorm, 'denorm')
            z_final_denorm = z_final_denorm.permute(0,2,1)
            draw_list_denorm = [self.revin_layer(f.permute(0,2,1), 'denorm').permute(0,2,1) for f in branch_forecasts]
            return z_final_denorm, draw_list_denorm, xxx

        return z_final, branch_forecasts, xxx

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


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
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
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

    def forward(self, x) -> Tensor:
        input_data = x
        for i in range(self.n_layers):
            output_ls = [self.encoder[i][j](input_data) for j in range(self.n_branches)]

            if i == self.n_layers - 1:
                break

            flattened_outputs = [out.flatten(start_dim=2) for out in output_ls]
            fused_output = torch.cat(flattened_outputs, dim=-1) # [bs, nvars, total_features]

            bs, nvars, total_features = fused_output.shape
            
            fused_reshaped = fused_output.reshape(bs * nvars, total_features)
            
            reprojected = self.bottle_neck(fused_reshaped) # [bs * nvars, seq_len]
            
            input_data = reprojected.view(bs, nvars, self.seq_len)

        return output_ls, []


# Cell

class VariableMixer(nn.Module):
    def __init__(self, d_model, n_vars, num_patch):
        super().__init__()
        # Use a KAN layer to mix information across the variable dimension
        self.var_kan_mixer = HahnPolynomials(n_vars, n_vars, 3, 1, 1, 7)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [bs, nvars, num_patch, d_model]
        
        # Permute to bring variables to the mixing dimension: [bs, num_patch, d_model, n_vars]
        x_permuted = x.permute(0, 2, 3, 1)
        
        # Apply KAN mixing across the n_vars dimension
        x_mixed = self.var_kan_mixer(x_permuted)
        
        # Permute back to original layout: [bs, nvars, num_patch, d_model]
        x_out = x_mixed.permute(0, 3, 1, 2)
        
        # Add residual connection and norm
        return self.norm(x + x_out)

class SSMMixerV2_Bidir(SSMMixerV2):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert D == self.d_model
        x0 = self.norm_in(x)
        x_m = self.pre_v(torch.relu(self.pre_u(x0)))
        gin = torch.sigmoid(self.gate_in(x0))
        xin = x_m * gin

        # Build causal kernel
        w_causal = self._build_kernel(L, device=x.device, dtype=x.dtype)   # [D,1,L]

        # Build anti-causal kernel by NOT reversing (conv will reverse it once)
        # -> easiest: flip causal back
        w_anti = torch.flip(w_causal, dims=[-1])                           # [D,1,L]

        xin_c = xin.transpose(1, 2)                                        # [B,D,L]
        y_c = F.conv1d(xin_c, weight=w_causal, stride=1, padding=L-1, groups=D)[:, :, :L]
        y_a = F.conv1d(xin_c, weight=w_anti,  stride=1, padding=L-1, groups=D)[:, :, -L:]
        y_ssm = (y_c + y_a).transpose(1, 2)                                # [B,L,D]

        y = y_ssm + x * self.d.view(1,1,D).to(x)
        y = self.post_v(torch.relu(self.post_u(y)))
        gout = torch.sigmoid(self.gate_out(y))
        y = y * gout
        y = self.norm_out(y)
        y = self.dropout(y)
        return x + self.res_scale * y


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
        # Instantiate the VariableMixer
        self.variable_mixer = VariableMixer(d_model, n_vars, q_len)

    def forward(self, x:Tensor):
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.size(0), x.size(1), -1, self.patch_len)
        # *** NEW: Calculate and append statistics ***
        patch_mean = x.mean(dim=-1, keepdim=True)
        patch_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-6)
        x = torch.cat([x, patch_mean, patch_std], dim=-1)
        x = self.W_P(x)
        # *** NEW: Apply Variable Mixer ***
        # x = self.variable_mixer(x)
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u + self.W_pos)
        output = self.layer(u)
        output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
        return output


class SSMMixerV2_ParallelConv(SSMMixerV2):
    def __init__(self, d_model, seq_len, K=4, rank=64, p_drop=0.05, kernel_size=5, dilation=2):
        super().__init__(d_model, seq_len, K, rank, p_drop)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                               padding=(kernel_size//2)*dilation, dilation=dilation, groups=d_model)
    def forward(self, x):
        y = super().forward(x)                     # [B,L,D]
        yl = self.local(x.transpose(1,2)).transpose(1,2)
        return y + 0.5 * yl                        # learnable scale optional


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, patch_num, d_ff=256,
                 norm='BatchNorm', xxx_dropout=0, dropout=0., bias=True, activation="gelu",
                 pre_norm=False, cfg=CN()):
        super().__init__()
        self.pre_norm = pre_norm
        self.kan = SSMMixerV2_ParallelConv(d_model, patch_num)

        # self.kan = HybridMixer(d_model, patch_num)
        self.dropout_xxx = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_xxx = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
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

    def forward(self, src:Tensor) -> Tensor:
        res = src
        if self.pre_norm:
            src = self.norm_xxx(src)
        src = self.kan(src)
        src = res + self.dropout_xxx(src)
        if not self.pre_norm:
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