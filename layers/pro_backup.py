# -*- coding: utf-8 -*-
# MTST_backbone_Pro: Big-swing variant for Patch-style forecasting
# Drop-in compatible with your shapes & training loop.

from typing import Optional, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, einsum

# Re-use your project utilities
from layers.PatchTST_layers import positional_encoding, Transpose, get_activation_fn
from layers.RevIN import RevIN
from yacs.config import CfgNode as CN

# ------------ Reuse your KAN mixer (already in your repo) ------------
# If KanMixer is defined elsewhere in your codebase, import it.
# Here’s a light wrapper that expects [B,L,D] (B=bs*nvars) and returns same.

from models.KANS.hahn import HahnPolynomials

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


class KanMixerAdapter(nn.Module):
    def __init__(self, d_model: int, patch_num: int, KanMixerCls=None):
        super().__init__()
        # assert KanMixerCls is not None, "Provide your KanMixer class to KanMixerAdapter"
        self.kan = KanMixer(d_model, patch_num)

    def forward(self, x: Tensor) -> Tensor:
        # your KanMixer already expects [B,L,D] per your code
        return self.kan(x)

# ---------------- Stronger SSM (bidirectional, multi-exponential) ----------------
class SSMMixerV2(nn.Module):
    """
    Multi-Exponential Diagonal SSM with content gating and low-rank mixing.
    Input/Output: [B, L, D]
    """
    def __init__(self, d_model: int, seq_len: int, K: int = 6, rank: int = 96, p_drop: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.K = K
        self.rank = min(rank, d_model)

        self.pre_u = nn.Linear(d_model, self.rank, bias=False)
        self.pre_v = nn.Linear(self.rank, d_model, bias=False)
        self.post_u = nn.Linear(d_model, self.rank, bias=False)
        self.post_v = nn.Linear(self.rank, d_model, bias=False)

        self.gate_in  = nn.Linear(d_model, d_model, bias=True)
        self.gate_out = nn.Linear(d_model, d_model, bias=True)

        self.a = nn.Parameter(torch.zeros(d_model, K))                        # tanh -> (-1,1)
        self.b = nn.Parameter(torch.randn(d_model, K) * 0.02)
        self.c = nn.Parameter(torch.randn(d_model, K) * 0.02)
        self.d = nn.Parameter(torch.ones(d_model))                            # skip

        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.norm_in  = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(p_drop)

        self.register_buffer("_arange_cache", torch.arange(0, seq_len).float(), persistent=False)

        # init for slow memory (helps long horizons)
        with torch.no_grad():
            self.a.copy_(torch.full_like(self.a, 0.9))

    def _build_kernel(self, L: int, device, dtype):
        if (self._arange_cache is None) or (self._arange_cache.numel() != L) or (self._arange_cache.device != device):
            self._arange_cache = torch.arange(0, L, device=device).float()
        k = self._arange_cache  # [L]

        a = torch.tanh(self.a).to(device=device, dtype=dtype)  # [D,K]
        b = self.b.to(device=device, dtype=dtype)
        c = self.c.to(device=device, dtype=dtype)

        eps = 1e-5
        mag = torch.clamp(torch.abs(a), min=eps)
        sign = torch.sign(a)
        logmag = torch.log(mag)
        a_pows = torch.exp(logmag.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0)) * (sign.unsqueeze(-1) ** k)
        h = (c.unsqueeze(-1) * a_pows) * b.unsqueeze(-1)  # [D,K,L]
        h = h.sum(dim=1)                                  # [D,L]
        h_rev = torch.flip(h, dims=[-1])                  # for causal conv
        return h_rev.unsqueeze(1)                         # [D,1,L]

    def _scan(self, xin: Tensor, weight: Tensor) -> Tensor:
        # xin: [B,L,D] -> conv expects [B,D,L]
        xin_c = xin.transpose(1, 2)  # [B,D,L]
        y = F.conv1d(xin_c, weight=weight, bias=None, stride=1, padding=xin.shape[1]-1, groups=self.d_model)
        y = y[:, :, :xin.shape[1]].transpose(1, 2)  # [B,L,D]
        return y

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        x0 = self.norm_in(x)
        x_m = self.pre_v(torch.relu(self.pre_u(x0)))
        gin = torch.sigmoid(self.gate_in(x0))
        xin = x_m * gin

        w_causal = self._build_kernel(L, x.device, x.dtype)         # [D,1,L]
        w_anti   = torch.flip(w_causal, dims=[-1])                   # anti-causal

        y_c = self._scan(xin, w_causal)
        y_a = self._scan(xin, w_anti)
        y_ssm = y_c + y_a

        y = y_ssm + x * self.d.view(1,1,D).to(x)
        y = self.post_v(torch.relu(self.post_u(y)))
        gout = torch.sigmoid(self.gate_out(y))
        y = y * gout
        y = self.norm_out(y)
        y = self.dropout(y)
        return x + self.res_scale * y

class RouterMixerLite(nn.Module):
    """
    Drop-in for HybridMixer in TSTEncoderLayer.
    Input/Output: [B, L, D] (B = bs*nvars). Prioritizes KAN; adds local-conv as assist.
    """
    def __init__(self, d_model: int, patch_num: int):
        super().__init__()
        self.kan = KanMixer(d_model, patch_num)  # your existing Hahn-based KAN :contentReference[oaicite:1]{index=1}
        self.local = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=5, padding=2, groups=d_model
        )
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # scalar logit per token
        )
        self.beta = nn.Parameter(torch.tensor(0.5))  # learnable scale on local path

    def forward(self, x):
        # x: [B,L,D]
        y_kan = self.kan(x)  # strong default
        y_loc = self.local(x.transpose(1, 2)).transpose(1, 2)

        # gate in [0,1], higher -> prefer KAN
        w = torch.sigmoid(self.router(x))  # [B,L,1]
        return w * y_kan + (1 - w) * (x + self.beta * y_loc)


# ---------------- Router: KAN ⟷ SSM per token ----------------
class RouterMixer(nn.Module):
    """
    Learns to mix KAN and SSM per token/channel. Drop-in for HybridMixer.
    Input/Output: [B, L, D] (B = bs*nvars).
    """
    def __init__(self, d_model: int, patch_num: int):
        super().__init__()
        self.kan = KanMixerAdapter(d_model, patch_num, KanMixerCls=KanMixer)
        self.ssm = SSMMixerV2(d_model, patch_num)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        y_kan = self.kan(x)     # [B,L,D]
        y_ssm = self.ssm(x)     # [B,L,D]
        logits = self.router(x) # [B,L,2]
        w = torch.softmax(logits, dim=-1)
        return w[...,0:1] * y_kan + w[...,1:2] * y_ssm

# ---------------- Variable Graph Mixer (sparse attention over variables) ----------------
class VariableGraphMixer(nn.Module):
    """
    Message passing across variables using attention with sparsification.
    Expects x: [bs, nvars, num_patch, d_model] and returns same shape.
    """
    def __init__(self, d_model: int, n_vars: int, num_patch: int, heads: int = 4, topk: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.topk = min(topk, n_vars)
        self.d_head = d_model // heads
        assert d_model % heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, V, P, D]
        B, V, P, D = x.shape
        q = self.q_proj(x)  # [B,V,P,D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # merge P into batch for attention over variables per-patch
        q = q.permute(0,2,1,3).reshape(B*P, V, D)
        k = k.permute(0,2,1,3).reshape(B*P, V, D)
        v = v.permute(0,2,1,3).reshape(B*P, V, D)

        # split heads
        q = q.reshape(B*P, V, self.heads, self.d_head).permute(0,2,1,3)  # [BP,H,V,dh]
        k = k.reshape(B*P, V, self.heads, self.d_head).permute(0,2,1,3)  # [BP,H,V,dh]
        v = v.reshape(B*P, V, self.heads, self.d_head).permute(0,2,1,3)  # [BP,H,V,dh]

        # attention logits: [BP,H,V,V]
        logits = einsum(q, k, 'b h i d, b h j d -> b h i j') / (self.d_head ** 0.5)

        # sparsify: keep top-k neighbors per query variable i
        topk = min(self.topk, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k=topk, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_idx, top_vals)
        attn = torch.softmax(mask, dim=-1)
        attn = self.drop(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')  # [BP,H,V,dh]
        out = out.permute(0,2,1,3).reshape(B*P, V, D)
        out = out.reshape(B, P, V, D).permute(0,2,1,3)        # [B,V,P,D]
        return self.norm(x + self.o_proj(out))

# ---------------- Encoder blocks ----------------
class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, patch_num, d_ff=256,
                 norm='BatchNorm', dropout=0., activation="gelu",
                 pre_norm=False, KanMixerCls=None, cfg=CN()):
        super().__init__()
        self.pre_norm = pre_norm
        # Big-swing: Router between KAN and SSM (per token)
        # assert KanMixerCls is not None, "Pass your KanMixer class into TSTEncoderLayer"
        self.mixer = RouterMixerLite(d_model, patch_num)

        self.dropout_xxx = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_xxx = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_xxx = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src: Tensor) -> Tensor:
        # src: [B,L,D] (B=bs*nvars)
        res = src
        if self.pre_norm:
            src = self.norm_xxx(src)
        src = self.mixer(src)
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

class TSTEncoder(nn.Module):
    """
    One branch (given patch config). Adds:
      - VariableGraphMixer across variables.
      - RouterMixer inside each layer (KAN ⟷ SSM).
      - Patch tokenization with mean/std stats (compatible with your code).
    """
    def __init__(self, n_vars, learn_pe, pe, q_len, patch_len, stride, padding_patch, d_model, d_ff=None,
                 norm='BatchNorm', dropout=0., activation='gelu', pre_norm=False,
                 KanMixerCls=None, cfg=CN()):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.n_vars = n_vars

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            q_len += 1

        # Project raw patch + mean/std -> d_model
        self.W_P = nn.Linear(patch_len + 2, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer = TSTEncoderLayer(
            q_len, d_model=d_model, patch_num=q_len, d_ff=d_ff, norm=norm,
            dropout=dropout, activation=activation, pre_norm=pre_norm,
            KanMixerCls=KanMixerCls, cfg=cfg
        )
        # Graph mixer across variables
        self.variable_mixer = VariableGraphMixer(d_model, n_vars, q_len, heads=4, topk=min(8, n_vars), dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, nvars, seq]
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [bs,nvars,num_patch,patch_len]
        x = x.reshape(x.size(0), x.size(1), -1, self.patch_len)
        # Append patch stats (mean/std) as in your code
        patch_mean = x.mean(dim=-1, keepdim=True)
        patch_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-6)
        x = torch.cat([x, patch_mean, patch_std], dim=-1)                   # [bs,nvars,num_patch,patch_len+2]
        x = self.W_P(x)                                                     # -> d_model

        # Variable graph mixer (across variables)
        # x = self.variable_mixer(x)                                          # [bs,nvars,num_patch,d_model]

        # Fold variables into batch, add positional encoding, and run layer
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u + self.W_pos)
        output = self.layer(u)                                              # [B*V,P,D]
        output = rearrange(output, '(b n) p d -> b n p d', n=n_vars)
        return output

class TSTiEncoder(nn.Module):
    """
    Multi-branch encoder (e.g., different patch sizes). After each layer except last,
    features are projected back to full sequence length (bottleneck) as in your code.
    """
    def __init__(self, learn_pe, pe, c_in, patch_num, patch_len, stride, max_seq_len=1024, padding_patch='end',
                 n_layers=1, n_branches=3, d_model=128, d_ff=256, norm='BatchNorm',
                 dropout=0., act="gelu", padding_var=None, pre_norm=False, KanMixerCls=None, cfg=CN()):
        super().__init__()
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.seq_len = cfg.get('seq_len', 336)
        self.n_vars = cfg.get('c_in', c_in)
        self.head_dropout = cfg.get('head_dropout', 0)

        if padding_patch == 'end':
            patch_num_for_bottleneck = [p_n + 1 for p_n in patch_num]
        else:
            patch_num_for_bottleneck = patch_num
        self.head_nf = d_model * sum(patch_num_for_bottleneck)

        self.encoder = nn.ModuleList([
            nn.ModuleList([
                TSTEncoder(
                    n_vars=c_in, learn_pe=learn_pe, pe=pe, q_len=patch_num[j],
                    patch_len=patch_len[j], stride=stride[j], padding_patch=padding_patch,
                    d_model=d_model, d_ff=d_ff, norm=norm, dropout=dropout,
                    activation=act, pre_norm=pre_norm, KanMixerCls=KanMixerCls, cfg=cfg
                ) for j in range(n_branches)
            ])
            for _ in range(n_layers)
        ])

        self.bottle_neck = nn.Sequential(
            nn.Linear(self.head_nf, self.head_nf // 2),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_nf // 2, self.seq_len)
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List]:
        input_data = x  # [bs,nvars,seq]
        for i in range(self.n_layers):
            output_ls = [self.encoder[i][j](input_data) for j in range(self.n_branches)]  # list of [bs,nvars,p,d]

            if i == self.n_layers - 1:
                break

            # Fuse for next layer input via bottleneck (same logic as your file)
            flattened = [out.flatten(start_dim=2) for out in output_ls]         # each [bs,nvars, p*d]
            fused = torch.cat(flattened, dim=-1)                                # [bs,nvars, total_features]
            bs, nvars, total = fused.shape
            fused_reshaped = fused.reshape(bs * nvars, total)
            reprojected = self.bottle_neck(fused_reshaped)                      # [bs*nvars, seq_len]
            input_data = reprojected.view(bs, nvars, self.seq_len)

        return output_ls, []

# ---------------- Fusion over branches (MoE-style router) ----------------
class RouterFusionHead(nn.Module):
    """
    Replaces single-query attention fusion with a small router MoE
    that scores each branch per variable using summary stats.
    """
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.n_branches = n_branches
        self.pool = lambda t: reduce(t, 'b v p d -> b v d', 'mean')  # mean over patches
        self.router = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_branches)
        )

    def forward(self, branch_outputs_list):
        # Each tensor: [bs, nvars, p, d]
        pooled = [reduce(b, 'b v p d -> b v d', 'mean') for b in branch_outputs_list]  # [bs,nvars,d] per branch

        # Dispersion over patches: std across dim=2 (the patch dimension)
        disp   = [torch.sqrt(torch.var(b, dim=2, unbiased=False) + 1e-6) for b in branch_outputs_list]  # [bs,nvars,d]

        # Simple summary (mean of branch-wise stats) -> routing features
        fused_feat = torch.stack(pooled, dim=2).mean(dim=2)   # [bs,nvars,d]
        disp_feat  = torch.stack(disp,   dim=2).mean(dim=2)   # [bs,nvars,d]
        router_in  = torch.cat([fused_feat, disp_feat], dim=-1)  # [bs,nvars,2d]

        logits  = self.router(router_in)       # [bs,nvars,n_branches]
        weights = torch.softmax(logits, dim=-1)
        return weights


# ---------------- Heads ----------------
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList([nn.Linear(nf, target_window) for _ in range(n_vars)])
            self.dropouts = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(n_vars)])
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(n_vars)])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.individual:
            outs = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                outs.append(z)
            return torch.stack(outs, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
            return x

class MDNResidualHead(nn.Module):
    """
    Predicts mixture of Gaussians over residuals: r = y_true - y_point.
    Returns corrected mean for point forecast at inference.
    """
    def __init__(self, n_vars: int, target_window: int, K: int = 5, d_hidden: int = 128):
        super().__init__()
        self.n_vars = n_vars
        self.tw = target_window
        self.K = K
        D = n_vars * target_window
        self.net = nn.Sequential(
            nn.Linear(D, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU()
        )
        self.pi = nn.Linear(d_hidden, D * K)     # mixture logits
        self.mu = nn.Linear(d_hidden, D * K)     # component means
        self.logsig = nn.Linear(d_hidden, D * K) # component log std

    def forward(self, residual: Tensor) -> Tuple[Tensor, dict]:
        # residual: [bs, nvars, tw]
        B = residual.size(0)
        flat = residual.reshape(B, -1)
        h = self.net(flat)
        pi = self.pi(h).reshape(B, self.n_vars, self.tw, self.K)
        mu = self.mu(h).reshape(B, self.n_vars, self.tw, self.K)
        logsig = self.logsig(h).reshape(B, self.n_vars, self.tw, self.K).clamp(-7, 3)

        w = torch.softmax(pi, dim=-1)                     # [B,V,T,K]
        mdn_mean = (w * mu).sum(dim=-1)                   # [B,V,T]
        out = {
            "pi": pi, "mu": mu, "logsig": logsig, "w": w,
        }
        return mdn_mean, out

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




# ---------------- Main Backbone ----------------
class MTST_backbone_Pro(nn.Module):
    """
    Big-swing MTST backbone:
      - RevIN (as in your code)
      - Multi-branch Patch encoders with RouterMixer (KAN ⟷ SSM) and VariableGraphMixer
      - MoE-style RouterFusionHead
      - Optional MDN residual head on top of point forecast

    Keeps your call signature and output shapes compatible.
    """
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len, stride, max_seq_len:Optional[int]=1024,
                 n_layers:int=1, n_branches:int=3, d_model=128,
                 d_ff:int=256, norm:str='BatchNorm', xxx_dropout:float=0.05, dropout:float=0.05, act:str="gelu",
                 padding_var:Optional[int]=None,  pre_norm:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout:float = 0.05, padding_patch = None,
                 pretrain_head:bool=False, head_type:str = 'flatten', individual:bool = False, revin:bool = True,
                 affine:bool = True, subtract_last:bool = False,
                 cfg:CN = CN(),
                 KanMixerCls=None,          # pass your existing KanMixer class here
                 use_mdn:bool = False,      # turn on MDN residual head
                 point_only:bool = True,    # if True, ignore MDN outputs
                 **kwargs):
        super().__init__()

        # assert KanMixerCls is not None, "Please pass your KanMixer class (from your repo) as KanMixerCls"
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        if isinstance(patch_len, str): patch_len = [int(i) for i in patch_len.split(',')]
        if isinstance(stride, str):    stride    = [int(i) for i in stride.split(',')]
        patch_num = [int((context_window - p_len) / s + 1) for p_len, s in zip(patch_len, stride)]

        self.backbone = TSTiEncoder(
            learn_pe, pe, c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
            padding_patch=padding_patch, n_layers=n_layers, n_branches=n_branches, d_model=d_model, d_ff=d_ff,
            norm=norm, dropout=dropout, act=act, padding_var=padding_var, pre_norm=pre_norm, KanMixerCls=KanMixerCls, cfg=cfg
        )

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

        # MoE-style fusion across branches
        # self.fusion_head = AttentionFusionHead(d_model, n_branches)
        self.fusion_head = RouterFusionHead(d_model, n_branches)

        # Forecast heads (per branch)
        if self.pretrain_head:
            self.head = nn.Sequential(nn.Dropout(fc_dropout), nn.Conv1d(sum(self.head_nf), c_in, 1))
        elif head_type == 'flatten':
            self.heads = nn.ModuleList([
                Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout)
                for i in range(self.n_branches)
            ])

        # Optional MDN residual head (on top of fused point forecast)
        self.use_mdn = use_mdn
        self.point_only = point_only
        if self.use_mdn:
            self.mdn = MDNResidualHead(self.n_vars, target_window, K=5, d_hidden=max(128, d_model))

    def forward(self, z: Tensor):
        # z: [bs, seq, nvars] or [bs, nvars, seq]? Your repo uses [bs, seq, nvars] -> it permutes before RevIN. :contentReference[oaicite:3]{index=3}
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # Backbone encoders
        z_backbone, _ = self.backbone(z)                              # list of [bs,nvars,p,d]

        # Per-branch point forecasts
        branch_forecasts = [self.heads[i](z_backbone[i]) for i in range(self.n_branches)]  # each [bs,nvars,tw]
        stacked_forecasts = torch.stack(branch_forecasts, dim=2)                             # [bs,nvars,B,tw]

        # MoE fusion over branches
        fusion_weights = self.fusion_head(z_backbone)                 # [bs,nvars,B]
        z_point = (stacked_forecasts * fusion_weights.unsqueeze(-1)).sum(dim=2)  # [bs,nvars,tw]

        outputs = {"point": z_point}

        # Optional MDN residual correction (train with NLL or CRPS-like if desired)
        if self.use_mdn:
            # During training, you’d pass y_true to compute residuals; here we expose the head.
            # For inference without y_true, we can return the MDN parameters given residual=0.
            residual_input = torch.zeros_like(z_point)
            mdn_mean, mdn_info = self.mdn(residual_input)
            z_point_corr = z_point + mdn_mean
            outputs.update({"mdn_mean": mdn_mean, "point_corrected": z_point_corr, "mdn": mdn_info})

        # Denormalize via RevIN
        if self.revin:
            point_denorm = self.revin_layer(z_point.permute(0,2,1), 'denorm').permute(0,2,1)
            draw_list_denorm = [self.revin_layer(f.permute(0,2,1), 'denorm').permute(0,2,1) for f in branch_forecasts]

            if self.use_mdn:
                point_corr_denorm = self.revin_layer(outputs["point_corrected"].permute(0,2,1), 'denorm').permute(0,2,1)
                outputs["point_denorm"] = point_denorm
                outputs["point_corrected_denorm"] = point_corr_denorm
                return outputs, draw_list_denorm, []
            else:
                return point_denorm, draw_list_denorm, []

        # If RevIN disabled
        if self.use_mdn:
            return outputs, branch_forecasts, []
        else:
            return z_point, branch_forecasts, []
