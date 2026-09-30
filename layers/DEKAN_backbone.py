__all__ = ['DEKAN_backbone']

import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, reduce

from layers.PatchTST_layers import Transpose, get_activation_fn, positional_encoding
from layers.RevIN import RevIN
from layers.KAN import build_kan
from layers.SMamba import SMambaEncoderLayer, SMambaEncoder


class DEKAN_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride, n_layers=1, n_branches=3,
                 d_model=128, d_ff=256, dropout=0., head_dropout=0., padding_patch='end',
                 revin=True, affine=False, subtract_last=False, decomp_mode='full',
                 kernel_sizes=(25, 73, 169), periods=(24, 168), harmonics=2, **kwargs):
        super().__init__()

        # RevIN
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # decomp_mode: full | none | residual_only | seasonal_only | trend_only
        self.decomp_mode = decomp_mode
        self.use_residual = decomp_mode not in ('seasonal_only', 'trend_only')
        self.use_trend = decomp_mode in ('full', 'trend_only')
        self.use_seasonal = decomp_mode in ('full', 'seasonal_only')
        self.n_branches = n_branches
        self.context_window = context_window
        self.target_window = target_window

        patch_num = [int((context_window - p) / s + 1) for p, s in zip(patch_len, stride)]

        # Residual branches
        if self.use_residual:
            self.backbone = Encoder(c_in, context_window, patch_num, patch_len, stride, n_layers=n_layers,
                                    n_branches=n_branches, d_model=d_model, d_ff=d_ff, dropout=dropout,
                                    head_dropout=head_dropout, padding_patch=padding_patch, **kwargs)
        if padding_patch == 'end':
            patch_num = [p + 1 for p in patch_num]
        self.head_nf = [d_model * p for p in patch_num]

        if self.use_residual:
            self.fusion_head = BranchFusion(d_model, n_branches, hidden=d_model)

        # Decomposition
        if decomp_mode != 'none':
            self.decomp = SeriesDecomp(context_window, kernel_sizes, periods, harmonics, decomp_mode)

        # Trend head
        if self.use_trend:
            if target_window > context_window:
                self.time_proj_trend = TimeProjector(context_window, target_window, rank=64)
            self.alpha_t = nn.Parameter(torch.tensor(0.0))

        # Seasonal head
        if self.use_seasonal:
            self.seasonal_head = SeasonalHead(self.decomp, target_window)
            self.alpha_s = nn.Parameter(torch.tensor(0.0))

        if self.use_residual:
            self.heads = nn.ModuleList([Flatten_Head(self.head_nf[i], target_window, head_dropout=head_dropout)
                                        for i in range(n_branches)])

        if self.use_trend and target_window <= context_window:
            self.ar_head = ARHead(context_window, target_window, c_in, kernel=25)

    def forward(self, z):                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # decomposition
        if self.decomp_mode != 'none':
            res, trend, seasonal = self.decomp(z)
        else:
            res, trend, seasonal = z, None, None

        y = None
        if self.use_residual:
            outputs = self.backbone(res)                                    # n_branches x [bs x nvars x patch_num x d_model]
            preds = torch.stack([self.heads[i](outputs[i]) for i in range(self.n_branches)], dim=2)
            w = torch.softmax(self.fusion_head(outputs), dim=-1)            # [bs x nvars x n_branches]
            y = (preds * w.unsqueeze(-1)).sum(dim=2)                        # [bs x nvars x target_window]
        res_pred = y

        if self.use_trend:
            if self.target_window <= self.context_window:
                trend_pred = self.ar_head(trend.transpose(1, 2)).transpose(1, 2)
            else:
                trend_pred = self.time_proj_trend(trend)
            term = torch.sigmoid(self.alpha_t) * trend_pred
            y = term if y is None else y + term

        if self.use_seasonal:
            seasonal_pred = self.seasonal_head(seasonal)
            term = torch.sigmoid(self.alpha_s) * seasonal_pred
            y = term if y is None else y + term

        # per-component forecasts, read by the a1 loss
        aux = {}
        if self.decomp_mode == 'full':
            aux = {'trend': trend_pred, 'seasonal': seasonal_pred, 'residual': res_pred,
                   'alpha_t': torch.sigmoid(self.alpha_t), 'alpha_s': torch.sigmoid(self.alpha_s)}

        # denorm
        if self.revin:
            y = y.permute(0, 2, 1)
            y = self.revin_layer(y, 'denorm')
            y = y.permute(0, 2, 1)
        return y, aux


class SeriesDecomp(nn.Module):
    def __init__(self, seq_len, kernel_sizes, periods, harmonics, decomp_mode='full', init_gate=0.2):
        super().__init__()
        self.L = seq_len
        self.kernel_sizes = tuple(kernel_sizes)
        self.periods = tuple(periods)
        self.H = harmonics
        for k in self.kernel_sizes:
            assert (k - 1) // 2 < seq_len, 'kernel size {} is too large for seq_len {}'.format(k, seq_len)

        self.trend_live = decomp_mode in ('full', 'trend_only', 'residual_only')
        self.seasonal_live = decomp_mode in ('full', 'seasonal_only', 'residual_only')
        self.residual_used = decomp_mode not in ('seasonal_only', 'trend_only')

        # trend: softmax mixture of moving averages
        if self.trend_live:
            self.trend_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

        # seasonal: projection onto a row-normalized Fourier basis [M x L]
        B = torch.zeros(0, seq_len)
        if self.seasonal_live and self.H > 0 and len(self.periods) > 0:
            t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
            B = []
            for P in self.periods:
                for h in range(1, self.H + 1):
                    ang = 2.0 * math.pi * h * t / float(P)
                    B += [torch.cos(ang), torch.sin(ang)]
            B = torch.cat(B, dim=0)
            B = B / B.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        self.register_buffer('fourier_basis', B)
        self.M = B.shape[0]

        # subtraction gates
        theta = math.log(init_gate / (1.0 - init_gate))
        if self.trend_live and self.residual_used:
            self.trend_gate = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        if self.seasonal_live and self.residual_used:
            self.season_gate = nn.Parameter(torch.tensor(theta, dtype=torch.float32))

    def moving_averages(self, x):                                           # x: [bs x nvars x seq_len]
        B, N, L = x.shape
        x = x.reshape(B * N, 1, L)
        out = []
        for k in self.kernel_sizes:
            w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
            pad = (k - 1) // 2
            xp = F.pad(x, (pad, pad), mode='reflect') if pad > 0 else x
            out.append(F.conv1d(xp, w))
        out = torch.cat(out, dim=1)
        return out.transpose(1, 2).reshape(B, N, L, len(self.kernel_sizes))

    def forward(self, x):
        trend = seasonal = None
        if self.trend_live:
            w = torch.softmax(self.trend_logits, dim=0)
            trend = (self.moving_averages(x) * w.view(1, 1, 1, -1)).sum(dim=-1)
        if self.seasonal_live:
            if self.M > 0:
                c = torch.matmul(x, self.fourier_basis.t())
                seasonal = torch.matmul(c, self.fourier_basis)
            else:
                seasonal = torch.zeros_like(x)

        if not self.residual_used:
            return None, trend, seasonal
        res = x
        if trend is not None:
            res = res - torch.sigmoid(self.trend_gate) * trend
        if seasonal is not None:
            res = res - torch.sigmoid(self.season_gate) * seasonal
        return res, trend, seasonal


class SeasonalHead(nn.Module):
    # harmonic continuation of the seasonal component over the horizon
    def __init__(self, decomp, target_window):
        super().__init__()
        L, T = decomp.L, target_window
        self.T = T
        self.register_buffer('B_in', decomp.fourier_basis.clone())
        self.M = self.B_in.shape[0]
        B_out = torch.zeros(0, T)
        if self.M > 0:
            t = torch.arange(L, L + T, dtype=torch.float32).unsqueeze(0)
            B_out = []
            for P in decomp.periods:
                for h in range(1, decomp.H + 1):
                    ang = 2.0 * math.pi * h * t / float(P)
                    B_out += [torch.cos(ang), torch.sin(ang)]
            B_out = torch.cat(B_out, dim=0)
            B_out = B_out / B_out.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        self.register_buffer('B_out', B_out)
        self.beta = nn.Parameter(torch.tensor(-2.2))

    def forward(self, x):                                                   # x: [bs x nvars x seq_len]
        if self.M == 0:
            return torch.zeros(x.shape[0], x.shape[1], self.T, device=x.device, dtype=x.dtype)
        C = torch.matmul(x, self.B_in.transpose(0, 1))
        return torch.sigmoid(self.beta) * torch.matmul(C, self.B_out)


class TimeProjector(nn.Module):
    # low-rank linear map over time, used for the trend when target_window > context_window
    def __init__(self, in_len, out_len, rank=64):
        super().__init__()
        r = min(rank, in_len, out_len)
        self.A = nn.Parameter(torch.randn(in_len, r) * 0.01)
        self.B = nn.Parameter(torch.randn(out_len, r) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_len))

    def forward(self, x):
        return (x @ self.A) @ self.B.transpose(0, 1) + self.bias.view(1, 1, -1)


class ARHead(nn.Module):
    # depthwise conv over time, keeps the last out_len steps
    def __init__(self, in_len, out_len, n_vars, kernel=25):
        super().__init__()
        self.out_len = out_len
        self.proj = nn.Conv1d(n_vars, n_vars, kernel_size=kernel, padding=kernel // 2, groups=n_vars, bias=False)

    def forward(self, x):                                                   # x: [bs x seq_len x nvars]
        y = self.proj(x.transpose(1, 2))
        return y[:, :, -self.out_len:].transpose(1, 2)


class BranchFusion(nn.Module):
    # scores each branch from the mean and std of its patch embeddings
    def __init__(self, d_model, n_branches, hidden=None, dropout=0.):
        super().__init__()
        hidden = d_model if hidden is None else hidden
        self.scorer = nn.Sequential(nn.LayerNorm(2 * d_model),
                                    nn.Linear(2 * d_model, hidden),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden, 1))

    def forward(self, outputs):
        feats = []
        for b in outputs:                                                   # b: [bs x nvars x patch_num x d_model]
            std = (b.var(2, unbiased=False) + 1e-6).sqrt()
            feats.append(torch.cat([b.mean(2), std], dim=-1))
        feats = torch.stack(feats, dim=2)                                   # [bs x nvars x n_branches x 2*d_model]
        B, V, N, F_ = feats.shape
        scores = self.scorer(feats.reshape(B * V * N, F_)).reshape(B, V, N)
        return torch.softmax(scores, dim=2)


class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                                   # x: [bs x nvars x patch_num x d_model]
        return self.dropout(self.linear(self.flatten(x)))


class Encoder(nn.Module):
    def __init__(self, c_in, seq_len, patch_num, patch_len, stride, n_layers=1, n_branches=3, d_model=128,
                 d_ff=256, dropout=0., head_dropout=0., padding_patch='end',
                 bottleneck_type='mlp', bottleneck_dim='0', **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.n_branches = n_branches
        self.seq_len = seq_len

        self.encoder = nn.ModuleList([
            nn.ModuleList([PatchEncoder(patch_num[j], patch_len[j], stride[j], padding_patch, d_model, d_ff,
                                        dropout=dropout, **kwargs) for j in range(n_branches)])
            for _ in range(n_layers)])

        # maps concatenated branch outputs back to seq_len between stacked layers
        if n_layers > 1:
            head_nf = d_model * sum(p + 1 if padding_patch == 'end' else p for p in patch_num)
            if bottleneck_type == 'linear':
                self.bottle_neck = nn.Sequential(nn.Linear(head_nf, seq_len), nn.Dropout(head_dropout))
            else:
                hidden = seq_len if bottleneck_dim == 'seq_len' else (int(bottleneck_dim) or head_nf // 2)
                self.bottle_neck = nn.Sequential(nn.Linear(head_nf, hidden), nn.GELU(),
                                                 nn.Dropout(head_dropout), nn.Linear(hidden, seq_len))

    def forward(self, x):                                                   # x: [bs x nvars x seq_len]
        for i in range(self.n_layers):
            outputs = [self.encoder[i][j](x) for j in range(self.n_branches)]
            if i == self.n_layers - 1:
                break
            z = torch.cat([out.flatten(start_dim=2) for out in outputs], dim=-1)
            bs, nvars, nf = z.shape
            x = self.bottle_neck(z.reshape(bs * nvars, nf)).view(bs, nvars, self.seq_len)
        return outputs


class PatchEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, stride, padding_patch, d_model, d_ff, dropout=0., **kwargs):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # patch values plus their mean and std
        self.W_P = nn.Linear(patch_len + 2, d_model)
        self.W_pos = positional_encoding('zeros', True, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer = EncoderLayer(patch_num, d_model, d_ff=d_ff, dropout=dropout, **kwargs)

    def forward(self, x):                                                   # x: [bs x nvars x seq_len]
        n_vars = x.shape[1]
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [bs x nvars x patch_num x patch_len]
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-6)
        x = self.W_P(torch.cat([x, mean, std], dim=-1))                    # x: [bs x nvars x patch_num x d_model]
        u = rearrange(x, 'b n p d -> (b n) p d')
        u = self.dropout(u + self.W_pos)
        z = self.layer(u)
        return rearrange(z, '(b n) p d -> b n p d', n=n_vars)


class EncoderLayer(nn.Module):
    def __init__(self, patch_num, d_model, d_ff=256, dropout=0., activation='gelu', backbone_type='kan',
                 kan_basis='krawtchouk', kan_degree=3, kan_grid_size=5, kan_path='both', **kwargs):
        super().__init__()
        if backbone_type == 'kan':
            self.mixer = DualPathKAN(d_model, patch_num, kan_basis, kan_degree, kan_grid_size, kan_path)
        elif backbone_type == 'transformer':
            nhead = next(h for h in (16, 8, 4, 2, 1) if d_model % h == 0)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout,
                                               batch_first=True, norm_first=True, activation=activation)
            self.mixer = nn.TransformerEncoder(layer, num_layers=1)
        elif backbone_type == 'smamba':
            layer = SMambaEncoderLayer(d_model=d_model, dim_feedforward=d_ff, dropout=dropout, norm_first=True,
                                       batch_first=True, mamba_cfg={'d_state': 16, 'd_conv': 4, 'expand': 2})
            self.mixer = SMambaEncoder(layer, num_layers=1, norm=nn.LayerNorm(d_model))
        elif backbone_type == 'mlpmixer':
            self.mixer = MLPMixer(d_model, patch_num, d_ff=d_ff, dropout=dropout)
        elif backbone_type == 'linear':
            self.mixer = LinearMixer(d_model, patch_num, dropout=dropout)
        else:
            raise ValueError('Unknown backbone_type: {}'.format(backbone_type))

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, src: Tensor) -> Tensor:                               # src: [bs*nvars x patch_num x d_model]
        src = self.norm_attn(src + self.dropout_attn(self.mixer(src)))
        src = self.norm_ffn(src + self.dropout_ffn(self.ff(src)))
        return src


class DualPathKAN(nn.Module):
    # ab: intra-patch then inter-patch, ba: the reverse
    def __init__(self, dim, length, kan_basis='krawtchouk', degree=3, grid_size=5, kan_path='both'):
        super().__init__()
        self.kan_path = kan_path
        self.intrapatch_kan = build_kan(kan_basis, dim, dim, degree, grid_size)
        self.interpatch_kan = build_kan(kan_basis, length, length, degree, grid_size)
        if kan_path == 'both':
            self.gating_layer = nn.Linear(dim, 2)

    def forward(self, x):                                                   # x: [bs*nvars x patch_num x d_model]
        if self.kan_path in ('both', 'ab'):
            ab = self.intrapatch_kan(x).permute(0, 2, 1)
            ab = self.interpatch_kan(ab).permute(0, 2, 1)
        if self.kan_path in ('both', 'ba'):
            ba = self.interpatch_kan(x.permute(0, 2, 1)).permute(0, 2, 1)
            ba = self.intrapatch_kan(ba)

        if self.kan_path == 'ab':
            return x + ab
        if self.kan_path == 'ba':
            return x + ba
        g = F.softmax(self.gating_layer(reduce(x, 'b l d -> b d', 'mean')), dim=-1)
        return x + g[:, 0, None, None] * ab + g[:, 1, None, None] * ba


class MLPMixer(nn.Module):
    def __init__(self, dim, length, d_ff=256, dropout=0.):
        super().__init__()
        self.norm_token = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(length, length), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(length, length))
        self.norm_chan = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, d_ff), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(d_ff, dim))

    def forward(self, x):                                                   # x: [bs*nvars x patch_num x d_model]
        x = x + self.token_mlp(self.norm_token(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.norm_chan(x))


class LinearMixer(nn.Module):
    # MLPMixer without the nonlinearity
    def __init__(self, dim, length, dropout=0.):
        super().__init__()
        self.norm_token = nn.LayerNorm(dim)
        self.token_lin = nn.Linear(length, length)
        self.norm_chan = nn.LayerNorm(dim)
        self.chan_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.token_lin(self.norm_token(x).transpose(1, 2))).transpose(1, 2)
        return x + self.dropout(self.chan_lin(self.norm_chan(x)))
