from typing import Optional

import torch
from torch import nn, Tensor

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FallbackSsmBlock(nn.Module):
    # gated causal depthwise conv, used when mamba_ssm is not installed
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size - 1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):
        x_ = x.transpose(1, 2)
        y = self.dw(x_)
        if is_causal:
            y = y[:, :, :x_.size(-1)]
        y = self.pw(y)
        y = self.gate(y) * y
        return self.dropout(y.transpose(1, 2))


class SMambaEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, norm_first=True,
                 batch_first=False, mamba_cfg: Optional[dict] = None):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.norm_first = norm_first

        if _HAS_MAMBA:
            cfg = dict(d_model=d_model, d_state=16, d_conv=4, expand=2)
            if mamba_cfg is not None:
                cfg.update(mamba_cfg)
            self.mixer = Mamba(**cfg)
        else:
            self.mixer = FallbackSsmBlock(d_model, kernel_size=7, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFFN(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor) -> Tensor:
        x = src if self.batch_first else src.transpose(0, 1)
        if self.norm_first:
            x = x + self.dropout1(self.mixer(self.norm1(x)))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.mixer(x)))
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x if self.batch_first else x.transpose(0, 1)


class SMambaEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer if i == 0 else
                                     SMambaEncoderLayer(d_model=encoder_layer.d_model,
                                                        dim_feedforward=encoder_layer.ffn.net[0].out_features,
                                                        dropout=encoder_layer.dropout1.p,
                                                        norm_first=encoder_layer.norm_first,
                                                        batch_first=encoder_layer.batch_first)
                                     for i in range(num_layers)])
        self.norm = norm

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output
