# S-Mamba Encoder: a drop-in replacement for nn.TransformerEncoder
# Requires: torch>=1.13.1. Optional: pip install mamba-ssm

import math
from typing import Optional
import torch
from torch import nn, Tensor

# ---- Optional Mamba import (falls back gracefully) --------------------------
_HAS_MAMBA = True
try:
    from mamba_ssm import Mamba
except Exception:
    _HAS_MAMBA = False


# ---- Utility: Feedforward (same as Transformer FFN) -------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---- Fallback “SSM-ish” block when mamba_ssm is not available ---------------
class FallbackSsmBlock(nn.Module):
    """
    Lightweight gated depthwise-conv mixer (causal) as a stand-in if Mamba isn't installed.
    Not a faithful Mamba, but keeps the interface and works as a placeholder.
    """
    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        # Depthwise temporal conv (causal padding)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=kernel_size - 1, groups=d_model)
        # Pointwise mix
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        # x: [B, L, D]
        x_ = x.transpose(1, 2)  # [B, D, L]
        y = self.dw(x_)
        if is_causal:
            # Strict causality: remove future leakage from padding
            ks = self.dw.kernel_size[0]
            y = y[:, :, : x_.size(-1)]
            # After causal conv, lengths match; nothing else needed

        y = self.pw(y)
        y = self.gate(y) * y
        y = y.transpose(1, 2)  # [B, L, D]
        return self.dropout(y)


# ---- S-Mamba encoder layer (TransformerEncoderLayer-like) -------------------
class SMambaEncoderLayer(nn.Module):
    """
    A single encoder layer that swaps self-attention for a Mamba (selective SSM) block.
    API mirrors nn.TransformerEncoderLayer as closely as possible.

    Args:
        d_model: embedding size (same as Transformer)
        dim_feedforward: FFN hidden size
        dropout: dropout prob
        norm_first: if True, use pre-norm; else post-norm
        batch_first: if True, expect [B, L, D]; else [L, B, D]
        mamba_cfg: dict passed to Mamba (if available), e.g. {'d_state': 16, 'd_conv': 4, 'expand': 2}
                    See mamba-ssm docs. Reasonable defaults are used if None.
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        batch_first: bool = False,
        mamba_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.norm_first = norm_first

        # Core sequence mixer: Mamba if available, otherwise fallback block
        if _HAS_MAMBA:
            cfg = dict(d_model=d_model, d_state=16, d_conv=4, expand=2)
            if mamba_cfg is not None:
                cfg.update(mamba_cfg)
            self.mixer = Mamba(**cfg)  # expects [B, L, D]
        else:
            self.mixer = FallbackSsmBlock(d_model, kernel_size=7, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFFN(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        """
        Args:
            src: [L, B, D] if batch_first=False (default), else [B, L, D]
            src_mask: (L, L) or (B, L, L). Not used by Mamba; kept for API compatibility.
            src_key_padding_mask: [B, L], True for padding positions.
            is_causal: if True, enforce causality in the mixer (Mamba is naturally causal).
        """
        x = src
        bf = self.batch_first

        if not bf:
            x = x.transpose(0, 1)  # [B, L, D]

        # Apply key padding mask by zeroing out embeddings (common SSM practice)
        if src_key_padding_mask is not None:
            # src_key_padding_mask: True means PAD; zero those tokens
            mask = src_key_padding_mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
            x = x * (1.0 - mask)

        # Mamba (or fallback) + residual + norm
        if self.norm_first:
            y = self.mixer(self.norm1(x), is_causal=True if is_causal is None else is_causal)
            x = x + self.dropout1(y)
            y = self.ffn(self.norm2(x))
            x = x + self.dropout2(y)
        else:
            y = self.mixer(x, is_causal=True if is_causal is None else is_causal)
            x = self.norm1(x + self.dropout1(y))
            y = self.ffn(x)
            x = self.norm2(x + self.dropout2(y))

        if not bf:
            x = x.transpose(0, 1)  # back to [L, B, D]
        return x


# ---- S-Mamba encoder stack (TransformerEncoder-like) ------------------------
class SMambaEncoder(nn.Module):
    """
    Stack of SMambaEncoderLayer with the same external API as nn.TransformerEncoder.

    Args:
        encoder_layer: an instance of SMambaEncoderLayer
        num_layers: how many layers to stack
        norm: optional final LayerNorm(d_model)
    """
    def __init__(self, encoder_layer: SMambaEncoderLayer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer if i == 0 else
                                     SMambaEncoderLayer(
                                         d_model=encoder_layer.d_model,
                                         dim_feedforward=encoder_layer.ffn.net[0].out_features,
                                         dropout=encoder_layer.dropout1.p,
                                         norm_first=encoder_layer.norm_first,
                                         batch_first=encoder_layer.batch_first,
                                         mamba_cfg=None  # inherit defaults; customize per-layer if needed
                                     )
                                     for i in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        """
        Args:
            src: [L, B, D] (default) or [B, L, D] if the layer was created with batch_first=True
            mask: kept for API symmetry (not used by Mamba)
            src_key_padding_mask: [B, L] boolean, True for padding
            is_causal: pass through to layers (Mamba is causal by design)
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           is_causal=is_causal)
        if self.norm is not None:
            if isinstance(self.layers[0], SMambaEncoderLayer) and self.layers[0].batch_first:
                output = self.norm(output)
            else:
                # norm expects last dim = d_model; shape is [L,B,D] or [B,L,D] either is fine
                output = self.norm(output)
        return output


# ---- Example usage ----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D = 4, 128, 256

    # Choose layout: batch_first=False to mirror nn.TransformerEncoder default
    layer = SMambaEncoderLayer(d_model=D, dim_feedforward=1024, dropout=0.1,
                               norm_first=True, batch_first=False,
                               mamba_cfg={"d_state": 16, "d_conv": 4, "expand": 2})
    enc = SMambaEncoder(layer, num_layers=6, norm=nn.LayerNorm(D))

    x = torch.randn(L, B, D)  # [L, B, D]
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[0, -10:] = True  # last 10 are padding for batch 0

    y = enc(x, src_key_padding_mask=pad_mask)  # shape: [L, B, D]
    print(y.shape)  # torch.Size([128, 4, 256])
