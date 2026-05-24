import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward

class WaveletKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, wavelet='haar', J=3, addbias=True):
        super().__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.J = J
        self.addbias = addbias

        # DWT1DForward only supports 1D signals (per channel)
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')

        # Learnable weights for each wavelet component
        # Each signal is transformed into approx + detail_coeffs
        # We'll learn a weighted sum from these features
        # Flattened coefficients: J levels * 2 (high/low) per input channel
        coeffs_per_channel = 2 * J

        self.weight = nn.Parameter(torch.randn(outdim, inputdim * coeffs_per_channel))
        if addbias:
            self.bias = nn.Parameter(torch.zeros(outdim))

    def forward(self, x):
        """
        x: Tensor of shape (B, inputdim, T), where T >= 2**J
        returns: Tensor of shape (B, outdim)
        """
        B, C, T = x.shape
        assert C == self.inputdim, f"Expected inputdim={self.inputdim}, got {C}"

        # Apply DWT to each channel
        approx, detail = self.dwt(x)  # approx: (B, C, T'), detail: list of J tensors

        # Stack all detail coefficients and final approximation
        features = [approx]
        for d in detail:
            features.append(d)

        # Flatten across all channels and levels
        flat_feats = torch.cat(features, dim=-1)  # shape: (B, C, total_T)
        flat_feats = flat_feats.reshape(B, -1)    # shape: (B, C * coeffs_per_channel)

        # Linear transformation
        out = flat_feats @ self.weight.T
        if self.addbias:
            out += self.bias
        return out
