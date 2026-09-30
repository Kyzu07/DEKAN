import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyKAN(nn.Module):
    """y_o = sum_i sum_r c[i, o, r] * P_r(x_i)"""
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def basis(self, x):
        raise NotImplementedError

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.input_dim)
        y = torch.einsum('bid,iod->bo', self.basis(x), self.coeffs)
        return y.reshape(*shape[:-1], self.output_dim)


class KrawtchoukKAN(PolyKAN):
    def __init__(self, input_dim, output_dim, degree, q=0.6, N=255):
        super().__init__(input_dim, output_dim, degree)
        self.q = q
        self.N = N

    def basis(self, x):
        x = torch.tanh(x)
        q, N, eps = self.q, self.N, 1e-8
        P = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device, dtype=x.dtype)
        P[:, :, 0] = 1.0
        if self.degree > 0:
            P[:, :, 1] = (q * N - x) / (q * N + eps)
        for n in range(2, self.degree + 1):
            k = n - 1
            A = q * (N - k) + eps
            B = q * (N - k) + k * (1.0 - q)
            C = k * (1.0 - q)
            P[:, :, n] = ((B - x) * P[:, :, n - 1].clone() - C * P[:, :, n - 2].clone()) / A
        return P


class LucasKAN(PolyKAN):
    def basis(self, x):
        x = torch.tanh(x)
        P = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        P[:, :, 0] = 2
        if self.degree > 0:
            P[:, :, 1] = x
        for i in range(2, self.degree + 1):
            P[:, :, i] = x * P[:, :, i - 1].clone() + P[:, :, i - 2].clone()
        return P


class ChebyshevKAN(PolyKAN):
    def basis(self, x):
        x = torch.tanh(x)
        P = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            P[:, :, 1] = x
        for i in range(2, self.degree + 1):
            P[:, :, i] = 2 * x * P[:, :, i - 1].clone() - P[:, :, i - 2].clone()
        return P


class MonomialKAN(PolyKAN):
    def basis(self, x):
        x = torch.sigmoid(x)
        P = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        P[:, :, 0] = 1
        if self.degree > 0:
            P[:, :, 1] = x
        for i in range(2, self.degree + 1):
            P[:, :, i] = x * P[:, :, i - 1].clone()
        return P


class LegendreKAN(PolyKAN):
    def basis(self, x):
        x = torch.tanh(x)
        P = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device, dtype=x.dtype)
        P[:, :, 0] = 1.0
        if self.degree > 0:
            P[:, :, 1] = x
        for n in range(2, self.degree + 1):
            P[:, :, n] = ((2.0 * n - 1.0) * x * P[:, :, n - 1].clone()
                          - (n - 1.0) * P[:, :, n - 2].clone()) / float(n)
        return P


class BernsteinKAN(PolyKAN):
    def basis(self, x):
        t = (0.5 * (torch.tanh(x) + 1.0)).unsqueeze(-1)
        # de Casteljau degree elevation
        P = torch.ones(x.shape + (1,), device=x.device, dtype=x.dtype)
        for _ in range(self.degree):
            pad = torch.zeros_like(P[..., :1])
            P = torch.cat([P, pad], dim=-1) * (1.0 - t) + torch.cat([pad, P], dim=-1) * t
        return P


class HahnKAN(PolyKAN):
    def __init__(self, input_dim, output_dim, degree, alpha=1.0, beta=1.0, N=7):
        super().__init__(input_dim, output_dim, degree)
        self.a = alpha
        self.b = beta
        self.N = N

    def basis(self, x):
        x = torch.tanh(x)
        a, b, N, eps = self.a, self.b, self.N, 1e-8
        P = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device, dtype=x.dtype)
        P[:, :, 0] = 1.0
        if self.degree > 0:
            P[:, :, 1] = 1 - (((a + b + 2) * x) / ((a + 1) * N))
        for n in range(2, self.degree + 1):
            m = n - 1
            A = (m + a + b + 1) * (m + a + 1) * (N - m)
            A /= (m + m + a + b + 1)
            A /= (m + m + a + b + 2)
            C = m * (m + a + b + N + 1) * (m + b)
            C /= (m + m + a + b)
            C /= (m + m + a + b + 1)
            P[:, :, n] = ((A + C - x) * P[:, :, n - 1].clone()) - (C * P[:, :, n - 2].clone())
            P[:, :, n] /= (A + eps)
        return P


class FourierKAN(nn.Module):
    # https://github.com/GistNoesis/FourierKAN
    def __init__(self, inputdim, outdim, gridsize=300):
        super().__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        outshape = x.shape[0:-1] + (self.outdim,)
        x = x.reshape(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.reshape(x.shape[0], 1, x.shape[1], 1)
        c = torch.reshape(torch.cos(k * xrshp), (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(torch.sin(k * xrshp), (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        return y.reshape(outshape)


class KANLinear(nn.Module):
    # B-spline KAN layer from https://github.com/Blealtan/efficient-kan
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) \
                  + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
                                     if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


def build_kan(kan_basis, in_dim, out_dim, degree, grid_size=5):
    if kan_basis == 'krawtchouk':
        return KrawtchoukKAN(in_dim, out_dim, degree)
    elif kan_basis == 'lucas':
        return LucasKAN(in_dim, out_dim, degree)
    elif kan_basis == 'chebyshev':
        return ChebyshevKAN(in_dim, out_dim, degree)
    elif kan_basis == 'monomials':
        return MonomialKAN(in_dim, out_dim, degree)
    elif kan_basis == 'legendre':
        return LegendreKAN(in_dim, out_dim, degree)
    elif kan_basis == 'bernstein':
        return BernsteinKAN(in_dim, out_dim, degree)
    elif kan_basis == 'hahn':
        return HahnKAN(in_dim, out_dim, degree)
    elif kan_basis == 'bspline':
        return KANLinear(in_dim, out_dim, grid_size=grid_size, spline_order=3)
    elif kan_basis == 'fourier':
        return FourierKAN(in_dim, out_dim, gridsize=grid_size)
    raise ValueError('Unknown kan_basis: {}'.format(kan_basis))
