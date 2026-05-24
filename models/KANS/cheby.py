import torch
import torch.nn as nn


class ChebyshevPolynomials(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevPolynomials, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        a, b ,c = x.shape
        x = torch.reshape(x, (-1, self.inputdim))
        x = torch.tanh(x)
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)
        y = y.view(a, b, self.outdim)
        return y