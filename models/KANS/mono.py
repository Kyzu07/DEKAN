import torch
import torch.nn as nn
import numpy as np

class Monomials(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(Monomials, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.mono_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.mono_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # print('khaisi mamu')
        four = False
        three = False
        two = False
        if len(x.shape) == 4:
            four = True
            a, b, c, d = x.shape
        elif len(x.shape) == 3:
            a, b, c = x.shape
            three = True
        else:
            a, b = x.shape
            two = True

        x = x.reshape(-1, self.input_dim)
        # x = x.view(-1, self.input_dim)
        x = torch.sigmoid(x)

        mono = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        mono[:, :, 0] = 1
        if self.degree > 0:
            mono[:, :, 1] = x

        for i in range(2, self.degree + 1):
            mono[:, :, i] = x * mono[:, :, i - 1].clone()

        y = torch.einsum('bid,iod->bo', mono, self.mono_coeffs)
        # y = y.view(a, b, self.output_dim)
        if four == True:
            y = y.reshape(a, b, c, self.output_dim)
        elif three == True:
            y = y.reshape(a, b, self.output_dim)
        else:
            y = y.reshape(a, self.output_dim)

        return y