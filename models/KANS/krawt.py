import torch
import torch.nn as nn

class KrawtchoukPolynomials(nn.Module):
    """
    Drop-in replacement for your HahnPolynomials module.

    Notes:
    - Uses the q-ary Krawtchouk recurrence:
        x K_k = - q (N - k) K_{k+1} + ( q (N - k) + k (1 - q) ) K_k - k (1 - q) K_{k-1}
      solved for K_{k+1}.
    - We treat `alpha` as `q` (alphabet-size style parameter); `beta` is unused.
    - Degree should satisfy degree <= N to avoid division by zero at k = N.
    """
    def __init__(self, input_dim, output_dim, degree, q, N):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.degree     = degree
        self.q          = q      # use alpha as q-parameter
        self.N          = N

        self.kraw_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.kraw_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
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

        # match your Hahn pre-nonlinearity for parity
        x = torch.tanh(x)  # still works as a continuous input to the discrete polynomial basis

        Bsz = x.size(0)
        D_in = self.input_dim
        deg = self.degree
        q = self.q
        N = self.N
        eps = 1e-8

        kraw = torch.zeros(Bsz, D_in, deg + 1, device=x.device, dtype=x.dtype)

        # Base cases
        kraw[:, :, 0] = 1.0
        if deg > 0:
            # From the recurrence with k=0:  x*1 = -qN K1 + qN*1  =>  K1 = (qN - x) / (qN)
            denom = q * N + eps
            kraw[:, :, 1] = (q * N - x) / denom

        # Three-term recurrence for k = 1..deg-1:
        # K_{k+1} = (( q(N-k) + k(1-q) - x ) * K_k - k(1-q) * K_{k-1}) / ( q(N-k) )
        for n in range(2, deg + 1):
            k = n - 1
            A = q * (N - k) + eps
            B = q * (N - k) + k * (1.0 - q)
            C = k * (1.0 - q)
            k_prev = kraw[:, :, n - 1].clone()
            k_prev2 = kraw[:, :, n - 2].clone()
            kraw[:, :, n] = ((B - x) * k_prev - C * k_prev2) / A

        # Linear combination across degrees
        y = torch.einsum('bid,iod->bo', kraw, self.kraw_coeffs)

        if four:
            y = y.reshape(a, b, c, self.output_dim)
        elif three:
            y = y.reshape(a, b, self.output_dim)
        else:
            y = y.reshape(a, self.output_dim)

        return y
