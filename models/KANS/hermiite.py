# hermite_kan_super_safe.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HermiteKAN(nn.Module):
    """
    Probabilists' Hermite KAN, ultra-conservative for autograd safety.
    He_0=1, He_1=x, He_{n+1} = x*He_n - n*He_{n-1}

    Safety:
      - No in-place ops, no .view/.transpose_.
      - No flatten/unflatten across batch/time dims (operates on last-dim features).
      - Optional clone on input (copy_in) and output (copy_out) to break aliasing.
      - Basis via list+stack (no slice writes).
      - per-channel z = (x - mu) / (softplus(sigma_raw)+eps)
      - Optional 1/sqrt(n!) orthonormal scaling.
      - LayerNorm over degree axis (last dim = deg+1) + dropout.
      - Zero-init gated residual skip x->y (safe).

    Shapes: accepts [..., D_in] and returns [..., D_out].
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 degree: int = 8,
                 ln_basis: bool = True,
                 basis_drop: float = 0.0,
                 use_skip: bool = True,
                 orthonormal: bool = True,
                 copy_in: bool = True,
                 copy_out: bool = True):
        super().__init__()
        assert degree >= 1
        self.D_in, self.D_out, self.deg = input_dim, output_dim, degree
        self.orthonormal = orthonormal
        self.copy_in = copy_in
        self.copy_out = copy_out
        self.eps = 1e-6

        # per-channel standardization to ~N(0,1)
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.sigma_raw = nn.Parameter(torch.zeros(input_dim))  # sigma = softplus + eps

        # coefficients: [D_out, D_in, deg+1]
        self.coeff = nn.Parameter(torch.empty(output_dim, input_dim, degree + 1))
        nn.init.normal_(self.coeff, mean=0.0, std=(1.0 / (input_dim * (degree + 1)))**0.5)

        # LN over degree axis (last dim)
        self.ln_basis = nn.LayerNorm(degree + 1) if ln_basis else None
        self.dropout = nn.Dropout(p=basis_drop) if basis_drop > 0 else nn.Identity()

        # zero-init gated residual skip
        self.use_skip = use_skip
        if use_skip:
            self.skip_W = nn.Parameter(torch.zeros(output_dim, input_dim))
            self.skip_b = nn.Parameter(torch.zeros(output_dim))
            self.skip_gate = nn.Parameter(torch.zeros(1))  # starts at 0

        # precompute 1/sqrt(n!)
        if orthonormal:
            vals = []
            acc = 0.0
            for n in range(degree + 1):
                if n > 0: acc += math.log(n)
                vals.append(math.exp(-0.5 * acc))
            self.register_buffer('inv_sqrt_fact', torch.tensor(vals, dtype=torch.float32))
        else:
            self.register_buffer('inv_sqrt_fact', torch.ones(degree + 1))

    def _hermite_basis(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [..., D_in]  ->  H: [..., D_in, deg+1]
        """
        H_list = []
        H0 = torch.ones_like(z)              # He_0
        H_list.append(H0)
        if self.deg >= 1:
            H1 = z                           # He_1
            H_list.append(H1)
        for n in range(1, self.deg):
            Hnm1 = H_list[-1]
            Hnm2 = H_list[-2]
            Hnp1 = z * Hnm1 - float(n) * Hnm2
            H_list.append(Hnp1)
        H = torch.stack(H_list, dim=-1)      # [..., D_in, deg+1]

        if self.orthonormal:
            # broadcast 1/sqrt(n!) onto last dim
            shape = (1,) * (H.ndim - 1) + (self.deg + 1,)
            H = H * self.inv_sqrt_fact.view(*shape)

        if self.ln_basis is not None:
            H = self.ln_basis(H)             # normalize along deg+1 axis

        return self.dropout(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., D_in]  ->  y: [..., D_out]
        """
        # Defensive copy at entry (break alias with caller)
        x_local = x.clone() if self.copy_in else x

        # per-channel standardization
        sigma = F.softplus(self.sigma_raw) + self.eps     # [D_in]
        z = (x_local - self.mu) / sigma                   # [..., D_in]

        # basis
        H = self._hermite_basis(z)                        # [..., D_in, deg+1]

        # contract basis -> outputs; einsum works on arbitrary leading dims
        y_basis = torch.einsum('...id,oid->...o', H, self.coeff)  # [..., D_out]

        if self.use_skip:
            # F.linear supports N-D input (last dim=in_features)
            y_skip = F.linear(x_local, self.skip_W, self.skip_b)  # [..., D_out]
            y = y_basis + self.skip_gate * y_skip
        else:
            y = y_basis

        # Defensive copy at exit (avoid downstream in-place mutating our saved tensors)
        return y.clone() if self.copy_out else y
