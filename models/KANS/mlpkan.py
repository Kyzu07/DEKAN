import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPKAN(nn.Module):
    """
    Kolmogorov–Arnold inner maps with per-edge univariate MLPs:
        χ_{q,p}(x_p) = Linear(1,1) -> GELU -> Linear(1,1)
    - Unique parameters for every (q, p) edge (no sharing).
    - Input shape: (..., n_vars)
    - Output shape: (..., n_outer) containing s_q(x) = sum_p χ_{q,p}(x_p)
    """
    def __init__(self, n_vars: int, n_outer: int, init: str = "identity"):
        super().__init__()
        self.n_vars = n_vars
        self.n_outer = n_outer

        # Parameters for all edges stacked as [Q, P]
        self.w1 = nn.Parameter(torch.empty(n_outer, n_vars))
        self.b1 = nn.Parameter(torch.empty(n_outer, n_vars))
        self.w2 = nn.Parameter(torch.empty(n_outer, n_vars))
        self.b2 = nn.Parameter(torch.empty(n_outer, n_vars))

        self.reset_parameters(init)

    def reset_parameters(self, init: str = "identity"):
        if init == "identity":
            # Near-identity χ so s_q starts near sum of inputs
            nn.init.ones_(self.w1)
            nn.init.zeros_(self.b1)
            nn.init.ones_(self.w2)
            nn.init.zeros_(self.b2)
        elif init == "xavier":
            g = 1.0
            for W in (self.w1, self.w2):
                fan_in = 1.0
                bound = g * (6.0 / fan_in) ** 0.5
                nn.init.uniform_(W, -bound, bound)
            nn.init.zeros_(self.b1)
            nn.init.zeros_(self.b2)
        else:
            nn.init.normal_(self.w1, mean=1.0, std=0.05)
            nn.init.zeros_(self.b1)
            nn.init.normal_(self.w2, mean=1.0, std=0.05)
            nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., n_vars)
        returns: (..., n_outer) with s_q(x) = sum_p χ_{q,p}(x_p)
        """
        assert x.shape[-1] == self.n_vars, f"Expected last dim {self.n_vars}, got {x.shape[-1]}"

        # Broadcast x to (..., Q, P)
        # x_exp[..., q, p] = x[..., p]
        x_exp = x.unsqueeze(-2)  # (..., 1, P)
        # Apply per-edge MLP in a vectorized way
        y = x_exp * self.w1 + self.b1         # (..., Q, P)
        y = F.gelu(y)
        y = y * self.w2 + self.b2             # (..., Q, P)

        # Sum over variables p -> s_q(x)
        s = y.sum(dim=-1)                     # (..., Q)
        return s

