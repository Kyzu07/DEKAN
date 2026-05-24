import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Hahn-KAN++ : stable, flexible
# -----------------------------
class HahnKANPP(nn.Module):
    """
    Better Hahn-based KAN.
    - Domain-aligned: learned affine maps x -> [0, N] (sigmoid), not hard tanh.
    - Stable 3-term Hahn recurrence with eps guards, vectorized.
    - Optional LayerNorm along degree axis to keep features well-scaled.
    - Zero-init residual skip (linear over time/features) with learnable gate.
    - Works for 2D [B,D], 3D [B,L,D], 4D [B,V,P,D].

    Args:
        input_dim:   D_in (last-dim features)
        output_dim:  D_out
        degree:      max polynomial degree (>=1)
        alpha,beta:  Hahn parameters (α, β)
        N:           Hahn support size (integer; x ∈ {0,...,N})
        ln_basis:    apply LayerNorm over degree axis (per input dim)
        basis_drop:  dropout on basis features (over degree axis)
        use_skip:    add zero-init residual linear skip from x -> y
        input_map:   'sigmoid' (to [0,N]) or 'identity' (assume already scaled)
    """
    def __init__(self, input_dim, output_dim, degree=8,
                 alpha=0.0, beta=0.0, N=255,
                 ln_basis=True, basis_drop=0.0,
                 use_skip=True, input_map='sigmoid'):
        super().__init__()
        assert degree >= 1, "degree must be >= 1"
        self.D_in, self.D_out = input_dim, output_dim
        self.deg = degree
        self.a, self.b = float(alpha), float(beta)
        self.N = float(N)
        self.input_map = input_map

        # per-channel affine to align domain before Hahn (x_affine -> [0,N])
        self.affine_scale = nn.Parameter(torch.ones(input_dim))
        self.affine_bias  = nn.Parameter(torch.zeros(input_dim))

        # coefficients for combining basis -> outputs
        # shape [D_out, D_in, deg+1]
        self.coeff = nn.Parameter(torch.empty(output_dim, input_dim, degree + 1))
        nn.init.normal_(self.coeff, mean=0.0, std=1.0 / (input_dim * (degree + 1))**0.5)

        # optional normalization/dropout on degree axis
        self.ln_basis = nn.LayerNorm(degree + 1) if ln_basis else None
        self.dropout = nn.Dropout(p=basis_drop) if basis_drop > 0 else nn.Identity()

        # zero-init residual skip (safe)
        self.use_skip = use_skip
        if use_skip:
            self.skip_W = nn.Parameter(torch.zeros(output_dim, input_dim))
            self.skip_b = nn.Parameter(torch.zeros(output_dim))
            self.skip_gate = nn.Parameter(torch.zeros(1))  # starts at 0

        # small epsilon to avoid div-by-zero in recurrence
        self.eps = 1e-9

    # -------------- utilities --------------
    @staticmethod
    def _reshape_in(x):
        """Return (x2d, shape_info) where x2d is [B*, D_in] and shape_info lets us restore."""
        four = three = False
        if x.ndim == 4:      # [B,V,P,D]
            B, V, P, D = x.shape
            four = True
            x2d = x.reshape(B * V * P, D)
            info = ('4', B, V, P, D)
        elif x.ndim == 3:    # [B,L,D]
            B, L, D = x.shape
            three = True
            x2d = x.reshape(B * L, D)
            info = ('3', B, L, D)
        elif x.ndim == 2:    # [B,D]
            B, D = x.shape
            x2d = x
            info = ('2', B, D)
        else:
            raise RuntimeError(f"Unsupported input shape {tuple(x.shape)}; last dim must be features")
        return x2d, info

    @staticmethod
    def _reshape_out(y2d, info):
        """Restore to original batch/time/vars with last dim = D_out."""
        tag = info[0]
        if tag == '4':
            _, B, V, P, _Din = info
            return y2d.view(B, V, P, -1)
        elif tag == '3':
            _, B, L, _Din = info
            return y2d.view(B, L, -1)
        else:
            _, B, _Din = info
            return y2d.view(B, -1)

    def _map_to_domain(self, x2d):
        """
        Map raw x (any real) to Hahn domain in [0, N].
        Uses per-channel affine + optional sigmoid squashing.
        """
        # per-channel affine
        x_aff = x2d * self.affine_scale + self.affine_bias  # [B*, D]
        if self.input_map == 'sigmoid':
            x_unit = torch.sigmoid(x_aff)                   # [0,1]
            x_scaled = x_unit * self.N                      # [0,N]
        elif self.input_map == 'identity':
            x_scaled = x_aff
        else:
            raise ValueError("input_map must be 'sigmoid' or 'identity'")
        return x_scaled

    # -------------- Hahn basis --------------
    def _hahn_basis(self, x_scaled):
        """
        Compute Hahn basis Q_n(x; a,b,N) for n=0..deg at points x_scaled ∈ [0,N] (continuous extension).
        Returns tensor H: [B*, D_in, deg+1].
        Recurrence is vectorized, with eps guards.
        """
        Bflat, D = x_scaled.shape
        H = x_scaled.new_zeros(Bflat, D, self.deg + 1)

        # Q_0(x) = 1
        H[..., 0] = 1.0

        # Q_1(x) per your recurrence
        if self.deg >= 1:
            denom1 = (self.a + 1.0) * (self.N + self.eps)
            H[..., 1] = 1.0 - ((self.a + self.b + 2.0) * x_scaled) / denom1

        # constants A_m, C_m depend on m (n-1), not on x
        # vectorize over m: m = 1..(deg-1) to compute Q_{m+1}
        for n in range(2, self.deg + 1):
            m = n - 1.0

            A = (m + self.a + self.b + 1.0) * (m + self.a + 1.0) * (self.N - m)
            A = A / (2.0 * m + self.a + self.b + 1.0 + self.eps)
            A = A / (2.0 * m + self.a + self.b + 2.0 + self.eps)

            C = m * (m + self.a + self.b + self.N + 1.0) * (m + self.b)
            C = C / (2.0 * m + self.a + self.b + self.eps)
            C = C / (2.0 * m + self.a + self.b + 1.0 + self.eps)

            # Q_n(x) = ((A + C - x) * Q_{n-1} - C * Q_{n-2}) / A
            H[..., n] = ((A + C - x_scaled) * H[..., n - 1] - C * H[..., n - 2]) / (A + self.eps)

        # (optional) normalize across degree axis per input dim for stability
        if self.ln_basis is not None:
            assert H.shape[-1] == self.deg + 1, f"Expected last dim={self.deg+1}, got {H.shape}"
            H = self.ln_basis(H)   # H is [B*, D_in, deg+1]


        H = self.dropout(H)  # basis dropout along degree axis
        return H  # [B*, D_in, deg+1]

    # -------------- forward --------------
    def forward(self, x):
        """
        x: [B,D] or [B,L,D] or [B,V,P,D] (last dim = input_dim)
        returns: same batch dims with last dim = output_dim
        """
        x2d, info = self._reshape_in(x)                    # [B*, D_in]
        x_scaled = self._map_to_domain(x2d)                # [B*, D_in] in [0,N]
        H = self._hahn_basis(x_scaled)                     # [B*, D_in, deg+1]

        # basis -> outputs: y_basis[b,o] = sum_{i,d} H[b,i,d] * coeff[o,i,d]
        y_basis = torch.einsum('bid,oid->bo', H, self.coeff)

        if self.use_skip:
            y_skip = F.linear(x2d, self.skip_W, self.skip_b)    # [B*, D_out]
            y = y_basis + self.skip_gate * y_skip
        else:
            y = y_basis

        return self._reshape_out(y, info)
