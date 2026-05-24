
class RelPosConv(nn.Module):
    """Depthwise long 1D conv (relative-pos filter) with tiny gated residual."""
    def __init__(self, d_model, kernel=129):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel,
                            padding=kernel//2, groups=d_model, bias=False)
        nn.init.zeros_(self.dw.weight)           # start as identity (no effect)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # small learnable gate

    def forward(self, x):  # x: [B, L, D]
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        return x + self.alpha * y



# --- DCT low-pass probe fusion head (smooth ONLY along P) --------------------
import math
import torch
from torch import nn
import torch.nn.functional as F

class _OrthoDCT(nn.Module):
    """
    Orthonormal DCT-II transform matrix for a given length P.
    With 'ortho' normalization, inverse is simply the transpose.
    """
    def __init__(self, P: int):
        super().__init__()
        n = torch.arange(P, dtype=torch.float32)                              # [P]
        k = torch.arange(P, dtype=torch.float32).unsqueeze(1)                 # [P,1]
        # DCT-II basis: C[k,n] = alpha(k) * cos(pi*(n+0.5)*k/P)
        C = torch.cos(math.pi * (n + 0.5) * k / P)                            # [P,P]
        alpha = torch.ones(P)
        alpha[0] = 1.0 / math.sqrt(P)
        alpha[1:] = math.sqrt(2.0 / P)
        C = (alpha.unsqueeze(1) * C).to(torch.float32)                        # [P,P]
        self.register_buffer("C", C, persistent=False)                        # moves with .to()

    def forward(self, x: torch.Tensor, keep_k: int) -> torch.Tensor:
        """
        x: [N, P, D]  -> returns low-pass reconstruction y: [N, P, D]
        keep_k: number of lowest DCT coeffs to keep (1..P)
        """
        N, P, D = x.shape
        C = self.C.to(device=x.device, dtype=x.dtype)                          # [P,P]
        # DCT: X = C @ x
        X = torch.einsum('kp,npd->nkd', C, x)                                  # [N,P,D]
        # Zero high frequencies
        K = max(1, min(int(keep_k), P))
        if K < P:
            X[:, K:, :] = 0
        # IDCT: y = C^T @ X
        y = torch.einsum('pk,nkd->npd', C, X)                                  # [N,P,D]
        return y

class GumbelTopKRouter(nn.Module):
    """
    Gumbel Top-K router.
    Inputs:  list of N branch tensors, each [B, V, P_j, D]
    Output:  weights [B, V, N] (K-hot during train if hard=True; soft at eval unless hard_eval=True)
    """
    def __init__(self, d_model: int, n_branches: int,
                 hidden: int | None = None, topk: int = 2,
                 tau: float = 1.0, hard: bool = True, hard_eval: bool = True,
                 detach_probe: bool = True, eps: float = 1e-12):
        super().__init__()
        self.n_branches = n_branches
        self.topk = max(1, min(topk, n_branches))
        self.tau = tau
        self.hard = hard
        self.hard_eval = hard_eval
        self.detach_probe = detach_probe
        feat_dim = d_model * 2  # mean + std over patches (P)
        hidden = d_model if hidden is None else hidden
        self.scorer = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.eps = eps

    @staticmethod
    def _feat(b: torch.Tensor) -> torch.Tensor:
        # b: [B, V, P, D] -> [B, V, 2D] via mean+std over P
        mean = b.mean(dim=2)
        std  = torch.sqrt(b.var(dim=2, unbiased=False) + 1e-6)
        return torch.cat([mean, std], dim=-1)

    @staticmethod
    def _sample_gumbel(shape, device, dtype):
        u = torch.rand(shape, device=device, dtype=dtype).clamp_(1e-9, 1 - 1e-9)
        return -torch.log(-torch.log(u))

    def _gumbel_topk(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, V, N] -> K-hot, renormalized weights [B, V, N]
        if self.training:
            g = self._sample_gumbel(logits.shape, logits.device, logits.dtype)
            y = (logits + g) / max(self.tau, 1e-6)
        else:
            y = logits  # deterministic at eval

        # choose Top-K on perturbed scores (or plain logits at eval)
        topk_val, topk_idx = torch.topk(y, k=self.topk, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)  # K-hot

        # soft distribution for gradients; renormalize only on the selected set
        y_soft = torch.softmax(y, dim=-1)
        y_masked = (y_soft * mask)
        y_masked = y_masked / (y_masked.sum(dim=-1, keepdim=True) + self.eps)

        if self.training and self.hard:
            # straight-through: forward uses hard mask; backward uses soft y_masked
            y_out = mask + (y_masked - y_masked.detach())
        else:
            # eval: optionally keep it hard too
            y_out = mask if self.hard_eval else y_masked
        return y_out

    def forward(self, branch_outputs_list):
        # branch_outputs_list: length N, each [B, V, P_j, D]
        feats = [self._feat(b.detach() if self.detach_probe else b)
                 for b in branch_outputs_list]                 # N * [B, V, 2D]
        feats = torch.stack(feats, dim=2)                      # [B, V, N, 2D]
        B, V, N, F = feats.shape
        logits = self.scorer(feats.reshape(B * V * N, F)).reshape(B, V, N)
        return self._gumbel_topk(logits)                       # [B, V, N]


class DCTRouterFusionHead(nn.Module):
    """
    DCT-based scale router (SCI1 variant).
    - For each branch [B,V,P,D], detach, low-pass along P via orthonormal DCT (keep first K),
      reconstruct y, then compute:
        residual (normalized), first/second diffs on y (normalized), magnitude, logP
    - Tiny MLP -> per-branch scores -> softmax (optional Top-K) -> weights [B,V,N]
    """
    def __init__(self, d_model: int, n_branches: int,
                 keep_k: int | None = None, keep_ratio: float = 0.25,
                 topk: int | None = 2, hidden: int = 64,
                 eps: float = 1e-6, temperature: float = 1.0):
        super().__init__()
        self.n_branches = n_branches
        self.keep_k = keep_k
        self.keep_ratio = keep_ratio
        self.topk = topk
        self.eps = eps
        self.temperature = temperature
        self._dct_by_P = nn.ModuleDict()  # lazy cache of _OrthoDCT(P)

        # Features: [resid, lips_n, curv_n, mag, logP] -> score
        self.scorer = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _get_dct(self, P: int, device, dtype):
        key = str(P)
        if key not in self._dct_by_P:
            self._dct_by_P[key] = _OrthoDCT(P).to(device=device, dtype=dtype)
        else:
            self._dct_by_P[key] = self._dct_by_P[key].to(device=device, dtype=dtype)
        return self._dct_by_P[key]

    def _choose_K(self, P: int) -> int:
        if self.keep_k is not None:
            return max(1, min(self.keep_k, P))
        # default: keep first ~25% of frequencies, but at least 3
        return max(3, min(P, int(math.ceil(P * self.keep_ratio))))

    def _metrics(self, x: torch.Tensor, y: torch.Tensor):
        """
        x,y: [BV, P, D]  -> scalars per row [BV] : e, lips_n, curv_n, mag
        """
        res = x - y
        e = (res.pow(2).mean(dim=(1, 2))) / (x.pow(2).mean(dim=(1, 2)) + self.eps)

        # First diffs on smoothed y along P
        if y.size(1) >= 2:
            d1 = y[:, 1:, :] - y[:, :-1, :]                                   # [BV,P-1,D]
            lips = d1.norm(dim=2).mean(dim=1)                                  # [BV]
        else:
            lips = torch.zeros_like(e)

        # Second diffs on smoothed y along P
        if y.size(1) >= 3:
            d2 = d1[:, 1:, :] - d1[:, :-1, :]                                  # [BV,P-2,D]
            curv = d2.norm(dim=2).mean(dim=1)                                  # [BV]
        else:
            curv = torch.zeros_like(e)

        mag = y.pow(2).mean(dim=(1, 2)).sqrt()                                 # [BV]
        lips_n = lips / (mag + self.eps)
        curv_n = curv / (mag + self.eps)
        return e, lips_n, curv_n, mag

    def forward(self, branch_outputs_list):
        """
        branch_outputs_list: list length N, each [B, V, P_j, D]
        returns weights: [B, V, N]
        """
        B, V = branch_outputs_list[0].shape[:2]
        feats_list = []
        for b_out in branch_outputs_list:
            _, _, P, D = b_out.shape
            x = b_out.reshape(B * V, P, D).detach()                             # stop-grad
            dct = self._get_dct(P, x.device, x.dtype)
            K = self._choose_K(P)
            y = dct(x, keep_k=K)                                                # [BV,P,D]

            e, lips_n, curv_n, mag = self._metrics(x, y)
            logP = x.new_full((x.size(0),), float(math.log(max(P, 1))))
            feats = torch.stack([e, lips_n, curv_n, mag, logP], dim=-1)         # [BV,5]
            feats_list.append(feats)

        feats = torch.stack(feats_list, dim=1)                                  # [BV,N,5]
        scores = self.scorer(feats).squeeze(-1)                                  # [BV,N]

        # Optional Top-K sparsification
        if self.topk is not None and self.topk < self.n_branches:
            topk_vals, topk_idx = torch.topk(scores, k=self.topk, dim=1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        w = torch.softmax(scores / self.temperature, dim=1)                      # [BV,N]
        return w.view(B, V, self.n_branches)
# --- end DCT router -----------------------------------------------------------


# --- Savitzky–Golay scale router (smooth ONLY along P) -----------------------
import math
import torch
from torch import nn
import torch.nn.functional as F

class SavitzkyGolay1D(nn.Module):
    """
    Fixed Savitzky–Golay smoother along the sequence axis (P).
    Applies the same central SG kernel to every position using reflect padding.
    Input:  [N, D, P]   (channels=D, length=P)
    Output: [N, D, P]
    """
    def __init__(self, window: int = 7, polyorder: int = 3):
        super().__init__()
        assert window % 2 == 1, "SG window must be odd"
        assert polyorder < window, "polyorder must be < window"
        self.base_window = window
        self.polyorder = polyorder
        self._kern_cache = {}  # {k: 1D kernel tensor of length k}

    def _kernel(self, k: int, device, dtype):
        if k not in self._kern_cache:
            m = (k - 1) // 2
            t = torch.arange(-m, m + 1, device=device, dtype=dtype)           # [-m..m]
            A = torch.stack([t ** j for j in range(self.polyorder + 1)], 1)   # [k, r+1]
            # coefficients c = (A^T A)^{-1} A^T y, smoothed value at center x=0 is c0
            AtA = A.transpose(0, 1) @ A
            pinv = torch.linalg.pinv(AtA) @ A.transpose(0, 1)                 # [(r+1) x k]
            w = pinv[0]                                                       # [k] weights for center
            w = w / (w.sum() + 1e-12)                                         # unity DC gain
            self._kern_cache[k] = w.detach().cpu()
        return self._kern_cache[k].to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D, P]
        N, D, P = x.shape
        k = min(self.base_window, P if P % 2 == 1 else P - 1)
        if k < 3 or self.polyorder >= k:
            return x  # too short to smooth; return input
        w = self._kernel(k, x.device, x.dtype)          # [k]
        pad = (k - 1) // 2
        xpad = F.pad(x, (pad, pad), mode="reflect")     # pad along P
        weight = w.view(1, 1, k).repeat(D, 1, 1)        # [D, 1, k] per-channel
        y = F.conv1d(xpad, weight, groups=D)            # [N, D, P]
        return y

class SGSavGolRouterFusionHead(nn.Module):
    """
    Savitzky–Golay probe router (SCI1 variant).
    - Looks at each branch tensor [B,V,P,D]
    - Smooths ONLY along P via fixed SG (no learning), per channel
    - Features: normalized residual, normalized first/second diffs, magnitude, log(P)
    - Returns soft (optionally Top-K) weights: [B, V, N]
    """
    def __init__(self, d_model: int, n_branches: int,
                 window: int = 7, polyorder: int = 3,
                 topk: int | None = 2, hidden: int = 64,
                 eps: float = 1e-6, temperature: float = 1.0):
        super().__init__()
        self.n_branches = n_branches
        self.sg = SavitzkyGolay1D(window=window, polyorder=polyorder)
        self.topk = topk
        self.eps = eps
        self.temperature = temperature
        # [residual, lips_n, curv_n, mag, logP] -> score
        self.scorer = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _metrics(self, x_bvpd: torch.Tensor, y_bvpd: torch.Tensor):
        """
        x,y: [BV, P, D]  (BV = B*V)
        Returns per-row scalars: e, lips_n, curv_n, mag  each [BV]
        """
        res = x_bvpd - y_bvpd
        e = (res.pow(2).mean(dim=(1, 2))) / (x_bvpd.pow(2).mean(dim=(1, 2)) + self.eps)

        # first differences on smoothed y along P
        if y_bvpd.size(1) >= 2:
            d1 = y_bvpd[:, 1:, :] - y_bvpd[:, :-1, :]        # [BV, P-1, D]
            lips = d1.norm(dim=2).mean(dim=1)                # [BV]
        else:
            lips = torch.zeros_like(e)

        # second differences on smoothed y along P
        if y_bvpd.size(1) >= 3:
            d2 = d1[:, 1:, :] - d1[:, :-1, :]                # [BV, P-2, D]
            curv = d2.norm(dim=2).mean(dim=1)                # [BV]
        else:
            curv = torch.zeros_like(e)

        mag = y_bvpd.pow(2).mean(dim=(1, 2)).sqrt()          # [BV]
        lips_n = lips / (mag + self.eps)
        curv_n = curv / (mag + self.eps)
        return e, lips_n, curv_n, mag

    def forward(self, branch_outputs_list):
        """
        branch_outputs_list: list of length N, each tensor [B, V, P_j, D]
        returns fusion weights: [B, V, N]
        """
        B, V = branch_outputs_list[0].shape[:2]
        feats_per_branch = []
        for b_out in branch_outputs_list:
            _, _, P, D = b_out.shape
            # Flatten BV, detach so the probe never backprops into the backbone
            x = b_out.reshape(B * V, P, D).detach()          # [BV, P, D]
            # SG smoothing along P (per channel)
            x_chfirst = x.transpose(1, 2)                    # [BV, D, P]
            y_chfirst = self.sg(x_chfirst)                   # [BV, D, P]
            y = y_chfirst.transpose(1, 2)                    # [BV, P, D]

            e, lips_n, curv_n, mag = self._metrics(x, y)
            logP = x.new_full((x.size(0),), float(math.log(max(P, 1))))
            feats = torch.stack([e, lips_n, curv_n, mag, logP], dim=-1)  # [BV, 5]
            feats_per_branch.append(feats)

        feats = torch.stack(feats_per_branch, dim=1)         # [BV, N, 5]
        scores = self.scorer(feats).squeeze(-1)              # [BV, N]

        # optional Top-K sparsification
        if self.topk is not None and self.topk < self.n_branches:
            topk_vals, topk_idx = torch.topk(scores, k=self.topk, dim=1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        w = torch.softmax(scores / self.temperature, dim=1)  # [BV, N]
        return w.view(B, V, self.n_branches)
# --- end SG router ------------------------------------------------------------


class KANRouterFusionHead(nn.Module):
    def __init__(self, d_model: int, n_branches: int, topk: int = None, hidden: int = 64, freeze_kan: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches
        self.topk = topk
        self.freeze_kan = freeze_kan
        self.kan_by_P = nn.ModuleDict()
        self.scorer = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _get_kan(self, P: int, device, dtype):
        key = str(P)
        if key not in self.kan_by_P:
            mod = KanMixer(self.d_model, P).to(device=device, dtype=dtype)  # <<< move to correct device/dtype
            if self.freeze_kan:
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False
            self.kan_by_P[key] = mod
        else:
            # ensure the cached module is on the right device/dtype (important if created on CPU earlier)
            self.kan_by_P[key] = self.kan_by_P[key].to(device=device, dtype=dtype)
            if self.freeze_kan:
                self.kan_by_P[key].eval()
        return self.kan_by_P[key]

    @staticmethod
    def _feat_from_y(y: torch.Tensor):
        d1 = y[:, 1:, :] - y[:, :-1, :]
        lips = d1.norm(dim=-1).mean(dim=1)
        d2 = d1[:, 1:, :] - d1[:, :-1, :]
        curv = d2.norm(dim=-1).mean(dim=1) if d2.numel() > 0 else torch.zeros_like(lips)
        mag = y.norm(dim=-1).mean(dim=1)
        return lips, curv, mag

    def forward(self, branch_outputs_list):
        B, V = branch_outputs_list[0].shape[:2]
        feats_per_branch = []
        for b_out in branch_outputs_list:                 # each [B, V, P, D]
            _, _, P, D = b_out.shape
            x = b_out.reshape(B * V, P, D)
            kan = self._get_kan(P, device=x.device, dtype=x.dtype)  # <<< pass device/dtype
            with torch.set_grad_enabled(not self.freeze_kan):
                y = kan(x)                               # [B*V, P, D]
            res = x - y
            e = (res.pow(2).mean(dim=(1, 2))).clamp_min(1e-12)
            lips, curv, mag = self._feat_from_y(y)
            logP = torch.full_like(e, float(np.log(max(P, 1))))
            f = torch.stack([e, lips, curv, mag, logP], dim=-1)  # [B*V, 5]
            feats_per_branch.append(f)

        feats = torch.stack(feats_per_branch, dim=1)      # [B*V, N, 5]
        scores = self.scorer(feats).squeeze(-1)           # [B*V, N]
        if self.topk is not None and self.topk < self.n_branches:
            topk_vals, topk_idx = torch.topk(scores, k=self.topk, dim=1)
            mask = torch.full_like(scores, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask
        weights = torch.softmax(scores, dim=1).view(B, V, self.n_branches)
        return weights



# fusion_heads.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_shapes(branch_outputs: List[torch.Tensor]):
    assert len(branch_outputs) >= 2, "Need at least 2 branches"
    B, V, P, D = branch_outputs[0].shape
    for t in branch_outputs[1:]:
        assert t.shape == (B, V, P, D), \
            f"All branches must have same shape; got {t.shape} vs {(B,V,P,D)}"
    return B, V, P, D


# -----------------------------------------------------------------------------
# 0) Uniform average (param-free baseline)
# -----------------------------------------------------------------------------
class UniformAverageFusion(nn.Module):
    """T_fused = mean_n T^{(n)}"""
    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B, V, P, D = _check_shapes(branch_outputs)
        stack = torch.stack(branch_outputs, dim=2)            # [B,V,N,P,D]
        return stack.mean(dim=2).contiguous()                 # [B,V,P,D]


# -----------------------------------------------------------------------------
# 1) Token-wise branch-axis attention (MLP scorer per token)
#    This is a token-level version of your AttentionFusionHeadMLP.
# -----------------------------------------------------------------------------
class TokenWiseBranchAttentionMLP(nn.Module):
    """
    For each (b,v,p), score each branch from its local feature x_{n} in R^D,
    softmax over branches, and mix the branch features.
    """
    def __init__(self, d_model: int, n_branches: int, hidden: Optional[int] = None, p_drop: float = 0.0):
        super().__init__()
        self.n = n_branches
        h = d_model if hidden is None else hidden
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h, bias=True),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(h, 1, bias=True)
        )

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B, V, P, D = _check_shapes(branch_outputs)
        # [B,V,N,P,D]
        x = torch.stack(branch_outputs, dim=2).contiguous()
        # score each branch per token: flatten to [B*V*P*N, D] for the MLP
        scores = self.scorer(x.view(B*V*P*self.n, D)).view(B, V, P, self.n, 1)  # [B,V,P,N,1]
        w = F.softmax(scores, dim=3)                                            # [B,V,P,N,1]
        fused = (w * x).sum(dim=2)                                              # [B,V,P,D]
        return fused.contiguous()


# -----------------------------------------------------------------------------
# 2) Concat + 1x1 linear (token-wise feature fusion; keeps temporal alignment)
#    Equivalent to a learned mixing across branches per token.
# -----------------------------------------------------------------------------
class ConcatLinearFusion(nn.Module):
    """
    Concatenate features from all branches along channel axis and project back to D.
    """
    def __init__(self, d_model: int, n_branches: int, p_drop: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * n_branches),
            nn.Linear(d_model * n_branches, d_model, bias=True),
            nn.Dropout(p_drop)
        )

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B, V, P, D = _check_shapes(branch_outputs)
        # [B,V,P,N,D] -> [B,V,P,N*D]
        cat = torch.stack(branch_outputs, dim=3).contiguous().view(B, V, P, -1)
        out = self.proj(cat)                                                    # [B,V,P,D]
        return out.contiguous()


# -----------------------------------------------------------------------------
# 3) MTST-style flatten-concat-linear (classic "Fuse")
#    Flatten token+channel per branch, concat across branches, project back,
#    then reshape to [P,D]. Mirrors MTST's flatten→concat→linear idea.
# -----------------------------------------------------------------------------
class MTSTStyleFuse(nn.Module):
    """
    Fuse by flattening per-branch [P,D] → [P*D], concat across N, project back to [P*D].
    """
    def __init__(self, d_model: int, n_branches: int, n_patches: int, p_drop: float = 0.0):
        super().__init__()
        self.P, self.D, self.N = n_patches, d_model, n_branches
        in_dim  = n_branches * n_patches * d_model
        out_dim = n_patches   * d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=True),
            nn.Dropout(p_drop)
        )

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B, V, P, D = _check_shapes(branch_outputs)
        assert P == self.P and D == self.D, f"Init with P={self.P},D={self.D} but got P={P},D={D}"
        # stack: [B,V,N,P,D] -> flatten per branch: [B,V,N,P*D] -> concat across N: [B,V,N*P*D]
        stacked = torch.stack(branch_outputs, dim=2).contiguous().view(B, V, self.N, P*D)
        concat  = stacked.view(B, V, self.N * P * D)
        fused   = self.proj(concat)                                             # [B,V,P*D]
        return fused.view(B, V, P, D).contiguous()


# -----------------------------------------------------------------------------
# 4) Gumbel top-k (hard selection) with straight-through estimator
#    If k=1 and N=2, this behaves like hard pick between the two branches.
# -----------------------------------------------------------------------------
class GumbelTopKFusion(nn.Module):
    """
    Hard-select top-k branches per (b,v,p) using Gumbel-Softmax (straight-through),
    then mix the selected branches (default k=1). Temperature tau controls hardness.
    """
    def __init__(self, d_model: int, n_branches: int, hidden: Optional[int] = None, k: int = 1, tau: float = 1.0):
        super().__init__()
        self.n, self.k, self.tau = n_branches, k, tau
        h = d_model if hidden is None else hidden
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h), nn.GELU(),
            nn.Linear(h, 1)
        )

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B, V, P, D = _check_shapes(branch_outputs)
        x = torch.stack(branch_outputs, dim=2).contiguous()          # [B,V,N,P,D]
        logits = self.scorer(x.view(B*V*P*self.n, D)).view(B, V, P, self.n)  # [B,V,P,N]

        # Straight-through Gumbel top-k
        g = -torch.empty_like(logits).exponential_().log()           # Gumbel(0,1)
        y_soft = F.softmax((logits + g) / self.tau, dim=3)           # [B,V,P,N]
        if self.k == 1:
            idx = y_soft.argmax(dim=3, keepdim=True)                 # [B,V,P,1]
            y_hard = torch.zeros_like(y_soft).scatter_(3, idx, 1.0)  # one-hot
        else:
            topk = torch.topk(y_soft, k=self.k, dim=3)
            y_hard = torch.zeros_like(y_soft).scatter_(3, topk.indices, 1.0 / self.k)

        # straight-through
        y = (y_hard - y_soft).detach() + y_soft                      # [B,V,P,N]
        fused = (y.unsqueeze(-1) * x).sum(dim=2)                     # [B,V,P,D]
        return fused.contiguous()


class AttentionFusionHead(nn.Module):
    """
    IMPROVEMENT 2B: Attention module to fuse multi-scale branch outputs.
    """
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.n_branches = n_branches
        # *** EINSUM FIX ***: Define the query as a simple vector of shape [d_model].
        # The previous shape [1, 1, d_model] required unsupported '()' notation in einsum.
        self.attention_query = nn.Parameter(torch.randn(d_model))

    def forward(self, branch_outputs_list):
        # branch_outputs_list: A list of tensors, each of shape [bs, nvars, patch_num, d_model]
        pooled_outputs = [reduce(branch, 'b v p d -> b v d', 'mean') for branch in branch_outputs_list]
        stacked_pools = torch.stack(pooled_outputs, dim=2) # [bs, nvars, n_branches, d_model]

        # *** EINSUM FIX ***: Use a supported einsum pattern.
        # 'd, b v n d -> b v n' performs a dot product between the query (d) and
        # the last dimension of the stacked pools (d), resulting in the desired attention scores.
        attn_scores = einsum(self.attention_query, stacked_pools, 'd, b v n d -> b v n')
        attn_weights = F.softmax(attn_scores, dim=-1) # [bs, nvars, n_branches]
        return attn_weights

class RouterMixerLite(nn.Module):
    """
    Drop-in for HybridMixer in TSTEncoderLayer.
    Input/Output: [B, L, D] (B = bs*nvars). Prioritizes KAN; adds local-conv assist.
    """
    def __init__(self, d_model: int, patch_num: int):
        super().__init__()
        self.kan = KanMixer(d_model, patch_num)  # your Hahn-based KAN
        self.local = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # scalar logit per token
        )
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B,L,D]
        y_kan = self.kan(x)
        y_loc = self.local(x.transpose(1, 2)).transpose(1, 2)
        w = torch.sigmoid(self.router(x))  # [B,L,1]; higher -> prefer KAN
        return w * y_kan + (1 - w) * (x + self.beta * y_loc)



# class MTST_backbone(nn.Module):
#     def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
#                  n_layers:int=1, n_branches:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
#                  d_ff:int=256, norm:str='BatchNorm', xxx_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
#                  padding_var:Optional[int]=None, xxx_mask:Optional[Tensor]=None,  pre_norm:bool=False, store_xxx:bool=False,
#                  pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
#                  pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
#                  verbose:bool=False, res_attention: bool = True,
#                  cfg=CN(),
#                  **kwargs
#                  ):

#         super().__init__()

#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

#         # Patching
#         if isinstance(patch_len, str):
#             patch_len= patch_len.split(',')
#             patch_len= [int(i) for i in patch_len]
#         if isinstance(stride, str):
#             stride = stride.split(',')
#             stride = [int(i) for i in stride]

#         patch_num = [int((context_window - patch_len[j]) / stride[j] + 1) for j in range(n_branches)]
#         if padding_patch == 'end':
#             patch_num =[p_n + 1 for p_n in patch_num]

#         # Backbone
#         self.backbone = TSTiEncoder(learn_pe, pe, c_in, patch_num=patch_num, patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, padding_patch=padding_patch,
#                                 n_layers=n_layers, n_branches=n_branches, d_model=d_model, d_ff=d_ff,
#                                 xxx_dropout=xxx_dropout, dropout=dropout, act=act, padding_var=padding_var, pre_norm=pre_norm,
#                                cfg=cfg, **kwargs)

#         # Head
#         self.head_nf = [d_model * p_n for p_n in patch_num] # to be modified
#         self.n_vars = c_in
#         self.pretrain_head = pretrain_head
#         self.head_type = head_type
#         self.head_dropout = head_dropout
#         self.individual = individual
#         self.target_window = target_window
#         self.n_layers = n_layers
#         self.n_branches = n_branches


#         if self.pretrain_head:
#             self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
#         elif head_type == 'flatten':
#             self.heads = nn.ModuleList()
#             for i in range(self.n_branches):
#                 self.heads.append(Flatten_Head(self.individual, self.n_vars, self.head_nf[i], target_window, head_dropout=head_dropout))


#     def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
#         # norm
#         if self.revin:
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'norm')
#             z = z.permute(0,2,1)

#         # do patching in each layer
#         # ------- Encoder ------
#         z, xxx = self.backbone(z)                                                      # z: [bs x nvars x seq_len]

#         draw_list = [self.heads[i](z[i]) for i in range(len(z))]  # 3 branches of the last layer to diff linear layer
#         z = torch.stack(draw_list, dim=-1).sum(dim=-1, keepdim = False)

#         # denorm
#         if self.revin:
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'denorm')
#             draw_list = [self.revin_layer(z_.permute(0,2,1), 'denorm') for z_ in draw_list]
#             # draw_list = [z_.permute(0,2,1) for z_ in draw_list]
#             z = z.permute(0,2,1)
#         return z, draw_list, xxx

#     def create_pretrain_head(self, head_nf, vars, dropout):
#         return nn.Sequential(nn.Dropout(dropout),
#                     nn.Conv1d(head_nf, vars, 1)
#                     )

class SeasonalTrendDecomp(nn.Module):
    """Low-pass via orthonormal DCT on the time axis (seq_len); seasonal = x - trend."""
    def __init__(self, seq_len: int, keep_ratio: float = 0.25, keep_k: int | None = None):
        super().__init__()
        self.seq_len = seq_len
        self.keep_ratio = keep_ratio
        self.keep_k = keep_k
        self._dct = _OrthoDCT(seq_len)  # reuse your existing DCT

    def forward(self, x):  # x: [B, N_vars, L]
        B, N, L = x.shape
        assert L == self.seq_len, f"got L={L}, expected {self.seq_len}"
        # reshape to match [N, P, D] = [B*N, L, 1]
        xn = x.reshape(B * N, L, 1)
        K = self.keep_k if self.keep_k is not None else max(3, int(round(L * self.keep_ratio)))
        trend = self._dct(xn, keep_k=K).reshape(B, N, L)
        seasonal = x - trend
        return seasonal, trend


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Cell

import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrAwareMixer(nn.Module):
    """
    Turns variable mixing ON only when inter-var correlation is high.
    - Hard threshold + EMA stabilizer
    - Identity at init; bounded effect when ON
    """
    def __init__(self, base_mixer: nn.Module, n_vars: int,
                 tau: float = 0.25,          # correlation threshold
                 ema: float = 0.9,           # stability for the gate
                 max_gate: float = 0.8):     # cap the effect even when ON
        super().__init__()
        self.mixer = base_mixer
        self.tau = tau
        self.ema = ema
        self.max_gate = max_gate
        self.register_buffer("s_ema", torch.tensor(0.0))

    @torch.no_grad()
    def _corr_strength(self, x):  # x: [B,N,P,D]
        B,N,P,D = x.shape
        u = x.permute(0,1,3,2).reshape(B,N,P*D)
        u = (u - u.mean(-1,keepdim=True)) / (u.std(-1,keepdim=True)+1e-6)
        C = torch.einsum("bni,bmi->bnm", u, u) / u.size(-1)     # [B,N,N]
        C = C.mean(0)
        off = C - torch.diag_embed(torch.diagonal(C))
        return off.abs().mean().clamp(0,1)

    def forward(self, x):         # x: [B,N,P,D]
        with torch.no_grad():
            s = self._corr_strength(x)
            self.s_ema = self.ema * self.s_ema + (1 - self.ema) * s
            gate = (self.s_ema - self.tau).clamp(min=0) / (1 - self.tau + 1e-6)
            gate = gate.clamp(0, self.max_gate)                # [0, max_gate]

        y = self.mixer(x)                                      # same shape
        return x + gate * (y - x)                              # identity when gate==0


class VariableMixerAdaptive(nn.Module):
    """
    Adaptive variant of your working VariableMixer.
    Mixes across variables (N) and scales the residual by a data-driven gate.

    Input/Output: x in [B, N, P, D] -> [B, N, P, D]
    """
    def __init__(self, d_model: int, n_vars: int,
                 dropout: float = 0.1,
                 ema_decay: float = 0.9,    # for dataset-level smoothing of the gate
                 base_init: float = 0.2,    # initial learnable base strength (alpha)
                 gate_temp: float = 1.0,    # >1 makes the gate respond more aggressively
                 max_gate: float = 0.8):    # cap to avoid over-mixing
        super().__init__()
        self.n_vars = n_vars
        self.norm = nn.LayerNorm(d_model)          # norm over D
        self.mix_vars = nn.Linear(n_vars, n_vars, bias=False)  # your original
        self.dropout = nn.Dropout(dropout)

        # Learnable base (like your alpha), but we’ll multiply it by a data gate
        self.alpha = nn.Parameter(torch.tensor(base_init))

        # Gate controls
        self.ema_decay = ema_decay
        self.gate_temp = gate_temp
        self.max_gate = max_gate

        # Running EMA of "correlation strength" (dataset-level adaptivity)
        self.register_buffer("gate_ema", torch.tensor(0.0))

        # Small init so the mixer starts near-identity
        nn.init.normal_(self.mix_vars.weight, mean=0.0, std=1e-3)

    @torch.no_grad()
    def _corr_strength(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate inter-variable dependency from the batch.
        We standardize over time/features and compute mean |off-diagonal correlation|.
        Returns scalar s in [0, 1] (clipped).
        x: [B, N, P, D]
        """
        B, N, P, D = x.shape
        # Flatten time/features to a single axis and z-score per variable
        u = x.permute(0, 1, 3, 2).reshape(B, N, P * D)          # [B, N, PD]
        u = u - u.mean(dim=-1, keepdim=True)
        u = u / (u.std(dim=-1, keepdim=True) + 1e-6)

        # Batch-averaged correlation matrix across variables
        # cov ≈ (u @ u^T) / (PD); diagonal ~ 1 after z-scoring
        C = torch.einsum("bni,bmi->bnm", u, u) / u.size(-1)     # [B, N, N]
        C = C.mean(dim=0)                                       # average over batch -> [N, N]

        # Mean absolute off-diagonal correlation
        off = C - torch.diag_embed(torch.diagonal(C, dim1=0, dim2=1))
        s = off.abs().mean()                                    # scalar
        return s.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D]
        B, N, P, D = x.shape
        assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"

        # 1) Compute/update dataset-level dependency gate (no grad)
        with torch.no_grad():
            s = self._corr_strength(x)                          # instantaneous score
            self.gate_ema = self.ema_decay * self.gate_ema + (1 - self.ema_decay) * s
            # temperature to accentuate high-correlation datasets (Weather) and damp low ones (ETT)
            s_eff = (self.gate_ema ** self.gate_temp).clamp(0.0, 1.0)

        # 2) Your original mixing over N (with pre-norm over D)
        z = self.norm(x)                                        # [B, N, P, D]
        z_perm = z.permute(0, 2, 3, 1)                          # [B, P, D, N]
        y = self.mix_vars(z_perm)                               # [B, P, D, N]
        y = y.permute(0, 3, 1, 2)                               # [B, N, P, D]

        # 3) Residual scaled by (learnable alpha) * (data gate)
        gate = (self.alpha.abs() * s_eff).clamp(max=self.max_gate)
        out = x + gate * self.dropout(y - z)                    # mix the deviation from identity
        return out



class SafeVariableMixer(nn.Module):
    """
    [B,N,P,D] -> [B,N,P,D], mixes across N but starts as identity.
    """
    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.1, alpha_init: float = 0.0):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        # deviation from identity: W = I + epsilon * (Q @ P)
        rank = min(16, n_vars)
        self.P = nn.Linear(n_vars, rank, bias=False)   # N -> r
        self.Q = nn.Linear(rank, n_vars, bias=False)   # r -> N
        nn.init.zeros_(self.P.weight)  # start with zero deviation
        nn.init.zeros_(self.Q.weight)
        self.dropout = nn.Dropout(dropout)
        self.alpha   = nn.Parameter(torch.tensor(alpha_init))  # starts at 0 → identity
        self.register_buffer('I', torch.eye(n_vars))           # identity buffer

    def forward(self, x):  # x: [B,N,P,D]
        B,N,P,D = x.shape
        z = self.norm(x)
        y = z.permute(0,2,3,1)               # [B,P,D,N]
        dev = self.Q(self.P(y))              # [B,P,D,N], small at init
        y  = y + self.alpha.tanh() * dev     # I + gate*dev
        y  = y.permute(0,3,1,2)              # [B,N,P,D]
        return x + self.dropout(y - z)       # residual

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableMixerID(nn.Module):
    """
    Mix across the variables axis N at every (patch, feature) location.
    Starts at (near) identity so it won't hurt ETT, but can learn deviation for Weather.
    Input/Output: [B, N, P, D]
    """
    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.1, alpha_init: float = 0.2):
        super().__init__()
        self.n_vars = n_vars
        self.norm = nn.LayerNorm(d_model)           # norm over D (stable)
        self.dropout = nn.Dropout(dropout)
        # Parameterize deviation from identity: W = I + E, with E small at init
        self.E = nn.Parameter(torch.zeros(n_vars, n_vars))   # learnable deviation
        # small residual gate; you can cosine-warm it from 0 → alpha_init over a few epochs
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Optional: tiny random init to let it move if needed
        nn.init.normal_(self.E, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D]
        B, N, P, D = x.shape
        assert N == self.n_vars, f"Expected N={self.n_vars}, got {N}"

        z = self.norm(x)                        # [B, N, P, D]
        Z = z.permute(0, 2, 3, 1)               # [B, P, D, N]

        # W = I + E, act along N
        I = torch.eye(self.n_vars, device=Z.device, dtype=Z.dtype)
        W = I + self.E                          # [N, N]

        Y = torch.matmul(Z, W)                  # [B, P, D, N]
        Y = Y.permute(0, 3, 1, 2)               # [B, N, P, D]

        # Mix only the deviation from identity: (Y - z)
        out = x + self.alpha * self.dropout(Y - z)
        return out


class VariableMixer(nn.Module):
    """
    Mixes across the variable/channel axis (N) with a shared linear over N
    applied at every (patch, feature). Shape preserved: [B, N, P, D] -> [B, N, P, D].
    """
    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.1):
        super().__init__()
        # project over the variables dimension N (last dim for nn.Linear)
        self.mix_vars = nn.Linear(n_vars, n_vars, bias=False)
        # pre/post norm for stability (norm over feature dim D, Transformer-style)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # small residual gate to avoid swamping early training
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D]
        B, N, P, D = x.shape

        # Pre-norm on features (stable even though we mix over N)
        x_norm = self.norm(x)  # LayerNorm over last dim D

        # Move N to the last position so Linear(N->N) can act on it
        x_perm = x_norm.permute(0, 2, 3, 1)  # [B, P, D, N]
        y = self.mix_vars(x_perm)            # mix across variables
        y = y.permute(0, 3, 1, 2)            # back to [B, N, P, D]

        return x + self.alpha * self.dropout(y)

class VariableMixerLR(nn.Module):
    def __init__(self, d_model, n_vars, rank=16, dropout=0.1, alpha=0.2):
        super().__init__()
        r = min(rank, n_vars)
        self.P = nn.Linear(n_vars, r, bias=False)   # N -> r
        self.Q = nn.Linear(r, n_vars, bias=False)   # r -> N
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        # start near identity
        nn.init.normal_(self.P.weight, std=1e-3)
        nn.init.normal_(self.Q.weight, std=1e-3)

    def forward(self, x):  # x: [B,N,P,D]
        B,N,P,D = x.shape
        xN = self.norm(x)
        y = xN.permute(0,2,3,1)          # [B,P,D,N]
        y = self.Q(self.P(y))            # mix over N
        y = y.permute(0,3,1,2)           # [B,N,P,D]
        return x + self.alpha * self.dropout(y)

# class VariableMixer(nn.Module):
#     def __init__(self, d_model, n_vars, num_patch):
#         super().__init__()
#         # Use a KAN layer to mix information across the variable dimension
#         # self.var_kan_mixer = HahnPolynomials(n_vars, n_vars, 3, 1, 1, 7)
#         self.var_kan_mixer = HermiteKAN(n_vars, n_vars)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         # x shape: [bs, nvars, num_patch, d_model]
        
#         # Permute to bring variables to the mixing dimension: [bs, num_patch, d_model, n_vars]
#         x_permuted = x.permute(0, 2, 3, 1)
        
#         # Apply KAN mixing across the n_vars dimension
#         x_mixed = self.var_kan_mixer(x_permuted)
        
#         # Permute back to original layout: [bs, nvars, num_patch, d_model]
#         x_out = x_mixed.permute(0, 3, 1, 2)
        
#         # Add residual connection and norm
#         return self.norm(x + x_out)

class CalendarFourier(nn.Module):
    def __init__(self, seq_len:int, steps_per_day:int=24, steps_per_week:int=168,
                 per_variable_gate:bool=True, n_vars:int=None, init_gate:float=0.1):
        super().__init__()
        self.L  = int(seq_len)
        self.SD = int(steps_per_day)
        self.SW = int(steps_per_week)
        gate_shape = (1, n_vars, 1, 4) if (per_variable_gate and n_vars is not None) else (1, 1, 1, 4)

        inv = lambda p: math.log(p/(1-p))
        self.gate = nn.Parameter(torch.full(gate_shape, inv(init_gate), dtype=torch.float32))

        t = torch.arange(self.L, dtype=torch.float32).unsqueeze(-1)   # [L,1]
        h = 2*math.pi * t / max(self.SD, 1)
        w = 2*math.pi * t / max(self.SW, 1)
        base = torch.cat([torch.sin(h), torch.cos(h), torch.sin(w), torch.cos(w)], dim=-1)  # [L,4]
        self.register_buffer("base", base)

    def _pooled_calendar(self, P:int, patch_len:int, stride:int, device, dtype):
        """
        Average sin/cos over each patch window using *actual* P.
        Handles padding_patch='end' by clamping indices to L-1.
        Returns [P,4].
        """
        L = self.base.size(0)
        starts = torch.arange(P, device=device) * stride                         # [P]
        offs   = torch.arange(patch_len, device=device)                          # [patch_len]
        idx    = starts.unsqueeze(1) + offs.unsqueeze(0)                         # [P,patch_len]
        idx    = torch.clamp(idx, max=L-1)                                       # reflect/replicate tail
        pooled = self.base.index_select(0, idx.reshape(-1))                      # [P*patch_len,4]
        pooled = pooled.view(P, patch_len, 4).mean(dim=1)                        # [P,4]
        return pooled.to(device=device, dtype=dtype)

    def make_feats(self, P:int, B:int, N:int, patch_len:int, stride:int, device, dtype):
        """
        Build gated calendar features aligned to current P.
        Returns [B,N,P,4].
        """
        pooled = self._pooled_calendar(P, patch_len, stride, device, dtype)      # [P,4]
        feats  = pooled.unsqueeze(0).unsqueeze(0).expand(B, N, P, 4)             # [B,N,P,4]
        g = torch.sigmoid(self.gate.to(device=device, dtype=dtype))              # [1,(N),1,4]
        if g.size(1) == 1 and N > 1:                                             # broadcast if not per-var
            g = g.expand(1, N, 1, 4)
        return feats * g

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: Assuming KanMixer and ConvMixer classes are already defined
# in your MTST_backbone.py file.
# from .mixers import KanMixer, ConvMixer

class HybridMixer(nn.Module):
    def __init__(self, d_model, patch_num):
        super().__init__()
        
        # Initialize both mixers.
        # These are the two branches the model can choose from.
        self.kan_mixer = KanMixer(d_model, patch_num)
        self.conv_mixer = ConvMixer(d_model)
        
        # This linear layer acts as our "router" or "gate".
        # It takes the input features and outputs a logit for each mixer.
        # The number of outputs is equal to the number of mixers (2).
        self.router = nn.Linear(d_model, 2)
        
        # A simple placeholder for the temperature parameter.
        # This should be annealed during training (see explanation below).
        # self.temperature = temp #1.0 

    def forward(self, x, temp = 0.1):
        
        # 1. Pass the input through each mixer branch.
        kan_out = self.kan_mixer(x)
        conv_out = self.conv_mixer(x)
        
        # 2. Get the logits for routing.
        # We take a mean over the sequence dimension to get a single logit vector.
        logits = self.router(x.mean(dim=1))
        
        # 3. Apply the Gumbel-Softmax trick.
        # This creates a one-hot-like selection vector.
        # A low temperature pushes the distribution towards a hard selection.
        selection_vector = F.gumbel_softmax(logits, tau=temp, hard=True)
        
        # 4. Use the selection vector to perform a "hard" routing.
        # The first element of selection_vector will gate the kan_out branch,
        # and the second will gate the conv_out branch.
        # For a hard selection, one will be 1 and the other 0.
        final_output = (selection_vector[:, 0].unsqueeze(1).unsqueeze(1) * kan_out +
                        selection_vector[:, 1].unsqueeze(1).unsqueeze(1) * conv_out)

        return final_output

# # --- Replace KanMixer with this HybridMixer in TSTEncoderLayer ---
# class HybridMixer(nn.Module):
#     def __init__(self, d_model, patch_num):
#         super().__init__()
#         # self.kan_mixer = KanMixer(d_model, patch_num)
#         # self.rel_pos_conv = RelPosConv(d_model)
#         self.conv_mixer = ConvMixer(d_model)
#         # A simple learnable gate to combine the two mixer outputs
#         # self.gate = nn.Parameter(torch.zeros(1, 1, d_model))

#     def forward(self, x):
#         # kan_out = self.kan_mixer(x)
#         # kan_out = self.rel_pos_conv(kan_out)
#         # return kan_out
#         conv_out = self.conv_mixer(x)
#         return conv_out
#         # # Use a sigmoid gate to blend the two outputs
#         # g = torch.sigmoid(self.gate)
#         # return g * kan_out + (1 - g) * conv_out
