# Shape losses used standalone: TILDE-Q, Soft-DTW, DILATE
import torch
import torch.nn as nn


# TILDE-Q (https://github.com/HyunWookL/TILDE-Q)
def amp_loss(outputs, targets):
    _, _, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim=-1)
    out_norm = torch.norm(outputs, dim=-1, keepdim=True)
    tgt_norm = torch.norm(targets, dim=-1, keepdim=True)

    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(T - 1):], auto_corr[..., :T]], dim=-1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim=-1).real
    cross_corr = torch.cat([cross_corr[..., -(T - 1):], cross_corr[..., :T]], dim=-1)
    nac_out = cross_corr / (tgt_norm * out_norm)
    return torch.mean(torch.abs(nac_tgt - nac_out))


def ashift_loss(outputs, targets):
    _, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim=-1)))


def phase_loss(outputs, targets):
    _, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim=-1)
    tgt_fourier = torch.fft.fft(targets, dim=-1)
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    mask = (tgt_fourier_sq > T).float()
    topk_indices = tgt_fourier_sq.topk(k=int(T ** 0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[..., 0] = 1.
    mask = torch.where(mask > 0, 1., 0.).bool()

    not_mask = (~mask).float()
    not_mask = not_mask / torch.mean(not_mask)
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)

    mask = mask.float()
    mask = mask / torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    return (torch.mean(zero_error) + torch.mean(ae)) / (T ** .5)


class TILDEQLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, true):
        outputs, targets = pred.permute(0, 2, 1), true.permute(0, 2, 1)
        loss = self.alpha * ashift_loss(outputs, targets) + (1 - self.alpha) * phase_loss(outputs, targets)
        if self.gamma:
            loss = loss + self.gamma * amp_loss(outputs, targets)
        return loss


# Soft-DTW, computed over anti-diagonals of the alignment lattice
def pairwise_sq_dists(x, y):
    x_sq = (x * x).sum(-1).unsqueeze(2)
    y_sq = (y * y).sum(-1).unsqueeze(1)
    return (x_sq + y_sq - 2.0 * torch.bmm(x, y.transpose(1, 2))).clamp_min(0.0)


def _gather_diag(diag, diag_lo, idx):
    pos = idx - diag_lo
    valid = (pos >= 0) & (pos < diag.shape[1])
    out = diag[:, pos.clamp(0, max(diag.shape[1] - 1, 0))]
    return torch.where(valid.unsqueeze(0), out, torch.full_like(out, float('inf')))


def soft_dtw(D, gamma=1.0):                                                   # D: [B x N x M]
    B, N, M = D.shape
    device, dtype = D.device, D.dtype
    prev2, prev2_lo = torch.zeros(B, 1, device=device, dtype=dtype), 0
    lo1, hi1 = max(0, 1 - M), min(N, 1)
    prev, prev_lo = torch.full((B, hi1 - lo1 + 1), float('inf'), device=device, dtype=dtype), lo1

    for k in range(2, N + M + 1):
        lo, hi = max(0, k - M), min(N, k)
        i = torch.arange(lo, hi + 1, device=device)
        j = k - i
        cands = torch.stack([_gather_diag(prev2, prev2_lo, i - 1),
                             _gather_diag(prev, prev_lo, i - 1),
                             _gather_diag(prev, prev_lo, i)], dim=0)
        if gamma > 0:
            finite = torch.isfinite(cands)
            m = torch.where(finite, cands, torch.full_like(cands, float('inf'))).amin(0)
            z = torch.where(finite, torch.exp(-(cands - m) / gamma), torch.zeros_like(cands))
            softmin = m - gamma * torch.log(z.sum(0))
        else:
            softmin = cands.amin(0)
        cost = D[:, (i - 1).clamp_min(0), (j - 1).clamp_min(0)]
        cur = torch.where(((i >= 1) & (j >= 1)).unsqueeze(0), cost + softmin, torch.full_like(cost, float('inf')))
        prev2, prev2_lo = prev, prev_lo
        prev, prev_lo = cur, lo
    return prev[:, -1]


class SoftDTWLoss(nn.Module):
    def __init__(self, gamma=0.01):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, true):
        D = pairwise_sq_dists(true, pred)
        return soft_dtw(D, self.gamma).mean() / pred.shape[1]


# DILATE (https://github.com/vincent-leguen/DILATE)
class DILATELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, true):
        B, T, _ = pred.shape
        D = pairwise_sq_dists(true, pred)
        per_sample = soft_dtw(D, self.gamma)
        loss_shape = per_sample.mean()
        # soft alignment path = d softDTW / dD
        if D.requires_grad:
            path = torch.autograd.grad(per_sample.sum(), D, retain_graph=True, create_graph=True)[0]
        else:
            D_path = D.detach().requires_grad_(True)
            with torch.enable_grad():
                path = torch.autograd.grad(soft_dtw(D_path, self.gamma).sum(), D_path)[0].detach()
        idx = torch.arange(1, T + 1, device=pred.device, dtype=pred.dtype).view(1, T, 1)
        loss_temporal = (path * pairwise_sq_dists(idx, idx)).sum() / (T * T * B)
        return (self.alpha * loss_shape + (1 - self.alpha) * loss_temporal) / T
