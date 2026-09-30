import torch
import torch.nn as nn

from utils.pca_basis import pca_torch


def base_loss(name):
    if name == 'huber':
        return nn.HuberLoss(delta=0.5)
    if name == 'mae':
        return nn.L1Loss()
    if name == 'mse':
        return nn.MSELoss()
    raise ValueError('Unknown loss_base: {}'.format(name))


# DBLoss (https://github.com/decisionintelligence/DBLoss)
class EMA(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class DBLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.ema = EMA(alpha)
        self.beta = beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        pred_trend, target_trend = self.ema(pred), self.ema(target)
        season_loss = self.mse(pred - pred_trend, target - target_trend)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss


# FreDF (https://github.com/Master-PLC/FreDF)
class FreDFLoss(nn.Module):
    def __init__(self, rec_lambda=0.2, auxi_lambda=0.8, base='mse'):
        super().__init__()
        self.rec_lambda = rec_lambda
        self.auxi_lambda = auxi_lambda
        self.base_loss = base_loss(base)

    def forward(self, pred, true):
        loss = self.rec_lambda * self.base_loss(pred, true)
        diff = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(true, dim=1)
        return loss + self.auxi_lambda * diff.abs().mean()


# TransDF / Time-o1 (https://github.com/Master-PLC/Time-o1)
class TransDFLoss(nn.Module):
    def __init__(self, pca_cache, rec_lambda=0.2, auxi_lambda=0.8, base='mse'):
        super().__init__()
        self.pca_cache = pca_cache
        self.rec_lambda = rec_lambda
        self.auxi_lambda = auxi_lambda
        self.base_loss = base_loss(base)

    def forward(self, pred, true):
        loss = self.rec_lambda * self.base_loss(pred, true)
        diff = pca_torch(pred, self.pca_cache) - pca_torch(true, self.pca_cache)
        return loss + self.auxi_lambda * diff.abs().mean()


# PSLoss (https://github.com/Dilfiraa/PS_Loss)
class PSLoss(nn.Module):
    def __init__(self, model, ps_lambda=3.0, patch_len_threshold=24, base='mse'):
        super().__init__()
        self.model = [model]
        self.ps_lambda = ps_lambda
        self.patch_len_threshold = patch_len_threshold
        self.base_loss = base_loss(base)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def head_params(self):
        model = self.model[0]
        model = getattr(model, 'module', model)
        return [p for p in model.model.heads.parameters() if p.requires_grad]

    @staticmethod
    def create_patches(x, patch_len, stride):
        x = x.permute(0, 2, 1)
        B, C, L = x.shape
        num_patches = (L - patch_len) // stride + 1
        return x.unfold(2, patch_len, stride).reshape(B, C, num_patches, patch_len)

    def fourier_based_adaptive_patching(self, true, pred):
        frequency_list = torch.abs(torch.fft.rfft(true, dim=1)).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = true.shape[1] // top_index
        patch_len = int(max(min(period // 2, self.patch_len_threshold), 2))
        stride = max(patch_len // 2, 1)
        return self.create_patches(true, patch_len, stride), self.create_patches(pred, patch_len, stride)

    def patch_wise_structural_loss(self, true_patch, pred_patch):
        true_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        true_std = torch.sqrt(torch.var(true_patch, dim=-1, keepdim=True, unbiased=False))
        pred_std = torch.sqrt(torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False))
        cov = torch.mean((true_patch - true_mean) * (pred_patch - pred_mean), dim=-1, keepdim=True)

        corr_loss = (1.0 - (cov + 1e-5) / (true_std * pred_std + 1e-5)).mean()
        var_loss = self.kl_loss(torch.log_softmax(pred_patch, dim=-1),
                                torch.softmax(true_patch, dim=-1)).sum(dim=-1).mean()
        mean_loss = torch.abs(true_mean - pred_mean).mean()
        return corr_loss, var_loss, mean_loss

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = ((cov + 1e-5) / (true_std * pred_std + 1e-5) + 1.0) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        params = self.head_params()

        def grads(term):
            g = torch.autograd.grad(term, params, create_graph=True, allow_unused=True)
            return [torch.zeros_like(p) if gi is None else gi for gi, p in zip(g, params)]

        def norm(g):
            return torch.sqrt(sum((gi ** 2).sum() for gi in g))

        corr_grads, var_grads, mean_grads = grads(corr_loss), grads(var_loss), grads(mean_loss)
        avg_norm = norm([(c + v + m) / 3.0 for c, v, m in zip(corr_grads, var_grads, mean_grads)]).detach()
        alpha = avg_norm / (norm(corr_grads).detach() + 1e-8)
        beta = avg_norm / (norm(var_grads).detach() + 1e-8)
        gamma = avg_norm / (norm(mean_grads).detach() + 1e-8)
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        return alpha, beta, gamma

    def forward(self, pred, true):
        loss = self.base_loss(pred, true)
        if not torch.is_grad_enabled():
            return loss
        true_patch, pred_patch = self.fourier_based_adaptive_patching(true, pred)
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)
        return loss + self.ps_lambda * (alpha * corr_loss + beta * var_loss + gamma * mean_loss)

