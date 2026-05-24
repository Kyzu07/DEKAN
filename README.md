# ModelX: A Decomposition-driven Time Series Forecasting Framework

> Preliminary work. Under review at AISTATS 2026.

ModelX is a decomposition-driven, patch-based forecasting framework that leverages **Kolmogorov–Arnold Networks (KANs)** to replace attention with efficient polynomial operators. It unifies multi-period seasonal-trend decomposition with patch-level modeling under a principled polynomial function space.

## Architecture Overview

ModelX processes each input series through five stages:

1. **RevIN** — Per-variable reversible instance normalization to prevent distribution leakage between train and test.
2. **MPSTDecomp** — Multi-Period Seasonal-Trend Decomposition that separates the signal into trend, seasonal, and residual components using a learnable bank of moving-average kernels and a fixed Fourier basis with multiple periods and harmonics. Learnable sigmoid gates control how much trend and seasonality is subtracted.
3. **Multi-branch Patching** — The residual is sent to *J* branches, each using a different patch length and stride, giving the model multi-resolution temporal coverage. Each patch is enriched with its local mean and standard deviation before linear projection and positional encoding.
4. **Dual-path KANMixer** — Each branch encodes patches through a two-path polynomial mixer: one path applies intra-patch then inter-patch KAN operators; the other reverses the order. An adaptive gating layer (mean pooling → softmax) fuses both paths with a learned convex combination. Krawtchouk polynomials parameterize each KAN layer.
5. **Softmax Fusion MLP** — Branch predictions are aggregated with a lightweight MLP scorer (mean + std features → LayerNorm → GELU → softmax weights).

The final forecast combines residual, trend, and seasonal predictions through two learned scalar gates before inverse RevIN.

## Key Contributions

- **MPSTDecomp**: multi-period Fourier basis decomposition with adaptive MA-bank trend extraction and learnable gating.
- **Dual-path KAN block**: Krawtchouk polynomial mixing of both intra-patch (feature dimension) and inter-patch (temporal dimension) dependencies, fused through adaptive gating.
- **Softmax Fusion MLP**: data-driven prioritization of patch scales using statistical branch descriptors.
- State-of-the-art results on 8 standard long-term forecasting benchmarks with up to **17.88%** improvement over strong baselines.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Datasets

Download the benchmark datasets and place the CSV files in `./dataset/`:

- ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity — available from the [Autoformer Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
- Illness (`national_illness.csv`) — same source

### Training

All experiment scripts are in `./scripts/LONG/`. Example:

```bash
sh ./scripts/LONG/weather.sh
sh ./scripts/LONG/etth1.sh
sh ./scripts/LONG/electricity.sh
```

Fair-comparison scripts (fixed look-back window) are in `./scripts/FAIR_LONG/`.

### Evaluation Only

```bash
python run_longExp.py --is_training 0 --model MTST ...
```

## Benchmarks

ModelX is evaluated on **Weather, Traffic, Electricity, Illness, ETTh1, ETTh2, ETTm1, ETTm2** with prediction horizons T ∈ {96, 192, 336, 720} (and T ∈ {24, 36, 48, 60} for Illness). Metrics: MSE and MAE (lower is better).

Across all 32 MSE evaluations ModelX attains **22 best** and **5 second-best** results; across 32 MAE evaluations **29 best** and **3 second-best**.

## Baselines

Comparisons include: MTST, PatchTST, N-HiTS, DLinear, MICN, TimesNet, FEDformer, Autoformer.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| KAN basis | Krawtchouk, q=0.6, N=255 |
| Polynomial degree | 3 (ablation includes 5) |
| Optimizer | Adam |
| Training loss | Huber (δ=0.5) |
| Max epochs | 100 |
| Early stopping patience | 10–20 |
| Learning rate | 1×10⁻⁴ (2.5×10⁻³ for Illness) |

## Acknowledgements

This codebase builds on the following open-source projects:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [MTST](https://github.com/yitianzhang/mtst) (Zhang et al., AISTATS 2024)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- [Autoformer](https://github.com/thuml/Autoformer)
- [FEDformer](https://github.com/MAZiqing/FEDformer)
- [Pyraformer](https://github.com/alipay/Pyraformer)
- [RevIN](https://github.com/ts-kim/RevIN)
