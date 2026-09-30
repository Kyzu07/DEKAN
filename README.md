# DEKAN

DEKAN is a decomposition-driven, patch-based model for long-term multivariate time series forecasting. The input is split into trend, seasonal and residual components. Trend and seasonality are extrapolated with lightweight heads, and the residual is modeled by parallel patch branches whose embeddings are mixed by a dual-path Krawtchouk KAN.

## Get Started

1. Install Python 3.10+ and the requirements: `pip install -r requirements.txt`.
2. Download the datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic) from the Google Drive linked in [Autoformer](https://github.com/thuml/Autoformer) and put the csv files directly in `./dataset/`.
3. Train and evaluate. Each script runs all four horizons of one dataset with look-back 336:

```
bash scripts/LONG/etth1.sh
```

Logs go to `logs/LongForecasting/`, metrics are appended to `result.txt`, and predictions are saved in `results/`.

## Ablations

`scripts/ablation/<name>/<dataset>.sh` use the same configuration as `scripts/LONG` and change one thing:

| name | what changes |
|---|---|
| `decomp_mode` | no decomposition, or only the residual / seasonal / trend path |
| `kan_basis` | Lucas, Chebyshev, monomial, Legendre, Bernstein, Hahn, B-spline and Fourier bases |
| `kan_degree` | polynomial degree 1-6 |
| `kan_path` | a single path (ab or ba) instead of the dual-path mixer |
| `backbone` | Transformer, S-Mamba, MLP-Mixer or linear mixer instead of the KAN mixer |
| `harmonics` | number of Fourier harmonics per seasonal period |
| `kernels` | moving average kernels of the trend |
| `bottleneck` | stacked encoder layers and the re-projection between them |
| `lookback` | look-back 96, 192, 512, 720 |
| `loss` | MSE, MAE, DBLoss, FreDF, TransDF, PSLoss, TILDE-Q, Soft-DTW, DILATE |
| `seed` | other random seeds |
| `L96` | look-back 96 |
| `zero_shot` | a model trained on one dataset, evaluated on the others (run `scripts/LONG` first) |
| `efficiency` | parameters, FLOPs, peak memory and latency |

S-Mamba uses `mamba-ssm` when it is installed, and a gated depthwise convolution otherwise.

## Baselines

```
bash scripts/baseline/setup.sh
```

clones TimeKAN, Time-TK, Amplifier, TimeMixer, PatchTST and Time-Series-Library into `./baseline/` at the commits we used and copies our look-back 336 scripts into them. Each baseline then runs from its own folder with its own requirements, for example `cd baseline/TimeKAN && bash scripts/l_336/ETTh1/ETTh1_96.sh`.

## Acknowledgement

We thank the authors of the following repositories for their code and datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/networkslab/MTST

https://github.com/thuml/Autoformer

https://github.com/ts-kim/RevIN

https://github.com/Blealtan/efficient-kan

https://github.com/GistNoesis/FourierKAN
