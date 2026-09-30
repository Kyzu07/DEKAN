#!/usr/bin/env bash
set -euo pipefail

# Robust to being launched from scripts/l_336 or from any other directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEQ_LEN=336
PRED_LENS=(96 192 336 720)
GPU="${GPU:-0}"
ITR="${ITR:-1}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_${SEQ_LEN}_${PRED_LEN} \
    --model Amplifier \
    --data ETTm2 \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --des Exp \
    --itr "${ITR}" \
    --gpu "${GPU}"
done
