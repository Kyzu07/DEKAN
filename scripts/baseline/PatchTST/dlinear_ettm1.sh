#!/bin/bash
# DLinear ETTm1, official config with --seq_len 336
set -eo pipefail

PRED_LENS="${@:-96 192 336 720}"
seq_len=336

mkdir -p logs/l336
for pred_len in $PRED_LENS
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_${seq_len}_${pred_len} \
      --model DLinear \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des Exp \
      --num_workers ${NUM_WORKERS:-6} \
      --itr 1 --batch_size 8 --learning_rate 0.0001 2>&1 | tee logs/l336/DLinear_ETTm1_${seq_len}_${pred_len}.log
done
