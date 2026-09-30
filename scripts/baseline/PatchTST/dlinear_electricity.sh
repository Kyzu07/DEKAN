#!/bin/bash
# DLinear Electricity, official config with --seq_len 336
set -eo pipefail
PRED_LENS="${@:-720}"
mkdir -p logs/l336
for pred_len in $PRED_LENS
do
    python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path electricity.csv \
      --model_id electricity_336_${pred_len} --model DLinear --data custom --features M \
      --seq_len 336 --pred_len $pred_len --enc_in 321 --des Exp --num_workers ${NUM_WORKERS:-6} \
      --itr 1 --batch_size 16 --learning_rate 0.001 2>&1 | tee logs/l336/DLinear_electricity_336_${pred_len}.log
done
