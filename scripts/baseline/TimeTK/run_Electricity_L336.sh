#!/bin/bash
# Time-TK Electricity, published config with --seq_len 336
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/l336

PY=python
RUN=run.py
ROOT=./dataset/

done_already() { [ -f "$1" ] && grep -q '^mse:' "$1"; }

LOG="logs/l336/TimeTK_Electricity_336_96.log"
if done_already "$LOG"; then
  echo "SKIP Electricity L=336 T=96 (already done)"
else
  echo "RUN  Electricity L=336 T=96 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "electricity.csv" \
    --model_id "L336_Electricity_H96" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 96 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --e_layers 2 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --d_model 512 \
    --dropout 0.1 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Electricity L=336 T=96 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Electricity_336_192.log"
if done_already "$LOG"; then
  echo "SKIP Electricity L=336 T=192 (already done)"
else
  echo "RUN  Electricity L=336 T=192 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "electricity.csv" \
    --model_id "L336_Electricity_H192" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 192 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --e_layers 4 \
    --batch_size 16 \
    --learning_rate 0.002 \
    --d_model 512 \
    --dropout 0.3 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Electricity L=336 T=192 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Electricity_336_336.log"
if done_already "$LOG"; then
  echo "SKIP Electricity L=336 T=336 (already done)"
else
  echo "RUN  Electricity L=336 T=336 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "electricity.csv" \
    --model_id "L336_Electricity_H336" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 336 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --e_layers 2 \
    --batch_size 32 \
    --learning_rate 0.002 \
    --d_model 512 \
    --dropout 0.3 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Electricity L=336 T=336 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Electricity_336_720.log"
if done_already "$LOG"; then
  echo "SKIP Electricity L=336 T=720 (already done)"
else
  echo "RUN  Electricity L=336 T=720 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "electricity.csv" \
    --model_id "L336_Electricity_H720" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 720 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --e_layers 3 \
    --batch_size 32 \
    --learning_rate 0.005 \
    --d_model 512 \
    --dropout 0.1 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Electricity L=336 T=720 -- see $LOG"
fi

echo "--- Electricity at L=336, authors' published hyperparameters complete ---"
