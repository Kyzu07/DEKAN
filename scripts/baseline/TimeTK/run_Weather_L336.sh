#!/bin/bash
# Time-TK Weather, published config with --seq_len 336
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs/l336

PY=python
RUN=run.py
ROOT=./dataset/

done_already() { [ -f "$1" ] && grep -q '^mse:' "$1"; }

LOG="logs/l336/TimeTK_Weather_336_96.log"
if done_already "$LOG"; then
  echo "SKIP Weather L=336 T=96 (already done)"
else
  echo "RUN  Weather L=336 T=96 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "weather.csv" \
    --model_id "L336_Weather_H96" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 96 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --d_model 64 \
    --dropout 0.2 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Weather L=336 T=96 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Weather_336_192.log"
if done_already "$LOG"; then
  echo "SKIP Weather L=336 T=192 (already done)"
else
  echo "RUN  Weather L=336 T=192 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "weather.csv" \
    --model_id "L336_Weather_H192" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 192 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 4 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --d_model 64 \
    --dropout 0.5 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Weather L=336 T=192 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Weather_336_336.log"
if done_already "$LOG"; then
  echo "SKIP Weather L=336 T=336 (already done)"
else
  echo "RUN  Weather L=336 T=336 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "weather.csv" \
    --model_id "L336_Weather_H336" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 336 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 1 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --d_model 128 \
    --dropout 0.1 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Weather L=336 T=336 -- see $LOG"
fi

LOG="logs/l336/TimeTK_Weather_336_720.log"
if done_already "$LOG"; then
  echo "SKIP Weather L=336 T=720 (already done)"
else
  echo "RUN  Weather L=336 T=720 -> $LOG"
  "$PY" -u "$RUN" \
    --is_training 1 \
    --root_path "$ROOT" \
    --data_path "weather.csv" \
    --model_id "L336_Weather_H720" \
    --model "TimeTK" \
    --data "custom" \
    --features "M" \
    --seq_len 336 \
    --pred_len 720 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 1 \
    --batch_size 64 \
    --learning_rate 0.0002 \
    --d_model 512 \
    --dropout 0.5 \
    --lradj "type1" \
    --train_epochs 30 \
    --itr 1 \
    > "$LOG" 2>&1 || echo "FAILED Weather L=336 T=720 -- see $LOG"
fi

echo "--- Weather at L=336, authors' published hyperparameters complete ---"
