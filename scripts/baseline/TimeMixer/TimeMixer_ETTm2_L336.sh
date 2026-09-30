#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_cell.sh"
for T in 96 192 336 720; do
  run_cell ETTm2 "$T" --data ETTm2 --data_path ETTm2.csv --enc_in 7 --dec_in 7 --c_out 7 --root_path ./dataset/ETT-small/ --d_model 32 --d_ff 32 --train_epochs 10 --learning_rate 0.001
done
