#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_cell.sh"
for T in 96 192 336 720; do
  LR=0.01; case "$T" in 192|720) LR=0.001 ;; esac
  run_cell Electricity "$T" --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --enc_in 321 --dec_in 321 --c_out 321 --d_model 16 --d_ff 32 --batch_size 32 --train_epochs 20 --patience 10 --learning_rate "$LR"
done
