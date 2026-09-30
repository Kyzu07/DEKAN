#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_cell.sh"
for T in 96 192 336 720; do
  run_cell Traffic "$T" --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --enc_in 862 --dec_in 862 --c_out 862 --d_model 32 --d_ff 64 --batch_size 8 --train_epochs 20
done
