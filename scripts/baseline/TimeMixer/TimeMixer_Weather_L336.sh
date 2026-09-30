#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_cell.sh"
for T in 96 192 336 720; do
  run_cell Weather "$T" --data custom --root_path ./dataset/weather/ --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --d_model 16 --d_ff 32 --train_epochs 20 --patience 10 --learning_rate 0.001
done
