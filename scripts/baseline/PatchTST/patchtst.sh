#!/bin/bash
# PatchTST/42, official configs with --seq_len 336
# usage: bash scripts/l336/patchtst.sh <dataset> [pred_len ...]
set -eo pipefail

ds=$1; shift
PRED_LENS="${@:-96 192 336 720}"
seq_len=336
NUM_WORKERS=${NUM_WORKERS:-6}

case $ds in
  ETTh1|ETTh2)
    data=$ds; csv=$ds.csv
    arch="--enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3"
    train="--train_epochs 100 --batch_size 128 --learning_rate 0.0001" ;;
  ETTm1|ETTm2)
    data=$ds; csv=$ds.csv
    arch="--enc_in 7 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2"
    train="--train_epochs 100 --patience 20 --lradj TST --pct_start 0.4 --batch_size 128 --learning_rate 0.0001" ;;
  weather)
    data=custom; csv=weather.csv
    arch="--enc_in 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2"
    train="--train_epochs 100 --patience 20 --batch_size 128 --learning_rate 0.0001" ;;
  electricity)
    data=custom; csv=electricity.csv
    arch="--enc_in 321 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2"
    train="--train_epochs 100 --patience 10 --lradj TST --pct_start 0.2 --batch_size 32 --learning_rate 0.0001" ;;
  traffic)
    data=custom; csv=traffic.csv
    arch="--enc_in 862 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 --fc_dropout 0.2"
    # batch 12 instead of 24 to fit in 24 GB
    train="--train_epochs 100 --patience 10 --lradj TST --pct_start 0.2 --batch_size 12 --learning_rate 0.0001" ;;
  *) echo "unknown dataset: $ds" >&2; exit 1 ;;
esac

mkdir -p logs/l336
for pred_len in $PRED_LENS
do
    python -u run_longExp.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $csv \
      --model_id ${ds}_${seq_len}_${pred_len} \
      --model PatchTST \
      --data $data \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      $arch \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des Exp \
      --num_workers $NUM_WORKERS \
      --itr 1 $train 2>&1 | tee logs/l336/PatchTST_${ds}_${seq_len}_${pred_len}.log
done
