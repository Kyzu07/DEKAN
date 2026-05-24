#!/usr/bin/env bash
set -euo pipefail

# ========= Logging setup =========
mkdir -p ./logs/LongForecasting

timestamp() {
  date +"%Y%m%d-%H%M%S"
}

logfile() {
  local model="$1" dataset="$2" seq_len="$3" pred_len="$4"
  echo "./logs/LongForecasting/${model}_${dataset}_${seq_len}_${pred_len}_$(timestamp).log"
}

PYTHON=${PYTHON:-python}   # allows override: PYTHON=python3 ./run_all_ablation.sh

# ========= Common constants (override per dataset where needed) =========
MODEL_NAME=MTST
ROOT=./dataset

# =========================================
# =============== ETTh1 ===================
# =========================================
seq_len=336
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2022

for pred_len in 96 192 336 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0.1 \
    --patch_len_ls '16, 48, 96' \
    --stride_ls '8, 24, 48' \
    --des 'Exp' \
    --padding_patch 'end' \
    --rel_pe 'rel_sin' \
    --train_epochs 120 \
    --patience 10 \
    --itr 1 --batch_size 400 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# =========================================
# =============== ETTh2 ===================
# =========================================
seq_len=336
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2021

# 96
for pred_len in 96; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0.3 \
    --patch_len_ls '16, 96, 192' \
    --stride_ls '8, 48, 96' \
    --des 'Exp' \
    --padding_patch 'end' \
    --train_epochs 130 \
    --patience 10 \
    --rel_pe 'rel_sin' \
    --itr 1 --batch_size 400 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# 192, 336
for pred_len in 192 336; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0.3 \
    --patch_len_ls '16, 96' \
    --stride_ls '8, 48' \
    --des 'Exp' \
    --padding_patch 'end' \
    --train_epochs 130 \
    --patience 10 \
    --rel_pe 'rel_sin' \
    --itr 1 --batch_size 400 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# 720
for pred_len in 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0.3 \
    --patch_len_ls '16, 96, 192' \
    --stride_ls '8, 48, 96' \
    --des 'Exp' \
    --padding_patch 'end' \
    --train_epochs 100 \
    --patience 10 \
    --rel_pe 'rel_sin' \
    --itr 1 --batch_size 400 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# =========================================
# =============== ETTm1 ===================
# =========================================
seq_len=336
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
random_seed=2021

for pred_len in 96 192 336 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '16, 96, 120' \
    --stride_ls '8, 48, 60' \
    --des 'Exp' \
    --padding_patch 'end' \
    --rel_pe 'rel_sin' \
    --train_epochs 130 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.4 \
    --itr 1 --batch_size 400 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# =========================================
# =============== ETTm2 ===================
# =========================================
seq_len=336
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
random_seed=2021

for pred_len in 96 192 336 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '16, 96' \
    --stride_ls '8, 48' \
    --rel_pe 'rel_sin' \
    --lradj 'TST' \
    --pct_start 0.4 \
    --des 'Exp' \
    --padding_patch 'end' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 \
    > "$LOG" 2>&1
done

# =========================================
# =============== Weather =================
# =========================================
seq_len=336
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2021

# 96
for pred_len in 96; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 21 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '24, 96, 192' \
    --stride_ls '12, 48, 96' \
    --des 'Exp' \
    --lradj 'TST' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# 192
for pred_len in 192; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 21 \
    --e_layers 3 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '24, 96, 192' \
    --stride_ls '12, 48, 96' \
    --des 'Exp' \
    --lradj 'TST' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# 336
for pred_len in 336; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 21 \
    --e_layers 2 \
    --n_branches 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '16, 96' \
    --stride_ls '8, 48' \
    --des 'Exp' \
    --lradj 'TST' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# 720
for pred_len in 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 21 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '16, 96, 192' \
    --stride_ls '8, 48, 96' \
    --des 'Exp' \
    --lradj 'TST' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# =========================================
# ============== Electricity ==============
# =========================================
seq_len=336
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
random_seed=2021

# 96, 192
for pred_len in 96 192; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 321 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 384 \
    --dropout 0.10 \
    --fc_dropout 0.10 \
    --head_dropout 0 \
    --patch_len_ls '24, 48, 96' \
    --stride_ls '12, 24, 48' \
    --rel_pe 'rel_sin' \
    --des 'Exp' \
    --train_epochs 120 \
    --patience 5 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 16 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# 336, 720
for pred_len in 336 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 321 \
    --e_layers 2 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 160 \
    --d_ff 480 \
    --dropout 0.15 \
    --fc_dropout 0.15 \
    --head_dropout 0 \
    --patch_len_ls '24, 96, 168' \
    --stride_ls '12, 48, 84' \
    --rel_pe 'rel_sin' \
    --des 'Exp' \
    --train_epochs 140 \
    --patience 5 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 12 --learning_rate 0.0004 \
    > "$LOG" 2>&1
done

# =========================================
# ================ Traffic ================
# =========================================
seq_len=336
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2021

# 96, 192
for pred_len in 96 192; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 862 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '8, 16, 48' \
    --stride_ls '4, 8, 24' \
    --des 'Exp' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 6 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# 336, 720
for pred_len in 336 720; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 862 \
    --e_layers 1 \
    --n_branches 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '8, 32, 96' \
    --stride_ls '4, 16, 48' \
    --des 'Exp' \
    --rel_pe 'rel_sin' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 6 --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

# =========================================
# ================= Illness ===============
# =========================================
# NOTE: Illness uses a different seq_len (104)
seq_len=104
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom
random_seed=2021

for pred_len in 24 36 48 60; do
  LOG=$(logfile "$MODEL_NAME" "$model_id_name" "$seq_len" "$pred_len")
  $PYTHON -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$ROOT/" \
    --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 1 \
    --n_branches 1 \
    --n_heads 8 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len_ls '24' \
    --stride_ls '2' \
    --rel_pe 'rel_sin' \
    --lradj 'TST' \
    --pct_start 0.4 \
    --des 'Exp' \
    --padding_patch 'end' \
    --train_epochs 500 \
    --patience 100 \
    --itr 1 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    > "$LOG" 2>&1
done

echo "All experiments dispatched. Logs are in ./logs/LongForecasting/"
