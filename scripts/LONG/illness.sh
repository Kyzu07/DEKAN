# ==== Illness (ILI) runs ====

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# ILI typically uses a short look-back
seq_len=104
model_name=MTST

root_path_name=./dataset/
# Change this if your filename differs (e.g., national_illness.csv)
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021

# Horizons for Illness
for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
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
      --learning_rate 0.0005
done
