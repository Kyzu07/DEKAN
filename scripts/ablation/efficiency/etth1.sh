# ETTh1, L=336, T=336, batch 32
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for pred_len in 336
do
    python -u run_longExp.py \
      --is_training 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --padding_patch 'end' \
      --batch_size 32 \
      --test_flop
done
