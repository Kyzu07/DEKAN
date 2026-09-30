if [ ! -d "logs/LongForecasting" ]; then
    mkdir -p logs/LongForecasting
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021

for pred_len in 96 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len_ls '8, 16, 48' \
      --stride_ls '4, 8, 24' \
      --padding_patch 'end' \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 6 \
      --learning_rate 0.0005 2>&1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len_ls '8, 32, 96' \
      --stride_ls '4, 16, 48' \
      --padding_patch 'end' \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 6 \
      --learning_rate 0.0005 2>&1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
