if [ ! -d "logs/ablation/L96" ]; then
    mkdir -p logs/ablation/L96
fi
seq_len=96
model_name=DEKAN

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

for pred_len in 96
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
      --enc_in 21 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '8, 24, 48' \
      --stride_ls '4, 12, 24' \
      --padding_patch 'end' \
      --harmonics 3 \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0005 2>&1 | tee logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192
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
      --enc_in 21 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '12, 24, 48' \
      --stride_ls '6, 12, 24' \
      --padding_patch 'end' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.0005 2>&1 | tee logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 21 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '8, 16, 32' \
      --stride_ls '4, 8, 16' \
      --padding_patch 'end' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0005 2>&1 | tee logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
