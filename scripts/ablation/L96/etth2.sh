if [ ! -d "logs/ablation/L96" ]; then
    mkdir -p logs/ablation/L96
fi
seq_len=96
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '8, 16, 32' \
      --stride_ls '4, 8, 16' \
      --padding_patch 'end' \
      --kan_degree 6 \
      --harmonics 3 \
      --des 'Exp' \
      --train_epochs 130 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '12, 24, 48' \
      --stride_ls '6, 12, 24' \
      --padding_patch 'end' \
      --kan_degree 5 \
      --harmonics 3 \
      --des 'Exp' \
      --train_epochs 130 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '12, 24, 48' \
      --stride_ls '6, 12, 24' \
      --padding_patch 'end' \
      --kan_degree 4 \
      --harmonics 3 \
      --des 'Exp' \
      --train_epochs 130 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
