if [ ! -d "logs/ablation/L96" ]; then
    mkdir -p logs/ablation/L96
fi
seq_len=96
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

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
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '12, 24, 48' \
      --stride_ls '6, 12, 24' \
      --padding_patch 'end' \
      --kan_degree 4 \
      --period_list '96,48' \
      --kernel_sizes '13,25,49' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 128 \
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
      --n_branches 4 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '8, 16, 32, 48' \
      --stride_ls '4, 8, 16, 24' \
      --padding_patch 'end' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
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
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '8, 16, 32' \
      --stride_ls '4, 8, 16' \
      --padding_patch 'end' \
      --harmonics 1 \
      --period_list '96,48' \
      --kernel_sizes '13,25,49' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 720
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
      --dropout 0.2 \
      --head_dropout 0.0 \
      --patch_len_ls '12, 24, 48' \
      --stride_ls '6, 12, 24' \
      --padding_patch 'end' \
      --harmonics 1 \
      --period_list '96,48' \
      --kernel_sizes '13,25,49' \
      --des 'Exp' \
      --lradj 'TST' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 >logs/ablation/L96/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
