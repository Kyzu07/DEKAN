if [ ! -d "logs/ablation/lookback" ]; then
    mkdir -p logs/ablation/lookback
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

seq_len=96

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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

seq_len=192

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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

seq_len=512

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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

seq_len=720

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
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 120 \
      --patience 10 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 >logs/ablation/lookback/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
