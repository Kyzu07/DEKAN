if [ ! -d "logs/ablation/bottleneck" ]; then
    mkdir -p logs/ablation/bottleneck
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

for e_layers in 2 3
do
for bottleneck_dim in 0 16 64 seq_len
do
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
      --e_layers $e_layers \
      --n_branches 2 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len_ls '16, 96' \
      --stride_ls '8, 48' \
      --lradj 'TST' \
      --pct_start 0.4 \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --bottleneck_dim $bottleneck_dim >logs/ablation/bottleneck/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_e'$e_layers'_h'$bottleneck_dim.log
done

done

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
      --e_layers $e_layers \
      --n_branches 2 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len_ls '16, 96' \
      --stride_ls '8, 48' \
      --lradj 'TST' \
      --pct_start 0.4 \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 100 \
      --patience 5 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --bottleneck_type linear >logs/ablation/bottleneck/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_e'$e_layers'_linear'.log
done

done
