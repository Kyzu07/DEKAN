if [ ! -d "logs/ablation/seed" ]; then
    mkdir -p logs/ablation/seed
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

for seed in 2022 2023
do
for pred_len in 96 192
do
    python -u run_longExp.py \
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
      --learning_rate 0.0001 \
      --random_seed $seed 2>&1 | tee logs/ablation/seed/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$seed.log
done

for pred_len in 336 720
do
    python -u run_longExp.py \
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
      --learning_rate 0.0001 \
      --random_seed $seed 2>&1 | tee logs/ablation/seed/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$seed.log
done

done
