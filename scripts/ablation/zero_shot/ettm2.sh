if [ ! -d "logs/ablation/zero_shot" ]; then
    mkdir -p logs/ablation/zero_shot
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

for target in 'ETTh1 ETTh1.csv 7 400' 'ETTh2 ETTh2.csv 7 400' 'ETTm1 ETTm1.csv 7 400' 'ETTm2 ETTm2.csv 7 128' 'custom weather.csv 21 64' 'custom electricity.csv 321 16' 'custom traffic.csv 862 6'
do
# data, data_path, enc_in, test batch size of the target
set -- $target
for pred_len in 96 192 336 720
do
    batch=$4
    if [ $2 = electricity.csv ] && [ $pred_len -ge 336 ]; then batch=12; fi
    python -u run_longExp.py \
      --random_seed $random_seed \
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
      --target_data $1 \
      --target_data_path $2 \
      --target_enc_in $3 \
      --target_batch_size $batch 2>&1 | tee logs/ablation/zero_shot/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_to_'$2.log
done

done
