if [ ! -d "logs/ablation/loss" ]; then
    mkdir -p logs/ablation/loss
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

for loss in mse mae dbloss fredf transdf psloss tildeq softdtw dilate
do
for pred_len in 96 192 336 720
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
      --loss $loss >logs/ablation/loss/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$loss.log
done

done

for pred_len in 96 192 336 720
do
    if [ $pred_len -eq 720 ]; then w=0.1; else w=0.05; fi
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
      --loss a1 \
      --a1_w_aux $w >logs/ablation/loss/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_a1'.log
done
