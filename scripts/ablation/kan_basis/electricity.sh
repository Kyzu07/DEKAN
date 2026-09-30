if [ ! -d "logs/ablation/kan_basis" ]; then
    mkdir -p logs/ablation/kan_basis
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021

for arm in 'lucas 5' 'chebyshev 5' 'monomials 5' 'legendre 5' 'bernstein 5' 'hahn 5' 'bspline 5' 'bspline 1' 'fourier 4'
do
# basis, grid size (bspline and fourier only)
set -- $arm
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
      --enc_in 321 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 384 \
      --dropout 0.10 \
      --head_dropout 0 \
      --patch_len_ls '24, 48, 96' \
      --stride_ls '12, 24, 48' \
      --padding_patch 'end' \
      --des 'Exp' \
      --train_epochs 120 \
      --patience 5 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --kan_basis $1 \
      --kan_grid_size $2 2>&1 | tee logs/ablation/kan_basis/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$1'_g'$2.log
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
      --enc_in 321 \
      --e_layers 1 \
      --n_branches 3 \
      --d_model 128 \
      --d_ff 384 \
      --dropout 0.10 \
      --head_dropout 0 \
      --patch_len_ls '24, 48, 96' \
      --stride_ls '12, 24, 48' \
      --padding_patch 'end' \
      --des 'Exp' \
      --train_epochs 120 \
      --patience 5 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --kan_basis $1 \
      --kan_grid_size $2 2>&1 | tee logs/ablation/kan_basis/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$1'_g'$2.log
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
      --enc_in 321 \
      --e_layers 2 \
      --n_branches 3 \
      --d_model 160 \
      --d_ff 480 \
      --dropout 0.15 \
      --head_dropout 0 \
      --patch_len_ls '24, 96, 168' \
      --stride_ls '12, 48, 84' \
      --padding_patch 'end' \
      --des 'Exp' \
      --train_epochs 140 \
      --patience 5 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 12 \
      --learning_rate 0.0004 \
      --kan_basis $1 \
      --kan_grid_size $2 2>&1 | tee logs/ablation/kan_basis/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$1'_g'$2.log
done

done
