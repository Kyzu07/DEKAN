if [ ! -d "logs/ablation/kan_basis" ]; then
    mkdir -p logs/ablation/kan_basis
fi
seq_len=336
model_name=DEKAN

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

for arm in 'lucas 5' 'chebyshev 5' 'monomials 5' 'legendre 5' 'bernstein 5' 'hahn 5' 'bspline 5' 'bspline 1' 'fourier 4'
do
# basis, grid size (bspline and fourier only)
set -- $arm
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
      --n_branches 3 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --patch_len_ls '16, 96, 120' \
      --stride_ls '8, 48, 60' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 130 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 400 \
      --learning_rate 0.0001 \
      --kan_basis $1 \
      --kan_grid_size $2 >logs/ablation/kan_basis/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$1'_g'$2.log
done

done
