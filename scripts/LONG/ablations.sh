if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=MTST

#################################################ETTh1####################################################

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

# mpstdecomp
# for 96:0.349 0.385
# for 192: 0.383 0.404
for pred_len in 96 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3\
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.1\
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 120\
      --patience 10 \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3\
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.1\
      --patch_len_ls '16, 48, 96' \
      --stride_ls '8, 24, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 120\
      --patience 10 \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

##############################################ETTh2########################################################

data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
# random_seed=2021

for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.3\
      --patch_len_ls '16, 96, 192' \
      --stride_ls '8, 48, 96' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 130\
      --patience 10 \
      --rel_pe 'rel_sin' \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.3\
      --patch_len_ls '16, 96' \
      --stride_ls '8, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 130\
      --patience 10 \
      --rel_pe 'rel_sin' \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.3\
      --patch_len_ls '16, 96' \
      --stride_ls '8, 48' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 130\
      --patience 10 \
      --rel_pe 'rel_sin' \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.3\
      --patch_len_ls '16, 96, 192' \
      --stride_ls '8, 48, 96' \
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 100\
      --patience 10 \
      --rel_pe 'rel_sin' \
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

####################################################ETTm1####################################################

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

# random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '16, 96, 120'\
      --stride_ls '8, 48, 60'\
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 130\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 400 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

###########################################ETTm2#################################################

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

# random_seed=2021

# L = 96 requires degree = 5, for others degree = 3
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '16, 96'\
      --stride_ls '8, 48'\
      --rel_pe 'rel_sin' \
      --lradj 'TST'\
      --pct_start 0.4\
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 100\
      --patience 5\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

#####################################################Weather####################################################

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

# random_seed=2021

for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '24, 96, 192' \
      --stride_ls '12, 48, 96' \
      --des 'Exp' \
      --lradj 'TST'\
      --rel_pe 'rel_sin' \
      --train_epochs 100\
      --patience 5\
      --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '24, 96, 192' \
      --stride_ls '12, 48, 96' \
      --des 'Exp' \
      --lradj 'TST'\
      --rel_pe 'rel_sin' \
      --train_epochs 100\
      --patience 5\
      --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 2 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '16, 96' \
      --stride_ls '8, 48' \
      --des 'Exp' \
      --lradj 'TST'\
      --rel_pe 'rel_sin' \
      --train_epochs 100\
      --patience 5\
      --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '16, 96, 192' \
      --stride_ls '8, 48, 96' \
      --des 'Exp' \
      --lradj 'TST'\
      --rel_pe 'rel_sin' \
      --train_epochs 100\
      --patience 5\
      --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

###################################################Electricity###################################################

# root_path_name=./dataset/
# data_path_name=electricity.csv
# model_id_name=Electricity
# data_name=custom

# # random_seed=2021

# for pred_len in 96
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id ${model_id_name}_${seq_len}_${pred_len} \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 1 \
#       --n_branches 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 384 \
#       --dropout 0.10 \
#       --fc_dropout 0.10 \
#       --head_dropout 0 \
#       --patch_len_ls '24, 48, 96' \
#       --stride_ls    '12, 24, 48' \
#       --rel_pe 'rel_sin' \
#       --des 'Exp' \
#       --train_epochs 120 \
#       --patience 5 \
#       --lradj 'TST' \
#       --pct_start 0.2 \
#       --itr 1 --batch_size 16 --learning_rate 0.0005 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
# done

# for pred_len in 192
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id ${model_id_name}_${seq_len}_${pred_len} \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 1 \
#       --n_branches 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 384 \
#       --dropout 0.10 \
#       --fc_dropout 0.10 \
#       --head_dropout 0 \
#       --patch_len_ls '24, 48, 96' \
#       --stride_ls    '12, 24, 48' \
#       --rel_pe 'rel_sin' \
#       --des 'Exp' \
#       --train_epochs 120 \
#       --patience 5 \
#       --lradj 'TST' \
#       --pct_start 0.2 \
#       --itr 1 --batch_size 16 --learning_rate 0.0005 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
# done

# for pred_len in 336 720
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id ${model_id_name}_${seq_len}_${pred_len} \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 2 \
#       --n_branches 3 \
#       --n_heads 16 \
#       --d_model 160 \
#       --d_ff 480 \
#       --dropout 0.15 \
#       --fc_dropout 0.15 \
#       --head_dropout 0 \
#       --patch_len_ls '24, 96, 168' \
#       --stride_ls    '12, 48, 84' \
#       --rel_pe 'rel_sin' \
#       --des 'Exp' \
#       --train_epochs 140 \
#       --patience 5 \
#       --lradj 'TST' \
#       --pct_start 0.2 \
#       --itr 1 --batch_size 12 --learning_rate 0.0004 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
# done

# #####################################################Traffic#####################################################

# root_path_name=./dataset/
# data_path_name=traffic.csv
# model_id_name=traffic
# data_name=custom

# # random_seed=2021

# for pred_len in 96 192
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 862 \
#       --e_layers 1 \
#       --n_branches 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 256 \
#       --dropout 0.2\
#       --fc_dropout 0.2\
#       --head_dropout 0\
#       --patch_len_ls '8, 16, 48' \
#       --stride_ls '4, 8, 24' \
#       --des 'Exp' \
#       --rel_pe 'rel_sin' \
#       --train_epochs 100\
#       --patience 10\
#       --lradj 'TST'\
#       --pct_start 0.2\
#       --itr 1 --batch_size 6 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# for pred_len in 336 720
# do
#   python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 862 \
#       --e_layers 1 \
#       --n_branches 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 256 \
#       --dropout 0.2\
#       --fc_dropout 0.2\
#       --head_dropout 0\
#       --patch_len_ls '8, 32, 96' \
#       --stride_ls '4, 16, 48' \
#       --des 'Exp' \
#       --rel_pe 'rel_sin' \
#       --train_epochs 100\
#       --patience 10\
#       --lradj 'TST'\
#       --pct_start 0.2\
#       --itr 1 --batch_size 6 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done

# ######################################################Illness####################################################

# # ILI uses a shorter look-back
# seq_len=104
# root_path_name=./dataset/
# data_path_name=national_illness.csv
# model_id_name=national_illness
# data_name=custom

# # random_seed=2021

# for pred_len in 24 36 48 60
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id ${model_id_name}_${seq_len}_${pred_len} \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 1 \
#       --n_branches 1 \
#       --n_heads 8 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.2 \
#       --fc_dropout 0.2 \
#       --head_dropout 0 \
#       --patch_len_ls '24' \
#       --stride_ls '2' \
#       --rel_pe 'rel_sin' \
#       --lradj 'TST' \
#       --pct_start 0.4 \
#       --des 'Exp' \
#       --padding_patch 'end' \
#       --train_epochs 500 \
#       --patience 100 \
#       --itr 1 \
#       --batch_size 64 \
#       --learning_rate 0.0005 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
# done
