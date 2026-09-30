# TimeKAN Traffic L=336. TimeKAN publishes no Traffic config, so this is its Electricity L=336 script
# (Electricity_720.sh) with the data swapped to traffic (enc_in 862) and TimeMixer's Traffic settings,
# since TimeKAN is built on TimeMixer's code: d_model 32, d_ff 64, lr 0.01.  Chosen by the user.
# Batch 8 -> 4: batch 8 ran out of memory on the 24 GB GPU (862 channels x d_model 32); batch 4 fits (user's call).
export CUDA_VISIBLE_DEVICES=0

model_name=TimeKAN
seq_len=336
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=4
train_epochs=10
patience=10


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_window $down_sampling_window\
  --down_sampling_layers $down_sampling_layers
