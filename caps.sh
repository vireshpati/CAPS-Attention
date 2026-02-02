#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

torch_compile=1    # whether to use the compile method in torch >= 2.0

e_layers=3         # number of layers (decoder here)
n_heads=4
d_model=0          # determined by $16\sqrt{C}$
dropout=0
max_grad_norm=1
patience=12
random_seed=2026
univariate=0
time_emb_dim=4
random_drop_training=0

data_names=("ETT-small/ETTm1.csv" "ETT-small/ETTm2.csv" "ETT-small/ETTh1.csv" "ETT-small/ETTh2.csv" "weather/weather.csv" "Solar/solar_AL.txt" "electricity/electricity.csv" "traffic/traffic.csv"  "PEMS/PEMS03.npz"   "PEMS/PEMS04.npz"   "PEMS/PEMS07.npz"   "PEMS/PEMS08.npz")
data_alias=("ETTm1"               "ETTm2"               "ETTh1"               "ETTh2"               "Weather"             "Solar"              "ECL"                         "Traffic"              "PEMS03"            "PEMS04"            "PEMS07"            "PEMS08")
data_types=("ETTm1"               "ETTm2"               "ETTh1"               "ETTh2"               "custom"              "Solar"              "custom"                      "custom"               "PEMS"              "PEMS"              "PEMS"              "PEMS")
enc_ins=(    7                     7                     7                     7                     21                    137                  321                           862                   358                 307                 883                 170)
batch_sizes=(32                    32                    32                    32                    32                    8                    2                             2                     8                   8                   2                   8)
grad_accums=(1                     1                     1                     1                     1                     4                    16                            16                    4                   4                   16                  4)
random_channel_dropouts=(1         1                     1                     1                     1                     0                    0                             0                     0                   0                   0                   0)
weight_decays=(0.0                 0.0                   0.1                   0.1                   0.1                   0                    0                             0                     0                   0                   0                   0)

for i in 0 1 2 3 4 5 6 8 9 11; do
data_name=${data_names[$i]}
data_alias_current=${data_alias[$i]}
data_type=${data_types[$i]}
enc_in=${enc_ins[$i]}
batch_size=${batch_sizes[$i]}
gradient_accumulation=${grad_accums[$i]}
random_channel_dropout=${random_channel_dropouts[$i]}
weight_decay=${weight_decays[$i]}

if [[ "$data_type" == "PEMS" ]]; then
    pred_lens=(96 48 24 12)
else
    pred_lens=(96 192 336 720)
fi

seq_lens=(96 96 96 96)

for j in $(seq 0 3)
do

pred_len=${pred_lens[$j]}
seq_len=${seq_lens[$j]}

for model_name in CAPS
do

for predictor in CAPS
do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_name \
  --model_id $random_seed'_'$model_name'_'$predictor'_'$data_alias_current'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_type \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --des 'Exp' \
  --scale 1 \
  --time_emb_dim $time_emb_dim \
  --plot_every 5 \
  --num_workers 6 \
  --train_epochs 100 \
  --max_grad_norm $max_grad_norm \
  --predictor $predictor \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --dropout $dropout \
  --patience $patience \
  --random_seed $random_seed \
  --gradient_accumulation $gradient_accumulation \
  --use_amp \
  --pct_start 0.0 \
  --compile $torch_compile \
  --univariate $univariate \
  --random_drop_training $random_drop_training \
  --random_channel_dropout $random_channel_dropout \
  --weight_decay $weight_decay \
  --itr 1 --batch_size $batch_size --learning_rate 0.0005 >logs/LongForecasting/$random_seed'_'$model_name'_'$predictor'_'$data_alias_current'_'$seq_len'_'$pred_len'.log'

done
done
done
done
