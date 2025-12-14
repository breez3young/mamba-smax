#!/bin/bash

#### env: 3s_vs_5z, 2s3z, 6h_vs_8z, 5m_vs_6m
#### all maps

### Available maps: 3m, 2s3z, 25m, 3s5z, 8m, 5m_vs_6m, 10m_vs_11m, 27m_vs_30m, 3s5z_vs_3s6z, 3s_vs_5z, 6h_vs_8z, smacv2_5_units, smacv2_10_units, smacv2_20_units


map_name="3s5z"
env="smax"
seed=2
steps=1500000
mode=online

CUDA_VISIBLE_DEVICES=1 python train.py \
    --env $env \
    --env_name $map_name \
    --n_workers 2 \
    --seed $seed \
    --steps $steps \
    --mode $mode