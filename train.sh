#!/bin/bash

#### env: 3s_vs_5z, 2s3z, 6h_vs_8z
map_name="3s_vs_5z"
env="smax"
seed=1
steps=1500000
mode=online

CUDA_VISIBLE_DEVICES=0 python train.py \
    --env $env \
    --env_name $map_name \
    --n_workers 2 \
    --seed $seed \
    --steps $steps \
    --mode $mode