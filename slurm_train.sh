#!/bin/bash

### Available maps: 5m_vs_6m, 8m, 10m_vs_11m, smacv2_5_units, smacv2_10_units

set -x
PARTITION=${PARTITION:-"optimal"}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

seeds=(
    1 2 3
)

env="smax"
if [ -z "$1" ]; then
    echo "Error: map_name argument is required."
    exit 1
fi
map_name=$1
steps=1000000

for seed in "${seeds[@]}"; do
    date_dir=$(date "+%Y-%m-%d")
    cur_date=$(date "+%H-%M-%S")
    OUTPUT_DIR=training-runs/$date_dir

    # log_name=$(echo "$map_name" | awk -F'_' '{print $(NF-2) "-" $(NF-1) "-" $NF}')
    log_name="$map_name-seed_$seed"
    echo $log_name
    note=

    mkdir -p $OUTPUT_DIR

    sbatch -p ${PARTITION} \
    -J ${log_name}${note:+-$note} \
    -N 1 \
    -n 6 \
    -o ${OUTPUT_DIR}/${cur_date}-${log_name}${note:+-$note}-%j.out \
    --gres=gpu:${GPUS_PER_NODE} \
    --wrap="python train.py \
            --env $env \
            --env_name $map_name \
            --n_workers 2 \
            --seed $seed \
            --steps $steps \
            --mode offline"
done