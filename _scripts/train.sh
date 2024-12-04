#!/bin/bash

gid=$1
model=$2
other=$3

project_dir=/nlp/projekty/mtlowre/COMEbyT5
models_dir="$project_dir/_models"
logs_dir="$project_dir/_logs"

tensorboard --logdir "$models_dir/$model" --port $((1000 + RANDOM % 9999)) &

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gid nohup $project_dir/comet/cli/train.py --cfg "$project_dir/configs/models/regression_model_$model.yaml" > "$logs_dir/$model.log"