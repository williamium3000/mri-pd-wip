#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet2d_cosinlr.yaml
train_id_path=data/pd_wip/pd_wip_2d.txt
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
save_path=work_dirs/unet_2d/unet2d_cosinlr_ssim

mkdir -p $save_path

srun --cpus-per-task=2 \
    --time=5:00:00 \
    --mem=50G \
    --job-name=train \
    --partition a100 \
    -A yqiao4_gpu \
    --gres=gpu:$1 \
    --ntasks-per-node=$1 \
    python sr_2d.py \
    --config=$config \
    --train-id-path $train_id_path \
    --val-id-path $val_id_path \
    --save --amp \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.txt