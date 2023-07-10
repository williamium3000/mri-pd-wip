#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix_2d/unet64_ngf64_basic-patchgan_ndf64_cosinlr.yaml
train_id_path=data/pd_wip/pd_wip_2d.txt
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
save_path=work_dirs/pix2pix_2d/unet64_ngf64_basic-patchgan_ndf64_cosinlr

mkdir -p $save_path

srun --cpus-per-task=4 \
    --time=48:00:00 \
    --mem=80G \
    --job-name=train \
    --partition ica100 \
    -A yqiao4_gpu \
    --gres=gpu:$1 \
    --ntasks-per-node=$1 \
    python pix2pix.py \
    --config=$config \
    --train-id-path $train_id_path \
    --val-id-path $val_id_path \
    --save \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.txt