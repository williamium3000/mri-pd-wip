#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/cyclegan/unet64_ngf64_basic-patchgan_ndf64_cosinlr.yaml
train_id_path=data/pd_wip/pd_wip_2d.txt
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
save_path=work_dirs/cycle_gan_2d/unet64_ngf64_basic-patchgan_ndf64_cosinlr_bs64

mkdir -p $save_path

srun --cpus-per-task=2 \
    --time=48:00:00 \
    --mem-per-cpu=20G \
    --job-name=train \
    --partition ica100 \
    -A ayuille1_gpu  \
    --gres=gpu:$1 \
    --ntasks-per-node=$1 \
    python cycle_gan.py \
    --config=$config \
    --train-id-path $train_id_path \
    --val-id-path $val_id_path \
    --save \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.txt