#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/msdbrain3d/unet3d_b64_1x4_adam_lr1e-4_cosine_w2e_dice_ce.yaml
train_id_path=split/msdbrain3d/train.txt
val_id_path=split/msdbrain3d/val.txt
save_path=exp/msdbrain3d/unet3d_b64_1x4_adam_lr1e-4_cosine_w2e_dice_ce/

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    supervised3d.py \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt