#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet2d_cosinlr.yaml
train_id_path=data/pd_wip/pd_wip_2d.txt
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
save_path=work_dirs/unet_2d/unet2d_cosinlr_ssim_loss

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    sr_2d_ssim.py \
    --save --amp \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt