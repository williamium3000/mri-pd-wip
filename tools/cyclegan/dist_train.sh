#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/cyclegan/unet64_ngf64_basic-patchgan_ndf64_b64_lr5e-3.yaml
train_id_path=data/pd_wip/pd_wip_2d.txt
val_id_path=data/pd_wip/pd_wip_2d.txt
save_path=work_dirs/cycle_gan_2d/unet64_ngf64_basic-patchgan_ndf64_b64_lr5e-3

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    cycle_gan.py \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt