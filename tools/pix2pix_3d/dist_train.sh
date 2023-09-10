#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix/unet3d-b32-residual_patchgan3d-ndf32_cosinlr_crop128_sample-max_50e.yaml
train_id_path=data/new_pd_wip/v6/pd_wip_3d_train.txt
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test.txt
save_path=work_dirs/pix2pix_3d/unet3d-b32-residual_patchgan3d-ndf32_cosinlr_crop128_sample-max_50e

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    pix2pix_3d.py \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save --save_feq 10 --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt