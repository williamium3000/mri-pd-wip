#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix_3d/unet-ngf16_basic-patchgan_ndf16_cosinlr.yaml
train_id_path=data/new_pd_wip/v6/pd_wip_3d_train_clean.txt
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test_clean.txt
save_path=work_dirs/pix2pix_3d_new/unet-ngf16_basic-patchgan_ndf16_cosinlr

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    pix2pix_3d.py \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt