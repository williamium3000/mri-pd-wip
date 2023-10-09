#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet3d_cosinlr_new_b64.yaml
train_id_path=data/new_pd_wip/v6/pd_wip_3d_train_clean.txt
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test_clean.txt
save_path=work_dirs/unet_3d/unet3d_cosinlr_new_b64

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    sr_3d.py \
    --save --amp \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt