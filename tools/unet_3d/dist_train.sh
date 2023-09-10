#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet3d-b32_cosinlr_50e_sample-max_crop128.yaml
train_id_path=data/new_pd_wip/v6/pd_wip_3d_train.txt
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test.txt
save_path=work_dirs/new_data/unet3d-b32_cosinlr_50e_sample-max_crop128

mkdir -p $save_path
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    sr_3d.py \
    --save --amp --clip-grad-norm 0.1 \
    --config=$config --train-id-path $train_id_path --val-id-path $val_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt