#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet3d_cosinlr.yaml
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test.txt
checkpoint=work_dirs/unet_3d/unet3d_cosinlr/best_21.3296.pth


CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/new_pd_wip/v6/pd_paired \
    --wip_root data/new_pd_wip/v6/wip_registration \
    --g-key model \
    --out-dir work_dirs/unet_3d/unet3d_cosinlr/results_new_data_no_scale \
    --save --eval-3d