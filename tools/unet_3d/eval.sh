#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet3d-b64_cosinlr_50e_dataset-max_crop128.yaml
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test_clean.txt
checkpoint=work_dirs/new_data/unet_3d/unet3d_cosinlr/best_9.9301.pth


CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/new_pd_wip/v6/pd_paired \
    --wip_root data/new_pd_wip/v6/wip_registration \
    --g-key model \
    --out-dir work_dirs/new_data/unet_3d/unet3d_cosinlr/new_results \
    --save --eval-3d