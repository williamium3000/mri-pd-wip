#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix/unet3d-residual_basic-patchgan3d-ndf32_cosinlr_200e.yaml
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test.txt
checkpoint=work_dirs/pix2pix_3d/unet3d-residual_basic-patchgan3d-ndf32_cosinlr_200e/best_27.6517.pth



python evaluate.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/new_pd_wip/v6/pd_paired \
    --wip_root data/new_pd_wip/v6/wip_registration \
    --g-key model_G \
    --out-dir work_dirs/pix2pix_3d/unet3d-residual_basic-patchgan3d-ndf32_cosinlr_200e/results_best \
    --save --eval-3d