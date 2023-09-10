#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix_2d/unet2d-b16-residual_basic-patchgan-ndf32_cosinlr.yaml
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
checkpoint=work_dirs/pix2pix_2d/unet2d-b16-residual_basic-patchgan-ndf32_cosinlr/best_22.8287.pth



python evaluate.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/pd_wip/pd_nifti_final/test\
    --wip_root data/pd_wip/wip_registration_nifti/test \
    --g-key model_G \
    --save-path work_dirs/pix2pix_2d/unet2d-b16-residual_basic-patchgan-ndf32_cosinlr \
    --save