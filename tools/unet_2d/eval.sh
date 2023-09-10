#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet2d_cosinlr_wo_residual.yaml
val_id_path=data/pd_wip/pd_wip_eval_3d.txt
checkpoint=work_dirs/unet_2d/unet2d_cosinlr_wo_residual/best_10.8962.pth



python evaluate.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/pd_wip/pd_nifti_final/test\
    --wip_root data/pd_wip/wip_registration_nifti/test \
    --g-key model \
    --save-path work_dirs/unet_2d/unet2d_cosinlr_wo_residual \
    --save