#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pix2pix_2d/unet64_ngf32_basic-patchgan_ndf32_cosinlr.yaml
val_id_path=data/pd_wip/pd_wip_eval_3d.txt



python eval_case.py \
    --config=$config --val-id-path $val_id_path \
    --case_dir work_dirs/unet_2d/unet2d_cosinlr_ssim/results \
    --pd_root data/pd_wip/pd_nifti_final/test \
    --wip_root data/pd_wip/wip_registration_nifti/test \