#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/unet/unet3d_cosinlr_new.yaml
val_id_path=data/new_pd_wip/v6/pd_wip_3d_test_clean.txt
checkpoint=work_dirs/unet_3d/unet3d_cosinlr_new/last.pth



python eval3d.py \
    --config=$config --val-id-path $val_id_path \
    --checkpoint $checkpoint \
    --pd_root data/new_pd_wip/v6/pd_paired \
    --wip_root data/new_pd_wip/v6/wip_registration \
    --g-key model \
    --save-path work_dirs/unet_3d/unet3d_cosinlr_new/results_ori_last \
    --save