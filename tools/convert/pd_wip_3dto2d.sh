python tools/convert/pd_wip_3dto2d.py \
    data/pd_wip/pd_nifti_final/train \
    data/pd_wip/wip_registration_nifti/train \
    --out-pd data/pd_wip/pd_nifti_final/train2d \
    --out-wip data/pd_wip/wip_registration_nifti/train2d \
    --out-list data/pd_wip/pd_wip_2d.txt $2 2>&1 | tee data/pd_wip/2d_stats.txt
