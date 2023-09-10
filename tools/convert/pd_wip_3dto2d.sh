python tools/convert/pd_wip_3dto2d.py \
    data/new_pd_wip/v6/pd_paired \
    data/new_pd_wip/v6/wip_registration \
    --out-pd data/new_pd_wip/v6/train2d/pd_paired \
    --out-wip data/new_pd_wip/v6/train2d/wip_paired \
    --out-list data/new_pd_wip/v6/pd_wip_2d.txt $2 2>&1 | tee data/new_pd_wip/v6/pd_wip_2d_stats.txt
