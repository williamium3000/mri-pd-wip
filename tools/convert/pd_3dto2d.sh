python tools/convert/pd_3dto2d.py \
    data/new_pd_wip/v6/pd_only \
    --out-pd data/new_pd_wip/v6/train2d/pd_only \
    --out-list data/new_pd_wip/v6/pd_2d.txt $2 2>&1 | tee data/new_pd_wip/v6/pd_2d_stats.txt
