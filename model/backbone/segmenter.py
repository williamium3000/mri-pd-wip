
backbone_cfg = {
    "vit-small":dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'),
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    "vit-tiny":dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'),
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=192,
        num_layers=12,
        num_heads=3,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),

}
    
