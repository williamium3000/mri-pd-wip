
backbone_norm_cfg = dict(type='LN', requires_grad=True)
backbone_cfg = {
        "swin-tiny":dict(
            type='SwinTransformer',
                init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'),
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg), # output chennels 96, 192, 384, 768
        "swin-small":dict(
                type='SwinTransformer',
                init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'),
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg), # output chennels 96, 192, 384, 768
    }
 