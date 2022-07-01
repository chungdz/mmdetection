_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]))
