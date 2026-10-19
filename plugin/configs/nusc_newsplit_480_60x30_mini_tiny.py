_base_ = [
    './nusc_newsplit_480_60x30_mini.py'
]

# Smallest runnable setting on nuScenes v1.0-mini (single GPU).
# - image encoder: ResNet-18 (same as nusc_newsplit_480_60x30_24e_r18.py)
# - map head: embed_dims 512 -> 256, decoder 6 -> 3 layers
# Data / schedule are the same as the mini config. Checkpoints every epoch so training can be resumed.

# taeu_outputs is owned by another account (read-only here); ann files are still read from there.
work_dir = '/nas2/Rotem_dataset/taewoo_outputs/StreamMapNet_outputs/work_dirs/nusc_newsplit_480_60x30_mini_tiny'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

bev_embed_dims = 256
embed_dims = 256
num_dec_layers = 3
num_points = 20

model = dict(
    backbone_cfg=dict(
        img_backbone=dict(
            _delete_=True,
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='pytorch',
            with_cp=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
        img_neck=dict(
            in_channels=[128, 256, 512]),
    ),
    head_cfg=dict(
        embed_dims=embed_dims,
        in_channels=bev_embed_dims,
        transformer=dict(
            encoder=dict(embed_dims=embed_dims),
            decoder=dict(
                num_layers=num_dec_layers,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                    ),
                    feedforward_channels=embed_dims*2,
                ),
            ),
        ),
    ),
)

train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim={{_base_.coords_dim}},
        roi_size={{_base_.roi_size}},
        sample_num={{_base_.num_points}},
        normalize=True,
        permute={{_base_.permute}},
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages', size={{_base_.img_size}}, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages', size={{_base_.img_size}}, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

data = dict(
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

checkpoint_config = dict(interval={{_base_.num_iters_per_epoch}})
