_base_ = [
    './nusc_newsplit_480_60x30_24e.py'
]

# Ablation: weaker image encoder (ResNet-18), everything else identical to the baseline.
# ResNet-18 uses BasicBlock, which does not support DCN, and torchvision weights expect RGB mean/std.

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
