_base_ = [
    './amap_student_r18_baseline.py'
]

# AMap student: the single-frame R18 student of amap_student_r18_baseline.py, distilled
# from the frozen future-aware teacher (amap_teacher_r50_future.py).
# The data pipeline normalizes for the teacher (caffe style); AMapDistiller re-normalizes
# the images for the student, so at test time the student sees the same input as the baseline.

work_dir = '/nas2/Rotem_dataset/taewoo_outputs/StreamMapNet_outputs/work_dirs/amap/student_r18_kd'

teacher_config = 'plugin/configs/amap/amap_teacher_r50_future.py'
teacher_ckpt = '/nas2/Rotem_dataset/taewoo_outputs/StreamMapNet_outputs/work_dirs/amap/teacher_r50_future/latest.pth'

teacher_img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    _delete_=True,
    type='AMapDistiller',
    student_cfg={{_base_.model}},
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    batch_size={{_base_.batch_size}},
    bev_channels={{_base_.bev_embed_dims}},
    data_img_norm=teacher_img_norm_cfg,
    student_img_norm={{_base_.img_norm_cfg}},
    distill_cfg=dict(
        basic_bev_weight=1.0,
        refined_bev_weight=1.0,
        query_weight=1.0,
        use_bev_mask=True,
        mask_thickness=3,       # BEV cells (0.6 m each)
        match_reg_weight=5.0,
        match_cls_weight=1.0,
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
    dict(type='Normalize3D', **teacher_img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages', size={{_base_.img_size}}, change_intrinsics=True),
    dict(type='Normalize3D', **teacher_img_norm_cfg),
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
