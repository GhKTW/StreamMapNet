_base_ = [
    '../nusc_newsplit_480_60x30_24e.py'
]

# Shared setting for the AMap-on-StreamMapNet check (full nuScenes newsplit, 2 GPUs).
# Every dataset split runs in reverse time order (reverse_time=True), so the streaming
# teacher's memory holds the future. Single-frame models are not affected by the order.
# Validation also reports A-mAP / R-mAP (ahead x>0 / rear x<0 halves).
# Values derived from batch size / epochs are recomputed here because mmcv does not
# re-evaluate them in the base file.

data_root = '/nas2/Rotem_dataset/nuscenes'
out_root = '/nas2/Rotem_dataset/taewoo_outputs/StreamMapNet_outputs'
ann_root = out_root + '/ann/nuscenes'
train_ann_file = ann_root + '/nuscenes_map_infos_train_newsplit.pkl'
val_ann_file = ann_root + '/nuscenes_map_infos_val_newsplit.pkl'

# reduced from 480x800 to save compute (multiple of 32, so no padding)
img_h = 384
img_w = 640
img_size = (img_h, img_w)

num_gpus = 2
batch_size = 4
num_iters_per_epoch = 27846 // (num_gpus * batch_size)
num_epochs = 12
num_epochs_single_frame = num_epochs // 6
total_iters = num_epochs * num_iters_per_epoch

# caffe-style normalization (ResNet-50 teacher); the R18 student re-defines it
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    head_cfg=dict(streaming_cfg=dict(batch_size=batch_size)),
    streaming_cfg=dict(batch_size=batch_size),
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
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

eval_config = dict(
    data_root=data_root,
    ann_file=val_ann_file,
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        reverse_time=True,
    ),
    val=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        eval_config=eval_config,
        reverse_time=True,
        eval_regions=True,
    ),
    test=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        eval_config=eval_config,
        reverse_time=True,
        eval_regions=True,
    ),
    shuffler_sampler=dict(
        num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
    ),
)

# the original lr 5e-4 is for a total batch of 32 (8 GPUs x 4). Keep that lr per sample:
# lr 5e-4 at a total batch of 8 diverged (grad norm 1e6+, loss stuck, mAP ~0)
optimizer = dict(lr=5e-4 * (num_gpus * batch_size / 32))

# one evaluation (sequential inference + mAP / A-mAP / R-mAP) takes up to ~1 h;
# rank 1 waits for rank 0 meanwhile, so raise the NCCL timeout (default 30 min)
dist_params = dict(backend='nccl', timeout_minutes=180)
evaluation = dict(interval=4*num_iters_per_epoch)
checkpoint_config = dict(interval=2*num_iters_per_epoch)
runner = dict(max_iters=total_iters)
