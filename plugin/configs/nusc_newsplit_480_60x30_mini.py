_base_ = [
    './nusc_newsplit_480_60x30_24e.py'
]

# Pipeline check on nuScenes v1.0-mini with a single GPU.
# Too little data for meaningful AP; only checks train -> checkpoint -> eval.

data_root = '/nas2/Rotem_dataset/nuscenes'
out_root = '/nas2/Rotem_dataset/taeu_outputs/StreamMapNet_outputs'
ann_root = out_root + '/ann/nuscenes_mini'
train_ann_file = ann_root + '/nuscenes_map_infos_train_newsplit.pkl'
val_ann_file = ann_root + '/nuscenes_map_infos_val_newsplit.pkl'

work_dir = out_root + '/work_dirs/nusc_newsplit_480_60x30_mini'

# batch_size 4 runs out of memory on a 16GB GPU
num_gpus = 1
batch_size = 2
num_iters_per_epoch = 323 // (num_gpus * batch_size)
num_epochs = 3
num_epochs_single_frame = 1
total_iters = num_epochs * num_iters_per_epoch

model = dict(
    head_cfg=dict(streaming_cfg=dict(batch_size=batch_size)),
    streaming_cfg=dict(batch_size=batch_size),
)

eval_config = dict(
    data_root=data_root,
    ann_file=val_ann_file,
)

data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        data_root=data_root,
        ann_file=train_ann_file,
    ),
    val=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        eval_config=eval_config,
    ),
    test=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        eval_config=eval_config,
    ),
    shuffler_sampler=dict(
        num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
    ),
)

optimizer = dict(lr=5e-4 * (batch_size / 4))
lr_config = dict(warmup_iters=50)

evaluation = dict(interval=total_iters)
checkpoint_config = dict(interval=total_iters)

runner = dict(max_iters=total_iters)

log_config = dict(interval=10)
