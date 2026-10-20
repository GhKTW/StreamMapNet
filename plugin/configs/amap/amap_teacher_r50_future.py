_base_ = [
    './amap_base.py'
]

# AMap teacher: the original StreamMapNet (R50, streaming BEV + query) trained on
# reverse-time sequences, so its GRU memory accumulates the future frames.
# Schedule as the original: first 1/6 of the epochs single-frame shuffled, then sequences.

work_dir = '/nas2/Rotem_dataset/taewoo_outputs/StreamMapNet_outputs/work_dirs/amap/teacher_r50_future'
