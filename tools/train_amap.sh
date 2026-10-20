#!/usr/bin/env bash
# AMap check on StreamMapNet: train one run on GPUs 1,2 (full nuScenes newsplit).
# If work_dir already has latest.pth, training resumes from it.
#   bash tools/train_amap.sh teacher    # future-aware teacher (R50, reverse-time streaming)
#   bash tools/train_amap.sh baseline   # single-frame student (R18), no KD
#   bash tools/train_amap.sh kd         # single-frame student (R18) + AMap KD; needs the teacher
#   GPUS=1,2 PORT=29510 bash tools/train_amap.sh teacher
set -e
cd "$(dirname "$0")/.."

case "$1" in
    teacher)  CONFIG=plugin/configs/amap/amap_teacher_r50_future.py ;;
    baseline) CONFIG=plugin/configs/amap/amap_student_r18_baseline.py ;;
    kd)       CONFIG=plugin/configs/amap/amap_student_r18_kd.py ;;
    *) echo "usage: $0 {teacher|baseline|kd}"; exit 1 ;;
esac

WORK_DIR=$(python -c "from mmcv import Config; print(Config.fromfile('${CONFIG}').work_dir)")

if [ "$1" = kd ]; then
    TEACHER_CKPT=$(python -c "from mmcv import Config; print(Config.fromfile('${CONFIG}').model.teacher_ckpt)")
    if [ ! -f "${TEACHER_CKPT}" ]; then
        echo "teacher checkpoint not found: ${TEACHER_CKPT} (train the teacher first)"; exit 1
    fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # nvidia-smi 번호와 일치
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
# num_gpus in amap_base.py sets iterations per epoch and lr
if [ "${NUM_GPUS}" != 2 ]; then
    echo "configs assume 2 GPUs (got ${CUDA_VISIBLE_DEVICES}); change num_gpus in amap_base.py first"; exit 1
fi

RESUME=()
if [ -f "${WORK_DIR}/latest.pth" ]; then
    RESUME=(--resume-from "${WORK_DIR}/latest.pth")
    echo "resume from ${WORK_DIR}/latest.pth"
fi

# no elastic auto-restart: it would start over from iteration 0; re-run this script to resume
PORT=${PORT:-29510}
PYTHONPATH=.:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} --max_restarts=0 \
    tools/train.py ${CONFIG} --launcher pytorch "${RESUME[@]}"
