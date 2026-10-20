#!/usr/bin/env bash
# Evaluate one AMap run (mAP, A-mAP, R-mAP) on nuScenes newsplit val with GPUs 1,2.
#   bash tools/test_amap.sh teacher|baseline|kd [checkpoint]   # default: work_dir/latest.pth
set -e
cd "$(dirname "$0")/.."

case "$1" in
    teacher)  CONFIG=plugin/configs/amap/amap_teacher_r50_future.py ;;
    baseline) CONFIG=plugin/configs/amap/amap_student_r18_baseline.py ;;
    kd)       CONFIG=plugin/configs/amap/amap_student_r18_kd.py ;;
    *) echo "usage: $0 {teacher|baseline|kd} [checkpoint]"; exit 1 ;;
esac

WORK_DIR=$(python -c "from mmcv import Config; print(Config.fromfile('${CONFIG}').work_dir)")
CKPT=${2:-${WORK_DIR}/latest.pth}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

PORT=${PORT:-29520} bash tools/dist_test.sh ${CONFIG} ${CKPT} ${NUM_GPUS} --eval
