#!/usr/bin/env bash
# nuScenes mini + 최소 모델(R18, head 256ch, decoder 3층) 단일 GPU 학습.
# work_dir(NAS)에 latest.pth가 있으면 거기서 이어서 학습.
#   bash tools/train_mini_tiny.sh            # GPU 2
#   GPU=0 bash tools/train_mini_tiny.sh
set -e
cd "$(dirname "$0")/.."

CONFIG=plugin/configs/nusc_newsplit_480_60x30_mini_tiny.py
WORK_DIR=$(python -c "from mmcv import Config; print(Config.fromfile('${CONFIG}').work_dir)")

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # nvidia-smi 번호와 일치
export CUDA_VISIBLE_DEVICES=${GPU:-2}

RESUME=()
if [ -f "${WORK_DIR}/latest.pth" ]; then
    RESUME=(--resume-from "${WORK_DIR}/latest.pth")
    echo "resume from ${WORK_DIR}/latest.pth"
fi

# config가 SyncBN=True라 분산 launcher(1 proc)로 실행
PORT=${PORT:-29502} bash tools/dist_train.sh ${CONFIG} 1 "${RESUME[@]}"
