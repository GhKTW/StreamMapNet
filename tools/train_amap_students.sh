#!/usr/bin/env bash
# Train the two AMap students one after the other on GPUs 1,2: KD student, then baseline.
# A run that crashes is resumed from its latest.pth (up to MAX_RETRIES times);
# a run that already reached the last iteration is skipped.
#   setsid nohup bash tools/train_amap_students.sh > logs/train_amap_students.out 2>&1 &
cd "$(dirname "$0")/.."

MAX_RETRIES=${MAX_RETRIES:-2}
mkdir -p logs

finished() {
    # $1: run name -> true if work_dir/latest.pth is the checkpoint of the last iteration
    python - "$1" <<'EOF'
import os, sys
from mmcv import Config
cfg = {'kd': 'plugin/configs/amap/amap_student_r18_kd.py',
       'baseline': 'plugin/configs/amap/amap_student_r18_baseline.py'}[sys.argv[1]]
cfg = Config.fromfile(cfg)
latest = os.path.join(cfg.work_dir, 'latest.pth')
done = os.path.islink(latest) and os.readlink(latest) == f'iter_{cfg.runner.max_iters}.pth'
sys.exit(0 if done else 1)
EOF
}

for RUN in kd baseline; do
    for ATTEMPT in $(seq 0 "${MAX_RETRIES}"); do
        if finished "${RUN}"; then
            echo "[$(date -u '+%F %T')] ${RUN}: finished"
            break
        fi
        echo "[$(date -u '+%F %T')] ${RUN}: start (attempt ${ATTEMPT})"
        bash tools/train_amap.sh "${RUN}" >> "logs/train_amap_${RUN}.out" 2>&1
        echo "[$(date -u '+%F %T')] ${RUN}: exit code $?"
    done
    if ! finished "${RUN}"; then
        echo "[$(date -u '+%F %T')] ${RUN}: not finished after ${MAX_RETRIES} retries, see logs/train_amap_${RUN}.out"
    fi
done
echo "[$(date -u '+%F %T')] all done"
