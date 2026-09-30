#!/bin/bash
# 컨테이너 생성: bash docker/run.sh
# 접속:         docker exec -it <CONTAINER_NAME> bash
set -e
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

if [ ! -f docker/.env ]; then
    echo "docker/.env가 없습니다. 먼저: cp docker/.env.example docker/.env 후 경로 수정"
    exit 1
fi
source docker/.env

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Container ${CONTAINER_NAME} already exists."
    echo "Remove it first: docker rm -f ${CONTAINER_NAME}"
    echo "Or start it:     docker start ${CONTAINER_NAME}"
    exit 1
fi

# 데이터셋은 코드의 ./datasets/<이름> 위치로 마운트 (config 경로 그대로 사용)
mkdir -p datasets   # 미리 만들어두지 않으면 docker가 root 소유로 생성함
DATA_MOUNTS=()
[ -n "${NAS_DIR}" ]      && DATA_MOUNTS+=(-v "${NAS_DIR}:/nas2")
[ -n "${NUSCENES_DIR}" ] && DATA_MOUNTS+=(-v "${NUSCENES_DIR}:/workspace/StreamMapNet/datasets/nuScenes")
[ -n "${AV2_DIR}" ]      && DATA_MOUNTS+=(-v "${AV2_DIR}:/workspace/StreamMapNet/datasets/av2")

if [ "${GPUS}" = "all" ]; then GPU_OPT="all"; else GPU_OPT="\"device=${GPUS}\""; fi

# 호스트 사용자 UID로 실행 → 마운트된 코드에 root 소유 파일이 생기지 않음
docker run -itd \
    --gpus "${GPU_OPT}" \
    --ipc=host \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e TORCH_HOME=/workspace/StreamMapNet/.cache/torch \
    -v "${REPO_DIR}:/workspace/StreamMapNet" \
    "${DATA_MOUNTS[@]}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE}"

echo ""
echo "      Container '${CONTAINER_NAME}' started."
echo "      to get entered: docker exec -it ${CONTAINER_NAME} bash"
