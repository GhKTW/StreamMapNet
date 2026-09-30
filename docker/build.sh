#!/bin/bash
# 이미지 빌드: bash docker/build.sh
set -e
cd "$(dirname "$0")/.."
source docker/.env
docker build -f docker/Dockerfile -t "${IMAGE}" .
