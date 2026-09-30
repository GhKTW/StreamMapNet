# StreamMapNet 실행 환경 (코드는 이미지에 넣지 않고 실행 시 마운트)
# - README 버전 그대로: Python 3.8 / torch 1.9.0+cu111 / mmcv-full 1.6.0 / mmdet 2.28.2 / mmdet3d 1.0.0rc6
# - sm_86 바이너리는 sm_89(RTX 40xx)에서도 돌아가므로 3090·4070 공용
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git openssh-client wget ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ninja-build \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel

# PyTorch
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
        -f https://download.pytorch.org/whl/torch_stable.html

# MMCV 계열 (mmcv-full은 미리 빌드된 wheel 사용)
RUN pip install mmcv-full==1.6.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && pip install mmdet==2.28.2 mmsegmentation==0.30.0 "yapf==0.40.1"

# mmdetection3d v1.0.0rc6 (CUDA op 빌드 포함)
RUN git clone --depth 1 --branch v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d \
    && cd /opt/mmdetection3d && pip install -e .

# StreamMapNet requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# av2가 numpy를 1.21+로 올리므로 mmdet3d가 고정한 numba 0.53과 충돌 → 호환 조합으로 맞춤
# (numpy 1.24부터 np.int 등이 제거되어 mmdet3d 코드가 깨지므로 1.23.x 유지)
RUN pip install "numpy==1.23.5" "numba==0.56.4"

WORKDIR /workspace/StreamMapNet
CMD ["bash"]
