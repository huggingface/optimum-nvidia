FROM tensorrt_llm/release:latest as BUILDER

ARG TARGET_CUDA_ARCHS="80-real;86-real;89-real;90-real"

COPY . /opt/optimum-nvidia

# Install tensorrt-llm
# TODO: Reduce the container size removing build artifacts
WORKDIR /opt/optimum-nvidia/third-party/tensorrt-llm
RUN python3 scripts/build_wheel.py -j --trt_root /usr/local/tensorrt --python_bindings --cuda_architectures=${TARGET_CUDA_ARCHS}

FROM tensorrt_llm/release

# Delete examples as they are quite big
RUN rm -rf /app/tensorrt_llm/examples

# Install TRTLLM
COPY --from=BUILDER /opt/optimum-nvidia/third-party/tensorrt-llm/build /opt/trtllm/build

RUN cp /opt/trtllm/build/lib/tensorrt_llm/bindings*.so /usr/local/lib/python3.10/dist-packages/tensorrt_llm/ && \
    python3 -m pip install --no-cache-dir /opt/trtllm/build/tensorrt_llm*.whl && \
    rm -rf /opt/trtllm

# Install NVIDIA Ammo Quantization Framework
ARG NVIDIA_AMMO_VERSION=0.3.0
ARG NVIDIA_AMMO_DOWNLOAD_URL=https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-${NVIDIA_AMMO_VERSION}.tar.gz

RUN cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | awk -F'[V.]' '{print $2$3}') && \
    # Obtain the python version from the system.
    python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}') && \
    # Download and install the AMMO package from the DevZone.
    wget ${NVIDIA_AMMO_DOWNLOAD_URL} && \
    tar -xzf nvidia_ammo-${NVIDIA_AMMO_VERSION}.tar.gz && \
    pip install --no-cache-dir nvidia_ammo-${NVIDIA_AMMO_VERSION}/nvidia_ammo-${NVIDIA_AMMO_VERSION}+cu$cuda_version-cp$python_version-cp$python_version-linux_x86_64.whl && \
    rm -rf nvidia_ammo-${NVIDIA_AMMO_VERSION}*

COPY . /opt/optimum-nvidia

# Install dependencies
RUN python -m pip install --upgrade --no-cache-dir datasets huggingface_hub hf-transfer transformers pynvml
ENV PYTHONPATH=/opt/optimum-nvidia/src:$PYTHONPATH

WORKDIR /opt/optimum-nvidia/examples
