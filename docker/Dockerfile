FROM tensorrt_llm/release:latest

# 70 = V100
# 75 = T4/RTX Quadro
# 80 = A100/A30
# 86 = A10/A40/RTX Axxx
# 89 = L4/L40/L40s/RTX Ada/4090
# 90 = H100/H200
#ARG TARGET_CUDA_ARCHS="70-real;75-real;80-real;86-real;89-real;90-real"

COPY . /opt/optimum-nvidia

# Install tensorrt-llm
# TODO: Reduce the container size removing build artifacts
WORKDIR /opt/optimum-nvidia/third-party/tensorrt-llm

# Install dependencies
RUN python -m pip install --upgrade --no-cache-dir datasets huggingface_hub hf-transfer optimum transformers pynvml
ENV PYTHONPATH=/opt/optimum-nvidia/src:$PYTHONPATH

#Add the project sources to the final layer
COPY . /opt/optimum-nvidia
WORKDIR /opt/optimum-nvidia/examples
