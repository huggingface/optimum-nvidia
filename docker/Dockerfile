FROM tensorrt_llm/release:latest

# 75 = T4/RTX Quadro
# 80 = A100/A30
# 86 = A10/A40/RTX Axxx
# 89 = L4/L40/L40s/RTX Ada/4090
# 90 = H100/H200
#ARG TARGET_CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real"

COPY . /opt/optimum-nvidia

# Install dependencies
RUN python -m pip install /opt/optimum-nvidia

# Let's put our users in the examples folder
WORKDIR /opt/optimum-nvidia/examples
