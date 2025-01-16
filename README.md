
<div align="center">

Optimum-NVIDIA
===========================
<h4> Optimized inference with NVIDIA and Hugging Face </h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://huggingface.co/docs/optimum/index)
[![python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/downloads/release/python-31013/)
[![cuda](https://img.shields.io/badge/cuda-12.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt-llm](https://img.shields.io/badge/TensorRT--LLM-0.15.0-green)](https://github.com/nvidia/tensorrt-llm)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

---
<div align="left">

Optimum-NVIDIA delivers the best inference performance on the NVIDIA platform through Hugging Face. Run LLaMA 2 at 1,200 tokens/second (up to 28x faster than the framework) by changing just a single line in your existing transformers code.

</div></div>

# Installation

## Pip

Pip installation flow has been validated on Ubuntu only at this stage.

```shell
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
python -m pip install --pre --extra-index-url https://pypi.nvidia.com optimum-nvidia
``` 

For developers who want to target the best performances, please look at the installation methods below.

## Docker container
You can use a Docker container to try Optimum-NVIDIA today. Images are available on the Hugging Face Docker Hub.

```bash
docker pull huggingface/optimum-nvidia
```

## Building from source
<!---
Currently, TRT LLM is built and run with Docker, so we should wait until pip installation is available;
Ideally the user doesn't need to use docker at all to build from source, they should be able to run something like
`git clone [...] && pip install -e optimum-nvidia`
-->

Instead of using the pre-built docker container, you can build Optimum-NVIDIA from source:
```bash
TARGET_SM="90-real;89-real"
git clone --recursive --depth=1 https://github.com/huggingface/optimum-nvidia.git
cd optimum-nvidia/third-party/tensorrt-llm
make -C docker release_build CUDA_ARCHS=$TARGET_SM
cd ../.. && docker build -t <organisation_name/image_name>:<version> -f docker/Dockerfile .
```

<!-- 
```bash
git clone git@github.com:huggingface/optimum-nvidia.git
cd optimum-nvidia
docker build Dockerfile
docker run optimum-nvidia
``` -->

# Quickstart Guide
## Pipelines

Hugging Face pipelines provide a simple yet powerful abstraction to quickly set up inference. If you already have a pipeline from transformers, you can unlock the performance benefits of Optimum-NVIDIA by just changing one line.

```diff
- from transformers.pipelines import pipeline
+ from optimum.nvidia.pipelines import pipeline

pipe = pipeline('text-generation', 'meta-llama/Llama-2-7b-chat-hf', use_fp8=True)
pipe("Describe a real-world application of AI in sustainable energy.")
```

## Generate

If you want control over advanced features like quantization and token selection strategies, we recommend using the `generate()` API. Just like with `pipelines`, switching from existing transformers code is super simple.

```diff
- from transformers import AutoModelForCausalLM
+ from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
  "meta-llama/Llama-2-7b-chat-hf",
+ use_fp8=True,
+ max_prompt_length=1024,
+ max_output_length=2048, # Must be at least size of max_prompt_length + max_new_tokens
+ max_batch_size=8,
)

model_inputs = tokenizer(["How is autonomous vehicle technology transforming the future of transportation and urban planning?"], return_tensors="pt").to("cuda")

generated_ids = model.generate(
    **model_inputs, 
    top_k=40, 
    top_p=0.7, 
    repetition_penalty=10,
)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

To learn more about text generation with LLMs, check out [this guide](https://huggingface.co/docs/transformers/llm_tutorial)!

<!-- For more details, read our [documentation](https://huggingface.com/docs/optimum/nvidia/index). -->

# Support Matrix
We test Optimum-NVIDIA on 4090, L40S, and H100 Tensor Core GPUs, though it is expected to work on any GPU based on the following architectures: 
* Ampere (A100/A30 are supported. Experimental support for A10, A40, RTX Ax000)
* Hopper
* Ada-Lovelace

Note that FP8 support is only available on GPUs based on Hopper and Ada-Lovelace architectures.

Optimum-NVIDIA works on Linux will support Windows soon.

Optimum-NVIDIA currently accelerates text-generation with LLaMAForCausalLM, and we are actively working to expand support to include more model architectures and tasks.

<!-- Optimum-NVIDIA supports the following model architectures and tasks:

| Model             | Tasks           |
| :----             | :----           |
| Gemma             | TextGeneration  |
| Llama             | TextGeneration  |
| Mistral           | TextGeneration  |
| Additional Models | Coming soon     | -->

# Contributing

Check out our [Contributing Guide](./CONTRIBUTING.md)
