<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 Optimum Nvidia

🤗 Optimum Nvidia provides seamless integrating for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) in the Hugging Face ecosystem.

While TensorRT-LLM provides the foundational blocks to ensure the greatest performances on NVIDIA GPUs, `optimum-nvidia` allows
to leverage the 🤗 to retrieve and load the weights directly inside TensorRT-LLM while maintaining a similar or identical API compared to `transformers` and others 🤗 libraries.

For NVIDIA Tensor Cores GPUs with `float8` hardware acceleration, `optimum-nvidia` allows to run all the necessary preprocessing steps required to target this datatype along with 
deploying the necessary technical blocks to ensure developer experience is fast and smooth for these architectures.