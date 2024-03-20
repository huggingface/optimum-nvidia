#  coding=utf-8
#  Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

DEFAULT_HF_HUB_TRT_REVISION: str = "trt-llm"

DEFAULT_ENGINE_FOLDER = ".engine"
TENSORRT_ENGINE_EXT = "engine"
TENSORRT_TIMINGS_FILE = "timings.cache"
OPTIMUM_NVIDIA_CONFIG_FILENAME = "build"
OPTIMUM_NVIDIA_CONFIG_FILE = f"{OPTIMUM_NVIDIA_CONFIG_FILENAME}.json"

# Nvidia Compute Capabilities
SM_VOLTA = 70
SM_TURING = 75
SM_AMPERE = 80
SM_AMPERE_NEXT = 86
SM_ADA_LOVELACE = 89
SM_HOPPER = 90
