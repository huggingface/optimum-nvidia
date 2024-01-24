#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from .builder import TensorRTEngineBuilder, TensorRTForCausalLMEngineBuilder, TensorRTForSpeechSeq2SeqEngineBuilder
from .logging import DEFAULT_LOGGING_FMT, setup_logging
from .models import AutoModelForCausalLM
from .runtime import TensorRTForCausalLM, TensorRTPreTrainedModel, TensorRTForSpeechSeq2Seq
from .version import VERSION, __version__

# TODO: probably no need to have these top level but that would be breaking to remove them
from .utils.constants import DEFAULT_HF_HUB_TRT_REVISION, TENSORRT_ENGINE_EXT, DEFAULT_ENGINE_FOLDER, TENSORRT_TIMINGS_FILE, OPTIMUM_NVIDIA_CONFIG_FILENAME, OPTIMUM_NVIDIA_CONFIG_FILE


# from .pipelines import pipeline
