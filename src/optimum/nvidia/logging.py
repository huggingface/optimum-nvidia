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
from logging import DEBUG, INFO, basicConfig


DEFAULT_LOGGING_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(verbose: bool = False, for_trtllm_logger: bool = False):
    basicConfig(format=DEFAULT_LOGGING_FMT, level=DEBUG if verbose else INFO)

    if for_trtllm_logger:
        from tensorrt_llm.logger import logger

        logger.set_level("verbose" if verbose else "info")
