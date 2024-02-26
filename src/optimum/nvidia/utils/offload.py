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

import functools
from logging import getLogger

from accelerate import cpu_offload_with_hook
from accelerate.hooks import remove_hook_from_module
from torch.nn import Module


LOGGER = getLogger(__name__)


# Copied from optimum.utils._modeling_utils
def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def maybe_offload_weights_to_cpu(model: Module):
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        if "disk" in devices:
            raise ValueError("disk offload is not supported with quantization")
        if "cpu" in devices and len(model.hf_device_map) > 1:
            hook = None
            for name, device in model.hf_device_map.items():
                if device == "cpu":
                    LOGGER.debug(f"Offloading {name} to device {device}")
                    module = recurse_getattr(model, name)
                    remove_hook_from_module(module, recurse=True)
                    module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)

    return model
