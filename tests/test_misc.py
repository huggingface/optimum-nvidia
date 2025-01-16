from typing import List

import torch

from optimum.nvidia.runtime import InferenceRuntimeBase


def test_inference_runtime_base__as_input_structure():
    outputs_not_batch_not_torch = InferenceRuntimeBase.as_inputs_structure(
        [0, 1, 2, 3], [5, 6, 7, 8]
    )
    assert (
        isinstance(outputs_not_batch_not_torch, List)
        and len(outputs_not_batch_not_torch) == 4
        and isinstance(outputs_not_batch_not_torch[0], int)
    )

    outputs_batch_not_torch = InferenceRuntimeBase.as_inputs_structure(
        [[0, 1], [2, 3]], [[5, 6], [7, 8]]
    )
    assert (
        isinstance(outputs_batch_not_torch, List)
        and len(outputs_batch_not_torch) == 2
        and isinstance(outputs_batch_not_torch[0], List)
        and isinstance(outputs_batch_not_torch[0][0], int)
    )

    outputs_not_batch_torch = InferenceRuntimeBase.as_inputs_structure(
        torch.tensor([0, 1, 2, 3]).to(torch.uint32),
        [5, 6, 7, 8],
    )

    assert (
        isinstance(outputs_not_batch_torch, torch.Tensor)
        and outputs_not_batch_torch.ndim == 1
        and outputs_not_batch_torch.numel() == 4
        and outputs_not_batch_torch.dtype == torch.uint32
    )

    outputs_batch_torch = InferenceRuntimeBase.as_inputs_structure(
        torch.tensor([[0, 1], [2, 3]]).to(torch.uint32),
        [[5, 6], [7, 8]],
    )
    assert (
        isinstance(outputs_batch_torch, torch.Tensor)
        and outputs_batch_torch.ndim == 2
        and outputs_batch_torch.numel() == 4
        and outputs_batch_torch.dtype == torch.uint32
    )
