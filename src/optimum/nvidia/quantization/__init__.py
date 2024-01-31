from typing import Tuple, Mapping, Union

import numpy as np
import torch

from .awq import to_awq_module
from .base import Calibration, HfDatasetCalibration
from .receipes import get_default_calibration_dataset
from .utils import unpack_int32_into_int8
from ..configs import QuantizationConfig

_AWQ_TENSOR_NAME_EXT = {"qweight", "qzeros", "scales"}
_GPTQ_TENSOR_NAME_EXT = {"g_idx"}

_AWQ_GPTQ_TENSOR_NAME_EXT = _AWQ_TENSOR_NAME_EXT.union(_GPTQ_TENSOR_NAME_EXT)


def quantizable(
    weights: Mapping[str, np.array],
    name: str,
    qconfig: QuantizationConfig
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Handle the retrieving of potentially quantized weights in various name and format
    :param weights: Weight dictionary holding all the weights
    :param name: Original name of the tensor to retrieve
    :param qconfig: The quantization configuration
    :return:
        - Single tensor: raw, high-precision tensor
        - Two tensors: quantized tensor and scales
        - Three tensors: quantized tensor, scales and zero-points
    """

    if qconfig.has_quantization_step:
        if qconfig.mode.is_weight_only():
            pack = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
            preprocess = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
            qdtype = torch.quint4x2 if qconfig.mode.is_int4_weight_only() else torch.int8

            tensor_name_parts = name.split(".")[:-1]
            qscales = weights[".".join(tensor_name_parts + ["scales"])]

            # Extract the weight stored in int32
            qweight_i32 = weights[".".join(tensor_name_parts + ["qweight"])]
            qweight_i8 = unpack_int32_into_int8(qweight_i32, center=True)
            qweight_packed_qi4x2 = preprocess(pack(qweight_i8), )

            if (tensor_zero_point_name := ".".join(tensor_name_parts + ["qzeros"])) in weights:
                qzeros = weights[tensor_zero_point_name]
                if qzeros.dtype == torch.int32:
                    qzeros = unpack_int32_into_int8(qzeros)

                return qweight_packed_qi4x2, qscales, qzeros
            else:
                return qweight_packed_qi4x2, qscales
    else:
        return weights[name]
