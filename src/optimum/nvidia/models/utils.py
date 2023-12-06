import numpy as np


def repeat_heads(tensor: np.array, factor: int, axis: int) -> np.array:
    """
    Repeat `factor` number of times the elements over the specified axis and return the new array
    :param tensor: The original tensor that needs to see its axis repeated
    :param factor: The number of time we should repeat elements over `axis`
    :param axis: Over which axis elements will be repeated
    :return: Tensor with `factor`-repeated axis
    """
    tensor_ = np.expand_dims(tensor, axis=axis).repeat(factor, axis=axis)
    return tensor_.reshape(-1, tensor.shape[-1])