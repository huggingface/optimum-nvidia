import numpy as np


def shard(tensor: np.array, rank: int, tp_degree: int, axis: int = 0) -> np.array:
    """
    Shard the specified tensor along the provided axis splitting it up into tp_degree chunks and returning the
    rank-th section
    :param tensor: The tensor to shard
    :param rank: Which section of the tensor to retrieve
    :param tp_degree: How many sections or chunks we need to split it up
    :param axis: On which axis we want the split to happen (default: 0)
    :return:
    """
    if tp_degree == 1:
        return tensor
    else:
        return np.ascontiguousarray(np.split(tensor, tp_degree, axis=axis)[rank])
