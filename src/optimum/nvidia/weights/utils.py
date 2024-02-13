from operator import itemgetter
from typing import List, Tuple, Union

import numpy as np


def shard(tensor: Union[np.array, List[np.array], Tuple[np.array]], rank: int, tp_degree: int, axis: int = 0) -> np.array:
    """
    Shard the specified tensor along the provided axis splitting it up into tp_degree chunks and returning the
    rank-th section
    :param tensor: The tensor to shard
    :param rank: Which section of the tensor to retrieve
    :param tp_degree: How many sections or chunks we need to split it up
    :param axis: On which axis we want the split to happen (default: 0)
    :return:
    """
    if isinstance(tensor, (List, Tuple)):
        return tuple(shard(x_i, rank, tp_degree, axis) for x_i in tensor)
    else:
        if tp_degree == 1:
            return tensor
        else:
            return np.ascontiguousarray(np.split(tensor, tp_degree, axis=axis)[rank])


def stack(tensors: Union[Tuple[np.array], Tuple[Tuple[np.array]]], axis: int = 0) -> Union[np.array, Tuple[np.array]]:
    if all((isinstance(t, np.ndarray) for t in tensors)):
        return np.stack(tensors, axis=axis)
    elif all((isinstance(t, tuple) for t in tensors)):
        length = set([len(t) for t in tensors])

        if not len(length) == 1:
            raise ValueError(f"Not all the provided tuples have the same number of items ({length})")

        packed = tuple(zip(*tensors))
        return tuple(np.stack(t, axis=axis) for t in packed)






