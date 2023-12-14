from unittest import TestCase

import numpy as np
from parameterized import parameterized

from optimum.nvidia.weights import shard


TENSOR_DIM_0 = 1024
TENSOR_DIM_1 = 4096


class MatrixShardingTestCase(TestCase):

    def setUp(self):
        self.tensor = np.random.rand(TENSOR_DIM_0, TENSOR_DIM_1)

    def test_no_sharding(self):
        sharded_tensor = shard(self.tensor, 0, 1, axis=0)
        self.assertTrue(np.array_equal(sharded_tensor, self.tensor))

    @parameterized.expand([1, 2, 4, 8])
    def test_sharding_tensor_parallelism_axis_0(self, tp_degree: int):
        shard_size = TENSOR_DIM_0 // tp_degree

        shards = [
            shard(self.tensor, rank, tp_degree, axis=0)
            for rank in range(tp_degree)
        ]

        for rank, tensor in enumerate(shards):
                self.assertTupleEqual(tensor.shape, (TENSOR_DIM_0 // tp_degree, TENSOR_DIM_1))
                self.assertTrue(np.array_equal(tensor, self.tensor[rank * shard_size: (rank + 1) * shard_size]))

    @parameterized.expand([1, 2, 4, 8])
    def test_sharding_tensor_parallelism_axis_1(self, tp_degree: int):
        shard_size = TENSOR_DIM_1 // tp_degree

        shards = [
            shard(self.tensor, rank, tp_degree, axis=1)
            for rank in range(tp_degree)
        ]

        for rank, tensor in enumerate(shards):
            self.assertTupleEqual(tensor.shape, (TENSOR_DIM_0, TENSOR_DIM_1 // tp_degree))
            self.assertTrue(np.array_equal(tensor, self.tensor[:, rank * shard_size: (rank + 1) * shard_size]))
