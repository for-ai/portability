# Owner(s): ["oncall: distributed"]

import copy
import math
import io
import itertools
import pickle
import sys
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.api import (
    shard_parameter,
    _shard_tensor,
    load_with_process_group,
    _collect_local_shard,
    _reshard_output,
)
from torch.distributed._shard.sharded_tensor import (
    custom_sharded_op_impl,
    pre_load_state_dict_hook,
    state_dict_hook,
    ShardedTensor,
    Shard
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device
)
from torch.distributed._shard.sharded_tensor.api import (
    TensorProperties,
    _create_tensor_from_params,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    TestCase,
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
    sandcastle_skip_if,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.distributed.remote_device import _remote_device
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
    MyShardedModel1,
)

if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan as torch + multiprocessing spawn have known issues", file=sys.stderr)
    sys.exit(0)




class TestShardParameter(ShardedTensorTestBase):

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_like(self):
        """ Test tensor like methods, i.e. torch.zeros_like(...), torch.full_like, etc. """

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 8
        expected_h = 2
        seed = 1234
        dtype = torch.double
        expected_device = torch.device(f"cuda:{self.rank}")
        st = sharded_tensor.rand(spec, (h, w), dtype=dtype)
        tensor_like_ops = {
            torch.randn_like: torch.randn,

        }
        for op, expect_local_op in tensor_like_ops.items():
            if op == torch.full_like:
                # special handle full/full_like as it needs to have additional fill_value arg
                expect_tensor = expect_local_op((expected_h, w), 8.8, device=expected_device, dtype=dtype)
                new_op_st = op(st, 8.8, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)
            elif op == torch.empty_like:
                # empty/empty_like we only compare the shape
                expect_tensor = expect_local_op(expected_h, w, device=expected_device, dtype=dtype)
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor().shape, expect_tensor.shape)
            else:
                torch.manual_seed(seed)
                expect_tensor = expect_local_op(expected_h, w, device=expected_device, dtype=dtype)
                torch.manual_seed(seed)
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)

    
if __name__ == '__main__':
    run_tests()
