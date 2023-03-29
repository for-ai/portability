# Owner(s): ["module: tests"]

import torch
import numpy as np

from itertools import product, combinations, permutations, chain
from functools import partial
import random
import warnings

from torch._six import nan
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, run_tests, torch_to_numpy_dtype_dict)
from torch.testing._internal.common_device_type import (
    onlyCPU, onlyCUDA, dtypes, onlyNativeDeviceTypes,
    dtypesIfCUDA, largeTensorTest)
from torch.testing._internal.common_dtype import all_types_and_complex_and, all_types, all_types_and
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

class TestShapeOps(TestCase):

    @dtypes(torch.float, torch.bool)
    def test_diag(self, device, dtype):
        if dtype is torch.bool:
            x = torch.rand(100, 100, device=device) >= 0.5
        else:
            x = torch.rand(100, 100, dtype=dtype, device=device)
        with pytorch_op_timer():
            res1 = torch.diag(x).to(device)
        res2 = torch.tensor((), dtype=dtype, device=device)
        with pytorch_op_timer():
            torch.diag(x, out=res2)
        self.assertEqual(res1, res2)

instantiate_device_type_tests(TestShapeOps, globals())

if __name__ == '__main__':
    run_tests()
