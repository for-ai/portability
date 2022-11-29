# -*- coding: utf-8 -*-
# Owner(s): ["module: tests"]

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCUDA, dtypes, skipMeta,
    onlyNativeDeviceTypes)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.utils.dlpack import from_dlpack, to_dlpack

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

class TestTorchDlPack(TestCase):
    exact_dtype = True

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        self.assertEqual(x, y)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack_noncontinguous(self, device, dtype):
        x = make_tensor((25,), dtype=dtype, device=device).reshape(5, 5)

        y1 = x[0]
        y1_dl = torch.from_dlpack(y1)
        self.assertEqual(y1, y1_dl)

        y2 = x[:, 0]
        y2_dl = torch.from_dlpack(y2)
        self.assertEqual(y2, y2_dl)

        y3 = x[1, :]
        y3_dl = torch.from_dlpack(y3)
        self.assertEqual(y3, y3_dl)

        y4 = x[1]
        y4_dl = torch.from_dlpack(y4)
        self.assertEqual(y4, y4_dl)

        y5 = x.t()
        y5_dl = torch.from_dlpack(y5)
        self.assertEqual(y5, y5_dl)


    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack_dtype(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        assert x.dtype == y.dtype

instantiate_device_type_tests(TestTorchDlPack, globals())

if __name__ == '__main__':
    run_tests()
