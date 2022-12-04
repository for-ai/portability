# Owner(s): ["module: nn"]
from functools import reduce
from itertools import repeat
import unittest
import subprocess
import sys
import os
import random
import itertools
import math

from torch._six import inf, nan
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_UBSAN, set_default_dtype, \
    instantiate_parametrized_tests, slowTest, parametrize as parametrize_test, subtest, skipIfMps
from torch.testing._internal.common_cuda import TEST_CUDA
from common_nn import NNTestCase, _test_bfloat16_ops, _test_module_empty_input
from torch.testing._internal.common_device_type import largeTensorTest, onlyNativeDeviceTypes, dtypes, \
    instantiate_device_type_tests, skipCUDAIfRocm, expectedFailureMeta, dtypesIfCUDA, onlyCPU, onlyCUDA, \
    TEST_WITH_ROCM
from torch.testing._internal.common_dtype import floating_types_and
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import gradcheck, gradgradcheck


class TestPoolingNNDeviceType(NNTestCase):
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_adaptive_pooling_zero_batch(self, dtype, device):
        inp = torch.ones(0, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool1d(5).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        inp = torch.ones(0, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool2d((5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

        inp = torch.ones(0, 10, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool3d((5, 5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    @onlyNativeDeviceTypes
    def test_adaptive_avg_pool3d_output_size_one(self, device):
        x = torch.randn((2, 3, 6, 6, 6), dtype=torch.float, device=device, requires_grad=True)

        net = torch.nn.AdaptiveAvgPool3d(1)
        out = net(x)
        ref_out = x.contiguous().mean((-1, -2, -3)).view(out.shape)

        out.sum().backward()    # make sure it doesn't crash

        self.assertEqual(out, ref_out)
        self.assertTrue(out.is_contiguous())
        c = out.size(1)
        self.assertEqual(out.stride(), [c, 1, 1, 1, 1])

    
    @onlyCUDA
    def test_pooling_bfloat16(self, device):
        _test_bfloat16_ops(self, torch.nn.AvgPool1d(3, stride=2), device, inp_dims=(8, 4, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AvgPool2d(3, stride=2), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AvgPool3d(3, stride=2), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool1d(3), device, inp_dims=(8, 4, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool2d((3, 5)), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool3d((3, 5, 7)), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)

    def test_adaptive_pool_invalid(self, device):
        inp_1d = (torch.randn(1, 1, 1, device=device), (-1,))
        inp_2d = (torch.randn(1, 1, 1, 1, device=device), (-1, 0))
        inp_3d = (torch.randn(1, 1, 1, 1, 1, device=device), (-1, 0, 2))
        module_input_dict = {torch.nn.AdaptiveAvgPool1d: inp_1d,
                             torch.nn.AdaptiveAvgPool2d: inp_2d,
                             torch.nn.AdaptiveAvgPool3d: inp_3d}

        for m, inp in module_input_dict.items():
            with self.assertRaisesRegex(RuntimeError,
                                        r"elements of output_size must be greater than or equal to 0"):
                t, output_size = inp
                m(output_size)(t)

instantiate_device_type_tests(TestPoolingNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()