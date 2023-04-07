# -*- coding: utf-8 -*-
# Owner(s): ["module: tests"]

import torch
import torch.utils.data
import numpy as np

import contextlib
import gc
import io
import inspect
import itertools
import math
import random
import re
import copy
import os
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import subprocess
import weakref
import sys
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch._six import inf, nan, string_classes
from itertools import product, combinations, permutations
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, TEST_WITH_ROCM, run_tests,
    IS_WINDOWS, IS_FILESYSTEM_UTF8_ENCODING, NO_MULTIPROCESSING_SPAWN,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, load_tests, slowTest,
    TEST_WITH_CROSSREF,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    skipIfNotRegistered, bytes_to_scalar, parametrize, skipIfMps)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta,
    PYTORCH_CUDA_MEMCHECK, largeTensorTest,
    expectedAlertNondeterministic, get_all_device_types, skipXLA)
from typing import Tuple
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, tf32_is_not_fp32, TEST_CUDNN)
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types, integral_types,
)

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()


class TestDevicePrecision(TestCase):
    @dtypes(torch.int64, torch.float32, torch.float64)
    @onlyNativeDeviceTypes
    def test_clamp(self, device, dtype):
        test_args = [
            *product(
                [(100, 50), (10, 64), (97,)],  # shape
                (True, False),  # non-contiguous
            )
        ]

        for shape, noncontig in test_args:
            x = make_tensor(shape, device=device, dtype=dtype,
                            noncontiguous=noncontig)
            ub = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)
            lb = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)

            expect = x.max(lb).min(ub)
            actual = x.clamp(lb, ub)
            self.assertEqual(expect, actual)

            expect = np.clip(
                x.cpu().numpy(), lb.cpu().numpy(), ub.cpu().numpy())
            self.assertEqual(expect, actual)

            expect = x.max(lb)
            actual = x.clamp(min=lb)
            self.assertEqual(expect, actual)

            expect = x.min(ub)
            actual = x.clamp(max=ub)
            self.assertEqual(expect, actual)

            # Test broadcasting min & max
            expect = x.max(lb[0]).min(ub[..., :1])
            actual = x.clamp(lb[0], ub[..., :1])
            self.assertEqual(expect, actual)

            # Test broadcasting x
            expect = x[..., :1].max(lb).min(ub)
            actual = x[..., :1].clamp(lb, ub)
            self.assertEqual(expect, actual)


instantiate_device_type_tests(TestDevicePrecision, globals())
if __name__ == '__main__':
    run_tests()
