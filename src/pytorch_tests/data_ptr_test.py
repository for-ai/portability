# FIXME: Port to a more appropriate test suite
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
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta,
    PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes,
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


class TestTorch(TestCase):
    def test_to(self):
        self._test_to_with_layout(torch.strided)
        is_cuda10_2_or_higher = (
            (torch.version.cuda is not None)
            and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))
        if is_cuda10_2_or_higher:  # in cuda10_1 sparse_csr is beta
            self._test_to_with_layout(torch.sparse_csr)

    def _test_to_with_layout(self, layout):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t),
                          non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(
                t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(
                t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append('cuda:{}'.format(
                        torch.cuda.current_device()))
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(
                    t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(
                    t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(
                    t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5)
        if layout == torch.sparse_csr:
            a = torch.tensor([[0, 1, 2], [2, 0, 3]]).to_sparse_csr()
        test_copy_behavior(a)
        self.assertEqual(a.device, a.to('cpu').device)
        self.assertEqual(a.device, a.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, a.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(a.device, a.to(torch.float32).device)
        self.assertIs(torch.float32, a.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(a), getter(a.to('cpu')))
            self.assertEqual(getter(a), getter(
                a.to(dtype=a.dtype, device=a.device, copy=False)))
            self.assertEqual(getter(a), getter(a.to('cpu', copy=False)))
            self.assertNotEqual(getter(a), getter(a.to('cpu', copy=True)))
        if layout == torch.sparse_csr:
            # TODO: compressed sparse tensors currently don't support data_ptr.
            # Exercising failure will allow us to widen coverage of this test once it does.
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer of Tensor that doesn't have storage"):
                a.data_ptr()
            # While compressed sparse tensors don't have a concept of data_ptr
            # the underlying tensors do. The implementation of to appropriately forwards
            # the call to the components, which is what we're test here.
            test_data_ptr(lambda a: a.values().data_ptr())
            test_data_ptr(lambda a: a.crow_indices().data_ptr())
            test_data_ptr(lambda a: a.col_indices().data_ptr())
        else:
            test_data_ptr(lambda a: a.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    test_copy_behavior(b, non_blocking)
                    self.assertEqual(b.device, b.to(
                        cuda, non_blocking=non_blocking).device)
                    self.assertEqual(a.device, b.to(
                        'cpu', non_blocking=non_blocking).device)
                    self.assertEqual(b.device, a.to(
                        cuda, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to(
                        'cpu', dtype=torch.int32, non_blocking=non_blocking).dtype)
                    self.assertEqual(a.device, b.to(
                        'cpu', dtype=torch.int32, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to(dtype=torch.int32).dtype)
                    self.assertEqual(b.device, b.to(dtype=torch.int32).device)


if __name__ == '__main__':
    run_tests()
