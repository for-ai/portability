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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests


class TestTorch(TestCase):
    def test_device(self, device):
        cpu = torch.device('cpu')
        self.assertEqual('cpu', str(cpu))
        self.assertEqual('cpu', cpu.type)
        self.assertEqual(None, cpu.index)

        cpu0 = torch.device('cpu:0')
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cpu0 = torch.device('cpu', 0)
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cuda = torch.device('cuda')
        self.assertEqual('cuda', str(cuda))
        self.assertEqual('cuda', cuda.type)
        self.assertEqual(None, cuda.index)

        cuda1 = torch.device('cuda:1')
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda1 = torch.device('cuda', 1)
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda90 = torch.device('cuda', 90)
        self.assertEqual('cuda:90', str(cuda90))
        self.assertEqual('cuda', cuda90.type)
        self.assertEqual(90, cuda90.index)

        self.assertRaises(RuntimeError, lambda: torch.device('cpu:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 '))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda: 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2?'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:?2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.232'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2+cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device(-1))

        self.assertRaises(RuntimeError, lambda: torch.device('other'))
        self.assertRaises(RuntimeError, lambda: torch.device('other:0'))

        device_set = {'cpu', 'cpu:0', 'cuda',
                      'cuda:0', 'cuda:1', 'cuda:10', 'cuda:100'}
        device_hash_set = set()
        for device in list(device_set):
            device_hash_set.add(hash(torch.device(device)))
        self.assertEqual(len(device_set), len(device_hash_set))

        def get_expected_device_repr(device):
            if device.index is not None:
                return "device(type='{type}', index={index})".format(
                    type=device.type, index=device.index)

            return "device(type='{type}')".format(type=device.type)

        for device in device_set:
            dev = torch.device(device)
            self.assertEqual(repr(dev), get_expected_device_repr(dev))

instantiate_device_type_tests(TestTorch, globals())

if __name__ == '__main__':
    run_tests()
