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
from torch._six import inf, nan, string_classes
from itertools import product, combinations, permutations
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, TEST_WITH_ROCM, run_tests, load_tests)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta,
    PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes,
    get_all_device_types, skipXLA)
from typing import Tuple
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, tf32_is_not_fp32, TEST_CUDNN)
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types,
)

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


# Protects against includes accidentally setting the default dtype
# assert torch.get_default_dtype() is torch.float32

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()


class TestTorch(TestCase):
    exact_dtype = True

    def test_is_same_size(self, device):
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, device=device)
        t3 = torch.empty(1, 9, 3, 3, device=device)
        t4 = torch.empty(3, 4, 9, 10, device=device)

        with pytorch_op_timer():
            result = t1.is_same_size(t2)
        self.assertFalse(result)
        with pytorch_op_timer():
            result = t1.is_same_size(t3)
        self.assertFalse(result)
        with pytorch_op_timer():
            result = t1.is_same_size(t4)
        self.assertTrue(result)


instantiate_device_type_tests(TestTorch, globals())

if __name__ == '__main__':
    run_tests()
