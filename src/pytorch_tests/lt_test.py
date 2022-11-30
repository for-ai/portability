# -*- coding: utf-8 -*-
# Owner(s): ["module: mps"]

import sys
import math
import random
import unittest
import warnings
import subprocess
import tempfile
import os
import pprint
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict
from torch._six import inf
from torch.nn import Parameter
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, run_tests, TestCase, download_file,
     TEST_WITH_UBSAN, TEST_WITH_ASAN, suppress_warnings)
from torch.testing import make_tensor
from torch.testing._comparison import TensorLikePair
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from functools import partial

from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    BinaryUfuncInfo,
)
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
import numpy as np
import torch
import torch.utils._pytree as pytree

class TestNLLLoss(TestCase):
    def test_lt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, mps_y)
            result_cpu = torch.lt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, 0.0)
            result_cpu = torch.lt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))


if __name__ == "__main__":
    run_tests()
