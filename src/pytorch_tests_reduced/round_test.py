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

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


# Copied from `test_ops.py` for the purposes of duplicating `test_numpy_ref`
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811
    NNTestCase = object  # noqa: F811

class TestRound(TestCase):
    def test_rounding(self, device):
        value = torch.tensor(1.5, device=device)
        with pytorch_op_timer():
            rounded_value = value.round()
        self.assertEqual(rounded_value, 2)
        value = torch.tensor(1.4, device=device)
        with pytorch_op_timer():
            rounded_value = value.round()
        self.assertEqual(rounded_value, 1)


instantiate_device_type_tests(TestRound, globals())

if __name__ == "__main__":
    run_tests()
