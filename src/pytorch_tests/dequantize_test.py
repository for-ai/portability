# Owner(s): ["oncall: quantization"]

import numpy as np
import math
import random
import torch
import io
import unittest
from copy import deepcopy
from hypothesis import given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TestCase, DeterministicGuard
import torch.testing._internal.hypothesis_utils as hu


hu.assert_deadline_disabled()

import itertools
import tempfile

class TestQuantizedTensor(TestCase):
    def _test_dequantize_fp16(self, device):
        data_orig = torch.randn(1, 2, 4, 4, dtype=torch.float, device=device)
        data_fp16 = data_orig.to(torch.float16)
        data_fp16_dequant = data_fp16.dequantize()
        data_fp16_fp32 = data_fp16.to(torch.float)
        self.assertTrue(data_fp16_dequant.dtype == torch.float)
        self.assertTrue(torch.allclose(data_fp16_fp32, data_fp16_dequant))

    def test_dequantize_fp16_cpu(self):
        self._test_dequantize_fp16(torch.device('cpu'))

    @unittest.skipIf(not TEST_CUDA, "No gpu is available.")
    def test_dequantize_fp16_cuda(self):
        self._test_dequantize_fp16(torch.device('cuda'))

    
if __name__ == '__main__':
    unittest.main()

