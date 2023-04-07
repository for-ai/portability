# Owner(s): ["module: named tensor"]

import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import get_all_device_types
from collections import namedtuple, OrderedDict
import itertools
import functools
import torch
from torch import Tensor
import torch.nn.functional as F
from multiprocessing.reduction import ForkingPickler
import pickle
import io
import sys
import warnings


class TestNamedTensor(TestCase):
    def test_bitwise_not(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.zeros(2, 3, names=names, dtype=torch.bool)
            result = torch.empty(0, dtype=torch.bool)

            self.assertEqual(tensor.bitwise_not().names, names)
            self.assertEqual(torch.bitwise_not(
                tensor, out=result).names, names)
            self.assertEqual(tensor.bitwise_not_().names, names)


if __name__ == '__main__':
    run_tests()
