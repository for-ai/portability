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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


class TestNamedTensor(TestCase):
    def test_bitwise_not(self, device):
        # for device in get_all_device_types():
        # names = ('N', 'D')
        # tensor = torch.zeros(2, 3, names=names, dtype=torch.bool, device=device)
        # result = torch.empty(0, dtype=torch.bool, device=device)

        # self.assertEqual(tensor.bitwise_not().names, names)
        # self.assertEqual(torch.bitwise_not(
        #     tensor, out=result).names, names)
        # self.assertEqual(tensor.bitwise_not_().names, names)
        with pytorch_op_timer():
            test = torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8, device=device))
        self.assertEqual(test,torch.tensor([ 0,  1, -4], dtype=torch.int8, device=device))
        

instantiate_device_type_tests(TestNamedTensor, globals())
if __name__ == '__main__':
    run_tests()
