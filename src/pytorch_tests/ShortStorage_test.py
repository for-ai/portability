# -*- coding: utf-8 -*-
# Owner(s): ["module: tests"]

import torch
import torch.utils.data

from torch.testing._internal.common_utils import (
    TestCase, run_tests)
import torch.backends.quantized
import torch.testing._internal.data

from ..utils.pytorch_device_decorators import instantiate_device_type_tests

# Protects against includes accidentally setting the default dtype


class TestTorch(TestCase):
    def test_element_size(self):
        byte = torch.ByteStorage().element_size()
        char = torch.CharStorage().element_size()
        short = torch.ShortStorage().element_size()
        int = torch.IntStorage().element_size()
        long = torch.LongStorage().element_size()
        float = torch.FloatStorage().element_size()
        double = torch.DoubleStorage().element_size()
        bool = torch.BoolStorage().element_size()
        bfloat16 = torch.BFloat16Storage().element_size()
        complexfloat = torch.ComplexFloatStorage().element_size()
        complexdouble = torch.ComplexDoubleStorage().element_size()

        self.assertEqual(byte, torch.ByteTensor().element_size())
        self.assertEqual(char, torch.CharTensor().element_size())
        self.assertEqual(short, torch.ShortTensor().element_size())
        self.assertEqual(int, torch.IntTensor().element_size())
        self.assertEqual(long, torch.LongTensor().element_size())
        self.assertEqual(float, torch.FloatTensor().element_size())
        self.assertEqual(double, torch.DoubleTensor().element_size())
        self.assertEqual(bool, torch.BoolTensor().element_size())
        self.assertEqual(bfloat16, torch.tensor(
            [], dtype=torch.bfloat16).element_size())
        self.assertEqual(complexfloat, torch.tensor(
            [], dtype=torch.complex64).element_size())
        self.assertEqual(complexdouble, torch.tensor(
            [], dtype=torch.complex128).element_size())

        self.assertGreater(byte, 0)
        self.assertGreater(char, 0)
        self.assertGreater(short, 0)
        self.assertGreater(int, 0)
        self.assertGreater(long, 0)
        self.assertGreater(float, 0)
        self.assertGreater(double, 0)
        self.assertGreater(bool, 0)
        self.assertGreater(bfloat16, 0)
        self.assertGreater(complexfloat, 0)
        self.assertGreater(complexdouble, 0)

        # These tests are portable, not necessarily strict for your system.
        self.assertEqual(byte, 1)
        self.assertEqual(char, 1)
        self.assertEqual(bool, 1)
        self.assertGreaterEqual(short, 2)
        self.assertGreaterEqual(int, 2)
        self.assertGreaterEqual(int, short)
        self.assertGreaterEqual(long, 4)
        self.assertGreaterEqual(long, int)
        self.assertGreaterEqual(double, float)


instantiate_device_type_tests(TestTorch, globals())
if __name__ == '__main__':
    run_tests()
