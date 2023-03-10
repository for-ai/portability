# Owner(s): ["module: nn"]
from itertools import product
import unittest
import random
import itertools


import torch
from torch.testing._internal.common_utils import run_tests, set_default_dtype, \
    instantiate_parametrized_tests
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_nn import NNTestCase, freeze_rng_state
from torch.testing._internal.common_device_type import expectedFailureXLA
import torch.nn.functional as F
import torch.nn as nn
from ..utils.timer_wrapper import pytorch_op_timer
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests

class TestDropoutNN(NNTestCase):
    def _test_alpha_dropout(self, cls, input):
        mean = input.mean()
        std = input.std()

        for p in [0.2, 0.5, 0.8]:
            module = cls(p)
            input_var = input.detach().clone().requires_grad_()
            output = module(input_var)
            # output mean should be close to input mean
            self.assertLess(abs(output.data.mean() - mean), 0.1)
            # output std should be close to input std
            self.assertLess(abs(output.data.std() - std), 0.1)
            output.backward(input)

    @onlyAcceleratedDeviceTypes
    def test_native_dropout_corner_case(self, device):
        for train in [True, False]:
            for p in [0.0, 1.0]:
                x = torch.randn(5).to(device=device).requires_grad_()
                x_ref = x.detach().requires_grad_()
                o = torch.native_dropout(x, p, train)[0]
                with pytorch_op_timer():
                    o_ref = torch.dropout(x_ref, p, train)
                o.sum().backward()
                o_ref.sum().backward()
                assert (o.equal(o_ref))
                assert (x.grad.equal(x_ref.grad))

    def test_invalid_dropout_p(self, device):
        v = torch.ones(1, device=device)

        with pytorch_op_timer():
            test_1 = lambda: F.dropout(v, -0.1).to(device)
        self.assertRaises(ValueError, test_1)
        with pytorch_op_timer():
            test_2 = lambda: F.dropout(v, 1.1).to(device)
        self.assertRaises(ValueError, test_2)


class TestDropoutNNDeviceType(NNTestCase):
    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(
            memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(
            memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def _test_dropout_discontiguous(self, cls, device, memory_format=torch.contiguous_format):
        # In this test, we verify that dropout preserves the layout and data for different memory formats.
        # We check whether, we get same values for the output of dropout, when the probability
        # of dropout is 0 or very close to 0.
        # Reference: https://github.com/pytorch/pytorch/issues/47176
        # Should be almost zero but not zero, as for p=0 different path is taken
        close_to_zero_p = 1e-10
        for p in [0, close_to_zero_p]:
            inp = torch.ones(2, 3, 3, 3, device=device)
            inp_discontiguous = torch.empty(
                2, 3, 3, 6, device=device, memory_format=memory_format)[..., ::2]
            inp_discontiguous.copy_(inp)
            mod = cls(p=p)
            out = mod(inp_discontiguous)
            if p != 0:  # Zero will keep strides as is based on input.
                # When prob == 0, input stride (54, 18, 6, 2) -> output stride (54, 18, 6, 2)
                # When prob != 0, input stride (54, 18, 6, 2) -> output stride (27, 9, 3, 1)
                self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(inp_discontiguous, out)

    def _test_dropout_stride_mean_preserve(self, cls, device):
        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2], d[3])

        inp = torch.ones(2, 3, 4, 5, device=device)
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for perm in itertools.permutations((0, 1, 2, 3), r=4):
            for shift in shifts:
                for p in [1e-10, 0.3, 0.5, 0.7]:
                    mod = cls(p=p)
                    permuted_inp = inp.permute(
                        perm).contiguous().permute(invert_perm(perm))
                    permuted_inp = permuted_inp[shift[0]:, shift[1]:, :, :]
                    out = mod(permuted_inp)

                    self.assertTrue(out.permute(perm).is_contiguous())
                    self.assertEqual(inp.mean(), out.mean(),
                                     rtol=0.5, atol=0.5)
                    if p == 1e-10:
                        self.assertEqual(permuted_inp, out)
                    else:
                        self.assertNotEqual(permuted_inp, out)

    #slow - run 1.5 hour, pass
    # def test_Dropout(self, device):
    #     input = torch.empty(1000, device=device)
    #     self._test_dropout(nn.Dropout, device, input)
    #     self._test_dropout_discontiguous(nn.Dropout, device)
    #     self._test_dropout_discontiguous(
    #         nn.Dropout, device, memory_format=torch.channels_last)
    #     self._test_dropout_stride_mean_preserve(nn.Dropout, device)
    #     if self.device_type == 'cuda' or self.device_type == 'cpu':
    #         input = input.bfloat16()
    #         self._test_dropout(nn.Dropout, device, input)

    def _test_dropoutNd_no_batch(self, dropout, input):
        input_clone = input.clone()
        with freeze_rng_state():
            res_no_batch = dropout(input)

        with freeze_rng_state():
            res_batched = dropout(input_clone.unsqueeze(0)).squeeze(0)

        self.assertEqual(res_no_batch, res_batched)

    def _test_dropoutNd_channel_zero(self, dropout, input):
        # Verify the number of zeros in a channel is 0 or the number of elements in the channel
        # for a fully positive input tensor
        shape = input.shape
        B = shape[0]
        C = shape[1]
        channel_numel = torch.tensor(shape[2:]).prod()
        result = dropout(input)

        for b, c in product(range(B), range(C)):
            self.assertTrue(result[b, c].count_nonzero() in (0, channel_numel))

    def test_empty_dropout(self, device):
        x = torch.tensor([]).to(device)
        with pytorch_op_timer():
            out = torch.nn.functional.dropout(x)
        self.assertEqual(out.size(), x.size())


instantiate_device_type_tests(TestDropoutNNDeviceType, globals())
instantiate_device_type_tests(TestDropoutNN, globals())

if __name__ == '__main__':
    run_tests()
