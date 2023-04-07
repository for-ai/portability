# Owner(s): ["module: nn"]
import math
import unittest
import itertools
import warnings
from itertools import product

import torch

import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_dtype import floating_types_and, floating_and_complex_types_and
from torch.testing._internal.common_utils import run_tests, \
    skipIfRocmVersionLessThan, TEST_SCIPY, TEST_WITH_ROCM, \
    download_file, parametrize as parametrize_test, subtest, \
    instantiate_parametrized_tests, set_default_dtype
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from .common_nn import NNTestCase, _test_module_empty_input
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIfRocmVersionLessThan, \
    onlyNativeDeviceTypes, largeTensorTest, skipMeta, \
    disableMkldnn, skipCPUIfNoMkldnn, disablecuDNN, skipCUDAIfMiopen, skipCUDAIfNoMiopen

from torch.testing import make_tensor
from torch.testing._internal.common_utils import gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()


if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

class TestConvolutionNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    
    def test_invalid_conv3d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble]:
            with pytorch_op_timer():
                module = torch.nn.Conv3d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            # Negative stride check
            with pytorch_op_timer():
                module = torch.nn.Conv3d(1, 1, kernel_size=3, stride=-2)
            input = torch.empty(1, 1, 4, 4, 4)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_Conv3d_module_same_padding(self):
        # Compare module against functional:
        x = torch.rand(1, 1, 4, 4, 4)
        # without dilation, both symmetric and asymmetric padding
        with pytorch_op_timer():
            module = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 3, 4),
                           padding='same')
        expect = F.conv3d(x, module.weight, module.bias, padding='same')
        self.assertEqual(expect, module(x))

        # with dilation, both symmetric and asymmetric padding
        with pytorch_op_timer():
            module = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 3, 4),
                           padding='same', dilation=(3, 2, 1))
        expect = F.conv3d(x, module.weight, module.bias, padding='same', dilation=(3, 2, 1))
        self.assertEqual(expect, module(x))

        # Test non-zero padding_mode, requiring explicit padding
        with pytorch_op_timer():
            module = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 3, 4),
                           padding='same', padding_mode='circular')
        x_padded = F.pad(x, [1, 2, 1, 1, 0, 1], mode='circular')
        expect = F.conv3d(x_padded, module.weight, module.bias, padding='valid')
        self.assertEqual(expect, module(x))
        self.assertEqual(x.size(), expect.size())

        # Test connstruction with invalid padding string raises
        with self.assertRaisesRegex(ValueError, 'Invalid padding string'):
            with pytorch_op_timer():
                module = nn.Conv3d(in_channels=3, out_channels=33, kernel_size=10, padding='foo')

    def test_conv_modules_raise_error_on_incorrect_input_size(self):
        for dtype in [torch.bfloat16, torch.double, torch.float]:
            with pytorch_op_timer():
                test_1 =  nn.Conv3d(3, 8, 3).to(dtype)
            modules = [nn.Conv1d(3, 8, 3).to(dtype), nn.ConvTranspose1d(3, 8, 3).to(dtype),
                       nn.Conv2d(3, 8, 3).to(dtype), nn.ConvTranspose2d(3, 8, 3).to(dtype),
                      test_1, nn.ConvTranspose3d(3, 8, 3).to(dtype)]

            invalid_input_dims = [(1, 4), (1, 4),
                                  (2, 5), (2, 5),
                                  (3, 6), (3, 6)]

            for invalid_dims, module in zip(invalid_input_dims, modules):
                for dims in invalid_dims:
                    input = torch.empty(torch.Size((3, ) * dims))
                    self.assertRaises(RuntimeError, lambda: module(input))

    def test_conv_shapecheck(self):
        def test(should_raise, module, input_size, dtype):
            input = torch.empty(3, *input_size).to(dtype)
            if should_raise:
                self.assertRaises(RuntimeError, lambda: module(input))
            else:
                # just run it to ensure no exception raised.
                module(input)

        for dtype in [torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble]:

            # Conv3D
            with pytorch_op_timer():
                test_1 = nn.Conv3d(1, 1, (3, 3, 3)).to(dtype)
            with pytorch_op_timer():
                test_2 = nn.Conv3d(1, 1, (3, 3, 3)).to(dtype)
            with pytorch_op_timer():
                test_3 = nn.Conv3d(1, 1, (3, 3, 3), padding=1).to(dtype)
            test(True, test_1, (1, 2, 2, 2), dtype)
            test(False, test_2, (1, 3, 3, 3), dtype)
            test(False, test_3, (1, 2, 2, 2), dtype)
    # CPU-only test for group conv3d fast implementation using bmm
    # See: https://github.com/pytorch/pytorch/pull/36355
    def test_Conv3d_groups_nobias(self):
        torch.manual_seed(123)
        with pytorch_op_timer():
            m = nn.Conv3d(4, 16, kernel_size=3, groups=2, bias=False).to("cpu", torch.float)
        i = torch.randn(2, 4, 6, 6, 6, device="cpu", dtype=torch.float, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 16, 4, 4, 4, device="cpu", dtype=torch.float)
        output.backward(grad_output)
        with pytorch_op_timer():
            m1 = nn.Conv3d(2, 8, kernel_size=3, bias=False).to("cpu", torch.float)
        m1.weight.data.copy_(m.weight.data[:8])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :8].contiguous())
        with pytorch_op_timer():
            m2 = nn.Conv3d(2, 8, kernel_size=3, bias=False).to("cpu", torch.float)
        m2.weight.data.copy_(m.weight.data[8:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 8:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         atol=dtype2prec_DONTUSE[torch.float], rtol=0)
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         atol=dtype2prec_DONTUSE[torch.float], rtol=dtype2prec_DONTUSE[torch.float])

    def test_Conv3d_groups_wbias(self):
        torch.manual_seed(123)
        with pytorch_op_timer():
            m = nn.Conv3d(4, 16, kernel_size=3, groups=2, bias=True).to("cpu", torch.float)
        i = torch.randn(2, 4, 6, 6, 6, device="cpu", dtype=torch.float, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 16, 4, 4, 4, device="cpu", dtype=torch.float)
        output.backward(grad_output)
        with pytorch_op_timer():
            m1 = nn.Conv3d(2, 8, kernel_size=3, bias=True).to("cpu", torch.float)
        m1.weight.data.copy_(m.weight.data[:8])
        m1.bias.data.copy_(m.bias.data[:8])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :8].contiguous())
        with pytorch_op_timer():
            m2 = nn.Conv3d(2, 8, kernel_size=3, bias=True).to("cpu", torch.float)
        m2.weight.data.copy_(m.weight.data[8:])
        m2.bias.data.copy_(m.bias.data[8:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 8:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         atol=dtype2prec_DONTUSE[torch.float],
                         rtol=dtype2prec_DONTUSE[torch.float])
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         atol=dtype2prec_DONTUSE[torch.float],
                         rtol=dtype2prec_DONTUSE[torch.float])
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         atol=dtype2prec_DONTUSE[torch.float], rtol=dtype2prec_DONTUSE[torch.float])

    

    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.half)
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    @tf32_on_and_off(0.005)
    def test_Conv3d_depthwise_naive_groups(self, device, dtype):
        for depth_multiplier in [1, 2]:
            with pytorch_op_timer():
                m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(device, dtype)
            i = torch.randn(2, 2, 6, 6, 6, device=device, dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier
            with pytorch_op_timer():
                m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())
            with pytorch_op_timer():
                m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())
            is_cuda_sm86 = device.startswith("cuda") and torch.cuda.get_device_capability(0) == (8, 6)
            atol, rtol = (3e-4, 3e-2) if dtype == torch.float32 and is_cuda_sm86 else (dtype2prec_DONTUSE[dtype], 0)

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             atol=atol, rtol=rtol)
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             atol=atol, rtol=rtol)


    @dtypes(torch.float, torch.cfloat)
    def test_conv_empty_channel(self, device, dtype):
        in_channels = 0
        # mod = torch.nn.Conv1d(in_channels, 8, 2, stride=2, dtype=dtype).to(device)
        # inp = torch.randn(2, 0, 15, device=device, dtype=dtype)
        # _test_module_empty_input(self, mod, inp, check_size=False)

        # with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
        #     inp = torch.randn(2, 1, 0, device=device, dtype=dtype)
        #     mod(inp)

        # mod = torch.nn.Conv2d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        # inp = torch.randn(2, 0, 50, 100, device=device, dtype=dtype)
        # _test_module_empty_input(self, mod, inp, check_size=False)

        # with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
        #     inp = torch.randn(2, 1, 40, 0, device=device, dtype=dtype)
        #     mod(inp)
        with pytorch_op_timer():
            mod = torch.nn.Conv3d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 50, 20, 40, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 50, 0, 40, device=device, dtype=dtype)
            mod(inp)

    
    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(8005)
    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_ndhwc(self, device, dtype):
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            input = torch.randint(-2, 2, (n, c, d, h, w), dtype=dtype, device=device)\
                .to(memory_format=torch.channels_last_3d)
            input.requires_grad_()
            with pytorch_op_timer():
                conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)\
                .to(device=device, dtype=dtype, memory_format=torch.channels_last_3d)
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            with pytorch_op_timer():
                ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(device=device, dtype=torch.double, memory_format=torch.contiguous_format)

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d))

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    # @onlyCUDA
    @largeTensorTest('40GB')
    @largeTensorTest('24GB', 'cpu')
    def test_conv3d_64bit_indexing(self, device):
        x = torch.rand(1, 32, 512, 512, 256)
        with pytorch_op_timer():
            m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False)
        yref = m(x)
        y = m.to(device=device)(x.to(device=device))
        self.assertEqual(yref, y)

instantiate_device_type_tests(TestConvolutionNN, globals())

if __name__ == '__main__':
    run_tests()