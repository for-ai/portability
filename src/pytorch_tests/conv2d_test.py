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
    skipIfRocmVersionLessThan, skipIfNotMiopenSuggestNHWC, TEST_SCIPY, TEST_WITH_ROCM, \
    download_file, parametrize as parametrize_test, subtest, \
    instantiate_parametrized_tests, set_default_dtype
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIfRocmVersionLessThan, skipCUDAIfNotMiopenSuggestNHWC, \
    onlyNativeDeviceTypes, largeTensorTest, skipMeta, \
    disableMkldnn, skipCPUIfNoMkldnn, disablecuDNN, skipCUDAIfMiopen, skipCUDAIfNoMiopen

from torch.testing import make_tensor
from torch.testing._internal.common_utils import gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()


if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

class TestConvolutionNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_Conv2d_module_same_padding(self):
        # Compare module against functional:
        # without strides/dilation, both symmetric and asymmetric padding
        x = torch.rand(1, 1, 9, 20)
        module = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 10),
                           padding='same')
        expect = F.conv2d(x, module.weight, module.bias, padding='same')
        self.assertEqual(expect, module(x))

        # with dilation, symmetric padding
        module = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 4),
                           padding='same', dilation=(1, 2))
        expect = F.conv2d(x, module.weight, module.bias, padding='same', dilation=(1, 2))
        self.assertEqual(expect, module(x))

        # Test non-zero padding_mode, requiring explicit padding
        module = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 4),
                           padding='same', padding_mode='reflect')
        x_padded = F.pad(x, [1, 2, 1, 1], mode='reflect')
        expect = F.conv2d(x_padded, module.weight, module.bias, padding='valid')
        self.assertEqual(expect, module(x))
        self.assertEqual(x.size(), expect.size())

        # Test connstruction with invalid padding string raises
        with self.assertRaisesRegex(ValueError, 'Invalid padding string'):
            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, padding='foo')

        # Test connstruction with same padding and strides raises
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, padding='same', stride=2)
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, padding='same', stride=(1, 3))
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, padding='same', stride=(4, 1))

    def test_Conv2d_inconsistent_types(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float)
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double)
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_Conv2d_inconsistent_types_on_GPU_without_cudnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        with torch.backends.cudnn.flags(enabled=False):
            # inconsistent types should raise an exception
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def test_Conv2d_1x1(self):
        in_channels = 2
        out_channels = 2
        mod = torch.nn.Conv2d(2, 2, 1, bias=False).to(dtype=torch.double)
        input = torch.randn(1, in_channels, 5, 5, requires_grad=True, dtype=torch.double)
        for enabled in (False, True):
            with torch.backends.mkldnn.flags(enabled=enabled):
                gradcheck(F.conv2d, (input, mod.weight))

    def test_Conv2d_OneDNN(self):
        def run_once(group_val=24, dilation=1):
            ifm = torch.ones([1, group_val, 6, 6], dtype=torch.float32)
            weights = torch.ones([group_val, 1, 3, 3], dtype=torch.float32)
            op = torch.nn.Conv2d(
                in_channels=group_val,
                out_channels=group_val,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[dilation, dilation],
                groups=group_val,
                bias=False,
                padding_mode='zeros'
            )

            op.weight.data = weights
            res = op(ifm)
            grad_in = torch.ones(res.shape, dtype=torch.float32)
            res.backward(grad_in)
            return op.weight.grad

        for gorup_val in (24, 48, 23, 25):
            for dilation in (1, 2):
                with torch.backends.mkldnn.flags(enabled=False):
                    without_onednn = run_once(gorup_val, dilation)

                with torch.backends.mkldnn.flags(enabled=True):
                    with_onednn = run_once(gorup_val, dilation)

                self.assertEqual(without_onednn, with_onednn)


    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_Conv2d_inconsistent_types_on_GPU_with_cudnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        with torch.backends.cudnn.flags(enabled=True):
            # inconsistent types should raise an exception
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def test_Conv2d_missing_argument(self):
        c = nn.Conv2d(3, 3, 3)
        self.assertRaises(TypeError, lambda: c(None))

    def test_Conv2d_backward_twice(self):
        input = torch.randn(2, 3, 5, 5)
        c = nn.Conv2d(3, 3, 3)
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: o1.sum().backward())


    # For https://github.com/pytorch/pytorch/pull/1273
    # Almost identical to the above `test_Conv2d_naive_groups`
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    def test_Conv2d_groups_nobias(self):
        dev_dtypes = [("cpu", torch.float)]
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        if AMPERE_OR_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype], rtol=0)

    # Almost identical to the above `test_Conv2d_naive_groups`
    # Covering special case when group > 1, input-channel / group < 16 and output-channel is multiple of 16
    # See also https://github.com/pytorch/pytorch/pull/18463#issuecomment-476563686
    # and https://github.com/pytorch/pytorch/pull/18463#issuecomment-477001024
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    def test_Conv2d_groups_nobias_v2(self):
        torch.manual_seed(123)
        dev_dtypes = [("cpu", torch.float)]
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        if AMPERE_OR_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:8])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :8].contiguous())

            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[8:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 8:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype], rtol=0)

    


class TestConvolutionNNDeviceType(NNTestCase):
    
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(*floating_and_complex_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else []))
    def test_Conv2d_deterministic_cudnn(self, device, dtype):
        inputs = torch.randn(2, 3, 5, 5, device=device, dtype=dtype, requires_grad=True)
        with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
            conv1 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2.bias.data.copy_(conv1.bias.data)
            conv2.weight.data.copy_(conv1.weight.data)
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            self.assertEqual(out1, out2, atol=0.0, rtol=0)
            y = torch.randn(out1.size(), device=device, dtype=dtype)
            out1.backward(y)
            out2.backward(y)
            self.assertEqual(conv1.bias.grad.data, conv2.bias.grad.data, atol=0.0, rtol=0)
            self.assertEqual(conv1.weight.grad.data, conv2.weight.grad.data, atol=0.0, rtol=0)


    @onlyCUDA
    @dtypes(*floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else []))
    def test_Conv2d_large_workspace(self, device, dtype):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            with torch.backends.cudnn.flags(benchmark=benchmark):
                conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device, dtype)
                for size in sizes:
                    x = torch.randn(size, device=device, dtype=dtype)
                    out = conv(x.detach().clone().requires_grad_())
                    out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)


    @onlyCUDA
    @tf32_on_and_off(0.01)
    @dtypes(torch.float, torch.double, torch.half)
    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    def test_Conv2d_depthwise_naive_groups(self, device, dtype):
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(device, dtype)
            i = torch.randn(2, 2, 6, 6, device="cuda", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)
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
                             atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    def test_Conv2d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.bias.grad.data, conv_cuda.bias.grad.data, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(conv_cpu.weight.grad.data, conv_cuda.weight.grad.data, atol=1e-5, rtol=0, exact_device=False)

    @dtypesIfCUDA(*floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else []))
    @dtypes(torch.float)
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    def test_Conv2d_naive_groups(self, device, dtype):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @dtypes(torch.double, torch.cdouble)
    def test_Conv2d_backward_depthwise(self, device, dtype):
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        def conv2d_depthwise(x, weight):
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2)

        for cudnn_enabled in [False, True]:
            with torch.backends.cudnn.flags(enabled=cudnn_enabled):
                torch.autograd.gradcheck(conv2d_depthwise, (x, weight))

    
instantiate_device_type_tests(TestConvolutionNNDeviceType, globals())
instantiate_parametrized_tests(TestConvolutionNN)

if __name__ == '__main__':
    run_tests()
