# Owner(s): ["oncall: quantization"]

from builtins import round

import copy
import itertools
import numpy as np
import unittest
import operator
import random

import torch
from torch import _VF
import torch.jit
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair

from hypothesis import settings, HealthCheck
from hypothesis import assume, given, note
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import IS_PPC, TEST_WITH_UBSAN, IS_MACOS, BUILD_WITH_CAFFE2
from torch.testing._internal.common_quantization import skipIfNoFBGEMM, skipIfNoQNNPACK
from torch.testing._internal.common_quantized import _quantize, _dequantize, _calculate_dynamic_qparams, \
    override_quantized_engine, supported_qengines, override_qengines, _snr
from torch.testing._internal.common_quantized import (
    qengine_is_qnnpack,
    qengine_is_onednn,
)
from torch.ao.quantization import PerChannelMinMaxObserver
from torch.testing._internal.common_cuda import TEST_CUDNN, TEST_CUDA
import torch.backends.xnnpack

from typing import Optional

np_dtype = {
    torch.quint8 : np.uint8,
    torch.qint8 : np.int8,
    torch.qint32 : np.int32
}

# Make sure we won't have overflows from vpmaddubsw instruction used in FBGEMM.
# On the current Intel x86 architecture, we need to utilize vpmaddubsw instruction
# for the 8-bit int multiplication. This instruction vertically multiplies each
# unsigned 8-bit integer from a with the corresponding signed 8-bit integer from
# b, producing intermediate signed 16-bit integers. This function modifies the
# weights to eliminate the overflow on the signed 16-bit integers.
def avoid_vpmaddubsw_overflow_linear(
    batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max
):
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min

    # Go through the same loop again to double check we don't have any overflow
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)


# Reference quantized Linear operator
def qlinear_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp, dtype=np.uint8):
    X_q = np.reshape(X_q, (-1, X_q.shape[X_q.ndim - 1]))
    row_offsets_ref = X_q.sum(axis=1).astype(np.int32).reshape((-1, 1))
    col_offsets_ref = W_q.sum(axis=1).astype(np.int32).reshape((1, -1))
    assert X_q.ndim == 2
    batch_size, input_channels = X_q.shape
    Prod_XqWq_ref = (
        np.matmul(X_q.astype(np.int32), W_q.astype(np.int32).T)
        - W_zp * row_offsets_ref
        - X_zp * col_offsets_ref
        + input_channels * X_zp * W_zp
    )
    if b_q is not None:
        Prod_XqWq_ref += b_q
    Y_q_ref = _quantize(Prod_XqWq_ref, Y_scale / (X_scale * W_scale), Y_zp, dtype=dtype)
    return Y_q_ref

"""Computes the output shape given pooling parameters."""
def pool_output_shape(input_size, kernel_size, padding, stride,
                      dilation, ceiling_mode=False):
    if stride is None:
        stride = kernel_size
    output_size = (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
         + (stride - 1 if ceiling_mode else 0)) // stride + 1)
    if (ceiling_mode and
            ((output_size - 1) * stride >= input_size + padding)):
        output_size -= 1
    return output_size

"""
Util for creating a random tensor and quantization params when Hypothesis
is undesirable.
"""
def _get_random_tensor_and_q_params(shapes, rand_scale, torch_type):
    X = (torch.rand(*shapes, dtype=torch.float) - 0.5) * rand_scale
    # Calculate reasonable quantization params
    min_val = torch.min(X)
    max_val = torch.max(X)
    if torch_type == torch.qint32:
        X_zero_point = int(torch.randint(-1 * (2 ** 31), 2 ** 31 - 1, (1,)))
        num_bins = 2 ** 32
        X_scale = float(max_val - min_val) / num_bins
    elif torch_type == torch.qint8:
        X_zero_point = int(torch.randint(-128, 127, (1,)))
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    else:  # torch.quint8
        X_zero_point = 127
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    if X_scale == 0:
        X_scale = 1e-10
    return X, X_scale, X_zero_point

class TestQuantizedOps(TestCase):

    
    @skipIfNoFBGEMM
    def test_group_norm(self):
        # hypothesis is flaky for this test, create test cases manually
        batches_list = (1, 7)
        num_groups_list = (1, 4)
        channels_per_groups = (1, 36, 72)
        elements_per_channels = (8, 128, 1024)
        torch_types = (torch.qint8, torch.quint8)
        y_scales = (0.1, 4.23)
        y_zero_points = (0, 1)
        channels_last_list = [True, False]
        affine_list = [True, False]
        combined = [batches_list, num_groups_list, channels_per_groups, elements_per_channels,
                    torch_types, y_scales, y_zero_points, channels_last_list, affine_list]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:

                batches, num_groups, channels_per_group, elements_per_channel, \
                    torch_type, Y_scale, Y_zero_point, channels_last, \
                    affine = test_case
                num_channels = num_groups * channels_per_group
                # minimum rank for channels_last
                shapes = (batches, num_channels, elements_per_channel, 1)

                # In the FP kernel, sums and sums of squares are calculated in floating point.
                # In the int8 and uint8 versions of the quantized kernel, they are
                # calculated in integer arithmetic (which is exact).
                # Because of this, the numerics do not always match exactly which is
                # expected and acceptable. We do the following to allow this failure
                # in this test:
                # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
                #    favors homogeneous inputs in its search strategies which isn't
                #    representative of the inputs we care about, and tends to maximize
                #    this particular numerics difference.
                # 2. allow a small % of off by Y_scale errors.  Even when the
                #    variance of the input is high, there can be off by one errors
                #    in the result if the input value happens to fall exactly on
                #    the bin boundary of the output scale.
                #
                # If we want the numerics to match we could switch to calculating
                # mean+var in floating point in the future, at the cost of speed.
                X, X_scale, X_zero_point = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)

                # Initialize the weights non-randomly for reproducibility
                if affine:
                    weight = torch.ones(num_channels).float() * 0.5
                    bias = torch.ones(num_channels).float()
                    for i in range(num_channels):
                        weight[i] *= i
                        bias[i] *= i
                else:
                    weight = None
                    bias = None

                eps = 0.001

                qX = torch.quantize_per_tensor(X, X_scale, X_zero_point, torch_type)
                if channels_last:
                    qX = qX.contiguous(memory_format=torch.channels_last)
                dqX = qX.dequantize()

                # Enforce non-homogeneous inputs
                for batch_idx in range(batches):
                    for group_idx in range(num_groups):
                        ch_start = group_idx * channels_per_group
                        ch_end = ch_start + channels_per_group
                        group_vals = dqX[batch_idx][ch_start:ch_end]
                        assume(
                            float(torch.unique(group_vals).shape[0]) / group_vals.numel() > 0.001
                            or group_vals.numel() < 5)

                qY = torch.ops.quantized.group_norm(qX, num_groups, weight, bias, eps, Y_scale, Y_zero_point)

                dqY_hat = F.group_norm(dqX, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
                qY_hat = torch.quantize_per_tensor(dqY_hat, Y_scale, Y_zero_point, torch_type)

                # Due to the numerics difference mentioned above between calculating
                # the variance in float vs int, the results can still be slightly
                # different.
                dqY = qY.dequantize()
                dqY_hat = qY_hat.dequantize()
                diff = dqY - dqY_hat

                # off-by-one errors are magnitude of Y_scale
                num_diff = torch.sum(diff > Y_scale * 1.0001)
                pct_diff = float(num_diff) / (diff.numel() + 1e-5)
                num_diff_off_by_one = torch.sum((diff > 0) * (diff <= Y_scale))
                pct_diff_off_by_one = float(num_diff_off_by_one) / (diff.numel() + 1e-5)

                self.assertTrue(pct_diff < 1e-6)
                self.assertTrue(pct_diff_off_by_one < 0.01)

    