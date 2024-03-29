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
# from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, run_tests, TestCase, download_file,
     TEST_WITH_UBSAN, TEST_WITH_ASAN, suppress_warnings)
from torch.testing import make_tensor
from torch.testing._comparison import TensorLikePair
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from functools import partial
from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

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
from ..utils.pytorch_device_decorators import onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


class TestNLLLoss(TestCase):

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self, device):
        x_size = 1024
        y_size = 256
        target = torch.rand(x_size, y_size, device=device)

        for reduction in ['none', 'mean', 'sum']:
            output_sig = torch.rand(x_size, y_size, device=device) - 0.5
            output_logits = output_sig.clone().detach()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = torch.rand(y_size, device=device)

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            with pytorch_op_timer():
                loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                    output_logits, target
                )

            # print(loss_logits, loss_sig)

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = torch.rand(x_size, y_size, device=device)
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_grad_at_zero(self, device):
        output = torch.zeros(3, 1, requires_grad=True, device=device)
        target = torch.zeros(3, 1, device=device)
        with pytorch_op_timer():
            nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device=device).fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self, device):
        target = torch.rand(16, 4, device=device)
        output = torch.rand(16, 4, device=device) - 0.5

        weight = torch.rand(4, device=device)
        with pytorch_op_timer():
            out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        with pytorch_op_timer():
            out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device=device)
        with pytorch_op_timer():
            out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        with pytorch_op_timer():
            out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self, device):
        target = torch.rand(64, 4, device=device)
        output = torch.rand(64, 4, device=device) - 0.5
        pos_weight = torch.ones(64, 4, device=device)

        with pytorch_op_timer():
            self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self, device):
        target = torch.rand(64, 4, device=device)
        output = torch.rand(64, 4, device=device) - 0.5
        pos_weight = torch.rand(4, device=device)
        with pytorch_op_timer():
            out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        with pytorch_op_timer():
            out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)

        with pytorch_op_timer():
            out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self, device):
        output = torch.zeros(3, 1, requires_grad=True, device=device)
        target = torch.zeros(3, 1, device=device)
        pos_weight = torch.ones(3, 1, device=device)

        with pytorch_op_timer():
            nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(
                output, target).backward()
        expected_grad = torch.empty(3, 1, device=device).fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self, device):
        output = torch.tensor([0., -120.], device=device)
        target = torch.tensor([0., 1.], device=device)
        pos_weight = torch.tensor([1., 1.], device=device)

        with pytorch_op_timer():
            out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(torch.isfinite(out1).all().item())

        with pytorch_op_timer():
            out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(torch.isfinite(out2).all().item())


instantiate_device_type_tests(TestNLLLoss, globals())

if __name__ == "__main__":
    run_tests()
