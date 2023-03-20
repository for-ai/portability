# Owner(s): ["module: optimizer"]

import warnings
import math
import unittest
import functools
import itertools
from copy import deepcopy
import torch.optim._multi_tensor as optim_mt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, SGD, Optimizer
from torch import sparse
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests
)
from typing import Dict, Any, Tuple
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


class TestOptim(TestCase):
    exact_dtype = True

    def _build_params_dict(self, weight, bias, **kwargs):
        return [{'params': [weight]}, dict(params=[bias], **kwargs)]

    def _test_basic_cases_template(
        self,
        weight_tensor,
        bias_tensor,
        input_tensor,
        constructor,
        scheduler_constructors,
        constructor_accepts_maximize=True,
        constructor_accepts_foreach=False,
        device=None
    ):
        maximize_options = set([False, constructor_accepts_maximize])
        foreach_options = set([False, constructor_accepts_foreach])

        four_arg_constructor = constructor
        if constructor_accepts_maximize and constructor_accepts_foreach:
            pass
        elif constructor_accepts_maximize:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(foreach)
                with pytorch_op_timer():
                    test_1 = constructor(weight, bias, maximize)
                return test_1
            
        elif constructor_accepts_foreach:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(maximize)
                with pytorch_op_timer():
                    test_2 = constructor(weight, bias, foreach)
                return test_2

        else:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(maximize or foreach)
                with pytorch_op_timer():
                    test_3 = constructor(weight, bias)
                return test_3

        for maximize, foreach in itertools.product(maximize_options, foreach_options):
            with torch.no_grad():
                weight = Parameter(weight_tensor.clone().detach())
                bias = Parameter(bias_tensor.clone().detach())
                input = input_tensor.clone().detach().requires_grad_()
            optimizer = four_arg_constructor(weight, bias, maximize, foreach)
            schedulers = []
            for scheduler_constructor in scheduler_constructors:
                schedulers.append(scheduler_constructor(optimizer))

            # to check if the optimizer can be printed as a string
            optimizer.__repr__()

            def fn():
                optimizer.zero_grad()
                y = weight.mv(input)
                if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                    y = y.cuda(bias.get_device())
                loss = (y + bias).pow(2).sum()
                loss.backward()
                return loss

            initial_value = fn().item()
            for _ in range(200):
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        val_loss = fn()
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                optimizer.step(fn)
            if maximize:
                self.assertGreater(fn().item(), initial_value)
            else:
                self.assertLess(fn().item(), initial_value)

    def _test_state_dict(self, weight, bias, input, constructor, atol=None, rtol=None, device=None):
        weight = Parameter(weight)
        bias = Parameter(bias)
        with torch.no_grad():
            input = input.clone().detach().requires_grad_()

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input.clone().detach().to(dtype=torch.float32, device=device) if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        with pytorch_op_timer():
            optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for _i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        with torch.no_grad():
            weight_c = Parameter(weight.clone().detach())
            bias_c = Parameter(bias.clone().detach())
        with pytorch_op_timer():    
            optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for _ in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)
        # Make sure state dict is deterministic with equal but not identical parameters
        self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
        # Make sure repeated parameters have identical representation in state dict
        optimizer_c.param_groups.extend(optimizer_c.param_groups)
        self.assertEqual(
            optimizer.state_dict()["param_groups"][-1],
            optimizer_c.state_dict()["param_groups"][-1],
        )

        # Make sure that optimizers that support maximize can load older models
        state_dict = optimizer.state_dict()
        if "maximize" in state_dict["param_groups"][0]:
            for group in state_dict["param_groups"]:
                del group["maximize"]
            optimizer.load_state_dict(state_dict)
            # Make sure we can still step
            optimizer.step()
        # Make sure that optimizers that support foreach can load older models
        state_dict = optimizer.state_dict()
        if "foreach" in state_dict["param_groups"][0]:
            for group in state_dict["param_groups"]:
                del group["foreach"]
            optimizer.load_state_dict(state_dict)
            # Make sure we can still step
            optimizer.step()

        # Make sure that loading optimizers with step not wrapped in tensor can work
        state_dict = optimizer.state_dict()
        if "step" in state_dict["state"][0] and torch.is_tensor(
            state_dict["state"][0]["step"]
        ):
            for state in state_dict["state"].values():
                state["step"] = state["step"].item()
            optimizer.load_state_dict(state_dict)
            optimizer.step()

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        with torch.no_grad():
            input_cuda = input.clone().detach().to(dtype=torch.float32, device=device)
            weight_cuda = Parameter(
                weight.clone().detach().to(dtype=torch.float32, device=device)
            )
            bias_cuda = Parameter(
                bias.clone().detach().to(dtype=torch.float32, device=device)
            )
        with pytorch_op_timer():
            optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(
            fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Make sure that device of state['step'] is still CPU
        new_state_dict = optimizer_cuda.state_dict()
        if "step" in state_dict["state"][0] and torch.is_tensor(
            state_dict["state"][0]["step"]
        ):
            for state in new_state_dict["state"].values():
                self.assertEqual(state["step"].device.type, "cpu")

        for _i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda, atol=atol, rtol=rtol)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return set(k for k in obj.__dict__ if not k.startswith("_"))

        self.assertEqual(getPublicAttr(optimizer),
                         getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(
        self,
        constructor,
        scheduler_constructors=None,
        ignore_multidevice=False,
        constructor_accepts_maximize=False,
        constructor_accepts_foreach=False,
        atol=None,
        rtol=None,
        device=None
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []

        def make_two_arg_constructor(
            constructor, maximize: bool = False, foreach: bool = False
        ):
            if constructor_accepts_maximize and constructor_accepts_foreach:
                return lambda weight, bias: constructor(weight, bias, maximize, foreach)
            if constructor_accepts_maximize:
                return lambda weight, bias: constructor(weight, bias, maximize)
            if constructor_accepts_foreach:
                return lambda weight, bias: constructor(weight, bias, foreach)
            return constructor

        for maximize, foreach in itertools.product(
            set([False, constructor_accepts_maximize]),
            set([False, constructor_accepts_foreach]),
        ):
            self._test_state_dict(
                torch.randn(10, 5).to(device),
                torch.randn(10).to(device),
                torch.randn(5).to(device),
                make_two_arg_constructor(constructor, maximize, foreach),
                atol=atol,
                rtol=rtol,
                device=device
            )
        self._test_basic_cases_template(
            torch.randn(10, 5).to(device),
            torch.randn(10).to(device),
            torch.randn(5).to(device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2).to(device)[..., 0],
            torch.randn(10, 2).to(device)[..., 0],
            torch.randn(5).to(device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # CUDA
        # if not torch.cuda.is_available():
        #     return
        self._test_basic_cases_template(
            torch.randn(10, 5).to(device),
            torch.randn(10).to(device),
            torch.randn(5).to(device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # Multi-GPU
        # if not torch.cuda.device_count() > 1 or ignore_multidevice:
        #     return
        self._test_basic_cases_template(
            torch.randn(10, 5).to(device),
            torch.randn(10).to(device),
            torch.randn(5).to(device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )

    # @skipIfRocm
    def test_rprop(self, device):
        for optimizer in [optim.Rprop, optim_mt.Rprop]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3), device=device
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3), device=device
            )
            with self.assertRaisesRegex(ValueError, "Invalid eta values: 1.0, 0.5"):
                optimizer(None, lr=1e-2, etas=(1.0, 0.5))
                
instantiate_device_type_tests(TestOptim, globals())

if __name__ == "__main__":
    run_tests()
