
import functools
from copy import deepcopy

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.testing._internal.common_utils import TestCase, run_tests
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

class TestOptim(TestCase):
    def _test_state_dict(self, weight, bias, input, constructor, device):
        weight = Variable(weight, requires_grad=True).to(device)
        bias = Variable(bias, requires_grad=True).to(device)
        input = Variable(input).to(device)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for _i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for _i in range(20):
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
        self.assertEqual(optimizer.state_dict()['param_groups'][-1],
                         optimizer_c.state_dict()['param_groups'][-1])

        # Make sure that optimizers that support maximize can load older models
        state_dict = optimizer.state_dict()
        if 'maximize' in state_dict['param_groups'][0]:
            for group in state_dict['param_groups']:
                del group['maximize']
            optimizer.load_state_dict(state_dict)
            # Make sure we can still step
            optimizer.step()
        # Make sure that optimizers that support foreach can load older models
        state_dict = optimizer.state_dict()
        if 'foreach' in state_dict['param_groups'][0]:
            for group in state_dict['param_groups']:
                del group['foreach']
            optimizer.load_state_dict(state_dict)
            # Make sure we can still step
            optimizer.step()

        # Make sure that loading optimizers with step not wrapped in tensor can work
        state_dict = optimizer.state_dict()
        if 'step' in state_dict['state'][0] and torch.is_tensor(state_dict['state'][0]['step']):
            for state in state_dict['state'].values():
                state['step'] = state['step'].item()
            optimizer.load_state_dict(state_dict)
            optimizer.step()

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
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
        if 'step' in state_dict['state'][0] and torch.is_tensor(state_dict['state'][0]['step']):
            for state in new_state_dict['state'].values():
                self.assertEqual(state['step'].device.type, 'cpu')

        for _i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return set(k for k in obj.__dict__ if not k.startswith('_'))
        self.assertEqual(getPublicAttr(optimizer),
                         getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(self, constructor, scheduler_constructors=None,
                          ignore_multidevice=False, constructor_accepts_maximize=False, device=False):
        if scheduler_constructors is None:
            scheduler_constructors = []

        def make_two_arg_constructor(constructor, maximize: bool = False):
            if constructor_accepts_maximize:
                with pytorch_op_timer():
                    return lambda weight, bias: constructor(weight, bias, maximize)
            return constructor

        for maximize in (True, False):
            self._test_state_dict(
                torch.randn(10, 5, device=device),
                torch.randn(10, device=device),
                torch.randn(5, device=device),
                make_two_arg_constructor(constructor, maximize),
                device=device
            )
        self._test_basic_cases_template(
            torch.randn(10, 5, device=device),
            torch.randn(10, device=device),
            torch.randn(5, device=device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            device=device
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2, device=device)[..., 0],
            torch.randn(10, 2, device=device)[..., 0],
            torch.randn(5, device=device),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            device=device
        )

    def _test_basic_cases_template(self, weight, bias, input, constructor,
                                   scheduler_constructors, constructor_accepts_maximize=True, device=False):
        maximize_options = set([False, constructor_accepts_maximize])
        if not constructor_accepts_maximize:
            def three_arg_constructor(weight, bias, maximize):
                self.assertFalse(maximize)
                return constructor(weight, bias)
        else:
            three_arg_constructor = constructor

        for maximize in maximize_options:
            weight = Variable(weight, requires_grad=True).to(device)
            bias = Variable(bias, requires_grad=True).to(device)
            input = Variable(input).to(device)
            with pytorch_op_timer():
                optimizer = three_arg_constructor(weight, bias, maximize)
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
            for _i in range(200):
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

    # @onlyNativeDeviceTypes
    def test_lbfgs(self, device):
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS([weight, bias]),
            ignore_multidevice=True,
            device=device
        )
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS(
                [weight, bias], line_search_fn="strong_wolfe"),
            ignore_multidevice=True,
            device=device
        )


instantiate_device_type_tests(TestOptim, globals())
if __name__ == '__main__':
    run_tests()
