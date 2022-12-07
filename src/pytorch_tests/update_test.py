# Owner(s): ["module: nn"]

from torch.types import _TensorOrTensors
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
import torch.testing._internal.hypothesis_utils as hu
from hypothesis import given
from torch.nn import MultiheadAttention
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, IS_WINDOWS
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.nn.parallel._functions import Broadcast
from torch.nn import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.utils.prune as prune
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd.forward_ad as fwAD
from torch._six import inf, nan
import contextlib
import math
import random
import string
import unittest
import io
import unittest.mock as mock
import itertools
import warnings
import pickle
from copy import deepcopy
from itertools import product
from functools import reduce, partial
from operator import mul
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import weakref
import gc

import torch

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import common_nn
    from common_nn import NNTestCase, NewModuleTest, CriterionTest, \
        module_tests, criterion_tests, loss_reference_fns, \
        ctcloss_reference, new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input

else:
    # uses current package visibility
    from . import common_nn
    from .common_nn import NNTestCase, NewModuleTest, CriterionTest, \
        module_tests, criterion_tests, loss_reference_fns, \
        ctcloss_reference, new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
    from scipy import stats
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNN(NNTestCase):

    def test_ModuleDict(self):
        modules = OrderedDict([
            ('act', nn.ReLU()),
            ('conv', nn.Conv2d(10, 10, 5)),
            ('fc', nn.Linear(5, 5)),
        ])

        module_dict = nn.ModuleDict(modules)

        def check():
            self.assertEqual(len(module_dict), len(modules))
            for k1, m2 in zip(modules, module_dict.children()):
                self.assertIs(modules[k1], m2)
            for k1, k2 in zip(modules, module_dict):
                self.assertIs(modules[k1], module_dict[k2])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k in module_dict.keys():
                self.assertIs(module_dict[k], modules[k])
            for k, v in module_dict.items():
                self.assertIs(modules[k], v)
            for k1, m2 in zip(modules, module_dict.values()):
                self.assertIs(modules[k1], m2)
            for k in modules.keys():
                self.assertTrue(k in module_dict)
        check()

        modules['conv'] = nn.Conv2d(3, 4, 3)
        module_dict['conv'] = modules['conv']
        check()

        next_modules = [
            ('fc2', nn.Linear(5, 5)),
            ('act', nn.Sigmoid()),
        ]
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = OrderedDict([
            ('fc3', nn.Linear(5, 5)),
            ('act2', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = {
            'fc4': nn.Linear(5, 5),
            'act3': nn.Sigmoid()
        }
        modules.update(next_modules.items())
        module_dict.update(next_modules)
        check()

        next_modules = nn.ModuleDict([
            ('fc5', nn.Linear(5, 5)),
            ('act4', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        del module_dict['fc']
        del modules['fc']
        check()

        with self.assertRaises(TypeError):
            module_dict.update(nn.ReLU())

        with self.assertRaises(TypeError):
            module_dict.update([nn.ReLU()])

        with self.assertRaises(ValueError):
            module_dict.update([[nn.ReLU()]])

        with self.assertRaises(TypeError):
            module_dict[1] = nn.ReLU()

        s = nn.Sequential(modules)
        module_dict = nn.ModuleDict(s.named_children())
        check()

        c = module_dict.pop('conv')
        self.assertIs(c, modules['conv'])
        modules.pop('conv')
        check()

        module_dict.clear()
        self.assertEqual(len(module_dict), 0)
        modules.clear()
        check()

        # verify the right exception is thrown when trying to "forward" through a ModuleDict
        self.assertRaises(NotImplementedError, module_dict)
        self.assertRaises(NotImplementedError, module_dict, torch.rand(1, 3))

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])

        parameter_dict = nn.ParameterDict(parameters)

        def check():
            self.assertEqual(len(parameter_dict), len(parameters))
            for i, (k1, (k2, m2)) in enumerate(zip(parameters, parameter_dict.named_parameters())):
                self.assertEqual(k1, k2)
                self.assertIs(parameters[k1], m2)
            for k1, k2 in zip(parameters, parameter_dict):
                self.assertIs(parameters[k1], parameter_dict[k2])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k in parameter_dict.keys():
                self.assertIs(parameter_dict[k], parameters[k])
            for k, v in parameter_dict.items():
                self.assertIs(v, parameters[k])
            for k1, m2 in zip(parameters, parameter_dict.values()):
                self.assertIs(parameters[k1], m2)
            for k in parameters.keys():
                self.assertTrue(k in parameter_dict)

        check()

        parameters['p4'] = Parameter(torch.randn(10, 10))
        parameter_dict['p4'] = parameters['p4']
        check()

        next_parameters = [
            ('p5', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
        ]
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = OrderedDict([
            ('p6', Parameter(torch.randn(10, 10))),
            ('p5', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = {
            'p8': Parameter(torch.randn(10, 10)),
            'p7': Parameter(torch.randn(10, 10))
        }
        parameters.update(sorted(next_parameters.items()))
        parameter_dict.update(next_parameters)
        check()

        next_parameters = nn.ParameterDict([
            ('p10', Parameter(torch.randn(10, 10))),
            ('p9', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        del parameter_dict['p3']
        del parameters['p3']
        check()

        with self.assertRaises(TypeError):
            parameter_dict.update(1)

        with self.assertRaises(TypeError):
            parameter_dict.update([1])

        with self.assertRaises(ValueError):
            parameter_dict.update(Parameter(torch.randn(10, 10)))

        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        check()

        # Check reverse works
        forward = list(iter(parameter_dict))
        backward = list(reversed(parameter_dict))
        self.assertEqual(len(forward), len(backward))
        n = len(forward)
        for i in range(n):
            self.assertIs(forward[i], backward[n - i - 1])
        check()

        # Check copy works
        copy = parameter_dict.copy()

        # Check all keys are present and have shallow copied values
        for key in parameter_dict:
            self.assertTrue(key in copy)
            self.assertEqual(parameter_dict[key], copy[key])
            self.assertIs(parameter_dict[key], copy[key])
        check()

        parameter_dict["p20"] = Parameter(torch.randn(10, 10))
        copy["p21"] = Parameter(torch.randn(9, 10))

        self.assertTrue("p20" in parameter_dict)
        self.assertFalse("p20" in copy)
        self.assertFalse("p21" in parameter_dict)
        self.assertTrue("p21" in copy)
        parameter_dict.pop("p20")
        check()

        p = Parameter(torch.randn(10, 10))
        parameter_dict['p12'] = p
        p_popitem = parameter_dict.popitem()
        self.assertEqual(p_popitem[0], 'p12')
        self.assertIs(p_popitem[1], p)
        check()

        # Unit test for set_default
        # 1. Ensure parameter is correctly inserted when
        #    the key is not present in `ParameterDict`
        assert 'p11' not in parameter_dict
        assert 'p11' not in parameters
        parameters['p11'] = Parameter(torch.randn(10, 10))
        p_setdefault = parameter_dict.setdefault('p11', parameters['p11'])
        self.assertIs(p_setdefault, parameters['p11'])
        self.assertIs(p_setdefault, parameter_dict['p11'])
        check()
        # 2. Ensure parameter is NOT inserted when the
        #    key is already present in `ParameterDict`
        p = Parameter(torch.randn(10, 10))
        self.assertFalse(parameter_dict.setdefault('p11', p) is p)
        check()
        # 3. Ensure `None` is inserted when the key is not
        #    present in `Parameter` and parameter is not specified
        self.assertIs(parameter_dict.setdefault('p26'), None)
        del parameter_dict['p26']
        check()

        parameters2 = OrderedDict([
            ('p13', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict()
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p14', Parameter(torch.randn(10, 10))),
            ('p15', Parameter(torch.randn(10, 10))),
            ('p13', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        # Check __or__ and __ror__ works
        parameters2 = OrderedDict([
            ('p20', Parameter(torch.randn(10, 10))),
            ('p21', Parameter(torch.randn(10, 10))),
            ('p22', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict = parameter_dict | parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p23', Parameter(torch.randn(10, 10))),
            ('p24', Parameter(torch.randn(10, 10))),
            ('p25', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters2.update(parameters)
        parameters = parameters2
        parameter_dict = parameter_dict2 | parameter_dict
        check()

        parameters['p17'] = Parameter(torch.randn(10, 10))
        parameter_dict['p17'] = parameters['p17']
        self.assertIs(parameters['p17'], parameter_dict.get('p17'))
        temp_param = Parameter(torch.randn(10, 10))
        self.assertIs(parameters['p17'], parameter_dict.get('p17', temp_param))
        self.assertIs(None, parameter_dict.get('p18'))
        self.assertIs(temp_param, parameter_dict.get('p18', temp_param))
        check()

        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'])
        self.assertEqual({'p19': None, 'p20': None}, parameter_dict2)
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'], temp_param)
        self.assertEqual(
            {'p19': temp_param, 'p20': temp_param}, parameter_dict2)
        check()

        parameter_dict['p21'] = torch.rand(2, 2)
        self.assertIsInstance(parameter_dict['p21'], Parameter)
        parameters['p21'] = parameter_dict['p21']

        parameter_dict.update({'p22': torch.rand(2, 2), 'foo': 'bar'})
        self.assertIsInstance(parameter_dict['p22'], Parameter)
        self.assertIsInstance(parameter_dict['foo'], str)
        parameters['p22'] = parameter_dict['p22']
        parameters['foo'] = parameter_dict['foo']

    def test_load_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv1 = nn.Conv2d(3, 3, 3, bias=True)
        block.conv2 = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)
        conv1_bias_dtype = block.conv1.bias.dtype

        state_dict = net.state_dict()
        state_dict.update({
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
        })
        # Also test if a DDP state_dict can be loaded from a local model.
        ddp_state_dict = net.state_dict()
        ddp_state_dict.update({
            'module.linear1.weight': torch.ones(5, 5),
            'module.block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'module.bn.running_mean': torch.randn(2),
        })
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            ddp_state_dict, 'module.')
        for sd in [state_dict, ddp_state_dict]:
            incompatible_keys = net.load_state_dict(sd)
            self.assertEqual(len(incompatible_keys.missing_keys), 0)
            self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
            self.assertNotIn('Incompatible', str(incompatible_keys))
            self.assertEqual(net.linear1.weight, sd['linear1.weight'])
            self.assertEqual(net.block.conv1.bias, sd['block.conv1.bias'])
            self.assertEqual(net.bn.running_mean, sd['bn.running_mean'])

        state_dict = net.state_dict()
        state_dict.update({'extra': torch.ones(5)})
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra', incompatible_keys.unexpected_keys)
        self.assertIn('Incompatible', str(incompatible_keys))

        state_dict = net.state_dict()
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        del state_dict['linear1.weight']
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        state_dict.update({'bn.running_mean': torch.rand(14, 4)})  # wrong size
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict))
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(
            state_dict, strict=False))

        state_dict = net.state_dict()
        old_state_dict = deepcopy(state_dict)
        state_dict = {
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
            'nonexistent_key': torch.rand(3)
        }
        net.load_state_dict(state_dict, strict=False)
        self.assertEqual(net.linear1.weight, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias, state_dict['block.conv1.bias'])
        self.assertEqual(net.bn.running_mean, state_dict['bn.running_mean'])
        new_state_dict = net.state_dict()
        del old_state_dict['linear1.weight']
        del old_state_dict['block.conv1.bias']
        del old_state_dict['bn.running_mean']
        for k, v, in old_state_dict.items():
            self.assertTrue(v.equal(new_state_dict[k]))


instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
