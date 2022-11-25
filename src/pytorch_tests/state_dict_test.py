# Owner(s): ["module: nn"]

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

from torch._six import inf, nan
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn import Parameter
from torch.nn.parallel._functions import Broadcast
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, IS_WINDOWS
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors


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

    def test_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)

        state_dict = net.state_dict()
        self.assertEqual(len(state_dict), 10)
        self.assertEqual(len(state_dict._metadata), 6)
        self.assertIn('', state_dict._metadata)
        self.assertIn('linear1', state_dict._metadata)
        self.assertIn('linear1.weight', state_dict)
        self.assertIn('linear1.bias', state_dict)
        self.assertIn('linear2', state_dict._metadata)
        self.assertIn('linear2.weight', state_dict)
        self.assertIn('linear2.bias', state_dict)
        self.assertIn('block', state_dict._metadata)
        self.assertIn('block.conv', state_dict._metadata)
        self.assertIn('block.conv.weight', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertNotIn('block.conv.bias', state_dict)
        self.assertIn('bn', state_dict._metadata)
        self.assertIn('bn.weight', state_dict)
        self.assertIn('bn.bias', state_dict)
        self.assertIn('bn.running_var', state_dict)
        self.assertIn('bn.running_mean', state_dict)
        self.assertIn('bn.num_batches_tracked', state_dict)
        self.assertFalse(any(k.startswith('empty') for k in state_dict.keys()))
        for k, v in state_dict.items():
            param = net
            for component in k.split('.'):
                param = getattr(param, component)
                if isinstance(param, Parameter):
                    param = param.data
            self.assertEqual(v.data_ptr(), param.data_ptr())

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertEqual(len(state_dict._metadata), 1)
        self.assertIn('', state_dict._metadata)
        self.assertTrue(state_dict._metadata['']['version'] >= 0)
        self.assertEqual(state_dict['weight'].data_ptr(), l.weight.data_ptr())
        self.assertEqual(state_dict['bias'].data_ptr(), l.bias.data_ptr())

        # Reference https://github.com/pytorch/pytorch/pull/75507#issuecomment-1110291545
        self.assertNotWarn(lambda: l.state_dict(destination=dict()), "Should not warn kwarg destination w/o _metadata")

def _hook_to_pickle(*args, **kwargs):
    pass
        
class TestStateDictHooks(TestCase):

    def test_load_state_dict_pre_hook(self):

        m = nn.Linear(10, 10)
        m_state_dict = m.state_dict()

        m_load = nn.Linear(10, 10)

        hook_called = 0

        def hook_without_module(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            nonlocal hook_called
            hook_called += 1

        def hook_with_module(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            self.assertTrue(m_load is module)
            nonlocal hook_called
            hook_called += 1

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_without_module)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(1, hook_called)

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_with_module, True)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(2, hook_called)

    def test_load_state_dict_module_pre_hook(self):
        hook_called = 0

        # Test with module instance method as hook
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_pre_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                nonlocal hook_called
                hook_called += 1

            def my_pre_load_hook_with_module(
                self,
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                assert self is module
                nonlocal hook_called
                hook_called += 1

        # Test that hooks registered on a submodule are also called
        # appropriately, i.e. with the submodule as module argument in
        # my_pre_load_hook_with_module.
        class MyModuleContainer(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

        for ctor in [MyModuleContainer, lambda x: x]:
            m = ctor(MyModule())
            state_dict = m.state_dict()
            if isinstance(m, MyModuleContainer):
                mod = m.mod
            else:
                mod = m

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook
            )
            m.load_state_dict(state_dict)
            self.assertEqual(1, hook_called)

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook_with_module, True
            )
            m.load_state_dict(state_dict)
            self.assertEqual(2, hook_called)

    def test_load_state_dict_post_hook(self):
        hook_called = 0

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_post_load_hook(self, module, incompatible_keys):
                assert module is self
                nonlocal hook_called
                incompatible_keys.missing_keys.append("foo")
                incompatible_keys.unexpected_keys.append("bar")
                hook_called += 1

        nested = MyModule()
        wrapped = nn.ModuleList([nested])
        handle = nested.register_load_state_dict_post_hook(
            nested.my_post_load_hook,
        )
        # Hook must be called even if it is wrapped
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(hook_called, 1)
        # Ensure that the hook modified missing_keys and unexpected_keys
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        self.assertEqual(missing, ["foo"])
        self.assertEqual(unexpected, ["bar"])
        # When called with strict=True, the error raised should mention the
        # missing and unexpected keys the hook added.
        with self.assertRaisesRegex(RuntimeError, "foo.*\n.*bar"):
            wrapped.load_state_dict(wrapped.state_dict(), strict=True)
        self.assertEqual(hook_called, 2)
        # Removing the hook via handle.remove() should cause it not to
        # fire anymore.
        handle.remove()
        # Hook did not run so it should not have added any keys
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(ret.missing_keys, [])
        self.assertEqual(ret.unexpected_keys, [])
        # hook_called should not have been incremented
        self.assertEqual(hook_called, 2)

        def load_hook_clear_incompatible(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        nested.register_load_state_dict_post_hook(load_hook_clear_incompatible)
        state_dict = wrapped.state_dict()
        state_dict["extra"] = torch.ones(1)
        # load state_dict with strict=True should not throw.
        ret = wrapped.load_state_dict(state_dict, strict=True)
        # explicitly ensure that the post hook clearned out incompatible_keys
        self.assertEqual([], ret.missing_keys)
        self.assertEqual([], ret.unexpected_keys)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_load_state_dict_post_hook_backward_compatibility(self):
        def my_post_load_hook(mod, _):
            nonlocal called
            called = True

        for m in [nn.Softmin(10), nn.Softmax(10), nn.LogSoftmax(10)]:
            called = False
            sd = deepcopy(m.state_dict())
            self.assertTrue(hasattr(m, '_load_state_dict_post_hooks'))
            # Simulate an older model that did not have this attr
            delattr(m, '_load_state_dict_post_hooks')
            # Save and load, and ensure that load_state_dict works (without proper
            # BC we would run into errors because this attribute would be expected).
            # In particular, Softmax runs into the issue described here:
            # https://github.com/pytorch/pytorch/issues/77280
            with NamedTemporaryFile() as f:
                # Note that torch.save / torch.load is not recommended to save/load
                # modules.
                torch.save(m, f.name)
                m = torch.load(f.name)
                m.load_state_dict(sd)
                self.assertFalse(called)

            # Ensure hooks can be registered and called.
            m.register_load_state_dict_post_hook(my_post_load_hook)
            m.load_state_dict(sd)
            self.assertTrue(called)


instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
