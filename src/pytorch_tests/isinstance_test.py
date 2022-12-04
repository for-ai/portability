# -*- coding: utf-8 -*-
# Owner(s): ["oncall: jit"]

import torch

# Torch
from torch import Tensor
from torch._C import TensorType, BoolType, parse_ir, _propagate_shapes
from torch.autograd import Variable
from torch.jit.annotations import BroadcastingList2, BroadcastingList3, Any  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck, make_tensor
import torch.autograd.profiler
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.nn as nn
import torch.nn.functional as F

# Testing utils
from torch.testing._internal import jit_utils
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    suppress_warnings, BUILD_WITH_CAFFE2, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, \
    freeze_rng_state, slowTest, TemporaryFileName, \
    enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs, \
    skipIfCrossRef, IS_MACOS
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, \
    _trace, do_input_map, get_execution_plan, make_global, \
    execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, \
    RUN_CUDA
from torch.testing._internal.jit_metaprogramming_utils import (
    get_script_args,
    create_input, unpack_variables,
    additional_module_tests, EXCLUDE_SCRIPT_MODULES,
    get_nn_module_name_from_kwargs, get_nn_mod_test_name, script_method_template)

from torch.testing._internal.common_nn import module_tests, new_module_tests, criterion_tests

# For testing truediv in python 2
from torch.testing._internal.test_module.future_div import div_int_future, div_float_future
from torch.testing._internal.test_module.no_future_div import div_int_nofuture, div_float_nofuture

# Standard library
from collections import defaultdict, namedtuple, OrderedDict
from copy import deepcopy
from itertools import product
from textwrap import dedent
from typing import List, Dict, NamedTuple, Optional, Tuple, Union
import copy
import functools
import inspect
import io
import itertools
import math
import numpy as np
import os
import pickle
import pickletools
import random
import re
import shutil
import string
import sys
import tempfile
import types
import typing
import unittest
import warnings
import zipfile


class TestJit(JitTestCase):
    def test_isinstance(self):
        # test isinstance operator for static type checking
        template = dedent('''
        def func(x):
            # type: ({type_hint}) -> bool
            return isinstance(x, {typ})
        ''')

        def test(inp, typ, type_hint):
            code = template.format(typ=typ, type_hint=type_hint)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(
                cu.func(inp),
                scope['func'](inp),
                msg="Failed with typ: {}"
                .format(typ)
            )

        inputs = [True, 1, 1.0, torch.tensor(1), [1, 2], (1.0,), [1, 2], 1]
        type_literals = ['bool', 'int', 'float', 'torch.Tensor', 'list', 'tuple',
                         '(list, tuple)', '(int, float, bool)']
        type_annotations = ['bool', 'int', 'float', 'Tensor', 'List[int]', 'Tuple[float]',
                            'List[int]', 'int']

        # do zipping to try different types
        for inp, typ, type_hint in zip(inputs, type_literals, type_annotations):
            test(inp, typ, type_hint)

        # test optional isinstance check
        @torch.jit.script
        def opt_func(x):
            # type: (Optional[int]) -> bool
            return isinstance(x, int)
        self.assertTrue(opt_func(3))
        self.assertFalse(opt_func(None))

    
    def test_isinstance_metacompile(self):
        @torch.jit.script
        def test_primitive_type(x):
            # type: (int) -> int
            if isinstance(x, int):
                return x + 1
            else:
                return x - 1

        self.assertEqual(test_primitive_type(1), 2)
        with self.assertRaisesRegex(Exception, "Expected a value of type"):
            test_primitive_type(1.5)

        _MyNamedTuple = namedtuple('_MyNamedTuple', ['value'])

        @torch.jit.script
        def test_non_primitive_types(x):
            # type: (_MyNamedTuple) -> Tensor
            if isinstance(1, _MyNamedTuple):
                return 10

            if isinstance(x, _MyNamedTuple):
                return x.value + 1
            else:
                return 1

        out = test_non_primitive_types(_MyNamedTuple(value=torch.tensor(5.0)))
        self.assertEqual(out, torch.tensor(6.0))


    def test_isinstance_dynamic(self):
        @torch.jit.script
        def foo(a):
            # type: (Optional[List[int]]) -> int
            b = 0
            if isinstance(a, (int, (float,), list, str)):
                b += 1
            if isinstance(a, (int, str)):
                b += 1
            if isinstance(a, List[int]):
                b += 1
            return b
        self.assertEqual(foo([3, 4]), 2)
        self.assertEqual(foo(None), 0)

    def test_isinstance_refinement(self):
        @torch.jit.script
        def foo(a):
            # type: (Optional[int]) -> int
            if isinstance(a, int):
                return a + 3
            else:
                return 4
        self.assertEqual(foo(4), 7)
        self.assertEqual(foo(None), 4)

        @torch.jit.script
        def foo2(a, b):
            # type: (Optional[int], Optional[int]) -> int
            if not isinstance(a, int) or not isinstance(b, int):
                return 0
            else:
                return a + b
        self.assertEqual(foo2(3, 4), 7)
        self.assertEqual(foo2(None, 4), 0)
        self.assertEqual(foo2(4, None), 0)

        @torch.jit.script
        def any_refinement(a, b):
            # type: (Any, Any) -> int
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            return 0

        self.assertEqual(any_refinement(3, 4), 7)
        self.assertEqual(any_refinement(3, "hi"), 0)

        @torch.jit.script
        def any_refinement2(a):
            # type: (Any) -> Tensor
            if isinstance(a, Tensor):
                return a
            return torch.tensor(3)

        self.assertEqual(any_refinement2(3), torch.tensor(3))
        self.assertEqual(any_refinement2(torch.tensor(5)), torch.tensor(5))



if __name__ == '__main__':
    run_tests()
    import jit.test_module_interface
    suite = unittest.findTestCases(jit.test_module_interface)
    unittest.TextTestRunner().run(suite)
