# Owner(s): ["module: dataloader"]

import math
import sys
import errno
import os
import ctypes
import faulthandler
import torch
import gc
import time
import signal
import unittest
import itertools
import warnings
import tempfile
import torch.utils.data.datapipes as dp
from torch import multiprocessing as mp
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    IterDataPipe,
    Subset,
    TensorDataset,
    _utils
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from torch.utils.data.datapipes.iter import IterableWrapper
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_utils import (TestCase, run_tests, TEST_NUMPY, IS_WINDOWS,
                                                  NO_MULTIPROCESSING_SPAWN, skipIfRocm, slowTest,
                                                 load_tests, TEST_WITH_ASAN, TEST_WITH_TSAN, IS_SANDCASTLE,
                                                  IS_MACOS)

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

try:
    import dill
    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False
skipIfNoDill = unittest.skipIf(not HAS_DILL, "no dill")


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA from torch.testing._internal.common_cuda here, because if we do that,
# the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
if TEST_CUDA:
    dev_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    IS_JETSON = 'xavier' in dev_name or 'nano' in dev_name or 'jetson' in dev_name or 'tegra' in dev_name
else:
    IS_JETSON = False

if not NO_MULTIPROCESSING_SPAWN:
    # We want to use `spawn` if able because some of our tests check that the
    # data loader terminiates gracefully. To prevent hanging in the testing
    # process, such data loaders are run in a separate subprocess.
    #
    # We also want to test the `pin_memory=True` configuration, thus `spawn` is
    # required to launch such processes and they initialize the CUDA context.
    #
    # Mixing different start method is a recipe for disaster (e.g., using a fork
    # `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
    # to avoid bugs.
    #
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers `RuntimeError`.
    mp = mp.get_context(method='spawn')


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


supported_multiprocessing_contexts = [None] + list(torch.multiprocessing.get_all_start_methods())

class CountingIterableDataset(IterableDataset):
    def __init__(self, n):
        super(CountingIterableDataset, self).__init__()
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n

@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestConcatDataset(TestCase):

    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())

    def test_iterable_dataset_err(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        it1 = CountingIterableDataset(5)
        it2 = CountingIterableDataset(10)

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([d1, it2, it1])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it2])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it1, d1])



if __name__ == '__main__':
    run_tests()
