import math
import sys
import errno
import multiprocessing
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
from torch import multiprocessing as mp
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    DataLoader2,
    Dataset,
    IterableDataset,
    IterDataPipe,
    Subset,
    TensorDataset,
    communication,
    _utils
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.map import SequenceWrapper
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_utils import (TestCase, run_tests, TEST_NUMPY, IS_WINDOWS,
                                                  IS_IN_CI, NO_MULTIPROCESSING_SPAWN, skipIfRocm, slowTest,
                                                  load_tests, TEST_WITH_ASAN, TEST_WITH_TSAN, IS_SANDCASTLE,
                                                  IS_MACOS)
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests


class TestDatasetRandomSplit(TestCase):
    def test_slicing_of_subset_of_dataset(self, device):
        # Testing slicing a subset initialized with a dataset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5], device=device))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_dataset[:], dataset[:])
        self.assertEqual(subset_of_dataset[1:2], dataset[1:2])
        self.assertEqual(subset_of_dataset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset from random split
        subset1, subset2 = random_split(dataset, [3, 2])
        self.assertEqual(subset1[:], dataset[subset1.indices[:]])
        self.assertEqual(subset1[0:2], dataset[subset1.indices[0:2]])
        self.assertEqual(subset1[0:-1:2], dataset[subset1.indices[0:-1:2]])

    def test_slicing_of_subset_of_subset(self, device):
        # Testing slicing a subset initialized with a subset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5], device=device))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        subset_of_subset = Subset(subset_of_dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_subset[:], dataset[:])
        self.assertEqual(subset_of_subset[0:2], dataset[0:2])
        self.assertEqual(subset_of_subset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset of subset from random split
        subset1, subset2 = random_split(dataset, [4, 1])
        subset_of_subset1, subset_of_subset2 = random_split(subset1, [3, 1])
        idx = [subset1.indices[i] for i in subset_of_subset1.indices]
        self.assertEqual(subset_of_subset1[:], dataset[idx[:]])
        self.assertEqual(subset_of_subset1[0:2], dataset[idx[0:2]])
        self.assertEqual(subset_of_subset1[0:-1:2], dataset[idx[0:-1:2]])

instantiate_device_type_tests(TestDatasetRandomSplit, globals())

if __name__ == '__main__':
    run_tests()
