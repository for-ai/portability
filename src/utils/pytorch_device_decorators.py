from functools import wraps
import unittest
import torch

from typing import List, Any, ClassVar
from torch.testing._internal.common_device_type import CUDATestBase, CPUTestBase, DeviceTypeTestBase, MPSTestBase, filter_desired_device_types
import os
import copy
import inspect

from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN, IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, TEST_WITH_ROCM
try:
    import torch_xla.core.xla_model as xm
    ImportTPU = True
except ImportError:
    ImportTPU = False
NATIVE_DEVICES = {'cpu', 'cuda', 'xla'}
ACCELERATED_DEVICES = {'cuda', 'xla'}
PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = 'PYTORCH_TESTING_DEVICE_ONLY_FOR'
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = 'PYTORCH_TESTING_DEVICE_EXCEPT_FOR'


# Custom decorator that runs on TPUs and GPUs
def onlyGPU(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ['cuda']:
            reason = "onlyAcceleratedDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn

def onlyTPU(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ['xla']:
            reason = "onlyAcceleratedDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn

def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

# Skips a test on CUDA if cuDNN is unavailable or its version is lower than requested.
def skipCUDAIfCudnnVersionLessThan(version=0):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda' or self.device_type == 'xla':
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = "cuDNN version {0} is available but {1} required".format(self.cudnn_version, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

# Custom decorator that runs on TPUs and GPUs
def onlyAcceleratedDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ACCELERATED_DEVICES:
            reason = "onlyAcceleratedDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


# Custom decorator that runs on TPUs, GPUs, CPUs
def onlyNativeDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in NATIVE_DEVICES:
            reason = "onlyNativeDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


class TPUTestBase(DeviceTypeTestBase):
    device_type = 'xla'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(
            num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized

        # Determines if cuDNN is available and its version
        # Acquires the current device as the primary (test) device
        t = torch.ones(1).to('xla:1')
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.primary_device = 'xla:1'

# Adds available device-type-specific test base classes


def get_device_type_test_bases():
    # set type to List[Any] due to mypy list-of-union issue:
    # https://github.com/python/mypy/issues/3351
    test_bases: List[Any] = list()

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            # Skip if sanitizer is enabled
            if not TEST_WITH_ASAN and not TEST_WITH_TSAN and not TEST_WITH_UBSAN:
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)
        elif ImportTPU and xm.xla_device():
            test_bases.append(TPUTestBase)
        # Disable MPS testing in generic device testing temporarily while we're
        # ramping up support.
        # elif torch.backends.mps.is_available():
        #   test_bases.append(MPSTestBase)

    return test_bases

# Almost the same as the method implemented by pytorch but it uses our custom test bases


def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    # Creates an 'empty' version of the generic_test_class
    # Note: we don't inherit from the generic_test_class directly because
    #   that would add its tests to our test classes and they would be
    #   discovered (despite not being runnable). Inherited methods also
    #   can't be removed later, and we can't rely on load_tests because
    #   pytest doesn't support it (as of this writing).
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    # Acquires members names
    # See Note [Overriding methods in generic tests]
    generic_members = set(generic_test_class.__dict__.keys()
                          ) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]

    # MPS backend support is disabled in `get_device_type_test_bases` while support is being ramped
    # up, so allow callers to specifically opt tests into being tested on MPS, similar to `include_lazy`
    test_bases = device_type_test_bases.copy()
    if allow_mps and torch.backends.mps.is_available() and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)
    # Filter out the device types based on user inputs
    desired_device_type_test_bases = filter_desired_device_types(
        test_bases, except_for, only_for)

    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    env_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))

    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
                                                                 env_except_for, env_only_for)

    # Creates device-specific test cases
    for base in desired_device_type_test_bases:
        # Special-case for ROCm testing -- only test for 'cuda' i.e. ROCm device by default
        # The except_for and only_for cases were already checked above. At this point we only need to check 'cuda'.
        if TEST_WITH_ROCM and base.device_type != 'cuda':
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupport runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, empty_class), {})
        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(
                    device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(
                        name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(
                        name, copy.deepcopy(test))
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(
                    name)
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)
        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


device_type_test_bases = get_device_type_test_bases()
