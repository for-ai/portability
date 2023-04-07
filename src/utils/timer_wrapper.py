import contextlib
# import torch
import time
import functools
import pytest
import os

try:
    # Import the TPUProfiler class from the torch_xla package
    from torch_xla import TPUProfiler
    import torch_xla.core.xla_model as xm
    import torch
except ImportError:
    # torch_xla is not installed, so TPUProfiler is not available
    TPUProfiler = None


@contextlib.contextmanager
def tensorflow_timer(record_function):
    start = time.perf_counter()

    yield

    # Stop the timer
    end = time.perf_counter()

    # Print the elapsed time
    record_function(end - start)
    # print("***TIME", end - start)  # seconds


@contextlib.contextmanager
def tensorflow_op_timer():
    with tensorflow_timer(
            lambda x: pytest.tensorflow_test_times[pytest.test_name]['operations'].append(x)):
        yield


def assign_pytorch_test_time(x):
    # print(pytest.pytorch_test_times)
    pytest.pytorch_test_times[pytest.test_name]['test_time'] = x


def assign_tensorflow_test_time(x):
    # print(pytest.tensorflow_test_times)
    pytest.tensorflow_test_times[pytest.test_name]['test_time'] = x


@contextlib.contextmanager
def tensorflow_test_timer():
    with tensorflow_timer(assign_tensorflow_test_time):
        yield


@contextlib.contextmanager
def pytorch_op_timer():
    with pytorch_timer(
            lambda x: pytest.pytorch_test_times[pytest.test_name]['operations'].append(x)):
        yield


@contextlib.contextmanager
def pytorch_test_timer():
    with pytorch_timer(assign_pytorch_test_time):
        yield


@contextlib.contextmanager
def pytorch_timer(record_function):
    if torch.cuda.is_available():
        # Use CUDA events to measure time on a GPU
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()

        # Waits for all CUDA operations to finish running
        torch.cuda.synchronize()

        record_function(start.elapsed_time(end))
        # pytest.test_times[pytest.test_name]['operations'].append(
        # start.elapsed_time(end))
        # print(start.elapsed_time(end))  # milliseconds
    elif TPUProfiler != None and xm.xla_device():
        # Use TPUProfiler to measure time on a TPU
        with TPUProfiler('pytorch_timer') as prof:
            yield
        # Print the time elapsed for TPU operations
        # pytest.test_times[pytest.test_name]['operations'].append(
        #     prof.total_time_ms())
        record_function(prof.total_time_ms())
        # print(prof.total_time_ms())
    else:
        # Use Python's time module to measure time on a CPU
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        # pytest.test_times[pytest.test_name]['operations'] = end - start
        record_function(end - start)
        # print(end - start)  # seconds
    pytest.test_i += 1


"""
- not sure if this is the best way, but it is a place to start
- for example, if we want to measure the running time for torch.sum, below is a way to do it. 
"""
# def cal_running_time(fn):


# def mysum(*args, **kwargs):
#     with pytorch_timer() as timer:
#         return temp(*args, **kwargs)

#     return timer.elapsed_time
#     # return mysum


# temp = torch.sum
# torch.sum = mysum

# # then every time we call torch.sum, it measures the time elapsed.
# print(torch.sum(torch.Tensor([1, 2, 3])))
