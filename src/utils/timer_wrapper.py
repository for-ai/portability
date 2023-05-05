import contextlib

# import torch
import time
import functools
import pytest
import os
import tensorflow as tf
from functools import partial

try:
    # Import the TPUProfiler class from the torch_xla package
    import torch
    from torch_xla import TPUProfiler
    import torch_xla.core.xla_model as xm
except ImportError:
    # torch_xla is not installed, so TPUProfiler is not available
    TPUProfiler = None


def assign_jax_test_time(x):
    # print(pytest.tensorflow_test_times)
    pytest.jax_test_times[pytest.test_name]["test_time"] = x


@contextlib.contextmanager
def jax_timer(record_function):
    start = time.perf_counter()
    # print("***START", start)

    yield
    # Stop the timer
    end = time.perf_counter()
    # print("***END", end)
    # Print the elapsed time
    record_function(end - start)
    # print("***TIME", end - start)  # seconds


@contextlib.contextmanager
def jax_op_timer():
    with jax_timer(
        lambda x: pytest.jax_test_times[pytest.test_name]["operations"].append(x)
    ):
        result = yield
        yield
        has_block = getattr(result, "block_until_ready", None)
        # print("***EAGER EXECUTE", tf.executing_eagerly())
        if callable(has_block):
            result.block_until_ready()


@contextlib.contextmanager
def jax_test_timer():
    with jax_timer(assign_jax_test_time):
        yield


def partial_timed(fn, *outer_args, **outer_kwargs):
    def wrapper(*inner_args, **inner_kwargs):
        args = outer_args + inner_args
        kwargs = {**outer_kwargs, **inner_kwargs}
        timer = jax_op_timer()
        with timer:
            result = fn(*args, **kwargs)
            timer.gen.send(result)
        return result

    return wrapper


@contextlib.contextmanager
def tensorflow_timer(record_function):
    start = time.perf_counter()
    # print("***START", start)

    yield
    # Stop the timer
    end = time.perf_counter()
    # print("***END", end)
    # Print the elapsed time
    record_function(end - start)
    # print("***TIME", end - start)  # seconds


@contextlib.contextmanager
def tensorflow_op_timer():
    with tensorflow_timer(
        lambda x: pytest.tensorflow_test_times[pytest.test_name]["operations"].append(x)
    ):
        result = yield
        yield
        before_sync = time.perf_counter()
        # print("***BEFORE SYNC", before_sync)
        has_numpy = getattr(result, "numpy", None)
        # print("***EAGER EXECUTE", tf.executing_eagerly())
        if callable(has_numpy):
            # print("***CALL NUMPY")
            result.numpy()


def assign_pytorch_test_time(x):
    # print(pytest.pytorch_test_times)
    pytest.pytorch_test_times[pytest.test_name]["test_time"] = x


def assign_tensorflow_test_time(x):
    # print(pytest.tensorflow_test_times)
    pytest.tensorflow_test_times[pytest.test_name]["test_time"] = x


@contextlib.contextmanager
def tensorflow_test_timer():
    with tensorflow_timer(assign_tensorflow_test_time):
        yield


@contextlib.contextmanager
def pytorch_op_timer():
    with pytorch_timer(
        lambda x: pytest.pytorch_test_times[pytest.test_name]["operations"].append(x)
    ):
        # Use CUDA events to measure time on a GPU
        yield

        # Waits for all CUDA operations to finish running

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif TPUProfiler is not None and xm.xla_device():
            xm.mark_step()


@contextlib.contextmanager
def pytorch_test_timer():
    with pytorch_timer(assign_pytorch_test_time):
        yield


@contextlib.contextmanager
def pytorch_timer(record_function):
    # if torch.cuda.is_available():
    #     # Use CUDA events to measure time on a GPU
    #     start = time.perf_counter()
    #     yield

    #     # Waits for all CUDA operations to finish running
    #     torch.cuda.synchronize()
    #     end = time.perf_counter()

    #     record_function(start.elapsed_time(end))
    #     # pytest.test_times[pytest.test_name]['operations'].append(
    #     # start.elapsed_time(end))
    #     # print(start.elapsed_time(end))  # milliseconds
    # elif TPUProfiler != None and xm.xla_device():
    #     # Use TPUProfiler to measure time on a TPU
    #     start = time.perf_counter()
    #     yield
    #     xm.mark_step()
    #     end = time.perf_counter()
    #     # Print the time elapsed for TPU operations
    #     # pytest.test_times[pytest.test_name]['operations'].append(
    #     #     prof.total_time_ms())
    #     record_function(start.elapsed_time(end))
    #     # print(prof.total_time_ms())
    # else:
    #     # Use Python's time module to measure time on a CPU
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # pytest.test_times[pytest.test_name]['operations'] = end - start
    # print(end - start)  # seconds
    record_function(end - start)
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
