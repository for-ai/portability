import contextlib
import time
import functools
import pytest
import os


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

def assign_tensorflow_test_time(x):
    # print(pytest.tensorflow_test_times)
    pytest.tensorflow_test_times[pytest.test_name]['test_time'] = x


@contextlib.contextmanager
def tensorflow_test_timer():
    with tensorflow_timer(assign_tensorflow_test_time):
        yield