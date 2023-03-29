
import pytest
import os
import json
from ..utils.tensorflow_timer_wrapper import tensorflow_test_timer
import gc
import tensorflow as tf
import contextlib

black_list = ["src/tensorflow_tests_reduced/run_test.py"]

_original_as_default = tf.Graph.as_default

def set_global_device():
    global global_device
    if os.environ['DEVICE'] == "tpu":
        global_device = "/device:TPU:0"
    elif os.environ['DEVICE'] == "gpu":
        global_device = "/device:GPU:0"
    else:
        global_device = "/device:CPU:0"

@contextlib.contextmanager
def device_context():
    with tf.device(global_device):
        yield
# Save the original as_default method
set_global_device()


@contextlib.contextmanager
def custom_as_default(self, include_device=True):
    # print("***DISABLE MONKEYPATCh", disable_monkeypatch)
    # Create a context manager using the original as_default method

    # print("***CONTEXT")
    if getattr(self, '_custom_device_set', False) or not include_device or getattr(self, '_custom_device_set', False):
        self._include_device = False
        # if not include_device:
        #     self._not_include_device = False
        # print("***SKIP")
        with _original_as_default(self):
            yield
    else:
        self._custom_device_set = True

        print("GLOBAL DEVICE", global_device)
        # Create a device context manager

        # Combine the two context managers

        with _original_as_default(self):
            with device_context():
                yield

def pytest_runtest_call(item):
            # class CombinedContext:
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(":")[0]
    if file_name not in black_list:
        tf.Graph.as_default = custom_as_default
    testfunction = item.obj
    print("ITEM", item)
    print("***PATH", os.environ.get('PYTEST_CURRENT_TEST').split(":")[0])


def pytest_runtest_call(item):
    testfunction = item.obj
    print("ITEM", item)

# @pytest.fixture(scope='session', autouse=True)
def pytest_configure():
    pytest.tensorflow_test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0

@pytest.fixture(autouse=True, scope="session")
def track_all():
    yield
    print("AFTER SESSION", pytest.tensorflow_test_times)
    f = open("tensorflow_test_timing.json", "w")
    f.write(json.dumps(pytest.tensorflow_test_times, indent=4, sort_keys=True))


@pytest.fixture(autouse=True)
def track_timing(request):
    pytest.test_name = os.environ.get(
        'PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    test_file = str(request.node.fspath).split("/")[-1]
    pytest.test_name = test_file + ":" + pytest.test_name
    tf.debugging.set_log_device_placement(True)

    pytest.tensorflow_test_times[pytest.test_name] = {"operations": []}
    tf.debugging.set_log_device_placement(True)
    if os.environ['DEVICE'] == "tpu":
        device_name = "/device:TPU:0"
    elif os.environ['DEVICE'] == "gpu":
        device_name = "/device:GPU:0"
    else:
        device_name = "/device:CPU:0"
    with tf.device(device_name):
        with tensorflow_test_timer():
            yield

    tf.Graph.as_default = _original_as_default
    if os.environ['DEVICE'] == "gpu":
        tf.keras.backend.clear_session()
    # print("***MEMORY FREE", tf.config.experimental.get_memory_info('GPU:0'))
