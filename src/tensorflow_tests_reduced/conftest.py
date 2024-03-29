import pytest
import os
import json
from ..utils.timer_wrapper import tensorflow_test_timer
import gc
import tensorflow as tf
import contextlib
from ..tensorflow_test import set_global_device, device_context, global_device

black_list = [
    "src/tensorflow_tests_reduced/run_test.py",
    "src/tensorflow_tests_reduced/get_global_step_test.py",
    "src/tensorflow_tests_reduced/eval_test.py",
    "src/tensorflow_tests_reduced/smart_cond_test.py",
    "src/tensorflow_tests_reduced/CheckpointSaverHook_test.py",
]

_original_as_default = tf.Graph.as_default

# Save the original as_default method
set_global_device()


@contextlib.contextmanager
def custom_as_default(self, include_device=True):
    # Create a context manager using the original as_default method
    if (
        getattr(self, "_custom_device_set", False)
        or not include_device
        or getattr(self, "_custom_device_set", False)
    ):
        self._include_device = False
        with _original_as_default(self):
            yield
    else:
        self._custom_device_set = True

        with _original_as_default(self):
            with device_context():
                yield


def pytest_runtest_call(item):
    # class CombinedContext:
    file_name = os.environ.get("PYTEST_CURRENT_TEST").split(":")[0]
    if file_name not in black_list:
        tf.Graph.as_default = custom_as_default
    testfunction = item.obj


def pytest_configure():
    pytest.tensorflow_test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0


@pytest.fixture(autouse=True, scope="session")
def track_all():
    yield
    f = open("tensorflow_test_timing.json", "w")
    f.write(json.dumps(pytest.tensorflow_test_times, indent=4, sort_keys=True))


@pytest.fixture(autouse=True)
def track_timing(request):
    pytest.test_name = (
        os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    )
    test_file = str(request.node.fspath).split("/")[-1]
    pytest.test_name = test_file + ":" + pytest.test_name

    pytest.tensorflow_test_times[pytest.test_name] = {"operations": []}
    # tf.debugging.set_log_device_placement(True)
    # tf.config.set_soft_device_placement(False)
    if os.environ["DEVICE"] == "tpu":
        device_name = "/device:TPU:0"
    elif os.environ["DEVICE"] == "gpu":
        device_name = "/device:GPU:0"
    else:
        device_name = "/device:CPU:0"
    with tf.device(device_name):
        with tensorflow_test_timer():
            yield

    tf.Graph.as_default = _original_as_default
    if os.environ["DEVICE"] == "gpu":
        tf.keras.backend.clear_session()
