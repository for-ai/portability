
import pytest
import os
import json
from ..utils.tensorflow_timer_wrapper import tensorflow_test_timer
import gc
import tensorflow as tf

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
    if os.environ['DEVICE'] == "tpu":
        device_name = "/device:TPU:0"
    elif os.environ['DEVICE'] == "gpu":
        device_name = "/device:GPU:0"
    else:
        device_name = "/device:CPU:0"
    with tf.device(device_name):
        with tensorflow_test_timer():
            yield

    if os.environ['DEVICE'] == "gpu":
        tf.keras.backend.clear_session()
    # print("***MEMORY FREE", tf.config.experimental.get_memory_info('GPU:0'))
