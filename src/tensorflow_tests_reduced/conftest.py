
import pytest
import os
import json
from ..utils.timer_wrapper import tensorflow_test_timer
import gc
import tensorflow as tf

# @pytest.fixture(scope='session', autouse=True)
def initialize_tpu(): 
    # if os.environ['DEVICE'] == "tpu":
    os.environ.TPU_NAME = 'local'
    print("*** TPU_NAME", os.environ.TPU_NAME)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_devices = tf.config.list_logical_devices('TPU')
    print("test device", tpu_devices)
    
initialize_tpu()
def pytest_configure():
    pytest.tensorflow_test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0
    
    
    if os.environ['DEVICE'] == "tpu":
        initialize_tpu()
    #     # Create a TPUClusterResolver
    #     resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    #     # Connect to the TPU system
    #     tf.config.experimental_connect_to_cluster(resolver)
    #     # # Initialize the TPU system
    #     tf.tpu.experimental.initialize_tpu_system(resolver)
    #     # # Get the list of TPU devices
    #     tpu_devices = tf.config.list_logical_devices('TPU')
    #     print("test device", tpu_devices)


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

    pytest.tensorflow_test_times[pytest.test_name] = {"operations": []}
    with tensorflow_test_timer():
        yield

    if os.environ['DEVICE'] == "gpu":
        tf.keras.backend.clear_session()
    # print("***MEMORY FREE", tf.config.experimental.get_memory_info('GPU:0'))
