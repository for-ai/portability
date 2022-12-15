
import pytest
import os
import json
from ..utils.timer_wrapper import tensorflow_test_timer


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

    pytest.tensorflow_test_times[pytest.test_name] = {"operations": []}
    with tensorflow_test_timer():
        yield
