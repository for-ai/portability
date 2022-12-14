import pytest
import os
import json


def pytest_configure():
    pytest.test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0


@pytest.fixture(autouse=True, scope="session")
def track_all():
    yield
    print("AFTER SESSION", pytest.test_times)
    f = open("torch_test_timing.json", "w")
    f.write(json.dumps(pytest.test_times, indent=4, sort_keys=True))


@pytest.fixture(autouse=True)
def track_timing():
    pytest.test_name = os.environ.get(
        'PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    yield
    pytest.test_i = 0
