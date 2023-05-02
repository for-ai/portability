import pytest
import os
import json
from ..utils.timer_wrapper import jax_test_timer
import gc
import contextlib
import jax
from jax.config import config
from jax._src import test_util as jtu


def blank(*args, **kwargs):
    return True


jtu.JaxTestCase._CompileAndCheck = blank


def pytest_runtest_call(item):
    # class CombinedContext:
    print("ITEM", item)


def pytest_configure():
    pytest.jax_test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0


@pytest.fixture(autouse=True, scope="session")
def track_all():
    yield
    f = open("jax_test_timing.json", "w")
    f.write(json.dumps(pytest.jax_test_times, indent=4, sort_keys=True))


@pytest.fixture(autouse=True)
def track_timing(request):
    config.update("jax_disable_jit", True)

    pytest.test_name = (
        os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    )
    test_file = str(request.node.fspath).split("/")[-1]
    pytest.test_name = test_file + ":" + pytest.test_name

    pytest.jax_test_times[pytest.test_name] = {"operations": []}
    if os.environ["DEVICE"] == "tpu":
        jax.default_device = "/device:TPU:0"
    elif os.environ["DEVICE"] == "gpu":
        jax.default_device = "/device:GPU:0"
    else:
        jax.default_device = "/device:CPU:0"

    with jax_test_timer():
        with jax.numpy_dtype_promotion("standard"):
            yield
