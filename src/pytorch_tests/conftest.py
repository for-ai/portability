import pytest
import os
import json
from ..utils.timer_wrapper import pytorch_test_timer
from torch.testing._internal.common_device_type import onlyCUDA
import torch


try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    xla_present = True
except ImportError:
    xla_present = False


def pytest_configure():
    pytest.pytorch_test_times = {}
    pytest.test_name = ""
    pytest.test_i = 0


def pytest_runtest_call(item):
    testfunction = item.obj
    print("ITEM", item)
    # item.obj = onlyCUDA(testfunction)  # Replace the item function with the decorated one


@pytest.fixture(autouse=True, scope="session")
def track_all():
    yield
    f = open("torch_test_timing.json", "w")
    f.write(json.dumps(pytest.pytorch_test_times, indent=4, sort_keys=True))


@pytest.fixture(autouse=True)
def track_timing(request):
    pytest.test_name = os.environ.get(
        'PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    test_file = str(request.node.fspath).split("/")[-1]
    pytest.test_name = test_file + ":" + pytest.test_name
    pytest.pytorch_test_times[pytest.test_name] = {"operations": []}
    if os.environ['DEVICE'] == "tpu" and xla_present:
        # with xm.xla_device():
            with pytorch_test_timer():
                yield
    else:
        with pytorch_test_timer():
            yield

    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    # f = r-a  # free inside reserved
    if os.environ['DEVICE'] == "cuda":
        torch.cuda.empty_cache()
        print("***MEMORY FREE", torch.cuda.mem_get_info())
