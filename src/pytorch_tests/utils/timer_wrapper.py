import contextlib
import torch
import time
import numpy as np

try:
    # Import the TPUProfiler class from the torch_xla package
    from torch_xla import TPUProfiler
    import torch_xla.core.xla_model as xm
except ImportError:
    # torch_xla is not installed, so TPUProfiler is not available
    TPUProfiler = None


@contextlib.contextmanager
def pytorch_timer():
    if torch.cuda.is_available():
        # Use CUDA events to measure time on a GPU
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()

        # Waits for all CUDA operations to finish running
        torch.cuda.synchronize()
        print(start.elapsed_time(end))  # milliseconds
    elif TPUProfiler != None and xm.xla_device():
        # Use TPUProfiler to measure time on a TPU
        with TPUProfiler('pytorch_timer') as prof:
            yield
        # Print the time elapsed for TPU operations
        print(prof.total_time_ms())
    else:
        # Use Python's time module to measure time on a CPU
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        print(end - start)  # seconds

"""
- not sure if this is the best way, but it is a place to start
- for example, if we want to measure the running time for np.sum, below is a way to do it. 
Then, in every place we use np.sum, we have to replace it with mysum.
- this will require a lot of work, so it would be great if we could find a better way to do it. 
"""
def mysum(*args, **kwargs):
    with pytorch_timer() as timer:
        return np.array(*args, **kwargs)

    return timer.elapsed_time

mysum([1,2,3])