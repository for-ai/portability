import contextlib
import torch
import time

try:
    # Import the TPUProfiler class from the torch_xla package
    from torch_xla import TPUProfiler
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
    elif device.type == 'xla':
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
