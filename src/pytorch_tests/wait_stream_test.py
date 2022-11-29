# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torch.distributed.pipeline.sync.stream import (
    CPUStream,
    current_stream,
    default_stream,
    get_device,
    is_cuda,
    new_stream,
    record_stream,
    use_device,
    use_stream,
    wait_stream,
)

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


class TestWaitStream:
    def _test_wait_stream(self, source, target, cuda_sleep=None):
        with use_stream(target):
            if is_cuda(target):
                cuda_sleep(0.5)
            x = torch.ones(100, 100, device=get_device(target))

        wait_stream(source, target)

        with use_stream(source):
            assert x.sum().item() == 10000

    def test_wait_stream_cpu_cpu(self):
        source = CPUStream
        target = CPUStream
        self._test_wait_stream(source, target)

    @skip_if_no_cuda
    def test_wait_stream_cpu_cuda(self, cuda_sleep):
        source = CPUStream
        target = new_stream(torch.device("cuda"))
        self._test_wait_stream(source, target, cuda_sleep)

    @skip_if_no_cuda
    def test_wait_stream_cuda_cpu(self, cuda_sleep):
        source = new_stream(torch.device("cuda"))
        target = CPUStream
        self._test_wait_stream(source, target, cuda_sleep)

    @skip_if_no_cuda
    def test_wait_stream_cuda_cuda(self, cuda_sleep):
        source = current_stream(torch.device("cuda"))
        target = new_stream(torch.device("cuda"))
        self._test_wait_stream(source, target, cuda_sleep)

