# Owner(s): ["module: tests"]

import torch
import numpy as np

from itertools import product, combinations, permutations, chain
from functools import partial
import random
import warnings

from torch._six import nan
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, run_tests, torch_to_numpy_dtype_dict)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, onlyCUDA, dtypes, onlyNativeDeviceTypes,
    dtypesIfCUDA, largeTensorTest)
from torch.testing._internal.common_dtype import all_types_and_complex_and, all_types, all_types_and

def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

class TestShapeOps(TestCase):
    def _rand_shape(self, dim, min_size, max_size):
        return tuple(torch.randint(min_size, max_size + 1, (dim,)))

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim_invalid(self, device, dtype):
        shape = self._rand_shape(4, min_size=5, max_size=10)
        x = _generate_input(shape, dtype, device, False)

        for fn in [torch.movedim, torch.moveaxis]:
            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

            # Mismatch in size of `source` and `destination`
            with self.assertRaisesRegex(RuntimeError, "movedim: Invalid source or destination dims:"):
                fn(x, (1, 0), (0, ))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `source`"):
                fn(x, (0, 0), (0, 1))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `source`"):
                fn(x, (0, 1, 0), (0, 1, 2))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `destination`"):
                fn(x, (0, 1), (1, 1))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `destination`"):
                fn(x, (0, 1, 2), (1, 0, 1))

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim(self, device, dtype):
        for fn in [torch.moveaxis, torch.movedim]:
            for nd in range(5):
                shape = self._rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        # Integer `source` and `destination`
                        torch_fn = partial(fn, source=src_dim, destination=dst_dim)
                        np_fn = partial(np.moveaxis, source=src_dim, destination=dst_dim)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

                    if nd == 0:
                        continue

                    def make_index_negative(sequence, idx):
                        sequence = list(sequence)
                        sequence[random_idx] = sequence[random_idx] - nd
                        return tuple(src_sequence)

                    for src_sequence in permutations(range(nd), r=random.randint(1, nd)):
                        # Sequence `source` and `destination`
                        dst_sequence = tuple(random.sample(range(nd), len(src_sequence)))

                        # Randomly change a dim to a negative dim representation of itself.
                        random_prob = random.random()
                        if random_negative and random_prob > 0.66:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)
                        elif random_negative and random_prob > 0.33:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                        elif random_negative:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)

                        torch_fn = partial(fn, source=src_sequence, destination=dst_sequence)
                        np_fn = partial(np.moveaxis, source=src_sequence, destination=dst_sequence)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Move dim to same position
            x = torch.randn(2, 3, 5, 7, 11)
            torch_fn = partial(fn, source=(0, 1), destination=(0, 1))
            np_fn = partial(np.moveaxis, source=(0, 1), destination=(0, 1))
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            torch_fn = partial(fn, source=1, destination=1)
            np_fn = partial(np.moveaxis, source=1, destination=1)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Empty Sequence
            torch_fn = partial(fn, source=(), destination=())
            np_fn = partial(np.moveaxis, source=(), destination=())
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

    
instantiate_device_type_tests(TestShapeOps, globals())

if __name__ == '__main__':
    run_tests()
