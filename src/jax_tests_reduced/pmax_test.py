# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from functools import partial
import itertools as it
from typing import Any, List, Optional, Callable, Union, TypeVar

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import lax
from jax._src.lax import parallel
from jax import random
from jax import jit, grad, jvp, vjp, make_jaxpr, jacfwd, jacrev, hessian
from jax import vmap
from jax.interpreters import batching
from jax.tree_util import register_pytree_node

from jax.config import config

config.parse_flags_with_absl()


# These are 'manual' tests for batching (vmap). The more exhaustive, more
# systematic tests are in lax_test.py's LaxVmapTest class.


class BatchingTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "_{}_vmap_names={}_collective_names={}".format(
                collective.__name__.replace(" ", ""),
                "".join(vmap_names),
                "".join(collective_names),
            ),
            "collective": collective,
            "bulk_op": bulk_op,
            "vmap_names": vmap_names,
            "collective_names": collective_names,
        }
        for collective, bulk_op in [(lax.pmax, jnp.max)]
        for vmap_names in [("i",), ("i", "j"), ("i", "j", "k")]
        for subset_size in range(1, len(vmap_names) + 1)
        for collective_subset in it.combinations(vmap_names, subset_size)
        for collective_names in it.permutations(collective_subset)
    )
    def testCommAssocCollective(
        self, collective, bulk_op, vmap_names, collective_names
    ):
        shape = (2, 2, 2)
        x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

        # To test relative permutations of the order in which the axis names appear
        # in the primitive call versus the order the vmaps are applied, we always
        # apply vmaps in the order of the `vmap_names` argument, and apply the
        # collective with names according to the `collective_names` argument.
        f = lambda x: x - collective(x, collective_names)
        # Use non-zero in and out axes to improve the coverage
        for i, axis_name in enumerate(vmap_names):
            f = vmap(f, axis_name=axis_name, in_axes=i, out_axes=i)
        pos_axis = [i for i, name in enumerate(vmap_names) if name in collective_names]
        self.assertAllClose(f(x), x - bulk_op(x, axis=pos_axis, keepdims=True))

        if collective is lax.psum:
            jtu.check_grads(f, (x,), 2, eps=1)
