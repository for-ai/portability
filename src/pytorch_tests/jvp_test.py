# Owner(s): ["module: autograd"]

import types
import unittest
import warnings

import torch
import torch.autograd.functional as autogradF

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    TestCase, run_tests, subtest, gradcheck, gradgradcheck, parametrize, instantiate_parametrized_tests)
from torch.testing._internal.logging_tensor import LoggingTensor

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu


# Utilities for parametrizing the tensor constructors used in autograd tests
#
# TODO: maybe move somewhere so other tests can also use
#
# NB: Not all factory functions included. A complete(?) list can be found here:
#     https://pytorch.org/cppdocs/notes/tensor_creation.html
base_ctors_dict = {
    "ones": torch.ones,
    "zeros": torch.zeros,
    "randn": torch.randn,
    "rand": torch.rand,
    "tensor": torch.tensor,
}
base_ctors = types.SimpleNamespace(**base_ctors_dict)

def wrap_with_logging_tensor(ctor):
    def wrapper(*args, **kwargs):
        requires_grad = kwargs.pop("requires_grad", False)
        return LoggingTensor(ctor(*args, **kwargs), requires_grad=requires_grad)
    return wrapper

logging_tensor_ctors_dict = {k: wrap_with_logging_tensor(ctor) for (k, ctor) in base_ctors_dict.items()}
logging_tensor_ctors = types.SimpleNamespace(**logging_tensor_ctors_dict)

base_and_logging_tensor = parametrize("ctors", [subtest(base_ctors, name="base_tensor"),
                                                subtest(logging_tensor_ctors, name="logging_tensor")])

FIXME_base_and_xfail_logging_tensor = parametrize("ctors", [subtest(base_ctors, name="base_tensor"),
                                                            subtest(logging_tensor_ctors, name="logging_tensor",
                                                                    decorators=[unittest.expectedFailure])])

# NB: This is equivalent to having both @parmetrize("vectorized", [True, False]) and
#     FIXME_base_and_xfail_logging_tensor, except the non-vectorized logging_tensor case is
#     actually expected to succeed
FIXME_xfail_vectorized_logging_tensor = (
    parametrize("vectorize,ctors", [subtest((True, base_ctors), name="vectorized_base_tensor"),
                                    subtest((False, base_ctors), name="base_tensor"),
                                    subtest((True, logging_tensor_ctors), name="vectorized_logging_tensor",
                                            decorators=[unittest.expectedFailure]),
                                    subtest((False, logging_tensor_ctors), name="logging_tensor")]))

vectorized_logging_tensor = (
    parametrize("vectorize,ctors", [subtest((True, base_ctors), name="vectorized_base_tensor"),
                                    subtest((False, base_ctors), name="base_tensor"),
                                    subtest((True, logging_tensor_ctors), name="vectorized_logging_tensor"),
                                    subtest((False, logging_tensor_ctors), name="logging_tensor")]))


class TestAutogradFunctional(TestCase):
    def _assert_same_struct(self, res, base):
        # base and res should be Tensors or tuple of Tensors with the same size
        if isinstance(base, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(base.size(), res.size())
        elif isinstance(base, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(base), len(res))
            for el_base, el_res in zip(base, res):
                self.assertTrue(isinstance(el_base, torch.Tensor))
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertEqual(el_base.size(), el_res.size())
        else:
            # Wrong base
            raise RuntimeError("The base given to `_assert_same_struct` doesn't have"
                               " the right structure.")

    def _assert_interleaved_struct(self, res, base1, base2):
        # base1 and base2 can be Tensors or tuples of Tensors.
        # If they are tuples, res should be a tuple as well.
        # The indexing works as follows for base1, base2 being
        # - tuple, tuple: res[i][j][k][l] = (base1[i][k], base2[j][l])
        # - tuple, Tensor: res[i][k][l] = (base1[i][k], base2[l])
        # - Tensor, tuple: res[i][j][l] = (base1[i], base2[j][l])
        # - Tensor, Tensor: res[k][l] = (base1[k], base2[l])
        if isinstance(base1, torch.Tensor) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(res.size(), base1.size() + base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base1, torch.Tensor))
                self.assertEqual(el_res.size(), el_base1.size() + base2.size())
        elif isinstance(base1, torch.Tensor) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base2))
            for el_res, el_base2 in zip(res, base2):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base2, torch.Tensor))
                self.assertEqual(el_res.size(), base1.size() + el_base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, tuple))
                self.assertEqual(len(res), len(base2))
                for el_el_res, el_base2 in zip(el_res, base2):
                    self.assertTrue(isinstance(el_el_res, torch.Tensor))
                    self.assertTrue(isinstance(el_base2, torch.Tensor))
                    self.assertEqual(el_el_res.size(), el_base1.size() + el_base2.size())
        else:
            # Wrong bases
            raise RuntimeError("The bases given to `_assert_interleaved_struct` don't have"
                               " the right structure.")


    @base_and_logging_tensor
    def test_jvp_err_check(self, ctors):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to jvp must be either a Tensor"):
            res = autogradF.jvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to jvp must"):
            res = autogradF.jvp(bar, inp, v)

        with self.assertRaisesRegex(RuntimeError, "The vector v can only be None if the input to the user-provided function"):
            res = autogradF.jvp(foo, inp)

        with self.assertRaisesRegex(RuntimeError, "The given v should contain a single Tensor."):
            res = autogradF.jvp(foo, inp, (v, v))

        with self.assertRaisesRegex(RuntimeError, "v has invalid size: should be torch.Size"):
            res = autogradF.jvp(foo, inp, v[:2])

        res = autogradF.jvp(foo, inp, v)[1]
        self._assert_same_struct(res, foo(inp))

    @base_and_logging_tensor
    def test_jvp_err_check_strict(self, ctors):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = ctors.rand(4)
        v = ctors.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.jvp(foo, inp, v, strict=True)
        res = autogradF.jvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.jvp(bar, inp, v, strict=True)
        res = autogradF.jvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.jvp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.jvp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    @base_and_logging_tensor
    def test_jvp_no_grad(self, ctors):
        def reducer(x):
            return x.sum(dim=1)
        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], ctors.zeros(4, 4))

    @base_and_logging_tensor
    def test_jvp_output(self, ctors):
        def reducer(x):
            return x.sum(dim=1)
        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.ones(2), ctors.ones(2))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y, x + y

        inputs = (ctors.rand(2), ctors.rand(2))
        v = (ctors.tensor([1., 0.]), ctors.tensor([1., 0.]))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

    @base_and_logging_tensor
    def test_jvp_scalar(self, ctors):
        def reducer(x):
            return x.sum()
        inputs = ctors.rand(4, 4)
        v = ctors.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[0], ctors.zeros([]))
        self._assert_same_struct(res[1], res[0])

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = ctors.rand([])
        v = ctors.ones([])
        res = autogradF.jvp(expander, inputs, v)
        self._assert_same_struct(res[0], ctors.zeros(4))
        self._assert_same_struct(res[1], res[0])

        res = autogradF.jvp(expander, inputs)
        self._assert_same_struct(res[0], ctors.zeros(4))
        self._assert_same_struct(res[1], res[0])

    @base_and_logging_tensor
    def test_jvp_create_graph(self, ctors):
        def reducer(x):
            return x.sum(dim=1)
        inputs = ctors.rand(2, 2, dtype=torch.double)
        v = ctors.ones(2, 2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))

        def adder(x, y):
            return 2 * x + 3 * y, x * y

        inputs = (ctors.rand(2, dtype=torch.double, requires_grad=True),
                  ctors.rand(2, dtype=torch.double, requires_grad=True))
        v = (ctors.tensor([1., 0.], dtype=torch.double, requires_grad=True),
             ctors.tensor([1., 0.], dtype=torch.double, requires_grad=True))

        gradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.jvp(adder, (x, y), v, create_graph=True)

            return val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    
    @base_and_logging_tensor
    def test_jacobian_match_vjp_jvp(self, ctors):
        def foo(x):
            return x ** 3 + x.sum()

        inputs = ctors.rand(4)
        v = ctors.rand(4)

        jac = autogradF.jacobian(foo, inputs)
        jvp = autogradF.jvp(foo, inputs, v)[1]
        vjp = autogradF.vjp(foo, inputs, v)[1]

        self.assertEqual(jvp, torch.mm(jac, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vjp, torch.mm(v.unsqueeze(0), jac).squeeze(0))

instantiate_parametrized_tests(TestAutogradFunctional)

if __name__ == '__main__':
    run_tests()
