# Owner(s): ["module: __torch_function__"]

import torch
import numpy as np
import inspect
import functools
import pprint
import pickle
import collections
import unittest

from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_CROSSREF
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    get_overridable_functions,
    get_testing_overrides,
    is_tensor_method_or_property,
    TorchFunctionMode,
)
from torch.utils._pytree import tree_map

Tensor = torch.Tensor


def unwrap(v):
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    return v._data if isinstance(v, Wrapper) else v

# wrap inputs if necessary
def wrap(v):
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi) for vi in v)

    return Wrapper(v) if isinstance(v, torch.Tensor) else v

class Wrapper:
    "Basic data container that knows how to unwrap itself"
    def __init__(self, data):
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        val = getattr(self._data, name)

        # If it's a method
        if callable(val):
            c = getattr(type(self._data), name)
            # Don't append self to args if classmethod/staticmethod
            if c is val:
                return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=a, kwargs=kw))
            # Otherwise append self to args
            return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=(self,) + a, kwargs=kw))

        return wrap(val)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, unwrap(value))

    def __setitem__(self, key, value):
        self._data[unwrap(key)] = unwrap(value)

    def __getitem__(self, key):
        return wrap(self._data[unwrap(key)])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Find an instance of this class in the arguments
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        assert len(args_of_this_cls) > 0
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return wrap(func(*args, **kwargs))

    def __add__(self, other):
        return self.__torch_function__(torch.add, (Wrapper,), (self, other))

    def __mul__(self, other):
        return self.__torch_function__(torch.mul, (Wrapper,), (self, other))

    def __sub__(self, other):
        return self.__torch_function__(torch.sub, (Wrapper,), (self, other))

    def __truediv__(self, other):
        return self.__torch_function__(torch.true_divide, (Wrapper,), (self, other))

    def __floordiv__(self, other):
        return self.__torch_function__(torch.floor_divide, (Wrapper,), (self, other))

    def __ge__(self, other):
        return self.__torch_function__(torch.ge, (Wrapper,), (self, other))

    def __gt__(self, other):
        return self.__torch_function__(torch.gt, (Wrapper,), (self, other))

    def __lt__(self, other):
        return self.__torch_function__(torch.lt, (Wrapper,), (self, other))

    def __le__(self, other):
        return self.__torch_function__(torch.le, (Wrapper,), (self, other))

    def __eq__(self, other):
        return self.__torch_function__(torch.eq, (Wrapper,), (self, other))

    def __ne__(self, other):
        return self.__torch_function__(torch.ne, (Wrapper,), (self, other))

    def __bool__(self):
        return self.__torch_function__(torch.Tensor.__bool__, (Wrapper,), (self,))

    def __int__(self):
        return self.__torch_function__(torch.Tensor.__int__, (Wrapper,), (self,))

    def __len__(self):
        return len(self._data)


class TestBroadcastAllOverride(TestCase):
    """ test for gh-37141 """
    def test_broadcast_all(self):
        from torch.distributions.utils import broadcast_all
        a = torch.tensor([1.2, 3.4, 5.6])
        a_w = Wrapper(a)
        b = torch.tensor(5.0)
        b_w = Wrapper(b)
        c = torch.tensor([5.0, 5.0, 5.0])

        o_1 = broadcast_all(a_w, b_w)
        self.assertTrue(isinstance(o_1[0], Wrapper))
        self.assertTrue(isinstance(o_1[1], Wrapper))
        self.assertEqual(o_1[0]._data, a)
        self.assertEqual(o_1[1]._data, c)

        o_2 = broadcast_all(a_w, b)
        self.assertTrue(isinstance(o_2[0], Wrapper))
        self.assertTrue(isinstance(o_2[1], Wrapper))
        self.assertEqual(o_2[0]._data, a)
        self.assertEqual(o_2[1]._data, c)



if __name__ == '__main__':
    run_tests()
