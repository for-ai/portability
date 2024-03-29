# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensorflow.python.client.session.Session."""
import collections
import os
import random
import sys
import threading
import time
import warnings

import numpy as np
import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as framework_device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
# Import gradients to resolve circular imports
from tensorflow.python.ops import gradients  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
import tensorflow as tf
from ..tensorflow_test import device_context
from ..utils.timer_wrapper import tensorflow_op_timer

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None

try:
  from frozendict import frozendict  # pylint:disable=g-import-not-at-top
except ImportError:
  frozendict = dict  # pylint:disable=invalid-name

defaultdict = collections.defaultdict  # pylint:disable=invalid-name


@test_util.with_eager_op_as_function
class SessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(SessionTest, self).setUp()
    warnings.simplefilter('always')

  def testSessionInterOpThreadPool(self):
    config_pb = config_pb2.ConfigProto()
    pool = config_pb.session_inter_op_thread_pool.add()
    with session.Session(config=config_pb) as s:
      with device_context():
        inp = constant_op.constant(10.0, name='W1')
        print("***INP", inp.device)
        timer = tensorflow_op_timer()
        with timer:
          results = s.run([inp])
          timer.gen.send(results)
        self.assertAllEqual([10.0], results)

    pool = config_pb.session_inter_op_thread_pool.add()
    pool.num_threads = 1
    with session.Session(config=config_pb) as s:
      with device_context():
        inp = constant_op.constant(20.0, name='W2')
        timer = tensorflow_op_timer()
        with timer:
          results = s.run([inp])
          timer.gen.send(results)
        self.assertAllEqual([20.0], results)

    pool = config_pb.session_inter_op_thread_pool.add()
    pool.num_threads = 1
    pool.global_name = 't1'
    run_options = config_pb2.RunOptions()
    run_options.inter_op_thread_pool = (
        len(config_pb.session_inter_op_thread_pool) - 1)
    with session.Session(config=config_pb) as s:
      with device_context():
        inp = constant_op.constant(30.0, name='W2')
        timer = tensorflow_op_timer()
        with timer:
          results = s.run([inp], options=run_options)
          timer.gen.send(results)
        self.assertAllEqual([30.0], results)

  def testErrorsReported(self):
    with session.Session() as s:
      with device_context():
        constant_op.constant(10.0, name='W1')
        with self.assertRaises(ValueError):
          s.run('foo:0')

 

  def testFetchNone(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0)
        with self.assertRaises(TypeError):
          s.run(None)
        with self.assertRaises(TypeError):
          s.run([None])
        with self.assertRaises(TypeError):
          s.run({'b': None})
        with self.assertRaises(TypeError):
          s.run({'a': a, 'b': None})

  def testFetchSingleton(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(a)
          timer.gen.send(res)
        self.assertEqual(42.0, res)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(a.op)  # An op, not a tensor.
          timer.gen.send(res)
        self.assertIsNone(res)
        tensor_runner = sess.make_callable(a)
        res = tensor_runner()
        self.assertEqual(42.0, res)
        op_runner = sess.make_callable(a.op)
        res = op_runner()
        self.assertIsNone(res)

  def testFetchSingletonByName(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(a.name)
          timer.gen.send(res)
        self.assertEqual(42.0, res)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(a.op)  # An op, not a tensor.
          timer.gen.send(res)
        self.assertIsNone(res)

  def testFetchList(self):
      with session.Session() as sess:
        with device_context():
          a = constant_op.constant(42.0)
          b = control_flow_ops.no_op()  # An op, not a tensor.
          c = constant_op.constant(44.0)
          v = variables.Variable([54.0])
          assign = v.assign([63.0])
          timer = tensorflow_op_timer()
          with timer:
            res = sess.run([a, b, c, a.name, assign.op])
            timer.gen.send(res)
          self.assertIsInstance(res, list)
          self.assertEqual([42.0, None, 44.0, 42.0, None], res)
          list_runner = sess.make_callable([a, b, c, a.name, assign.op])
          res = list_runner()
          self.assertIsInstance(res, list)
          self.assertEqual([42.0, None, 44.0, 42.0, None], res)

  def testFetchTuple(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        b = control_flow_ops.no_op()  # An op, not a tensor.
        c = constant_op.constant(44.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run((a, b, c, a.name))
          timer.gen.send(res)
        self.assertIsInstance(res, tuple)
        self.assertEqual((42.0, None, 44.0, 42.0), res)
        tuple_runner = sess.make_callable((a, b, c, a.name))
        res = tuple_runner()
        self.assertIsInstance(res, tuple)
        self.assertEqual((42.0, None, 44.0, 42.0), res)

  def testFetchNamedTuple(self):
    # pylint: disable=invalid-name
    ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
    # pylint: enable=invalid-name
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        b = control_flow_ops.no_op()  # An op, not a tensor.
        c = constant_op.constant(44.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(ABC(a, b, c))
          timer.gen.send(res)
        self.assertIsInstance(res, ABC)
        self.assertEqual(42.0, res.a)
        self.assertIsNone(res.b)
        self.assertEqual(44.0, res.c)
        namedtuple_runner = sess.make_callable(ABC(a, b, c))
        res = namedtuple_runner()
        self.assertIsInstance(res, ABC)
        self.assertEqual(42.0, res.a)
        self.assertIsNone(res.b)
        self.assertEqual(44.0, res.c)

  def testFetchDict(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        b = control_flow_ops.no_op()  # An op, not a tensor.
        c = constant_op.constant(44.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run({'a': a, 'b': b, 'c': c})
          timer.gen.send(res)
        self.assertIsInstance(res, dict)
        self.assertEqual(42.0, res['a'])
        self.assertIsNone(res['b'])
        self.assertEqual(44.0, res['c'])

  def testFetchOrderedDict(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(42.0)
        b = control_flow_ops.no_op()  # An op, not a tensor.
        c = constant_op.constant(44.0)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(collections.OrderedDict([(3, a), (2, b), (1, c)]))
          timer.gen.send(res)
        self.assertIsInstance(res, collections.OrderedDict)
        self.assertEqual([3, 2, 1], list(res.keys()))
        self.assertEqual(42.0, res[3])
        self.assertIsNone(res[2])
        self.assertEqual(44.0, res[1])

  @test_util.run_v1_only('b/120545219')
  def testFetchAttrs(self):
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class SampleAttr(object):
      field1 = attr.ib()
      field2 = attr.ib()

    val1 = np.array([1.2, 3.4, 5.6])
    val2 = np.array([[1, 2], [4, 3]])
    val3 = np.array([10, 20, 30])

    t1 = constant_op.constant(val1)
    t2 = constant_op.constant(val2)

    sample = SampleAttr(t1, t2)
    with session.Session() as sess:
      with device_context():
        timer = tensorflow_op_timer()
        with timer:
          result = sess.run(sample)
          timer.gen.send(result)
        self.assertIsInstance(result, SampleAttr)
        self.assertAllEqual(val1, result.field1)
        self.assertAllEqual(val2, result.field2)
        timer = tensorflow_op_timer()
        with timer:
          result = sess.run(sample, feed_dict={sample.field1: val3})
          timer.gen.send(result)
        self.assertIsInstance(result, SampleAttr)
        self.assertAllEqual(val3, result.field1)
        self.assertAllEqual(val2, result.field2)

  @test_util.run_v1_only('b/120545219')
  def testFetchNestedAttrs(self):
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class SampleAttr(object):
      field0 = attr.ib()
      field1 = attr.ib()

    v1 = 10
    v2 = 20
    v3 = np.float32(1.2)
    v4 = np.float32(3.4)
    v5 = np.float64(100.001)
    v6 = np.float64(-23.451)
    arr1 = np.array([1.2, 6.7, 3.4])
    arr2 = np.array([7, 11, 3])
    sample = SampleAttr(
        SampleAttr(
            SampleAttr(constant_op.constant(v1), constant_op.constant(v2)),
            SampleAttr(constant_op.constant(arr1), constant_op.constant(arr2))),
        {'A': SampleAttr(constant_op.constant(v3), constant_op.constant(v4)),
         'B': [SampleAttr(constant_op.constant(v5), constant_op.constant(v6))]})

    with session.Session() as sess:
      with device_context():
        timer = tensorflow_op_timer()
        with timer:
          result = sess.run(sample)
          timer.gen.send(result)
        self.assertIsInstance(result, SampleAttr)
        self.assertIsInstance(result.field0, SampleAttr)
        self.assertIsInstance(result.field0.field0, SampleAttr)
        self.assertIsInstance(result.field0.field1, SampleAttr)
        self.assertIsInstance(result.field0.field1.field0, np.ndarray)
        self.assertAllEqual(arr1, result.field0.field1.field0)
        self.assertIsInstance(result.field0.field1.field1, np.ndarray)
        self.assertAllEqual(arr2, result.field0.field1.field1)
        self.assertIsInstance(result.field1, dict)
        self.assertIn('A', result.field1)
        self.assertIn('B', result.field1)
        self.assertIsInstance(result.field1['A'], SampleAttr)
        self.assertAllEqual(
            [v3, v4],
            [result.field1['A'].field0, result.field1['A'].field1])
        self.assertIsInstance(result.field1['B'], list)
        self.assertEqual(1, len(result.field1['B']))
        self.assertIsInstance(result.field1['B'][0], SampleAttr)
        self.assertAllEqual(
            [v5, v6],
            [result.field1['B'][0].field0, result.field1['B'][0].field1])

  def testFetchNestingEmptyOneLevel(self):
    with session.Session() as sess:
      with device_context():
        a_val = 11.0
        a = constant_op.constant(a_val)
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run([[], tuple(), {}])
          timer.gen.send(res)
        self.assertIsInstance(res, list)
        self.assertEqual(3, len(res))
        self.assertIsInstance(res[0], list)
        self.assertEqual(0, len(res[0]))
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(0, len(res[1]))
        self.assertIsInstance(res[2], dict)
        self.assertEqual(0, len(res[2]))
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run([[], tuple(), {}, a])
          timer.gen.send(res)
        self.assertIsInstance(res, list)
        self.assertEqual(4, len(res))
        self.assertIsInstance(res[0], list)
        self.assertEqual(0, len(res[0]))
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(0, len(res[1]))
        self.assertIsInstance(res[2], dict)
        self.assertEqual(0, len(res[2]))
        self.assertEqual(a_val, res[3])

  def testFetchNestingOneLevel(self):
    with session.Session() as sess:
      with device_context():
        # pylint: disable=invalid-name
        ABC = collections.namedtuple('ABC', ['a', 'b', 'c'])
        DEFGHI = collections.namedtuple('DEFGHI', ['d', 'e', 'f', 'g', 'h', 'i'])
        # pylint: enable=invalid-name
        a_val = 42.0
        b_val = None
        c_val = 44.0
        a = constant_op.constant(a_val)
        b = control_flow_ops.no_op()  # An op, not a tensor.
        c = constant_op.constant(c_val)
        test_dct = {'a': a.name, 'c': c, 'b': b}
        test_dct_types = [dict, frozendict, defaultdict]
        # List of lists, tuples, namedtuple, dict, frozendict, and defaultdict
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run([
            [a, b, c],
            (a, b, c),
            ABC(a=a, b=b, c=c),
            dict(test_dct),
            frozendict(test_dct),
            defaultdict(str, test_dct),
          ])
          timer.gen.send(res)
        self.assertIsInstance(res, list)
        self.assertEqual(6, len(res))
        self.assertIsInstance(res[0], list)
        self.assertEqual(3, len(res[0]))
        self.assertEqual(a_val, res[0][0])
        self.assertEqual(b_val, res[0][1])
        self.assertEqual(c_val, res[0][2])
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(3, len(res[1]))
        self.assertEqual(a_val, res[1][0])
        self.assertEqual(b_val, res[1][1])
        self.assertEqual(c_val, res[1][2])
        self.assertIsInstance(res[2], ABC)
        self.assertEqual(a_val, res[2].a)
        self.assertEqual(b_val, res[2].b)
        self.assertEqual(c_val, res[2].c)
        for expected_type, r in zip(test_dct_types, res[3:]):
          self.assertIsInstance(r, expected_type)
          self.assertEqual(3, len(r))
          self.assertEqual(a_val, r['a'])
          self.assertEqual(b_val, r['b'])
          self.assertEqual(c_val, r['c'])
        self.assertEqual(res[5].default_factory, str)
        # Tuple of lists, tuples, namedtuple, dict, frozendict, and defaultdict
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(([a, b, c], (a.name, b, c), ABC(a=a, b=b,
                                                      c=c), dict(test_dct),
                        frozendict(test_dct), defaultdict(str, test_dct)))
          timer.gen.send(res)
        self.assertIsInstance(res, tuple)
        self.assertEqual(6, len(res))
        self.assertIsInstance(res[0], list)
        self.assertEqual(3, len(res[0]))
        self.assertEqual(a_val, res[0][0])
        self.assertEqual(b_val, res[0][1])
        self.assertEqual(c_val, res[0][2])
        self.assertIsInstance(res[1], tuple)
        self.assertEqual(3, len(res[1]))
        self.assertEqual(a_val, res[1][0])
        self.assertEqual(b_val, res[1][1])
        self.assertEqual(c_val, res[1][2])
        self.assertIsInstance(res[2], ABC)
        self.assertEqual(a_val, res[2].a)
        self.assertEqual(b_val, res[2].b)
        self.assertEqual(c_val, res[2].c)
        for expected_type, r in zip(test_dct_types, res[3:]):
          self.assertIsInstance(r, expected_type)
          self.assertEqual(3, len(r))
          self.assertEqual(a_val, r['a'])
          self.assertEqual(b_val, r['b'])
          self.assertEqual(c_val, r['c'])
        self.assertEqual(res[5].default_factory, str)

        # Namedtuple of lists, tuples, namedtuples, dict, frozendict, defaultdict
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run(
            DEFGHI(
                d=[a, b, c],
                e=(a, b, c),
                f=ABC(a=a.name, b=b, c=c),
                g=dict(test_dct),
                h=frozendict(test_dct),
                i=defaultdict(str, test_dct)))
          timer.gen.send(res)
        self.assertIsInstance(res, DEFGHI)
        self.assertIsInstance(res.d, list)
        self.assertEqual(3, len(res.d))
        self.assertEqual(a_val, res.d[0])
        self.assertEqual(b_val, res.d[1])
        self.assertEqual(c_val, res.d[2])
        self.assertIsInstance(res.e, tuple)
        self.assertEqual(3, len(res.e))
        self.assertEqual(a_val, res.e[0])
        self.assertEqual(b_val, res.e[1])
        self.assertEqual(c_val, res.e[2])
        self.assertIsInstance(res.f, ABC)
        self.assertEqual(a_val, res.f.a)
        self.assertEqual(b_val, res.f.b)
        self.assertEqual(c_val, res.f.c)
        self.assertIsInstance(res.g, dict)
        self.assertEqual(3, len(res.g))
        self.assertEqual(a_val, res.g['a'])
        self.assertEqual(b_val, res.g['b'])
        self.assertEqual(c_val, res.g['c'])
        self.assertIsInstance(res.h, frozendict)
        self.assertEqual(3, len(res.h))
        self.assertEqual(a_val, res.h['a'])
        self.assertEqual(b_val, res.h['b'])
        self.assertEqual(c_val, res.h['c'])
        self.assertIsInstance(res.i, defaultdict)
        self.assertEqual(3, len(res.i))
        self.assertEqual(a_val, res.i['a'])
        self.assertEqual(b_val, res.i['b'])
        self.assertEqual(c_val, res.i['c'])
        self.assertEqual(res.i.default_factory, str)
        # Dict of lists, tuples, namedtuples, dict, frozendict, defaultdict
        timer = tensorflow_op_timer()
        with timer:
          res = sess.run({
            'd': [a, b, c],
            'e': (a, b, c),
            'f': ABC(a=a, b=b, c=c),
            'g': dict(test_dct),
            'h': frozendict(test_dct),
            'i': defaultdict(str, test_dct),
          })
          timer.gen.send(res)
        self.assertIsInstance(res, dict)
        self.assertEqual(6, len(res))
        self.assertIsInstance(res['d'], list)
        self.assertEqual(3, len(res['d']))
        self.assertEqual(a_val, res['d'][0])
        self.assertEqual(b_val, res['d'][1])
        self.assertEqual(c_val, res['d'][2])
        self.assertIsInstance(res['e'], tuple)
        self.assertEqual(3, len(res['e']))
        self.assertEqual(a_val, res['e'][0])
        self.assertEqual(b_val, res['e'][1])
        self.assertEqual(c_val, res['e'][2])
        self.assertIsInstance(res['f'], ABC)
        self.assertEqual(a_val, res['f'].a)
        self.assertEqual(b_val, res['f'].b)
        self.assertEqual(c_val, res['f'].c)
        for expected_type, r_key in zip(test_dct_types, ('g', 'h', 'i')):
          r = res[r_key]
          self.assertIsInstance(r, expected_type)
          self.assertEqual(3, len(r))
          self.assertEqual(a_val, r['a'])
          self.assertEqual(b_val, r['b'])
          self.assertEqual(c_val, r['c'])
        self.assertEqual(res['i'].default_factory, str)

  def testFetchTensorObject(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[2, 3])
        c = math_ops.matmul(a, b)
        timer = tensorflow_op_timer()
        with timer:
          results_with_list = s.run([c])
          timer.gen.send(results_with_list)
        self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_list[0])
        timer = tensorflow_op_timer()
        with timer:
          results_with_single = s.run(c)
          timer.gen.send(results_with_single)
        self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_single)
        results_with_get = c.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_get)
        timer = tensorflow_op_timer()
        with timer:
          a_val, b_val = s.run([a, b])  # Test multiple fetches.
          timer.gen.send(a_val)
        self.assertAllEqual([[1.0, 1.0]], a_val)
        self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], b_val)
        timer = tensorflow_op_timer()
        with timer:
          results_with_dict = s.run({'a': [a], 'b': b, 'z': [a, b]})
          timer.gen.send(results_with_dict)
        self.assertAllEqual([[1.0, 1.0]], results_with_dict['a'][0])
        self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                            results_with_dict['b'])
        self.assertAllEqual(results_with_dict['a'][0], results_with_dict['z'][0])
        self.assertAllEqual(results_with_dict['b'], results_with_dict['z'][1])

        # Test nested structures
        timer = tensorflow_op_timer()
        with timer:
          results_with_nested_list = s.run([[[a, b], b], a, [a, b]])
          timer.gen.send(results_with_nested_list)
        self.assertAllEqual([[1.0, 1.0]], results_with_nested_list[0][0][0])
        self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                            results_with_nested_list[0][0][1])
        self.assertAllEqual(results_with_nested_list[0][0][0],
                            results_with_nested_list[1])
        self.assertAllEqual(results_with_nested_list[1],
                            results_with_nested_list[2][0])
        self.assertAllEqual(results_with_nested_list[0][0][1],
                            results_with_nested_list[0][1])
        self.assertAllEqual(results_with_nested_list[0][1],
                            results_with_nested_list[2][1])

  def testFetchScalar(self):
    with session.Session() as s:
      with device_context():
        for scalar in np.int32, np.int64, np.float16, np.float32, np.float64:
          x = scalar(7)
          y = scalar(8)
          tf_x = constant_op.constant(x, shape=[])
          tf_y = constant_op.constant(y)
          tf_xy = math_ops.add(tf_x, tf_y)
          # Single fetch
          timer = tensorflow_op_timer()
          with timer:
            xy = s.run(tf_xy)
            timer.gen.send(xy)
          self.assertEqual(scalar, type(xy))
          self.assertEqual(x + y, xy)
          # List fetch
          timer = tensorflow_op_timer()
          with timer:
            xy, = s.run([tf_xy])
            timer.gen.send(xy)
          self.assertEqual(scalar, type(xy))
          self.assertEqual(x + y, xy)
          # Dict fetch
          timer = tensorflow_op_timer()
          with timer:
            xy = s.run({'xy': tf_xy})['xy']
            timer.gen.send(xy)
          self.assertEqual(scalar, type(xy))
          self.assertEqual(x + y, xy)
          # Nested list fetch
          timer = tensorflow_op_timer()
          with timer:
            xy = s.run([[[tf_xy]], tf_xy, [tf_xy]])
            timer.gen.send(xy)
          self.assertAllEqual(xy, [[[x + y]], x + y, [x + y]])
          self.assertEqual(scalar, type(xy[0][0][0]))
          self.assertEqual(scalar, type(xy[1]))
          self.assertEqual(scalar, type(xy[2][0]))

  def testFetchOperationObject(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        v = variables.Variable(a, name='testFetchOperationObject_v')
        timer = tensorflow_op_timer()
        with timer:
          s.run(v.initializer)
          timer.gen.send(s)
        timer = tensorflow_op_timer()
        with timer:  
          v_val = s.run(v)
          timer.gen.send(v_val)
        self.assertAllEqual([[1.0, 1.0]], v_val)

  def testFetchSparseTensor(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        shape = np.array([7, 9, 2]).astype(np.int64)
        sp = sparse_tensor.SparseTensor(
            constant_op.constant(indices), constant_op.constant(values),
            constant_op.constant(shape))
        # Single fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run(sp)
          timer.gen.send(sp_out)
        indices_out, values_out, shape_out = sp_out
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Single fetch, use as SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run(sp)
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out.indices, indices)
        self.assertAllEqual(sp_out.values, values)
        self.assertAllEqual(sp_out.dense_shape, shape)
        # Tuple fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(sp)
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # List fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          (indices_out, values_out, shape_out), = s.run([sp])
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # List fetch, use as SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp_out, = s.run([sp])
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out.indices, indices)
        self.assertAllEqual(sp_out.values, values)
        self.assertAllEqual(sp_out.dense_shape, shape)
        # Dict fetch (single value), use as tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run({'sp': sp})['sp']
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Dict fetch (list value), use as tuple
        timer = tensorflow_op_timer()
        with timer:
          (indices_out, values_out, shape_out), = s.run({'sp': [sp]})['sp']
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Dict fetch, use as SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run({'sp': sp})['sp']
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out.indices, indices)
        self.assertAllEqual(sp_out.values, values)
        self.assertAllEqual(sp_out.dense_shape, shape)
        # Nested list fetch use as tuple
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run([[[sp]], sp])
          timer.gen.send(sp_out)
        indices_out, values_out, shape_out = sp_out[0][0][0]
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        indices_out, values_out, shape_out = sp_out[1]
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Nested list fetch, use as SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run([[[sp]], sp])
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out[0][0][0].indices, indices)
        self.assertAllEqual(sp_out[0][0][0].values, values)
        self.assertAllEqual(sp_out[0][0][0].dense_shape, shape)
        self.assertAllEqual(sp_out[1].indices, indices)
        self.assertAllEqual(sp_out[1].values, values)
        self.assertAllEqual(sp_out[1].dense_shape, shape)

  def testFeedSparseTensor(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        shape = np.array([7, 9, 2]).astype(np.int64)
        sp = sparse_tensor.SparseTensor(
            array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
            array_ops.placeholder(dtype=np.float32, shape=(2,)),
            array_ops.placeholder(dtype=np.int64, shape=(3,)),
        )
        sp_indices = array_ops.identity(sp.indices)
        sp_values = array_ops.identity(sp.values)
        sp_shape = array_ops.identity(sp.dense_shape)
        sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: (indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with tuple, fetch sp directly
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run(sp, {sp: (indices, values, shape)})
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out.indices, indices)
        self.assertAllEqual(sp_out.values, values)
        self.assertAllEqual(sp_out.dense_shape, shape)
        # Feed with SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: sparse_tensor.SparseTensorValue(indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with SparseTensorValue, fetch SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp2_out = s.run(sp2, {
            sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
          timer.gen.send(sp2_out)
        self.assertAllEqual(sp2_out.indices, indices)
        self.assertAllEqual(sp2_out.values, values)
        self.assertAllEqual(sp2_out.dense_shape, shape)
        # Feed SparseTensorValue and fetch sp directly.
        timer = tensorflow_op_timer()
        with timer:
          sp_out = s.run(sp, {
            sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
          timer.gen.send(sp_out)
        self.assertAllEqual(sp_out.indices, indices)
        self.assertAllEqual(sp_out.values, values)
        self.assertAllEqual(sp_out.dense_shape, shape)

  def testFeedSparsePlaceholder(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        shape = np.array([7, 9, 2]).astype(np.int64)
        sp = array_ops.sparse_placeholder(dtype=np.float32, name='placeholder1')
        sp_indices = array_ops.identity(sp.indices)
        sp_values = array_ops.identity(sp.values)
        sp_shape = array_ops.identity(sp.dense_shape)
        sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: (indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: sparse_tensor.SparseTensorValue(indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with SparseTensorValue, fetch SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp2_out = s.run(sp2, {
            sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
          timer.gen.send(sp2_out)
        self.assertAllEqual(sp2_out.indices, indices)
        self.assertAllEqual(sp2_out.values, values)
        self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderPartialShape(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        shape = np.array([7, 9, 2]).astype(np.int64)
        sp = array_ops.sparse_placeholder(
            shape=[None, 9, 2], dtype=np.float32, name='placeholder1')
        sp_indices = array_ops.identity(sp.indices)
        sp_values = array_ops.identity(sp.values)
        sp_shape = array_ops.identity(sp.dense_shape)
        sp2 = sparse_tensor.SparseTensor(sp_indices, sp_values, sp_shape)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: (indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: sparse_tensor.SparseTensorValue(indices, values, shape)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)
        # Feed with SparseTensorValue, fetch SparseTensorValue
        timer = tensorflow_op_timer()
        with timer:
          sp2_out = s.run(sp2, {
            sp: sparse_tensor.SparseTensorValue(indices, values, shape)
          })
          timer.gen.send(sp2_out)
        self.assertAllEqual(sp2_out.indices, indices)
        self.assertAllEqual(sp2_out.values, values)
        self.assertAllEqual(sp2_out.dense_shape, shape)

  def testFeedSparsePlaceholderConstantShape(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        shape = np.array([7, 9, 2]).astype(np.int64)
        sp = array_ops.sparse_placeholder(
            dtype=np.float32, shape=shape, name='placeholder1')
        self.assertAllEqual(sp.dense_shape.eval(session=s), shape)
        self.assertAllEqual(tensor_util.constant_value(sp.shape), shape)
        sp_indices = array_ops.identity(sp.indices)
        sp_values = array_ops.identity(sp.values)
        sp_shape = array_ops.identity(sp.dense_shape)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          indices_out, values_out, shape_out = s.run(
            [sp_indices, sp_values, sp_shape], {
                sp: (indices, values)
            })
          timer.gen.send(indices_out)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(shape_out, shape)

  def testFetchIndexedSlices(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        dense_shape = np.array([7, 9, 2]).astype(np.int64)
        ind = indexed_slices.IndexedSlices(
            constant_op.constant(values), constant_op.constant(indices),
            constant_op.constant(dense_shape))
        # Single fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          ind_out = s.run(ind)
          timer.gen.send(ind_out)
        values_out, indices_out, dense_shape_out = ind_out
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # Single fetch, use as IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          ind_out = s.run(ind)
          timer.gen.send(ind_out)
        self.assertAllEqual(ind_out.values, values)
        self.assertAllEqual(ind_out.indices, indices)
        self.assertAllEqual(ind_out.dense_shape, dense_shape)
        # Tuple fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out, dense_shape_out = s.run(ind)
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # List fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          (values_out, indices_out, dense_shape_out), = s.run([ind])
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # List fetch, use as IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          ind_out, = s.run([ind])
          timer.gen.send(ind_out)
        self.assertAllEqual(ind_out.values, values)
        self.assertAllEqual(ind_out.indices, indices)
        self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlices(self):
    with session.Session() as s:
      with device_context():
        values = np.array([1.0, 2.0]).astype(np.float32)
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        dense_shape = np.array([7, 9, 2]).astype(np.int64)
        ind = indexed_slices.IndexedSlices(
            array_ops.placeholder(dtype=np.float32, shape=(2,)),
            array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
            array_ops.placeholder(dtype=np.int64, shape=(3,)),
        )
        ind_values = array_ops.identity(ind.values)
        ind_indices = array_ops.identity(ind.indices)
        ind_dense_shape = array_ops.identity(ind.dense_shape)
        ind2 = indexed_slices.IndexedSlices(ind_values, ind_indices,
                                            ind_dense_shape)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out, dense_shape_out = s.run(
            [ind_values, ind_indices, ind_dense_shape], {
                ind: (values, indices, dense_shape)
            })
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # Feed with IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out, dense_shape_out = s.run([
            ind_values, ind_indices, ind_dense_shape
          ], {ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)})
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          ind2_out = s.run(ind2, {
            ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
          })
          timer.gen.send(ind2_out)
        self.assertAllEqual(ind2_out.values, values)
        self.assertAllEqual(ind2_out.indices, indices)
        self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testFetchIndexedSlicesWithoutDenseShape(self):
    with session.Session() as s:
      with device_context():
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        values = np.array([1.0, 2.0]).astype(np.float32)
        dense_shape = None
        ind = indexed_slices.IndexedSlices(
            constant_op.constant(values), constant_op.constant(indices), None)
        # Single fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          ind_out = s.run(ind)
          timer.gen.send(ind_out)
        values_out, indices_out, dense_shape_out = ind_out
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # Single fetch, use as IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          ind_out = s.run(ind)
          timer.gen.send(ind_out)
        self.assertAllEqual(ind_out.values, values)
        self.assertAllEqual(ind_out.indices, indices)
        self.assertAllEqual(ind_out.dense_shape, dense_shape)
        # Tuple fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out, dense_shape_out = s.run(ind)
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # List fetch, use as tuple
        timer = tensorflow_op_timer()
        with timer:
          (values_out, indices_out, dense_shape_out), = s.run([ind])
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        self.assertAllEqual(dense_shape_out, dense_shape)
        # List fetch, use as IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          ind_out, = s.run([ind])
          timer.gen.send(ind_out)
        self.assertAllEqual(ind_out.values, values)
        self.assertAllEqual(ind_out.indices, indices)
        self.assertAllEqual(ind_out.dense_shape, dense_shape)

  def testFeedIndexedSlicesWithoutDenseShape(self):
    with session.Session() as s:
      with device_context():
        values = np.array([1.0, 2.0]).astype(np.float32)
        indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
        dense_shape = None
        ind = indexed_slices.IndexedSlices(
            array_ops.placeholder(dtype=np.float32, shape=(2,)),
            array_ops.placeholder(dtype=np.int64, shape=(2, 3)), None)
        ind_values = array_ops.identity(ind.values)
        ind_indices = array_ops.identity(ind.indices)
        ind2 = indexed_slices.IndexedSlices(ind_values, ind_indices)
        # Feed with tuple
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out = s.run([ind_values, ind_indices], {
            ind: (values, indices)
          })
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        # Feed with IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
          values_out, indices_out = s.run([ind_values, ind_indices], {
            ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
          })
          timer.gen.send(values_out)
        self.assertAllEqual(values_out, values)
        self.assertAllEqual(indices_out, indices)
        # Feed with IndexedSlicesValue, fetch IndexedSlicesValue
        timer = tensorflow_op_timer()
        with timer:
         ind2_out = s.run(ind2, {
            ind: indexed_slices.IndexedSlicesValue(values, indices, dense_shape)
         })
         timer.gen.send(ind2_out)
        self.assertAllEqual(ind2_out.values, values)
        self.assertAllEqual(ind2_out.indices, indices)
        self.assertAllEqual(ind2_out.dense_shape, dense_shape)

  def testExtendWithStatelessOperations(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[2, 3])
        c = math_ops.matmul(a, b)
        timer = tensorflow_op_timer()
        with timer:
          c_val = s.run(c)
          timer.gen.send(c_val)
        self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)
        d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
        e = math_ops.matmul(c, d)
        # Extend will happen here.
        timer = tensorflow_op_timer()
        with timer:
          e_val = s.run(e)
          timer.gen.send(e_val)
        self.assertAllEqual([[24.0]], e_val)

  def testExtendWithStatefulOperations(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[2, 3])
        c = math_ops.matmul(a, b)
        v = variables.Variable(c, name='testExtendWithStatefulOperations_v')
        timer = tensorflow_op_timer()
        with timer:
          v.initializer.run()
          timer.gen.send(None)
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        d = constant_op.constant(3.0, shape=[2, 3])
        e = math_ops.matmul(a, d)
        assign_e_to_v = state_ops.assign(v, e)
        # Extend will happen here.
        e_val = e.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        timer = tensorflow_op_timer()
        with timer:
          s.run(assign_e_to_v)
          timer.gen.send(s)
        v_val = v.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)

  def testExtendWithGroupBy(self):
    with session.Session() as s:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        p = variables.Variable(a, name='testExtendWithGroupBy_p')
        a_val = a.eval()  # Force an Extend after this op.
        self.assertAllEqual([[1.0, 1.0]], a_val)

        b = constant_op.constant(2.0, shape=[1, 2])
        q = variables.Variable(b, name='testExtendWithGroupBy_q')
        # Extend will happen here.
        init = control_flow_ops.group(p.initializer, q.initializer)
        timer = tensorflow_op_timer()
        with timer:
          s.run(init)
          timer.gen.send(s)
        timer = tensorflow_op_timer()
        with timer:
          p_val, q_val = s.run([p, q])
          timer.gen.send(p_val)

        self.assertAllEqual([[1.0, 1.0]], p_val)
        self.assertAllEqual([[2.0, 2.0]], q_val)


  def testDefaultGraph(self):
    with session.Session() as s:
      with device_context():
        self.assertEqual(ops.get_default_graph(), s.graph)
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[2, 3])
        self.assertEqual(ops.get_default_graph(), a.graph)
        self.assertEqual(ops.get_default_graph(), b.graph)
        c = math_ops.matmul(a, b)
        v = variables.Variable(c, name='testDefaultGraph_v')
        timer = tensorflow_op_timer()
        with timer:
          v.initializer.run()
          timer.gen.send(None)
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        d = constant_op.constant(3.0, shape=[2, 3])
        e = math_ops.matmul(a, d)
        assign_e_to_v = state_ops.assign(v, e)
        e_val = e.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        timer = tensorflow_op_timer()
        with timer:
          s.run(assign_e_to_v)
          timer.gen.send(None)
        v_val = v.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
        self.assertEqual(ops.get_default_graph(), s.graph)

  def _testDefaultGraphInThread(self, constructed_event, continue_event, i):
    with session.Session() as s:
      with device_context():
        self.assertEqual(ops.get_default_graph(), s.graph)
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[2, 3])
        c = math_ops.matmul(a, b)
        v = variables.Variable(c, name='var_%d' % i)

        # Block here until all threads have constructed their graph.
        constructed_event.set()
        continue_event.wait()

        assign_c_to_v = state_ops.assign(v, c)
        timer = tensorflow_op_timer()
        with timer:
          v.initializer.run()
          timer.gen.send(None)
        assign_c_to_v.eval()
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        d = constant_op.constant(3.0, shape=[2, 3])
        e = math_ops.matmul(a, d)
        assign_e_to_v = state_ops.assign(v, e)
        e_val = e.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
        v_val = v.eval()
        self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
        timer = tensorflow_op_timer()
        with timer:
          s.run(assign_e_to_v)
          timer.gen.send(None)
        v_val = v.eval()
        self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
        self.assertEqual(ops.get_default_graph(), s.graph)




  # @staticmethod
  # def _build_graph():
  #   time.sleep(random.random() * 0.1)
  #   # Do some graph construction. Try to exercise non-trivial paths.
  #   graph = ops.get_default_graph()
  #   gdef = None
  #   for _ in range(10):
  #     x = array_ops.placeholder(dtype=dtypes.float32)
  #     with ops.colocate_with(x):
  #       y = array_ops.placeholder(dtype=dtypes.float32)
  #     with ops.device('/cpu:0'):
  #       z = control_flow_ops.while_loop(
  #           lambda x, y: x < 10, lambda x, y: (x + 1, x * y), [x, y])
  #     with graph._attr_scope({'_a': attr_value_pb2.AttrValue(b=False)}):
  #       gradients_impl.gradients(z, [x, y])
  #       if gdef is None:
  #         gdef = graph.as_graph_def()
  #       else:
  #         importer.import_graph_def(gdef, name='import')

  # @test_util.run_v1_only('b/120545219')
  # def testParallelRunAndSingleBuild(self):
  #   with session.Session() as sess:
  #     c = constant_op.constant(5.0)
  #     stop = threading.Event()

  #     def run_loop():
  #       while not stop.is_set():
  #         time.sleep(random.random() * 0.1)
  #         self.assertEqual(sess.run(c), 5.0)

  #     threads = [self.checkedThread(target=run_loop) for _ in range(10)]
  #     for t in threads:
  #       t.start()

  #     SessionTest._build_graph()

  #     stop.set()
  #     for t in threads:
  #       t.join()

  # @test_util.run_v1_only('b/120545219')
  # def testParallelRunAndParallelBuild(self):
  #   with session.Session() as sess:
  #     c = constant_op.constant(5.0)
  #     stop = threading.Event()

  #     def run_loop():
  #       while not stop.is_set():
  #         time.sleep(random.random() * 0.1)
  #         self.assertEqual(sess.run(c), 5.0)

  #     run_threads = [self.checkedThread(target=run_loop) for _ in range(10)]
  #     for t in run_threads:
  #       t.start()

  #     build_threads = [self.checkedThread(target=SessionTest._build_graph)
  #                      for _ in range(10)]
  #     for t in build_threads:
  #       t.start()
  #     for t in build_threads:
  #       t.join()

  #     # Let the run_threads run until the build threads are finished.
  #     stop.set()
  #     for t in run_threads:
  #       t.join()

  def testRunFeedDict(self):
    with session.Session() as s:
      with device_context():
        x = array_ops.zeros([2])
        timer = tensorflow_op_timer()
        with timer:
          y = s.run(2 * x, feed_dict={x: np.ones(2).astype(np.float32)})
          timer.gen.send(y)
        self.assertAllEqual(y, 2 * np.ones(2))
        timer = tensorflow_op_timer()
        with timer:
          y = s.run(2 * x, feed_dict={x.name: np.ones(2).astype(np.float32)})
          timer.gen.send(y)
        self.assertAllEqual(y, 2 * np.ones(2))
        timer = tensorflow_op_timer()
        with timer:
          y = s.run(2 * x, feed_dict={x: [1, 1]})
          timer.gen.send(y)
        assert (y == 2 * np.ones(2)).all()

        # Test nested tuple keys
        z = (((array_ops.zeros([2]),),), array_ops.zeros([2]),
            (array_ops.zeros([2]),))
        result = [z[0][0][0] * 2, z[1] * 2, z[2][0] * 2]
        values = (((np.array([1, 1]),),), np.array([2, 2]), (np.array([3, 3]),))
        timer = tensorflow_op_timer()
        with timer:
          result_value = s.run(result, feed_dict={z: values})
          timer.gen.send(result_value)
        self.assertAllEqual(result_value[0], 2 * np.ones(2))
        self.assertAllEqual(result_value[1], 2 * np.array([2, 2]))
        self.assertAllEqual(result_value[2], 2 * np.array([3, 3]))


  def testUseAfterClose(self):
    with session.Session() as sess:
      with device_context():
        c = constant_op.constant(5.0)
        self.assertAllEqual(sess.run(c), 5.0)
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda e: 'Attempted to use a closed Session.' in str(e)):
      sess.run(c)

  def testUseAfterCloseConcurrent(self):
    with session.Session() as sess:
      with device_context():
        c = constant_op.constant(5.0)
        self.assertAllEqual(sess.run(c), 5.0)

        def update_thread():
          with self.assertRaisesWithPredicateMatch(
              RuntimeError,
              lambda e: 'Attempted to use a closed Session.' in str(e)):
            while True:
              # with tensorflow_op_timer():
              sess.run(c)

        t = threading.Thread(target=update_thread)
        t.start()
        time.sleep(0.1)
        sess.close()
        t.join()

  def testUseEmptyGraph(self):
    with session.Session() as sess:
      with device_context():
        with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
          sess.run([])
        with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
          sess.run(())
        with self.assertRaisesRegex(RuntimeError, 'The Session graph is empty.'):
          sess.run({})

  # @test_util.run_v1_only('b/120545219')
  # def testNotEntered(self):
  #   # pylint: disable=protected-access
  #   self.assertIsNone(ops._default_session_stack.get_default())
  #   # pylint: enable=protected-access
  #   with ops.device('/cpu:0'):
  #     sess = session.Session()
  #     c_1 = constant_op.constant(5.0)
  #     with sess.graph.as_default():
  #       c_2 = constant_op.constant(5.0)
  #     self.assertEqual(c_1.graph, c_2.graph)
  #     self.assertEqual(sess.run(c_2), 5.0)
  #     with self.assertRaisesWithPredicateMatch(
  #         ValueError, lambda e: 'No default session is registered.' in str(e)):
  #       c_2.eval()

  @test_util.run_v1_only('b/120545219')
  def testMultipleInteractiveSessionsWarning(self):
    # Reinitialize the global state to ensure that the expected warnings will
    # be emitted.
    session.InteractiveSession._active_session_count = 0  # pylint: disable=protected-access

    sess = session.InteractiveSession()
    timer = tensorflow_op_timer()
    with timer:
      sess.run(constant_op.constant(4.0))  # Run so that the session is "opened".
      timer.gen.send(sess)
    sess.close()
    # Opening and closing interactive sessions serially should not warn.
    with warnings.catch_warnings(record=True) as w:
      sess = session.InteractiveSession()
      sess.close()
    self.assertEqual(0, len(w))

    with warnings.catch_warnings(record=True) as w:
      sess = session.InteractiveSession()
    self.assertEqual(0, len(w))
    with warnings.catch_warnings(record=True) as w:
      sess2 = session.InteractiveSession()
    self.assertEqual(1, len(w))
    self.assertIn('An interactive session is already active. This can cause '
                  'out-of-memory errors in some cases. You must explicitly '
                  'call `InteractiveSession.close()` to release resources '
                  'held by the other session(s).', str(w[0].message))
    sess2.close()
    sess.close()


  # @test_util.run_v1_only('b/120545219')
  # def testDefaultSessionPlacePrunedGraph(self):
  #   sess = session.Session()

  #   # Build a graph that has a bad op in it (no kernel).
  #   #
  #   # This test currently does not link in any GPU kernels,
  #   # which is why placing this is invalid.  If at some point
  #   # GPU kernels are added to this test, some other different
  #   # op / device combo should be chosen.
  #   with ops.device('/device:GPU:0'):
  #     _ = constant_op.constant(1.0, shape=[1, 2])

  #   b = constant_op.constant(1.0, shape=[1, 2])

  #   with self.assertRaises(errors.InvalidArgumentError):
  #     # Even though we don't run the bad op, we place the entire
  #     # graph, which should fail with a non-interactive session.
  #     sess.run(b)

  #   sess.close()

  # def testSharedGraph(self):
  #   with ops.Graph().as_default() as g, ops.device('/cpu:0'):
  #     with device_context():
  #       a = constant_op.constant(1.0, shape=[1, 2])
  #       b = constant_op.constant(2.0, shape=[2, 3])
  #       c = math_ops.matmul(a, b)

  #   with session.Session(graph=g) as sess1:
  #     with device_context():
  #       with session.Session(graph=g) as sess2:
  #         with device_context():
  #           with tensorflow_op_timer():
  #             test = sess1.run(c), sess2.run(c)
  #           self.assertAllEqual(sess1.run(c), sess2.run(c))

  def testDuplicatedInputs(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(1.0, shape=[1, 2])
        b = constant_op.constant(2.0, shape=[1, 3])
        timer = tensorflow_op_timer()
        with timer:
          a_val, b_val, a2_val = sess.run([a, b, a])
          timer.gen.send(a_val)
        self.assertAllEqual(a_val, [[1.0, 1.0]])
        self.assertAllEqual(b_val, [[2.0, 2.0, 2.0]])
        self.assertAllEqual(a2_val, [[1.0, 1.0]])

  def testFeedAndFetch(self):
    with session.Session() as sess:
      # with device_context():
        for dtype in [
            dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
            dtypes.uint8, dtypes.int16, dtypes.int8, dtypes.int64, dtypes.bool,
            dtypes.complex64, dtypes.complex128
        ]:
          for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
            np_dtype = dtype.as_numpy_dtype

            feed_t = array_ops.placeholder(dtype=dtype, shape=shape)
            out_t = array_ops.identity(feed_t)

            np_array = np.random.randint(-10, 10, shape)

            if dtype == dtypes.bool:
              np_array = np_array > 0
            elif dtype == dtypes.complex64:
              np_array = np.sqrt(np_array.astype(np_dtype))
            elif dtype == dtypes.complex64:
              np_array = np.sqrt(np_array.astype(np_dtype))
            else:
              np_array = np_array.astype(np_dtype)
            timer = tensorflow_op_timer()
            with timer:
              test = sess.run(out_t, feed_dict={
                                    feed_t: np_array
                                })
              timer.gen.send(test)
            timer = tensorflow_op_timer()
            with timer:
              sess.run(feed_t, feed_dict={
                                    feed_t: np_array
                                })
              timer.gen.send(sess)
            timer = tensorflow_op_timer()
            with timer:
              sess.run(
                [out_t, feed_t], feed_dict={
                    feed_t: np_array
                })
              timer.gen.send(sess)
            self.assertAllEqual(np_array,
                                sess.run(out_t, feed_dict={
                                    feed_t: np_array
                                }))
            # Check that we can also get the feed back.
            self.assertAllEqual(np_array,
                                sess.run(feed_t, feed_dict={
                                    feed_t: np_array
                                }))
            # Also check that we can get both back.
            out_v, feed_v = sess.run(
                [out_t, feed_t], feed_dict={
                    feed_t: np_array
                })
            self.assertAllEqual(np_array, out_v)
            self.assertAllEqual(np_array, feed_v)

            feed_fetch_runner = sess.make_callable([out_t, feed_t], [feed_t])
            out_v, feed_v = feed_fetch_runner(np_array)
            self.assertAllEqual(np_array, out_v)
            self.assertAllEqual(np_array, feed_v)

  def testMakeCallableOnOperationWithRunOptions(self):
    with session.Session() as sess:
      with device_context():
        a = variables.Variable(42.0)
        b = state_ops.assign_add(a, 1.0)
        timer = tensorflow_op_timer()
        with timer:
          sess.run(a.initializer)
          timer.gen.send(sess)
        tensor_runner = sess.make_callable(b.op, accept_options=True)
        run_options = config_pb2.RunOptions(
            trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        self.assertEqual(0, len(run_metadata.step_stats.dev_stats))
        tensor_runner(options=run_options, run_metadata=run_metadata)
        timer = tensorflow_op_timer()
        with timer:
          test = sess.run(a)
          timer.gen.send(test)
        self.assertEqual(43.0, sess.run(a))
        self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)



  def testFeedError(self):
    with session.Session() as sess:
      with device_context():
        feed_t = array_ops.placeholder(dtype=dtypes.float32)
        out_t = array_ops.identity(feed_t)
        feed_val = constant_op.constant(5.0)
        with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
          sess.run(out_t, feed_dict={feed_t: feed_val})
        with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
          out_t.eval(feed_dict={feed_t: feed_val})
        with self.assertRaisesRegex(TypeError, 'cannot be a tf.Tensor object'):
          out_t.op.run(feed_dict={feed_t: feed_val})

  def testFeedPrecisionLossError(self):
    with session.Session() as sess:
      with device_context():
        largest_int64 = np.iinfo(np.int64).max

        feed_int_implicit_int32 = constant_op.constant(1)
        feed_int_explicit_int32 = constant_op.constant(1, dtype=dtypes.int32)

        out_t = constant_op.constant(1.0)

        with self.assertRaisesRegex(TypeError,
                                    'is not compatible with Tensor type'):
          sess.run(out_t, feed_dict={feed_int_implicit_int32: largest_int64})
        with self.assertRaisesRegex(TypeError,
                                    'is not compatible with Tensor type'):
          sess.run(out_t, feed_dict={feed_int_explicit_int32: largest_int64})


  def testStringFeed(self):
    with session.Session() as sess:
      with device_context():
        for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
          size = 1
          for s in shape:
            size *= s
          c_list = np.array([compat.as_bytes(str(i)) for i in range(size)],
                            dtype=np.object_).reshape(shape)
          feed_t = array_ops.placeholder(dtype=dtypes.string, shape=shape)
          c = array_ops.identity(feed_t)
          timer = tensorflow_op_timer()
          with timer:
            test = sess.run(c, feed_dict={feed_t: c_list})
            timer.gen.send(test)
          self.assertAllEqual(sess.run(c, feed_dict={feed_t: c_list}), c_list)
          timer = tensorflow_op_timer()
          with timer:
            test = sess.run(feed_t, feed_dict={
                  feed_t: c_list
              })
            timer.gen.send(test)
          self.assertAllEqual(
              sess.run(feed_t, feed_dict={
                  feed_t: c_list
              }), c_list)
          timer = tensorflow_op_timer()
          with timer:
            c_v, feed_v = sess.run([c, feed_t], feed_dict={feed_t: c_list})
            timer.gen.send(c_v)
          self.assertAllEqual(c_v, c_list)
          self.assertAllEqual(feed_v, c_list)


  def testFetchByNameDifferentStringTypes(self):
    with session.Session() as sess:
      with device_context():
        c = constant_op.constant(42.0, name='c')
        d = constant_op.constant(43.0, name=u'd')
        e = constant_op.constant(44.0, name=b'e')
        f = constant_op.constant(45.0, name=r'f')

        self.assertIsInstance(c.name, six.text_type)
        self.assertIsInstance(d.name, six.text_type)
        self.assertIsInstance(e.name, six.text_type)
        self.assertIsInstance(f.name, six.text_type)
        timer = tensorflow_op_timer()
        with timer:
          test = sess.run('c:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test = sess.run(u'c:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(b'c:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(r'c:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run('d:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(u'd:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(b'd:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(r'd:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run('e:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test = sess.run(u'e:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(b'e:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(r'e:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run('f:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(u'f:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test =sess.run(b'f:0')
          timer.gen.send(test)
        timer = tensorflow_op_timer()
        with timer:
          test = sess.run(r'f:0')
          timer.gen.send(test)
        
        self.assertEqual(42.0, sess.run('c:0'))
        self.assertEqual(42.0, sess.run(u'c:0'))
        self.assertEqual(42.0, sess.run(b'c:0'))
        self.assertEqual(42.0, sess.run(r'c:0'))

        self.assertEqual(43.0, sess.run('d:0'))
        self.assertEqual(43.0, sess.run(u'd:0'))
        self.assertEqual(43.0, sess.run(b'd:0'))
        self.assertEqual(43.0, sess.run(r'd:0'))

        self.assertEqual(44.0, sess.run('e:0'))
        self.assertEqual(44.0, sess.run(u'e:0'))
        self.assertEqual(44.0, sess.run(b'e:0'))
        self.assertEqual(44.0, sess.run(r'e:0'))

        self.assertEqual(45.0, sess.run('f:0'))
        self.assertEqual(45.0, sess.run(u'f:0'))
        self.assertEqual(45.0, sess.run(b'f:0'))
        self.assertEqual(45.0, sess.run(r'f:0'))

  def testIncorrectGraph(self):
    with ops.Graph().as_default() as g_1:
      with device_context():
        c_1 = constant_op.constant(1.0, name='c')

    with ops.Graph().as_default() as g_2:
      with device_context():
        c_2 = constant_op.constant(2.0, name='c')

    self.assertEqual('c', c_1.op.name)
    self.assertEqual('c', c_2.op.name)

    with session.Session(graph=g_1) as sess_1:
      with device_context():
        self.assertEqual(1.0, sess_1.run(c_1))
        with self.assertRaises(ValueError):
          sess_1.run(c_2)
        with self.assertRaises(ValueError):
          sess_1.run(c_2.op)

    with session.Session(graph=g_2) as sess_2:
      with device_context():
        with self.assertRaises(ValueError):
          sess_2.run(c_1)
        with self.assertRaises(ValueError):
          sess_2.run(c_1.op)
        self.assertEqual(2.0, sess_2.run(c_2))

  def testFeedDictKeyException(self):
    with session.Session() as sess:
      with device_context():
        a = constant_op.constant(1.0, dtypes.float32, name='a')
        with self.assertRaisesRegex(TypeError, 'Cannot interpret feed_dict'):
          sess.run(a, feed_dict={'a': [2.0]})

  def testFeedShapeCompatibility(self):
    with session.Session() as sess:
      with device_context():
        some_tensor = constant_op.constant([2.0, 2.0, 2.0, 2.0])
        new_shape = constant_op.constant([2, 2])
        reshaped_tensor = array_ops.reshape(some_tensor, new_shape)

        with self.assertRaisesRegex(ValueError, 'Cannot feed value of shape'):
          sess.run(reshaped_tensor, feed_dict={some_tensor: [1.0, 2.0, 3.0]})

        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            'Input to reshape is a tensor with 4 values, '
            'but the requested shape has 21'):
          sess.run(reshaped_tensor, feed_dict={new_shape: [3, 7]})


  def testBuildCostModel(self):
    run_options = config_pb2.RunOptions()
    config_pb = config_pb2.ConfigProto(
        allow_soft_placement=True,
        graph_options=config_pb2.GraphOptions(build_cost_model=100))
    with session.Session(config=config_pb) as sess:
      with device_context():
        a = array_ops.placeholder(dtypes.float32, shape=[])
        b = math_ops.add(a, a)
        c = array_ops.identity(b)
        d = math_ops.multiply(c, c)
      for step in range(120):
        run_metadata = config_pb2.RunMetadata()
        timer = tensorflow_op_timer()
        with timer:
          sess.run(
            d,
            feed_dict={a: 1.0},
            options=run_options,
            run_metadata=run_metadata)
          timer.gen.send(sess)
        if step == 99:
          self.assertTrue(run_metadata.HasField('cost_graph'))
        else:
          self.assertFalse(run_metadata.HasField('cost_graph'))

  def runTestOutputPartitionGraphs(self, sess):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    a = constant_op.constant(1)
    run_metadata = config_pb2.RunMetadata()
    timer = tensorflow_op_timer()
    with timer:
      sess.run(a, options=run_options, run_metadata=run_metadata)
      timer.gen.send(sess)
    self.assertGreater(len(run_metadata.partition_graphs), 0)
    timer = tensorflow_op_timer()
    with timer:
      sess.run(a, run_metadata=run_metadata)
      timer.gen.send(sess)
    self.assertEqual(len(run_metadata.partition_graphs), 0)

  @test_util.run_v1_only('b/120545219')
  def testOutputPartitionGraphsDirect(self):
    self.runTestOutputPartitionGraphs(session.Session())

  @test_util.run_v1_only('b/120545219')
  def testOutputPartitionGraphsDistributed(self):
    server = server_lib.Server.create_local_server()
    self.runTestOutputPartitionGraphs(session.Session(server.target))


  @test_util.run_v1_only('b/120545219')
  def testTimeoutWithShortOperations(self):
    num_epochs = 5
    q = data_flow_ops.FIFOQueue(capacity=50, dtypes=[dtypes.int32], shapes=[()])
    enqueue_op = q.enqueue_many(constant_op.constant([1, 2]))

    # Use a 10-second timeout, which should be longer than any
    # non-blocking enqueue_many op.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=10000)
    with session.Session(config=config_pb) as sess:
      for _ in range(num_epochs):
        timer = tensorflow_op_timer()
        with timer:
          sess.run(enqueue_op)
          timer.gen.send(sess)
      timer = tensorflow_op_timer()
      with timer:
        test = sess.run(q.size())
        timer.gen.send(test)
      self.assertEqual(sess.run(q.size()), num_epochs * 2)

  @test_util.run_v1_only('b/120545219')
  def testRegisterFetchAndFeedConversionFunctions(self):

    class SquaredTensor(object):

      def __init__(self, tensor):
        self.sq = math_ops.square(tensor)

    fetch_fn = lambda squared_tensor: ([squared_tensor.sq], lambda val: val[0])
    feed_fn1 = lambda feed, feed_val: [(feed.sq, feed_val)]
    feed_fn2 = lambda feed: [feed.sq]

    session.register_session_run_conversion_functions(SquaredTensor, fetch_fn,
                                                      feed_fn1, feed_fn2)
    with self.assertRaises(ValueError):
      session.register_session_run_conversion_functions(SquaredTensor, fetch_fn,
                                                        feed_fn1, feed_fn2)
    with self.cached_session() as sess:
      np1 = np.array([1.0, 1.5, 2.0, 2.5])
      np2 = np.array([3.0, 3.5, 4.0, 4.5])
      squared_tensor = SquaredTensor(np2)
      timer = tensorflow_op_timer()
      with timer:
        squared_eval = sess.run(squared_tensor)
        timer.gen.send(squared_eval)
      self.assertAllClose(np2 * np2, squared_eval)
      timer = tensorflow_op_timer()
      with timer:
        squared_eval = sess.run(
          squared_tensor, feed_dict={
              squared_tensor: np1 * np1
          })
        timer.gen.send(squared_eval)
      self.assertAllClose(np1 * np1, squared_eval)
      partial_run = sess.partial_run_setup([squared_tensor], [])
      squared_eval = sess.partial_run(partial_run, squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)

  
  @test_util.run_v1_only('b/120545219')
  def testLocalMasterSessionTimeout(self):
    # Test that the timeout passed in a config to the session works correctly.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server()
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target, config=config_pb) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  @test_util.run_v1_only('b/120545219')
  def testDefaultServerTimeout(self):
    # Test that the default server config timeout gets used when no Session
    # config is provided.
    config_pb = config_pb2.ConfigProto(operation_timeout_in_ms=1000)
    server = server_lib.Server.create_local_server(config=config_pb)
    q = data_flow_ops.FIFOQueue(1, dtypes.float32)
    dequeued_t = q.dequeue()

    with session.Session(server.target) as sess:
      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaises(errors.DeadlineExceededError):
        sess.run(dequeued_t)

  def runTestBuildGraphError(self, sess):
    # Ensure that errors from building the graph get propagated.
    data = array_ops.placeholder(dtypes.float32, shape=[])
    # pylint: disable=protected-access
    enter_1 = gen_control_flow_ops.enter(data, 'foo_1', False)
    enter_2 = gen_control_flow_ops.enter(data, 'foo_2', False)
    # pylint: enable=protected-access
    res = math_ops.add(enter_1, enter_2)
    with self.assertRaisesOpError('has inputs from different frames'):
      sess.run(res, feed_dict={data: 1.0})

  @test_util.run_v1_only('b/120545219')
  def testBuildGraphErrorDirect(self):
    self.runTestBuildGraphError(session.Session())

  @test_util.run_v1_only('b/120545219')
  def testBuildGraphErrorDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestBuildGraphError(session.Session(server.target))

  # def runTestAddFunctionToSession(self, target=''):
  #   """Add a function to a session after the graph has already been run."""


  #   x = constant_op.constant(1.0)
  #   with session.Session(target=target) as sess:
  #     with ops.device('/device:GPU:0'):
  #       with tensorflow_op_timer():
  #         sess.run(x)
  #       f = foo(x)
  #       with tensorflow_op_timer():
  #         result = sess.run(f)
  #       self.assertEqual(result, 2.0)


  @test_util.run_v1_only('b/120545219')
  def testAutoConvertAndCheckData(self):
    with self.cached_session() as sess:
      a = array_ops.placeholder(dtype=dtypes.string)
      with self.assertRaisesRegex(
          TypeError, r'Type of feed value 1 with type <(\w+) \'int\'> is not'):
        sess.run(a, feed_dict={a: 1})



if __name__ == '__main__':
  googletest.main()
