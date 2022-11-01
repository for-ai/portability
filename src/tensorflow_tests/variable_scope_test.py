
import os
import re

import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import compat
from tensorflow.python.util import nest


class _GraphMerger(object):
    """GraphDef merging methods for testing purposes."""

    @staticmethod
    def merge_any(x1, x2, empty_fn):
        """Merges two values using the message's CopyFrom/MergeFrom methods."""
        merged = empty_fn()
        merged.CopyFrom(x1)
        merged.MergeFrom(x2)
        return merged

    @staticmethod
    def merge_nodes(node1, node2):
        """Merges two NodeDef messages."""
        merged = _GraphMerger.merge_any(node1, node2, node_def_pb2.NodeDef)
        merged_inputs = node1.input[:]
        merged_inputs.extend(
            [i for i in node2.input[:] if i not in merged_inputs])
        merged.input[:] = merged_inputs
        return merged

    @staticmethod
    def merge_lists(repeated1, repeated2, empty_fn, key_fn, merge_fn):
        """Merges two lists representing maps."""
        merged = {}
        xs1 = {key_fn(x): x for x in repeated1}
        xs2 = {key_fn(x): x for x in repeated2}
        for name in set().union(xs1.keys(), xs2.keys()):
            x1 = empty_fn() if name not in xs1 else xs1[name]
            x2 = empty_fn() if name not in xs2 else xs2[name]
            merged[name] = merge_fn(x1, x2)
        return sorted(merged.values(), key=key_fn)

    @staticmethod
    def merge_node_lists(repeated_nodes1, repeated_nodes2):
        """Merges two repeated node fields."""
        return _GraphMerger.merge_lists(repeated_nodes1, repeated_nodes2,
                                        node_def_pb2.NodeDef, lambda n: n.name,
                                        _GraphMerger.merge_nodes)

    @staticmethod
    def merge_functions(fn1, fn2):
        """Merges two FunctionDefs."""
        merged = _GraphMerger.merge_any(fn1, fn2, function_pb2.FunctionDef)

        del merged.signature.input_arg[:]
        merged.signature.input_arg.extend(
            _GraphMerger.merge_lists(
                fn1.signature.input_arg[:], fn2.signature.input_arg[:],
                op_def_pb2.OpDef.ArgDef, lambda a: a.name,
                lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

        del merged.signature.output_arg[:]
        merged.signature.output_arg.extend(
            _GraphMerger.merge_lists(
                fn1.signature.output_arg[:], fn2.signature.output_arg[:],
                op_def_pb2.OpDef.ArgDef, lambda a: a.name,
                lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

        del merged.node_def[:]
        merged.node_def.extend(
            _GraphMerger.merge_node_lists(fn1.node_def[:], fn2.node_def[:]))

        return merged

    @staticmethod
    def merge_graphs(graph1, graph2):
        """Merges two GraphDef messages."""
        merged = graph_pb2.GraphDef()
        merged.node.extend(
            _GraphMerger.merge_node_lists(graph1.node[:], graph2.node[:]))

        merged.library.function.extend(
            _GraphMerger.merge_lists(graph1.library.function,
                                     graph2.library.function,
                                     function_pb2.FunctionDef,
                                     lambda f: f.signature.name,
                                     _GraphMerger.merge_functions))

        return merged


class VariablesToConstantsTest(test.TestCase):
    def _assertGraphContains(self, graph, subgraph):
        """Asserts that the given subgraph is contained within the given graph."""

        def normalize_uids(msg):
            """Replace auto-id function names with something consistent."""
            # These functions have non-deterministic names, the non-determinism coming
            # from having an ops.uid() suffix in their names. We're replacing these
            # with new sequential IDs starting from 0 for each prefix, which is
            # is sufficient for tests.
            if isinstance(msg, graph_pb2.GraphDef):
                msg = text_format.MessageToString(msg)
            name_prefixes = ["case_cond_true.*", "case_cond_false.*"]
            name_regex = r"\b(" + "|".join(name_prefixes) + r")_([0-9]+)\b"
            names = {}
            for (name, index) in re.findall(name_regex, msg):
                names.setdefault(name, set()).add(int(index))
            for name, indices in names.items():
                for new_index, old_index in enumerate(sorted(list(indices))):
                    msg = re.sub(r"\b" + name + "_" + str(old_index) + r"\b",
                                 name + "_" + str(new_index), msg)
            return msg

        norm_graph = text_format.Parse(
            normalize_uids(graph), graph_pb2.GraphDef())
        norm_subgraph = text_format.Parse(
            normalize_uids(subgraph), graph_pb2.GraphDef())

        # Graph S is contained in C if and only if merge(C,S) == C.
        # We merge the input graph with an empty graph to normalize repeated fields:
        # assertProtoEquals is sensitive to ordering.
        norm_graph = _GraphMerger.merge_graphs(
            norm_graph, graph_pb2.GraphDef())
        merged_graph = _GraphMerger.merge_graphs(norm_graph, norm_subgraph)
        self.assertProtoEquals(norm_graph, merged_graph)

    def testConvertSingleVariable(self):
        """Tests that a single variable is properly converted to a constant."""

        with ops.Graph().as_default():
            with variable_scope.variable_scope("", use_resource=False):
                _ = variable_scope.get_variable("x", initializer=1.0)
            with session_lib.Session() as sess:
                sess.run(variables.global_variables_initializer())
                variable_graph_def = sess.graph.as_graph_def()
                constant_graph_def = (
                    convert_to_constants
                    .convert_variables_to_constants_from_session_graph(
                        sess, variable_graph_def, ["x/read"]))
                self._assertGraphContains(
                    constant_graph_def, """
              node {
                name: "x" op: "Const"
                attr { key: "dtype" value { type: DT_FLOAT } }
                attr {
                  key: "value"
                  value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
              }
              node {
                name: "x/read" op: "Identity" input: "x"
                attr { key: "T" value { type: DT_FLOAT } }
              }""")


if __name__ == "__main__":
    googletest.main()
