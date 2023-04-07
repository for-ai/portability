# Owner(s): ["oncall: package/deploy"]

from torch.package._digraph import DiGraph
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDiGraph(PackageTestCase):
    """Test the DiGraph structure we use to represent dependencies in PackageExporter"""

    def test_successors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("bar", list(g.successors("foo")))
        self.assertIn("baz", list(g.successors("foo")))
        self.assertEqual(len(list(g.successors("qux"))), 0)

    def test_predecessors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("foo", list(g.predecessors("bar")))
        self.assertIn("foo", list(g.predecessors("baz")))
        self.assertEqual(len(list(g.predecessors("qux"))), 0)


    def test_node_attrs(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1, other_attr=2)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)
        self.assertEqual(g.nodes["foo"]["other_attr"], 2)

    def test_node_attr_update(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

        g.add_node("foo", my_attr="different")
        self.assertEqual(g.nodes["foo"]["my_attr"], "different")


    def test_iter(self):
        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        nodes = set()
        for n in g:
            nodes.add(n)

        self.assertEqual(nodes, set([1, 2, 3]))

    def test_contains(self):
        g = DiGraph()
        g.add_node("yup")

        self.assertTrue("yup" in g)
        self.assertFalse("nup" in g)

    
if __name__ == "__main__":
    run_tests()
