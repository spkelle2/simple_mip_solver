from queue import PriorityQueue
import unittest

from simple_mip_solver.nodes.search.depth_first import DepthFirstSearchNode
from test_simple_mip_solver.example_models import small_branch

from test_simple_mip_solver.helpers import TestModels


class TestNode(TestModels):

    Node = DepthFirstSearchNode

    def test_init(self):
        node = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices)
        self.assertTrue(node.search_method == 'depth first')

    def test_lt(self):
        node1 = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices)
        node2 = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices, depth=1)

        self.assertTrue(node2 < node1)
        self.assertFalse(node1 < node2)
        self.assertRaises(TypeError, node1.__lt__, 5)

        # make sure if we put them in PQ that they come out in the right order
        q = PriorityQueue()
        q.put(node2)
        q.put(node1)
        self.assertTrue(q.get().depth == 1)
        self.assertTrue(q.get().depth == 0)

    def test_eq(self):
        node1 = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices, depth=1)
        node2 = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices)
        node3 = DepthFirstSearchNode(small_branch.lp, small_branch.integerIndices)

        self.assertTrue(node3 == node2)
        self.assertFalse(node1 == node2)
        self.assertRaises(TypeError, node1.__eq__, 5)

    def test_models(self):
        self.base_test_models()


if __name__ == '__main__':
    unittest.main()
