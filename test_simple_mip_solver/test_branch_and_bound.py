from cylp.cy import CyClpSimplex
from queue import PriorityQueue
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound
from test_simple_mip_solver.example_models import no_branch, small_branch, infeasible


class TestBranchAndBound(unittest.TestCase):

    def test_init(self):
        bb = BranchAndBound(small_branch)
        self.assertTrue(bb._global_upper_bound == float('inf'))
        self.assertTrue(bb._root_node)
        self.assertTrue(bb._node_queue.empty())
        self.assertTrue(bb.model)
        self.assertFalse(bb._best_solution)
        self.assertFalse(bb.solution)
        self.assertTrue(bb.status == 'unsolved')
        self.assertFalse(bb.objective_value)
        self.assertFalse(bb._current_node)

    def test_init_fails_asserts(self):
        lp = CyClpSimplex()
        bb = BranchAndBound(small_branch)
        queue = PriorityQueue()

        # model asserts
        self.assertRaisesRegex(AssertionError, 'model must be cuppy MILPInstance',
                               BranchAndBound, lp)

        # Node asserts
        self.assertRaisesRegex(AssertionError, 'Node must be a class',
                               BranchAndBound, small_branch, 'Node')
        for attribute in bb._node_attributes:

            class BadNode(BaseNode):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    delattr(self, attribute)

            self.assertRaisesRegex(AssertionError, f'Node needs a {attribute}',
                                   BranchAndBound, small_branch, BadNode)

        for func in bb._node_funcs:

            class BadNode(BaseNode):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.__dict__[func] = 5

            self.assertRaisesRegex(AssertionError, f'Node needs a {func}',
                                   BranchAndBound, small_branch, BadNode)

        # node_queue asserts
        for func in reversed(bb._queue_funcs):
            queue.__dict__[func] = 5
            self.assertRaisesRegex(AssertionError, f'node_queue needs a {func} function',
                                   BranchAndBound, small_branch, BaseNode, queue)

    def test_solve_optimal(self):
        bb = BranchAndBound(small_branch)
        bb.solve()
        self.assertTrue(bb.status == 'optimal')
        self.assertTrue(all(s.is_integer for s in bb.solution))
        self.assertTrue(bb.objective_value == -2)

    def test_solve_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb.solve()
        self.assertTrue(bb.status == 'infeasible')
        self.assertFalse(bb.solution)
        self.assertTrue(bb.objective_value == float('inf'))

    def test_evaluate_next_node_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb._node_queue.put(bb._root_node)
        bb._evaluate_next_node()

        self.assertTrue(bb._node_queue.empty(), 'inf model should create no nodes')
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')

    def test_evaluate_next_node_fractional(self):
        bb = BranchAndBound(small_branch)
        bb._node_queue.put(bb._root_node)
        bb._evaluate_next_node()

        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._node_queue.qsize() == 2, 'should branch and add two nodes')

    def test_evaluate_next_node_integer(self):
        bb = BranchAndBound(no_branch)
        bb._node_queue.put(bb._root_node)
        bb._evaluate_next_node()

        self.assertTrue(all(bb._best_solution == [1, 1, 0]))
        self.assertTrue(bb._global_upper_bound == -2)
        self.assertTrue(bb._node_queue.empty(), 'immediately optimal model should create no nodes')

    def test_evaluate_proper_nodes_pruned(self):
        bb = BranchAndBound(no_branch)
        bb._global_upper_bound = -2
        called_node = BaseNode(bb.model.lp, bb.model.integerIndices, -4)
        pruned_node = BaseNode(bb.model.lp, bb.model.integerIndices, 0)
        with patch.object(called_node, 'bound') as cnb , \
                patch.object(pruned_node, 'bound') as pnb:
            bb._node_queue.put(called_node)
            bb._node_queue.put(pruned_node)
            bb._evaluate_next_node()
            bb._evaluate_next_node()
            self.assertTrue(cnb.call_count == 1, 'first node should run')
            self.assertFalse(pnb.call_count, 'second node should get pruned')
            self.assertTrue(bb._node_queue.empty())


if __name__ == '__main__':
    unittest.main()
