from cylp.cy import CyClpSimplex
import inspect
from queue import PriorityQueue
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound, \
    PseudoCostBranchDepthFirstSearchNode as PCBDFSNode
from test_simple_mip_solver.example_models import no_branch, small_branch,\
    infeasible, unbounded


class TestBranchAndBound(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node.bound()
        cls.children = node.branch()

    def test_init(self):
        bb = BranchAndBound(small_branch)
        self.assertTrue(bb._global_upper_bound == float('inf'))
        self.assertTrue(bb._node_queue.empty())
        self.assertTrue(inspect.isclass(bb._Node))
        self.assertTrue(bb._root_node)
        self.assertTrue(bb.model)
        self.assertFalse(bb._unbounded)
        self.assertFalse(bb._best_solution)
        self.assertFalse(bb.solution)
        self.assertTrue(bb.status == 'unsolved')
        self.assertFalse(bb.objective_value)
        self.assertFalse(bb._pseudo_costs, 'should exist but be empty')
        self.assertTrue(bb._strong_branch_iters == 5)

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

        # strong branch iters asserts
        self.assertRaisesRegex(AssertionError,
                               'strong branching iterations must be positive integer',
                               BranchAndBound, small_branch, strong_branch_iters=-1)

    def test_solve_optimal(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(small_branch, Node=Node)
            bb.solve()
            self.assertTrue(bb.status == 'optimal')
            self.assertTrue(all(s.is_integer for s in bb.solution))
            self.assertTrue(bb.objective_value == -2)

    def test_solve_infeasible(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(infeasible, Node=Node)
            bb.solve()
            self.assertTrue(bb.status == 'infeasible')
            self.assertFalse(bb.solution)
            self.assertTrue(bb.objective_value == float('inf'))

    def test_solve_unbounded(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(unbounded, Node=Node)
            bb.solve()
            self.assertTrue(bb.status == 'unbounded')

        # check we quit even if node_queue nonempty
        with patch.object(bb, '_evaluate_node') as en:
            bb = BranchAndBound(unbounded)
            bb._unbounded = True
            bb.solve()
            self.assertFalse(en.called)

    def test_evaluate_node_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertTrue(bb._node_queue.empty(), 'inf model should create no nodes')
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb._root_node, 'bound') as bd, \
                patch.object(bb._root_node, 'branch') as bh:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)

    def test_evaluate_node_fractional(self):
        bb = BranchAndBound(small_branch, Node=PCBDFSNode)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._node_queue.qsize() == 2, 'should branch and add two nodes')
        self.assertTrue(bb._pseudo_costs, 'something should be set')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb._root_node, 'bound') as bd, \
                patch.object(bb._root_node, 'branch') as bh:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)  # direct calls
            self.assertTrue(pbr.call_count == 1)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 1)

    def test_evaluate_node_integer(self):
        bb = BranchAndBound(no_branch)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertTrue(all(bb._best_solution == [1, 1, 0]))
        self.assertTrue(bb._global_upper_bound == -2)
        self.assertTrue(bb._node_queue.empty(), 'immediately optimal model should create no nodes')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb._root_node, 'bound') as bd, \
                patch.object(bb._root_node, 'branch') as bh:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)

    def test_evaluate_node_unbounded(self):
        bb = BranchAndBound(unbounded)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertTrue(bb._unbounded)

    def test_evaluate_node_properly_prunes(self):
        bb = BranchAndBound(no_branch)
        bb._global_upper_bound = -2
        called_node = BaseNode(bb.model.lp, bb.model.integerIndices, -4)
        pruned_node = BaseNode(bb.model.lp, bb.model.integerIndices, 0)
        with patch.object(called_node, 'bound') as cnb, \
                patch.object(pruned_node, 'bound') as pnb:
            cnb.return_value = {}
            pnb.return_value = {}
            bb._node_queue.put(called_node)
            bb._node_queue.put(pruned_node)
            bb._evaluate_node(bb._node_queue.get())
            bb._evaluate_node(bb._node_queue.get())
            self.assertTrue(cnb.call_count == 1, 'first node should run')
            self.assertFalse(pnb.call_count, 'second node should get pruned')
            self.assertTrue(bb._node_queue.empty())

    def test_process_rtn_fails_asserts(self):
        bb = BranchAndBound(small_branch)
        self.assertRaisesRegex(AssertionError, 'rtn must be a dictionary',
                               bb._process_rtn, 'fish')

    def test_process_rtn(self):
        bb = BranchAndBound(small_branch)
        bb._process_rtn({'_pseudo_costs': 5})
        self.assertTrue(bb._pseudo_costs == 5)

    def test_process_branch_rtn_fails_asserts(self):
        bb = BranchAndBound(small_branch)
        self.assertRaisesRegex(AssertionError, 'rtn must be a dictionary',
                               bb._process_rtn, 'fish')
        rtn = {'up': 5, 'down': 5}
        self.assertRaisesRegex(AssertionError, 'value must be type',
                               bb._process_branch_rtn, rtn)
        del rtn['up']
        self.assertRaisesRegex(AssertionError, 'must be in the returned',
                               bb._process_branch_rtn, rtn)

    def test_process_branch_rtn(self):
        bb = BranchAndBound(small_branch)
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node.bound()
        rtn = node.branch()
        bb._process_branch_rtn(rtn)

        # check attributes
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(bb._node_queue.empty())

        # check function calls
        bb = BranchAndBound(small_branch)
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node.bound()
        rtn = node.branch()
        with patch.object(bb, '_process_rtn') as pr:
            bb._process_branch_rtn(rtn)
            self.assertTrue(pr.call_count == 1, 'should call rtn')


if __name__ == '__main__':
    unittest.main()
