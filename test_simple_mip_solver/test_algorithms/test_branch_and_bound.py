from queue import PriorityQueue
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound, \
    PseudoCostBranchDepthFirstSearchNode as PCBDFSNode
from simple_mip_solver.algorithms.utils import Utils
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
        self.assertTrue(isinstance(bb, Utils))
        self.assertTrue(bb._global_upper_bound == float('inf'))
        self.assertTrue(bb._node_queue.empty())
        self.assertFalse(bb._unbounded)
        self.assertFalse(bb._best_solution)
        self.assertFalse(bb.solution)
        self.assertTrue(bb.status == 'unsolved')
        self.assertFalse(bb.objective_value)

    def test_init_fails_asserts(self):
        bb = BranchAndBound(small_branch)
        queue = PriorityQueue()

        # node_queue asserts
        for func in reversed(bb._queue_funcs):
            queue.__dict__[func] = 5
            self.assertRaisesRegex(AssertionError, f'node_queue needs a {func} function',
                                   BranchAndBound, small_branch, BaseNode, queue)

    def test_solve_optimal(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(small_branch, Node=Node, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'optimal')
            self.assertTrue(all(s.is_integer for s in bb.solution))
            self.assertTrue(bb.objective_value == -2)

    def test_solve_infeasible(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(infeasible, Node=Node, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'infeasible')
            self.assertFalse(bb.solution)
            self.assertTrue(bb.objective_value == float('inf'))

    def test_solve_unbounded(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(unbounded, Node=Node, pseudo_costs={})
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
                patch.object(bb._root_node, 'branch') as bh, \
                patch.object(bb, '_update_lower_bound') as ulb:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)
            self.assertTrue(ulb.called)

    def test_evaluate_node_fractional(self):
        bb = BranchAndBound(small_branch, Node=PCBDFSNode, pseudo_costs={},
                            strong_branch_iters=5)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._node_queue.qsize() == 2, 'should branch and add two nodes')
        self.assertTrue(bb._kwargs['pseudo_costs'], 'something should be set')
        self.assertTrue(bb._kwargs['strong_branch_iters'], 'something should be set')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb._root_node, 'bound') as bd, \
                patch.object(bb._root_node, 'branch') as bh, \
                patch.object(bb, '_update_lower_bound') as ulb:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)  # direct calls
            self.assertTrue(pbr.call_count == 1)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 1)
            self.assertTrue(ulb.called)

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
                patch.object(bb._root_node, 'branch') as bh, \
                patch.object(bb, '_update_lower_bound') as ulb:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)
            self.assertTrue(ulb.called)

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
                patch.object(pruned_node, 'bound') as pnb, \
                patch.object(bb, '_update_lower_bound') as ulb:
            cnb.return_value = {}
            pnb.return_value = {}
            bb._node_queue.put(called_node)
            bb._node_queue.put(pruned_node)
            bb._evaluate_node(bb._node_queue.get())
            bb._evaluate_node(bb._node_queue.get())
            self.assertTrue(cnb.call_count == 1, 'first node should run')
            self.assertFalse(pnb.call_count, 'second node should get pruned')
            self.assertTrue(bb._node_queue.empty())

    def test_update_lower_bound_fails_asserts(self):
        bb = BranchAndBound(small_branch)
        self.assertRaises(AssertionError, bb._update_lower_bound, bb._root_node)

    def test_update_lower_bound(self):
        bb = BranchAndBound(small_branch)
        bb._root_node.bound()
        bb._update_lower_bound(bb._root_node)
        # first one should update bc nothing else in queue
        self.assertTrue(bb._global_lower_bound == bb._root_node.objective_value)
        bb._process_branch_rtn(bb._root_node.branch())
        glb = bb._global_lower_bound
        cur_node = bb._node_queue.get()
        cur_node.bound()
        bb._update_lower_bound(cur_node)
        # shouldnt change with same lower bound sibling in queue
        self.assertTrue(bb._global_lower_bound < cur_node.objective_value)
        self.assertTrue(bb._global_lower_bound == glb)
        # again doesnt change
        glb = bb._global_lower_bound
        cur_node = bb._node_queue.get()
        cur_node.bound()
        bb._update_lower_bound(cur_node)
        self.assertTrue(bb._global_lower_bound == cur_node.objective_value)
        self.assertTrue(bb._global_lower_bound == glb)

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
