from coinor.gimpy.tree import BinaryTree
from queue import PriorityQueue
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound, \
    PseudoCostBranchDepthFirstSearchNode as PCBDFSNode
from simple_mip_solver.algorithms.utils import Utils
from test_simple_mip_solver.example_models import no_branch, small_branch,\
    infeasible, unbounded


class TestBranchAndBound(unittest.TestCase):

    def setUp(self) -> None:
        self.bb = BranchAndBound(small_branch)
        self.unbounded_root = BaseNode(small_branch.lp, small_branch.integerIndices)
        self.bound_root = BaseNode(small_branch.lp, small_branch.integerIndices)
        self.bound_root.bound()
        self.root_branch_rtn = self.bound_root.branch()

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
        self.assertTrue(isinstance(bb._tree, BinaryTree))
        self.assertTrue(list(bb._tree.nodes.keys()) == [0])
        self.assertTrue(bb._tree.nodes[0].attr['node'] is bb._root_node)

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

    def test_solve_past_node_limit(self):
        bb = BranchAndBound(unbounded, node_limit=10)
        # check we quit even if node_queue nonempty
        with patch.object(bb, '_evaluate_node') as en:
            bb._evaluated_nodes = 10
            bb.solve()
            self.assertFalse(en.called, 'were past the node limit')

    def test_evaluate_node_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertTrue(bb._node_queue.empty(), 'inf model should create no nodes')
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._global_lower_bound == -float('inf'), 'shouldnt change')
        self.assertTrue(bb._evaluated_nodes == 1, 'only one node should be evaluated')

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
        bb = BranchAndBound(small_branch, Node=PCBDFSNode, pseudo_costs={},
                            strong_branch_iters=5)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._global_lower_bound > -float('inf'), 'should change')
        self.assertTrue(bb._node_queue.qsize() == 2, 'should branch and add two nodes')
        self.assertTrue(bb._kwargs['pseudo_costs'], 'something should be set')
        self.assertTrue(bb._kwargs['strong_branch_iters'], 'something should be set')
        self.assertTrue(bb._evaluated_nodes == 1, 'only one node should be evaluated')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb._root_node, 'bound') as bd, \
                patch.object(bb._root_node, 'branch') as bh:
            bb._evaluate_node(bb._root_node)
            self.assertTrue(pr.call_count == 1)  # direct calls
            self.assertTrue(pbr.call_count == 1)
            self.assertTrue(0 == pbr.call_args.args[0], 'root node id should be first call arg')
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 1)

    def test_evaluate_node_integer(self):
        bb = BranchAndBound(no_branch)
        bb._evaluate_node(bb._root_node)

        # check attributes
        self.assertTrue(all(bb._best_solution == [1, 1, 0]))
        self.assertTrue(bb._global_upper_bound == -2)
        self.assertTrue(bb._global_lower_bound == -float('inf'), 'shouldnt change')
        self.assertTrue(bb._node_queue.empty(), 'immediately optimal model should create no nodes')
        self.assertTrue(bb._evaluated_nodes == 1, 'only one node should be evaluated')

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
        self.assertTrue(bb._evaluated_nodes == 1, 'only one node should be evaluated')

    def test_evaluate_node_properly_prunes(self):
        bb = BranchAndBound(no_branch)
        bb._global_upper_bound = -2
        called_node = BaseNode(bb.model.lp, bb.model.integerIndices, lower_bound=-4)
        pruned_node = BaseNode(bb.model.lp, bb.model.integerIndices, lower_bound=0)
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
            self.assertTrue(bb._evaluated_nodes == 1,
                            'only one node should be evaluated since other pruned')

    def test_process_branch_rtn_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'rtn must be a dictionary',
                               self.bb._process_branch_rtn, 0, 'fish')
        rtn = {'right': 5, 'left': 5}
        self.assertRaisesRegex(AssertionError, 'must be int',
                               self.bb._process_branch_rtn, '0', rtn)
        self.assertRaisesRegex(AssertionError, 'must already exist in tree',
                               self.bb._process_branch_rtn, 1, rtn)
        self.assertRaisesRegex(AssertionError, 'value must be type',
                               self.bb._process_branch_rtn, 0, rtn)
        del rtn['left']
        self.assertRaisesRegex(AssertionError, 'must be in the returned',
                               self.bb._process_branch_rtn, 0, rtn)
        self.root_branch_rtn['right'].idx = 0
        rtn = self.bound_root.branch(next_node_idx=0)
        self.assertRaisesRegex(AssertionError, 'give unique node ID',
                               self.bb._process_branch_rtn, 0, rtn)

    def test_process_branch_rtn(self):
        bb = BranchAndBound(small_branch)
        node = BaseNode(small_branch.lp, small_branch.integerIndices, idx=0)
        node.bound()
        rtn = node.branch(next_node_idx=1)
        left_node = rtn['left']
        right_node = rtn['right']
        bb._process_branch_rtn(node.idx, rtn)

        # check attributes
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(bb._node_queue.empty())
        children = bb._tree.get_children(node.idx)
        self.assertTrue(len(children) == 2, 'there should be two kids created')
        for child in children:
            self.assertFalse(bb._tree.get_children(child), 'children shouldnt have kids')

        self.assertTrue(bb._tree.get_node(1).attr['node'] is left_node)
        self.assertTrue(bb._tree.get_node(2).attr['node'] is right_node)

        # check function calls
        bb = BranchAndBound(small_branch)
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        node.bound()
        rtn = node.branch()
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb._tree, 'add_left_child') as alc, \
                patch.object(bb._tree, 'add_right_child') as arc:
            bb._process_branch_rtn(0, rtn)
            self.assertTrue(pr.call_count == 1, 'should call process rtn')
            self.assertTrue(alc.call_count == 1, 'should call add left child')
            self.assertTrue(arc.call_count == 1, 'should call add right child')


if __name__ == '__main__':
    unittest.main()
