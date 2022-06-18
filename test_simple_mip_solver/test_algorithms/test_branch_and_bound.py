from coinor.cuppy.milpInstance import MILPInstance
from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
import inspect
from math import isclose
import numpy as np
import os
from queue import PriorityQueue
import re
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound, \
    PseudoCostBranchDepthFirstSearchNode as PCBDFSNode, PseudoCostBranchNode
from simple_mip_solver.algorithms.branch_and_bound import BranchAndBoundTree
from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from test_simple_mip_solver.example_models import no_branch, small_branch, infeasible, \
    unbounded, infeasible2, h3p1, h3p1_0, h3p1_1, h3p1_2, h3p1_3, h3p1_4, h3p1_5, \
    small_branch_copy
from test_simple_mip_solver import example_models


skip_longs = False


class TestBranchAndBoundTree(unittest.TestCase):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'small_branch_std': small_branch, 'infeasible_std': infeasible,
                        'no_branch_std': no_branch}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = BaseAlgorithm._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    def test_get_leaves_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, "subtree_root_id must belong to the tree",
                               bb.tree.get_leaves, 20)
        self.assertRaisesRegex(AssertionError, "depth is a nonnegative integer",
                               bb.tree.get_leaves, subtree_root_id=0, depth=1.5)
        self.assertRaisesRegex(AssertionError, "keep is one of 'all', 'feasible', or 'not infeasible'",
                               bb.tree.get_leaves, subtree_root_id=0, keep=False)

    def test_get_leaves(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False, node_limit=1)
        bb.solve()

        # feasible vs not infeasible
        leaves = bb.tree.get_leaves(0, keep='not infeasible')
        self.assertTrue(len(leaves) == 2)
        leaves = bb.tree.get_leaves(0, keep='feasible')
        self.assertFalse(leaves)

        bb.node_limit = float('inf')
        bb.solve()

        # all leaves
        leaves = bb.tree.get_leaves(0)
        for node_id, node in bb.tree.nodes.items():
            if node_id in [n.idx for n in leaves]:
                self.assertFalse(bb.tree.get_children(node_id))
            else:
                self.assertTrue(len(bb.tree.get_children(node_id)) == 2)

        # all feasible leaves
        leaves = bb.tree.get_leaves(0, keep='feasible')
        for node_id, node in bb.tree.nodes.items():
            if node_id in [n.idx for n in leaves]:
                self.assertFalse(bb.tree.get_children(node_id))
                self.assertTrue(node.attr['node'].lp_feasible)
            else:
                self.assertTrue(len(bb.tree.get_children(node_id)) == 2 or not
                                node.attr['node'].lp_feasible)

        # depth 0 subtree
        leaves = bb.tree.get_leaves(2, depth=0)
        self.assertTrue(len(leaves) == 1)
        self.assertTrue(leaves[0].idx == 2)

        leaves = bb.tree.get_leaves(2, depth=0, keep='feasible')
        self.assertFalse(leaves)

        # depth 1 subtree
        leaves = bb.tree.get_leaves(0, depth=1)
        self.assertTrue(len(leaves) == 2)
        self.assertTrue(set(n.idx for n in leaves) == {1, 2})

        leaves = bb.tree.get_leaves(0, depth=1, keep='feasible')
        self.assertTrue(len(leaves) == 1)
        self.assertTrue(leaves[0].idx == 1)

        # depth 2 subtree
        leaves = bb.tree.get_leaves(1, depth=2)
        self.assertTrue({n.idx for n in leaves} == {5, 6, 7, 8})
        for node in leaves:
            self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)

        leaves = bb.tree.get_leaves(1, depth=2, keep='feasible')
        self.assertTrue({n.idx for n in leaves} == {5, 7})
        for node in leaves:
            self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)

        # depth 3 subtree
        leaves = bb.tree.get_leaves(1, depth=3)
        self.assertTrue({n.idx for n in leaves} == {5, 6, 8, 9, 10})
        for node in leaves:
            if node.idx <= 8:
                self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)
            else:
                self.assertTrue(
                    bb.tree.get_parent(bb.tree.get_parent(bb.tree.get_parent(node.idx))) == 1
                )

        leaves = bb.tree.get_leaves(1, depth=3, keep='feasible')
        self.assertTrue({n.idx for n in leaves} == {5, 9})

    def test_get_disjunction_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, "subtree_root_id must belong to the tree",
                               bb.tree.get_disjunction, 20)

    def test_get_disjunction(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        disjunction = bb.tree.get_disjunction(0)
        self.assertTrue(all(disjunction[5][0] == [0, 0, 0]))
        self.assertTrue(all(disjunction[5][1] == [0, 1, 1]))
        self.assertTrue(all(disjunction[11][0] == [1, 0, 0]))
        self.assertTrue(all(disjunction[11][1] == [1, 1, 0]))

    def test_get_node_instances_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, 'must be an integer or iterable',
                               bb.tree.get_node_instances, '1')
        self.assertRaisesRegex(AssertionError, 'node_ids are not in the tree',
                               bb.tree.get_node_instances, [20])
        del bb.tree.nodes[0].attr['node']
        self.assertRaisesRegex(AssertionError, 'must have an attribute for a node instance',
                               bb.tree.get_node_instances, [0])

    def test_get_node_instances(self):
        bb = BranchAndBound(small_branch_copy, gomory_cuts=False)
        bb.solve()

        # test list
        node1, node2 = bb.tree.get_node_instances([1, 2])
        self.assertTrue(node1.idx == 1, 'we should get node with matching id')
        self.assertTrue(isinstance(node1, BaseNode), 'we should get a node')
        self.assertTrue(node2.idx == 2, 'we should get node with matching id')
        self.assertTrue(isinstance(node2, BaseNode), 'we should get a node')

        # test singleton
        node1 = bb.tree.get_node_instances(1)
        self.assertTrue(node1.idx == 1, 'we should get node with matching id')
        self.assertTrue(isinstance(node1, BaseNode), 'we should get a node')

    def test_subtree_dual_bound_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        self.assertRaisesRegex(AssertionError, 'subtree_root_id must belong to the tree',
                               bb.tree.subtree_dual_bound, subtree_root_id=1)

    def test_subtree_dual_bound(self):
        bb = BranchAndBound(small_branch_copy, gomory_cuts=False, node_limit=1)

        # 0 nodes
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -float('inf'))

        # 1 node
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2.75)

        # 2 nodes
        bb.node_limit = 2
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2.75)

        # all nodes
        bb.node_limit = float('inf')
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2)
        self.assertTrue(bb.tree.subtree_dual_bound(2) == float('inf'))
        self.assertTrue(bb.tree.subtree_dual_bound(0, depth=1) == -2.75)


class TestBranchAndBound(unittest.TestCase):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'small_branch_std': small_branch, 'infeasible_std': infeasible,
                        'no_branch_std': no_branch}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = BaseAlgorithm._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

        self.bb = BranchAndBound(self.small_branch_std)
        self.unbounded_root = BaseNode(BaseAlgorithm._convert_constraints_to_greq(self.small_branch_std).lp,
                                       self.small_branch_std.integerIndices)
        self.bound_root = BaseNode(BaseAlgorithm._convert_constraints_to_greq(self.small_branch_std).lp,
                                   self.small_branch_std.integerIndices)
        for constr in self.bound_root.lp.constraints:
            if self.bound_root.cut_name_pattern.match(constr.name):
                self.bound_root.lp.removeConstraint(constr.name)
        self.bound_root.bound(gomory_cuts=False)
        self.root_branch_rtn = self.bound_root.branch()

    def test_init(self):
        bb = BranchAndBound(self.small_branch_std)
        self.assertTrue(isinstance(bb, BaseAlgorithm))
        self.assertTrue(bb.primal_bound == float('inf'))
        self.assertTrue(bb.dual_bound == -float('inf'))
        self.assertTrue(bb._node_queue.empty())
        self.assertFalse(bb._unbounded)
        self.assertFalse(bb._best_solution)
        self.assertFalse(bb.solution)
        self.assertTrue(bb.status == 'unsolved')
        self.assertFalse(bb.objective_value)
        self.assertTrue(isinstance(bb.tree, BranchAndBoundTree))
        self.assertTrue(list(bb.tree.nodes.keys()) == [0])
        self.assertTrue(bb.tree.nodes[0].attr['node'] is bb.root_node)
        self.assertFalse(bb.solve_time, 'solve time should exist and be 0')
        self.assertTrue(bb.mip_gap, 'mip gap should be an attribute')
        self.assertFalse(bb.logging)
        self.assertTrue(bb.max_run_time == float('inf'))

    def test_init_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std)
        queue = PriorityQueue()

        # node_queue asserts
        for func in reversed(bb._queue_funcs):
            queue.__dict__[func] = 5
            self.assertRaisesRegex(AssertionError, f'node_queue needs a {func} function',
                                   BranchAndBound, self.small_branch_std, BaseNode, queue)

        # node limit asserts
        self.assertRaisesRegex(AssertionError, f'node limit', BranchAndBound,
                               model=self.small_branch_std, node_limit=-5)

        # mip gap asserts
        self.assertRaisesRegex(AssertionError, f'mip_gap', BranchAndBound,
                               model=self.small_branch_std, mip_gap=-5)

        # logging asserts
        self.assertRaisesRegex(AssertionError, f'logging', BranchAndBound,
                               model=self.small_branch_std, logging=0)

        # run time assert
        self.assertRaisesRegex(AssertionError, f'max_run_time', BranchAndBound,
                               model=self.small_branch_std, max_run_time=0)

        # initial primal bound assert
        self.assertRaisesRegex(AssertionError, f'initial_primal_bound', BranchAndBound,
                               model=self.small_branch_std, initial_primal_bound=-float('inf'))

        # kwargs asserts
        self.assertRaisesRegex(AssertionError, 'saved for later use', BranchAndBound,
                               model=self.small_branch_std, right=-5)

    def test_current_gap(self):
        bb = BranchAndBound(self.small_branch_std, node_limit=1, gomory_cuts=False)
        bb.solve()
        self.assertTrue(bb.current_gap is None)
        bb.node_limit = 10
        bb.solve()
        self.assertTrue(bb.current_gap == .125)
        bb.node_limit = float('inf')
        bb.solve()
        self.assertTrue(bb.current_gap == 0)
        print()

    def test_solve_stopped_on_time(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(self.small_branch_std, Node=Node, max_run_time=.000001, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'stopped on iterations or time')
            self.assertTrue(bb.solve_time > .000001)

    def test_solve_stopped_on_iterations(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(self.small_branch_std, Node=Node, node_limit=1, pseudo_costs={},
                                gomory_cuts=False)
            bb.solve()
            self.assertTrue(bb.status == 'stopped on iterations or time')
            self.assertTrue(bb.solve_time)

    def test_solve_optimal(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(self.small_branch_std, Node=Node, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'optimal')
            self.assertTrue(all(s.is_integer for s in bb.solution))
            self.assertTrue(bb.objective_value == -2)
            self.assertTrue(bb.solve_time)

    def test_solve_infeasible(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(infeasible2, Node=Node, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'infeasible')
            self.assertFalse(bb.solution)
            self.assertTrue(bb.objective_value == float('inf'))
            self.assertTrue(bb.solve_time)

    def test_solve_unbounded(self):
        # check and make sure we're good with both nodes
        for Node in [BaseNode, PCBDFSNode]:
            bb = BranchAndBound(unbounded, Node=Node, pseudo_costs={})
            bb.solve()
            self.assertTrue(bb.status == 'unbounded')
            self.assertTrue(bb.solve_time)

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
            bb.evaluated_nodes = 10
            bb.solve()
            self.assertFalse(en.called, 'were past the node limit')

    def test_evaluate_node_infeasible(self):
        bb = BranchAndBound(infeasible2)
        bb._evaluate_node(bb.root_node)

        # check attributes
        self.assertTrue(bb._node_queue.empty(), 'inf model should create no nodes')
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb.primal_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb.dual_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb.evaluated_nodes == 1, 'only one node should be evaluated')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb.root_node, 'bound') as bd, \
                patch.object(bb.root_node, 'branch') as bh:
            bd.return_value = {}
            bb._evaluate_node(bb.root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)

    def test_evaluate_node_fractional(self):
        bb = BranchAndBound(self.small_branch_std, Node=PCBDFSNode, pseudo_costs={},
                            strong_branch_iters=5, gomory_cuts=False)
        bb._evaluate_node(bb.root_node)

        # check attributes
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb.primal_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb.dual_bound > -float('inf'), 'should change')
        self.assertTrue(bb._node_queue.qsize() == 2, 'should branch and add two nodes')
        self.assertTrue(bb._kwargs['pseudo_costs'], 'something should be set')
        self.assertTrue(bb._kwargs['strong_branch_iters'], 'something should be set')
        self.assertTrue(bb.evaluated_nodes == 1, 'only one node should be evaluated')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb.root_node, 'bound') as bd, \
                patch.object(bb.root_node, 'branch') as bh:
            bd.return_value = {}
            bb._evaluate_node(bb.root_node)
            self.assertTrue(pr.call_count == 1)  # direct calls
            self.assertTrue(pbr.call_count == 1)
            self.assertTrue(0 == pbr.call_args.args[0], 'root node id should be first call arg')
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 1)

    def test_evaluate_node_integer(self):
        bb = BranchAndBound(no_branch)
        bb._evaluate_node(bb.root_node)

        # check attributes
        self.assertTrue(all(bb._best_solution == [1, 1, 0]))
        self.assertTrue(bb.primal_bound == -2)
        self.assertTrue(bb.dual_bound == -2, 'should match upper bound when optimal')
        self.assertTrue(bb._node_queue.empty(), 'immediately optimal model should create no nodes')
        self.assertTrue(bb.evaluated_nodes == 1, 'only one node should be evaluated')

        # check function calls - recycle object since it has attrs already set
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb, '_process_branch_rtn') as pbr, \
                patch.object(bb.root_node, 'bound') as bd, \
                patch.object(bb.root_node, 'branch') as bh:
            bd.return_value = {}
            bb._evaluate_node(bb.root_node)
            self.assertTrue(pr.call_count == 1)
            self.assertTrue(pbr.call_count == 0)
            self.assertTrue(bd.call_count == 1)
            self.assertTrue(bh.call_count == 0)

    def test_evaluate_node_unbounded(self):
        bb = BranchAndBound(unbounded)
        bb._evaluate_node(bb.root_node)

        # check attributes
        self.assertTrue(bb._unbounded)
        self.assertTrue(bb.evaluated_nodes == 1, 'only one node should be evaluated')

    def test_evaluate_node_properly_prunes(self):
        bb = BranchAndBound(no_branch, initial_primal_bound=-2)
        called_node = BaseNode(bb.model.lp, bb.model.integerIndices, dual_bound=-4)
        pruned_node = BaseNode(bb.model.lp, bb.model.integerIndices, dual_bound=0)
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
            self.assertTrue(bb.evaluated_nodes == 1,
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

        # test node index against rest of tree
        rtn = self.bound_root.branch(next_node_idx=1)
        rtn['left'].idx = 0
        self.assertRaisesRegex(AssertionError, 'give unique node ID',
                               self.bb._process_branch_rtn, 0, rtn)

    def test_process_branch_rtn(self):
        bb = BranchAndBound(self.small_branch_std)
        node = BaseNode(BaseAlgorithm._convert_constraints_to_greq(self.small_branch_std).lp,
                        self.small_branch_std.integerIndices, idx=0)
        node.bound(gomory_cuts=False)
        rtn = node.branch(next_node_idx=1)
        left_node = rtn['left']
        right_node = rtn['right']
        bb._process_branch_rtn(node.idx, rtn)

        # check attributes
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(isinstance(bb._node_queue.get(), BaseNode))
        self.assertTrue(bb._node_queue.empty())
        children = bb.tree.get_children(node.idx)
        self.assertTrue(len(children) == 2, 'there should be two kids created')
        for child in children:
            self.assertFalse(bb.tree.get_children(child), 'children shouldnt have kids')

        self.assertTrue(bb.tree.get_node(1).attr['node'] is left_node)
        self.assertTrue(bb.tree.get_node(2).attr['node'] is right_node)

        # check function calls
        bb = BranchAndBound(self.small_branch_std)
        node = BaseNode(BaseAlgorithm._convert_constraints_to_greq(self.small_branch_std).lp,
                        self.small_branch_std.integerIndices, idx=0)
        node.bound(gomory_cuts=False)
        rtn = node.branch()
        with patch.object(bb, '_process_rtn') as pr, \
                patch.object(bb.tree, 'add_left_child') as alc, \
                patch.object(bb.tree, 'add_right_child') as arc:
            bb._process_branch_rtn(0, rtn)
            self.assertTrue(pr.call_count == 1, 'should call process rtn')
            self.assertTrue(alc.call_count == 1, 'should call add left child')
            self.assertTrue(arc.call_count == 1, 'should call add right child')

    def test_process_bound_rtn_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std)
        self.assertRaisesRegex(AssertionError, 'rtn must be a dictionary',
                               self.bb._process_bound_rtn, 'fish')

    def test_process_bound_rtn(self):
        cglp_bb = BranchAndBound(self.small_branch_std, node_limit=8)
        cglp_bb.solve()
        cglp = CutGeneratingLP(cglp_bb, cglp_bb.root_node.idx)
        pi, pi0 = cglp.solve()
        rtn = {'cuts': {'cut_cglp_0_0': (pi, pi0)}}

        with patch.object(cglp_bb, '_process_rtn') as pr:
            cglp_bb._process_bound_rtn(rtn)
            args, kwargs = pr.call_args
            self.assertTrue(len(args) == 1 and len(kwargs) == 0)
            self.assertFalse(args[0])
            for node in cglp_bb._node_queue.queue:
                self.assertTrue(len(node.cut_pool) == 1)
                self.assertTrue((node.cut_pool['node_0_cglp_cut'][0] == pi).all())
                self.assertTrue((node.cut_pool['node_0_cglp_cut'][1] == pi0).all())

    def test_find_parameterized_dual_bound_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        self.assertRaisesRegex(AssertionError, 'must solve this instance before',
                               bb.find_parameterized_dual_bound, CyLPArray([2.5, 4.5]))
        bb.solve()
        self.assertRaisesRegex(AssertionError, 'only works with CyLP arrays',
                               bb.find_parameterized_dual_bound, np.array([2.5, 4.5]))
        self.assertRaisesRegex(AssertionError, 'shape of the RHS being added should match',
                               bb.find_parameterized_dual_bound, CyLPArray([4.5]))

        bb = BranchAndBound(infeasible2)
        bb.root_node.lp += np.matrix([[0, -1, -1]]) * bb.root_node.lp.getVarByName('x') >= CyLPArray([-2.5])
        bb.solve()
        self.assertRaisesRegex(AssertionError, 'feature expects the root node to have a single constraint object',
                               bb.find_parameterized_dual_bound, CyLPArray([2.5, 4.5]))

    def test_find_parameterized_dual_bound(self):

        # Ensure that BranchAndBound.find_parameterized_dual_bound generates the dual function
        # that we saw in ISE 418 HW 3 problem 1
        bb = BranchAndBound(h3p1, gomory_cuts=False)
        bb.solve()
        bound = bb.find_parameterized_dual_bound(CyLPArray([3.5, -3.5]))
        self.assertTrue(bb.objective_value == bound, 'dual should be strong at original rhs')

        prob = {0: h3p1_0, 1: h3p1_1, 2: h3p1_2, 3: h3p1_3, 4: h3p1_4, 5: h3p1_5}
        sol_new = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
        sol_bound = {0: 0, 1: .5, 2: 1, 3: 2, 4: 2, 5: 2.5}
        for beta in range(6):
            new_bb = BranchAndBound(prob[beta])
            new_bb.solve()
            bound = bb.find_parameterized_dual_bound(CyLPArray(np.array([beta, -beta])))
            self.assertTrue(isclose(sol_new[beta], new_bb.objective_value, abs_tol=.01),
                            'new branch and bound objective should match expected')
            self.assertTrue(isclose(sol_bound[beta], bound),
                            'new dual bound value should match expected')
            self.assertTrue(bound <= new_bb.objective_value + .01,
                            'dual bound value should be at most the value function for this rhs')

        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        bound = bb.find_parameterized_dual_bound(CyLPArray([-2.5, -4.5]))

        # just make sure the dual bound works here too
        self.assertTrue(bound <= -5.99,
                        'dual bound value should be at most the value function for this rhs')

        # check function calls
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        bound_parameterized_duals = [bb._bound_parameterized_dual(n.lp) for n in bb.tree.get_node_instances([6, 12, 10, 8, 2])]
        with patch.object(bb, '_bound_parameterized_dual') as bd:
            bd.side_effect = bound_parameterized_duals
            bound = bb.find_parameterized_dual_bound(CyLPArray([3, 3]))
            self.assertTrue(bd.call_count == 5)

        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        bound = bb.find_parameterized_dual_bound(CyLPArray([3, 3]))
        with patch.object(bb, '_bound_parameterized_dual') as bd:
            bound = bb.find_parameterized_dual_bound(CyLPArray([1, 1]))
            self.assertFalse(bd.called)

    @unittest.skipIf(skip_longs, "debugging")
    def test_find_parameterized_dual_bound_many_times(self):
        pattern = re.compile('evaluation_(\d+).mps')
        fldr_pth = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(example_models))),
                                'example_value_functions')
        for count, sub_fldr in enumerate(os.listdir(fldr_pth)):
            print(f'dual bound {count}')
            sub_fldr_pth = os.path.join(fldr_pth, sub_fldr)
            evals = {}
            for file in os.listdir(sub_fldr_pth):
                eval_num = int(pattern.search(file).group(1))
                instance = MILPInstance(file_name=os.path.join(sub_fldr_pth, file))
                bb = BranchAndBound(instance, PseudoCostBranchNode, pseudo_costs={},
                                    gomory_cuts=False)
                bb.solve()
                evals[eval_num] = bb
            instance_0 = evals[0]
            for bb in evals.values():
                # all problems were given as <=, so their constraints were flipped by default
                self.assertTrue(instance_0.find_parameterized_dual_bound(CyLPArray(-bb.model.b)) <=
                                bb.objective_value + .01, 'dual_bound should be less')

    def test_bound_parameterized_dual(self):
        bb = BranchAndBound(infeasible2)
        bb.root_node.lp += np.matrix([[0, -1, -1]]) * bb.root_node.lp.getVarByName('x') >= CyLPArray([-2.5])
        bb.solve()
        terminal_nodes = bb.tree.get_leaves(0)
        infeasible_nodes = [n for n in terminal_nodes if n.lp_feasible is False]
        n = infeasible_nodes[0]
        lp = bb._bound_parameterized_dual(n.lp)

        # test that we get a CyClpSimplex object back
        self.assertTrue(isinstance(lp, CyClpSimplex), 'should return CyClpSimplex instance')

        # same variables plus extra 's'
        self.assertTrue({v.name for v in lp.variables} == {'x', 's_0', 's_1'},
                        'x should already exist and s_1 and s_2 should be added')
        old_x = n.lp.getVarByName('x')
        new_x, s_0, s_1 = lp.getVarByName('x'), lp.getVarByName('s_0'), lp.getVarByName('s_1')

        # same variable bounds, plus s >= 0
        self.assertTrue(all(new_x.lower == old_x.lower) and all(new_x.upper == old_x.upper),
                        'x should have the same bounds')
        self.assertTrue(all(s_0.lower == [0, 0]) and all(s_0.upper > [1e300, 1e300]), 's_0 >= 0')
        self.assertTrue(all(s_1.lower == [0]) and all(s_1.upper > 1e300), 's_1 >= 0')

        # same constraints, plus slack s
        self.assertTrue(lp.nConstraints == 3, 'should have same number of constraints')
        self.assertTrue((lp.constraints[0].varCoefs[new_x] == np.array([[-1, -1, 0], [0, 0, -1]])).all(),
                        'x coefs should stay same')
        self.assertTrue((lp.constraints[0].varCoefs[s_0] == np.matrix(np.identity(2))).all(),
                        's_0 should have coef of 2-D identity')
        self.assertTrue(all(lp.constraints[1].varCoefs[new_x] == np.array([0, -1, -1])),
                        'x coefs should stay same')
        self.assertTrue(lp.constraints[1].varCoefs[s_1] == np.matrix(np.identity(1)),
                        's_0 should have coef of 1-D identity')
        self.assertTrue(all(lp.constraints[0].lower == np.array([1, -1])) and
                        all(lp.constraints[0].upper >= np.array([1e300])),
                        'constraint bounds should remain same')
        self.assertTrue(lp.constraints[1].lower == np.array([-2.5]) and
                        lp.constraints[1].upper >= np.array([1e300]),
                        'constraint bounds should remain same')

        # same objective, plus large s coefficient
        self.assertTrue(all(lp.objective == np.array([-1, -1, 0, bb._M, bb._M, bb._M])))

        # problem is now feasible
        self.assertTrue(lp.getStatusCode() == 0, 'lp should now be optimal')

    def test_bound_parameterized_dual_fails_asserts(self):
        bb = BranchAndBound(self.infeasible_std)
        bb.solve()
        terminal_nodes = bb.tree.get_leaves(0)
        infeasible_nodes = [n for n in terminal_nodes if n.lp_feasible is False]
        n = infeasible_nodes[0]
        n.lp.addVariable('s_0', 1)
        self.assertRaisesRegex(AssertionError, "variable 's_0' is a reserved name",
                               bb._bound_parameterized_dual, n.lp)
        self.assertRaisesRegex(AssertionError, "must give CyClpSimplex instance",
                               bb._bound_parameterized_dual, n)


if __name__ == '__main__':
    unittest.main()
