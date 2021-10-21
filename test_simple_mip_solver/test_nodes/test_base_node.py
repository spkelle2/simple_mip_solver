from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray
# so pulp and pyomo don't believe reading in flat files is a worthwhile feature
# so we don't have much of an option here but to use some sort of commercial solver
try:  # if you don't have gurobipy installed, all tests except those using gurobi will run
    import gurobipy as gu
except ImportError:
    gu = None
from queue import PriorityQueue
import unittest
from unittest.mock import patch, PropertyMock

from simple_mip_solver import BaseNode
from test_simple_mip_solver.example_models import no_branch, small_branch, \
    infeasible, random, unbounded, cut2
from test_simple_mip_solver.helpers import TestModels


class TestNode(TestModels):

    Node = BaseNode  # define this for the TestModels attribute

    def test_init(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        self.assertTrue(node.lp, 'should get a model on proper instantiation')
        self.assertTrue(node._integer_indices == [0, 1, 2], 'should have list of integer indices')
        self.assertFalse(node.idx, 'idx should be None')
        self.assertTrue(node._var_indices == list(range(3)), 'should have list of var indices')
        self.assertTrue(node._row_indices == list(range(2)), 'should have list of row indices')
        self.assertTrue(node.lower_bound == -float('inf'))
        self.assertFalse(node.objective_value, 'should have obj but empty')
        self.assertFalse(node.solution, 'should have solution but empty')
        self.assertFalse(node.lp_feasible, 'should have lp_feasible but empty')
        self.assertFalse(node.unbounded, 'should have unbounded but empty')
        self.assertFalse(node.mip_feasible, 'should have mip_feasible but empty')
        self.assertTrue(node._epsilon > 0, 'should have epsilon > 0')
        self.assertFalse(node._b_dir, 'should have branch direction but empty')
        self.assertFalse(node._b_idx, 'should have branch index but empty')
        self.assertFalse(node._b_val, 'should have node value but empty')
        self.assertFalse(node.depth, 'should have depth but 0')
        self.assertTrue(node.branch_method == 'most fractional')
        self.assertTrue(node.search_method == 'best first')
        self.assertTrue(node.is_leaf, 'all nodes instantiate to being leaves')
        self.assertFalse(node.lineage, 'lineage should be None')

    def test_init_lineage(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, idx=0)
        self.assertTrue(node.lineage == (0,))

        node = BaseNode(small_branch.lp, small_branch.integerIndices, ancestors=(0,))
        self.assertTrue(node.lineage == (0,))

        node = BaseNode(small_branch.lp, small_branch.integerIndices, idx=3,
                        ancestors=(0, 1))
        self.assertTrue(node.lineage == (0, 1, 3))

    def test_init_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'lp must be CyClpSimplex instance',
                               BaseNode, small_branch, small_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'indices must match variables',
                               BaseNode, small_branch.lp, [4])
        self.assertRaisesRegex(AssertionError, 'node idx must be integer',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               idx=0.5)
        self.assertRaisesRegex(AssertionError, 'indices must be distinct',
                               BaseNode, small_branch.lp, [0, 1, 1])
        self.assertRaisesRegex(AssertionError, 'lower bound must be a float or an int',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               lower_bound='five')
        self.assertRaisesRegex(AssertionError, 'none are none or all are none',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               b_dir='up')
        self.assertRaisesRegex(AssertionError, 'branch index corresponds to integer variable if it exists',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               b_idx=4, b_dir='right', b_val=.5)
        self.assertRaisesRegex(AssertionError, 'we can only branch right or left',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               b_idx=1, b_dir='sideways', b_val=.5)
        self.assertRaisesRegex(AssertionError, 'branch val should be within 1 of both bounds',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               b_idx=1, b_dir='right', b_val=.5)
        self.assertRaisesRegex(AssertionError, 'depth is a positive integer',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               depth=2.5)
        self.assertRaisesRegex(AssertionError, 'node idx must be integer',
                               BaseNode, small_branch.lp, small_branch.integerIndices, .5)
        self.assertRaisesRegex(AssertionError, 'ancestors must be a tuple',
                               BaseNode, lp=small_branch.lp,
                               integer_indices=small_branch.integerIndices,
                               ancestors=[0])
        self.assertRaisesRegex(AssertionError, 'idx cannot be an ancestor of itself',
                               BaseNode, lp=small_branch.lp,
                               integer_indices=small_branch.integerIndices,
                               idx=0, ancestors=(0,))

    def test_base_bound_integer(self):
        node = BaseNode(no_branch.lp, no_branch.integerIndices)
        node._base_bound()
        self.assertTrue(node.objective_value == -2)
        self.assertTrue(all(node.solution == [1, 1, 0]))
        # integer solutions should come back as both lp and mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.mip_feasible)
        self.assertFalse(node.unbounded)

    def test_base_bound_fractional(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node._base_bound()
        self.assertTrue(node.objective_value == -2.75)
        self.assertTrue(all(node.solution == [0, 1.25, 1.5]))
        # fractional solutions should come back as lp but not mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertFalse(node.mip_feasible)
        self.assertFalse(node.unbounded)

    def test_base_bound_infeasible(self):
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        node._base_bound()
        # infeasible problems should come back as neither lp nor mip feasible
        self.assertFalse(node.lp_feasible)
        self.assertFalse(node.mip_feasible)
        self.assertFalse(node.unbounded)
        self.assertTrue(node.solution is None)
        self.assertTrue(node.objective_value == float('inf'))

    def test_base_bound_unbounded(self):
        node = BaseNode(unbounded.lp, unbounded.integerIndices)
        node._base_bound()

        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.unbounded)

    def test_bound(self):
        # check function calls
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        with patch.object(node, '_base_bound') as bb:
            node.bound(junk='stuff')  # should work with extra args
            self.assertTrue(bb.call_count == 1, 'should call base bound')

        # check return
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        rtn = node.bound()
        self.assertTrue(isinstance(rtn, dict), 'should return dict')
        self.assertFalse(rtn, 'dict should be empty')

    def test_base_branch_fails_asserts(self):
        # branching with bad next node idx should fail
        node = BaseNode(no_branch.lp, no_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'next node index should be integer',
                               node._base_branch, branch_idx=0, next_node_idx=.5)

        # branching before solving should fail
        node = BaseNode(no_branch.lp, no_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'must solve before branching',
                               node._base_branch, branch_idx=0)

        # branching on integer feasible node should fail
        node.bound()
        self.assertRaisesRegex(AssertionError, 'must branch on integer index',
                               node._base_branch, branch_idx=-1)

        # branching on non integer index should fail
        self.assertRaisesRegex(AssertionError, 'index branched on must be fractional',
                               node._base_branch, branch_idx=1)

    def test_base_branch(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node.bound()
        idx = 2
        out = {next_node_idx: node._base_branch(2, next_node_idx) for
               next_node_idx in [None, 1]}

        # confirm current node is no longer a leaf
        self.assertFalse(node.is_leaf)

        # check each node
        for next_node_idx, rtn in out.items():
            for name, n in rtn.items():
                if name not in ['left', 'right']:
                    continue
                self.assertTrue(all(n.lp.matrix.elements == node.lp.matrix.elements))
                self.assertTrue(all(n.lp.objective == node.lp.objective))
                self.assertTrue(all(n.lp.constraintsLower == node.lp.constraintsLower))
                self.assertTrue(all(n.lp.constraintsUpper == node.lp.constraintsUpper))
                self.assertTrue(n._integer_indices == node._integer_indices)
                self.assertTrue(n.lower_bound == node.objective_value)
                self.assertTrue(n._b_idx == idx)
                self.assertTrue(n._b_val == 1.5)
                self.assertTrue(n.depth == 1)
                if name == 'left':
                    self.assertTrue(all(n.lp.variablesUpper == [10, 10, 1]))
                    self.assertTrue(n.lp.variablesUpper[idx] == 1)
                    self.assertTrue(all(n.lp.variablesLower == node.lp.variablesLower))
                    self.assertTrue(n.lineage == (1, ) if next_node_idx else n.lineage is None)
                    self.assertTrue(n.idx == 1 if next_node_idx else n.idx is None)
                    self.assertTrue(n._b_dir == 'left')
                else:
                    self.assertTrue(all(n.lp.variablesUpper == node.lp.variablesUpper))
                    self.assertTrue(all(n.lp.variablesLower == [0, 0, 2]))
                    self.assertTrue(n.lineage == (2,) if next_node_idx else n.lineage is None)
                    self.assertTrue(n.idx == 2 if next_node_idx else n.idx is None)
                    self.assertTrue(n._b_dir == 'right')
                # check basis statuses work - i.e. are warm started
                for i in [0, 1]:
                    self.assertTrue(all(node.lp.getBasisStatus()[i] ==
                                        n.lp.getBasisStatus()[i]), 'bases should match')

            # check other returns
            self.assertTrue(rtn['next_node_idx'] == 3 if next_node_idx else
                            rtn['next_node_idx'] is None)

    def test_strong_branch_fails_asserts(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        node.bound()
        idx = node._most_fractional_index

        # test we stay within iters and improve bound/stay same
        self.assertRaisesRegex(AssertionError, 'iterations must be positive integer',
                               node._strong_branch, idx, 2.5)

    def test_strong_branch(self):
        iters = 5
        node = BaseNode(random.lp, random.integerIndices, 0)
        node.bound()
        idx = node._most_fractional_index

        # test we stay within iters and improve bound/stay same
        rtn = node._strong_branch(idx, iterations=iters)
        for direction, child_node in rtn.items():
            self.assertTrue(child_node.lp.iteration <= iters)
            if child_node.lp.getStatusCode() in [0, 3]:
                self.assertTrue(child_node.lp.objectiveValue >= node.objective_value)

        # test call base_branch
        node = BaseNode(random.lp, random.integerIndices)
        node.bound()
        idx = node._most_fractional_index
        children = node._base_branch(idx)
        with patch.object(node, '_base_branch') as bb:
            bb.return_value = children
            rtn = node._strong_branch(idx, iterations=iters)
            self.assertTrue(bb.called)

    def test_is_fractional_fails_asserts(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertRaisesRegex(AssertionError, 'value should be a number',
                               node._is_fractional, '5')

    def test_is_fractional(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertTrue(node._is_fractional(5.5))
        self.assertFalse(node._is_fractional(5))
        self.assertFalse(node._is_fractional(5.999999))
        self.assertFalse(node._is_fractional(5.000001))

    def test_get_fraction_fails_asserts(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertRaisesRegex(AssertionError, 'value should be a number',
                               node._get_fraction, '5.5')

    def test_get_fraction(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertTrue(.5 == node._get_fraction(5.5))
        
    def test_most_fractional_index(self):
        node = BaseNode(no_branch.lp, no_branch.integerIndices, 0)
        node.bound()
        self.assertFalse(node._most_fractional_index,
                         'int solution should have no fractional index')

        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        node.bound()
        self.assertTrue(node._most_fractional_index == 2)

    def test_branch(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        node.bound()

        # check function calls
        mock_pth = 'simple_mip_solver.nodes.base_node.BaseNode._most_fractional_index'
        with patch(mock_pth, new_callable=PropertyMock) as mfi, \
                patch.object(node, '_base_branch') as bb:
            bb_rtn = {'left': 'mock', 'right': 'another_mock', 'next_node_idx': 3}
            bb.return_value = bb_rtn
            branch_rtn = node.branch(junk='stuff')  # should work with extra args
            self.assertTrue(mfi.call_count == 1, 'should call most frac idx')
            self.assertTrue(bb.call_count == 1, 'should call base branch')
            self.assertTrue(branch_rtn == bb_rtn)

    def test_lt(self):
        node1 = BaseNode(small_branch.lp, small_branch.integerIndices, 0,  -float('inf'))
        node2 = BaseNode(small_branch.lp, small_branch.integerIndices, 0, 0)

        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)
        self.assertRaises(TypeError, node1.__lt__, 5)

        # make sure if we put them in PQ that they come out in the right order
        q = PriorityQueue()
        q.put(node2)
        q.put(node1)
        self.assertTrue(q.get().lower_bound < 0)
        self.assertTrue(q.get().lower_bound == 0)

    def test_eq(self):
        node1 = BaseNode(small_branch.lp, small_branch.integerIndices, 0, -float('inf'))
        node2 = BaseNode(small_branch.lp, small_branch.integerIndices, 0, 0)
        node3 = BaseNode(small_branch.lp, small_branch.integerIndices, 0, 0)

        self.assertTrue(node3 == node2)
        self.assertFalse(node1 == node2)
        self.assertRaises(TypeError, node1.__eq__, 5)

    def test_sense_fails_asserts(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        node.lp.addConstraint(-CyLPArray([1, 0, 0]) * node.lp.getVarByName('x') >= 1)
        with self.assertRaises(AssertionError):
            node._sense

    def test_sense(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertTrue(node._sense == '<=')
        milp = MILPInstance(A=small_branch.A, b=small_branch.b, c=small_branch.c,
                            sense=['Min', '>='], numVars=3,
                            integerIndices=small_branch.integerIndices)
        node = BaseNode(milp.lp, milp.integerIndices, 0)
        self.assertTrue(node._sense == '>=')

    def test_variables_nonnegative(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertTrue(node._variables_nonnegative)
        node = BaseNode(cut2.lp, cut2.integerIndices, 0)
        l = node.lp.variablesLower.copy()
        l[0] = -10
        node.lp.variablesLower = l
        self.assertFalse(node._variables_nonnegative)

    def test_models(self):
        self.base_test_models()


if __name__ == '__main__':
    unittest.main()
