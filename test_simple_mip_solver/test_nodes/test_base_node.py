import re

from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray
# so pulp and pyomo don't believe reading in flat files is a worthwhile feature
# so we don't have much of an option here but to use some sort of commercial solver
try:  # if you don't have gurobipy installed, all tests except those using gurobi will run
    import gurobipy as gu
except ImportError:
    gu = None
from math import isclose
import numpy as np
from queue import PriorityQueue
import unittest
from unittest.mock import patch, PropertyMock

from simple_mip_solver import BaseNode
from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from test_simple_mip_solver.example_models import no_branch, small_branch, \
    infeasible, random, unbounded, cut2, cut1, negative, cut3
from test_simple_mip_solver.helpers import TestModels


class TestBaseNode(TestModels):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'cut1_std': cut1, 'cut2_std': cut2, 'cut3_std': cut3,
                        'infeasible_std': infeasible, 'no_branch_std': no_branch,
                        'small_branch_std': small_branch}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = BaseAlgorithm._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    def test_init(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        self.assertTrue(node.lp, 'should get a model on proper instantiation')
        self.assertTrue(node._integer_indices == [0, 1, 2], 'should have list of integer indices')
        self.assertFalse(node.idx, 'idx should be None')
        self.assertTrue(node.dual_bound == -float('inf'))
        self.assertFalse(node.objective_value, 'should have obj but empty')
        self.assertFalse(node.solution, 'should have solution but empty')
        self.assertFalse(node.lp_feasible, 'should have lp_feasible but empty')
        self.assertFalse(node.unbounded, 'should have unbounded but empty')
        self.assertFalse(node.mip_feasible, 'should have mip_feasible but empty')
        self.assertFalse(node._b_dir, 'should have branch direction but empty')
        self.assertFalse(node._b_idx, 'should have branch index but empty')
        self.assertFalse(node._b_val, 'should have node value but empty')
        self.assertFalse(node.depth, 'should have depth but 0')
        self.assertTrue(node.branch_method == 'most fractional')
        self.assertTrue(node.search_method == 'best first')
        self.assertTrue(node.is_leaf, 'all nodes instantiate to being leaves')
        self.assertFalse(node.lineage, 'lineage should be None')
        self.assertFalse(node.cut_generation_iterations)
        self.assertRegexpMatches('cut_gomory_1_1_6', node.cut_name_pattern)
        self.assertFalse(node.cut_generation_stalled)
        self.assertFalse(node.iterations_gmic_created)
        self.assertFalse(node.number_gmic_created)
        self.assertFalse(node.iterations_gmic_added)
        self.assertFalse(node.number_gmic_added)
        self.assertFalse(node.iterations_gmic_removed)
        self.assertFalse(node.number_gmic_removed)
        self.assertTrue(isinstance(node.gmic_name_pattern, re.Pattern))
        self.assertFalse(node.cut_pool)

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
        self.assertRaisesRegex(AssertionError, 'dual bound must be a float or an int',
                               BaseNode, small_branch.lp, small_branch.integerIndices,
                               dual_bound='five')
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

    def test_cut_pool_setter_fails_asserts(self):
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        with self.assertRaisesRegex(AssertionError, 'idx should start with'):
            node.cut_pool = {'gomory_1_1_1': (CyLPArray([1, 0]), 0)}
        with self.assertRaisesRegex(AssertionError, 'pi should be CyLPArray'):
            node.cut_pool = {'cut_gomory_1_1_1': (np.array([1, 0]), 0)}
        with self.assertRaisesRegex(AssertionError, 'pi0 should be number'):
            node.cut_pool = {'cut_gomory_1_1_1': (CyLPArray([1, 0]), '0')}

    def test_bound(self):
        # check function calls
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        with patch.object(node, '_base_bound') as bb:
            bb.return_value = {}
            rtn = node.bound(junk='stuff')  # should work with extra args
            self.assertTrue(bb.call_count == 1, 'should call base bound')
            self.assertFalse(rtn)

    def test_base_bound_fails_asserts(self):
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        self.assertRaisesRegex(AssertionError, 'must be a positive integer',
                               node._base_bound, max_cut_generation_iterations=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_cut_generation_iterations=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_iterations_gmic_created=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_number_gmic_created=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_iterations_gmic_added=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_number_gmic_added=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_iterations_gmic_removed=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node._base_bound, total_number_gmic_removed=-1)

    def test_base_bound(self):
        node = BaseNode(infeasible.lp, infeasible.integerIndices)

        def patch_cgi_failed_iter():
            node.cut_generation_iterations += 1
            node.lp_feasible = False

        # check function calls
        # infeasible lp
        with patch.object(node, '_bound_lp') as bl, \
                patch.object(node, '_cut_generation_iteration', new=patch_cgi_failed_iter) as cgi:

            # initially infeasible
            node.lp_feasible = False
            node.mip_feasible = False

            node._base_bound()

            self.assertTrue(bl.called)
            self.assertFalse(node.cut_generation_iterations)  # would be 1 if mock called

            # infeasible after a cut generation iteration
            node.lp_feasible = True
            max_iters = 3
            node._base_bound(max_cut_generation_iterations=max_iters)
            # should get called once and then stopped
            self.assertTrue(bl.call_count == 2)
            self.assertFalse(node.lp_feasible)
            self.assertFalse(node.mip_feasible)
            self.assertFalse(node.cut_generation_stalled)
            self.assertTrue(node.cut_generation_iterations == 1)

        def patch_cgi_iter():
            node.cut_generation_iterations += 1

        # feasible lp but infeasible mip stop on iterations
        with patch.object(node, '_bound_lp') as bl, \
                patch.object(node, '_cut_generation_iteration', new=patch_cgi_iter) as cgi:
            node.lp_feasible = True
            max_cut_generation_iterations = 3

            node._base_bound(max_cut_generation_iterations=max_cut_generation_iterations)

            self.assertTrue(bl.called)
            self.assertTrue(node.lp_feasible)
            self.assertFalse(node.mip_feasible)
            self.assertFalse(node.cut_generation_stalled)
            self.assertTrue(node.cut_generation_iterations == max_cut_generation_iterations)

        node = BaseNode(infeasible.lp, infeasible.integerIndices)

        def patch_cgi_stall():
            node.cut_generation_stalled = True

        # feasible lp but infeasible mip stop on stall
        with patch.object(node, '_bound_lp') as bl, \
                patch.object(node, '_cut_generation_iteration', new=patch_cgi_stall) as cgi:
            node.lp_feasible = True
            node._base_bound(max_cut_generation_iterations=max_cut_generation_iterations)

            self.assertTrue(bl.called)
            self.assertTrue(node.lp_feasible)
            self.assertFalse(node.mip_feasible)
            self.assertTrue(node.cut_generation_stalled)
            self.assertTrue(node.cut_generation_iterations < max_cut_generation_iterations)

        node = BaseNode(infeasible.lp, infeasible.integerIndices)

        def patch_cgi_mip_feasible():
            node.mip_feasible = True

        # feasible lp but infeasible mip becomes feasible
        with patch.object(node, '_bound_lp') as bl, \
                patch.object(node, '_cut_generation_iteration', new=patch_cgi_mip_feasible) as cgi:
            node.lp_feasible = True
            node._base_bound(max_cut_generation_iterations=max_cut_generation_iterations)

            self.assertTrue(bl.called)
            self.assertTrue(node.lp_feasible)
            self.assertTrue(node.mip_feasible)
            self.assertFalse(node.cut_generation_stalled)
            self.assertTrue(node.cut_generation_iterations < max_cut_generation_iterations)

        # feasible mip
        with patch.object(node, '_bound_lp') as bl, \
                patch.object(node, '_cut_generation_iteration') as cgi:
            node.mip_feasible = True

            node._base_bound(max_cut_generation_iterations=max_cut_generation_iterations)

            self.assertTrue(bl.called)
            self.assertFalse(cgi.called)

        # do normal run to make sure we're ok
        node = BaseNode(self.cut2_std.lp, self.cut2_std.integerIndices)
        node._bound_lp()
        obj = node.objective_value
        constrs = node.lp.nConstraints
        rtn = node._base_bound(gomory_cuts=True, total_iterations_gmic_created=1,
                               total_number_gmic_created=1, total_iterations_gmic_added=1,
                               total_number_gmic_added=1, total_iterations_gmic_removed=1,
                               total_number_gmic_removed=1, total_cut_generation_iterations=10)
        self.assertTrue(node.lp_feasible)
        self.assertFalse(node.cut_generation_stalled)
        self.assertTrue(-2.01 < obj - node.objective_value < -2)
        self.assertTrue(node.lp.nConstraints > constrs)
        self.assertTrue(node.mip_feasible)
        self.assertTrue(rtn['total_iterations_gmic_created'] == 3)
        self.assertTrue(rtn['total_number_gmic_created'] == 5)
        self.assertTrue(rtn['total_iterations_gmic_added'] == 3)
        self.assertTrue(rtn['total_number_gmic_added'] == 5)
        self.assertTrue(rtn['total_iterations_gmic_removed'] == 1)
        self.assertTrue(rtn['total_number_gmic_removed'] == 1)
        self.assertTrue(rtn['total_cut_generation_iterations'] == 12)

    def test_bound_lp_fails_asserts(self):
        node = self.make_multivariable_node()
        self.assertRaisesRegex(AssertionError, 'x must be our only variable',
                               node._bound_lp)
    
    def test_bound_lp_integer(self):
        node = BaseNode(no_branch.lp, no_branch.integerIndices)
        node._bound_lp()
        self.assertTrue(node.objective_value == -2)
        self.assertTrue(all(node.solution == [1, 1, 0]))
        # integer solutions should come back as both lp and mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.mip_feasible)
        self.assertFalse(node.unbounded)

    def test_bound_lp_fractional(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices)
        node._bound_lp()
        self.assertTrue(node.objective_value == -2.75)
        self.assertTrue(all(node.solution == [0, 1.25, 1.5]))
        # fractional solutions should come back as lp but not mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertFalse(node.mip_feasible)
        self.assertFalse(node.unbounded)

    def test_bound_lp_infeasible(self):
        node = BaseNode(infeasible.lp, infeasible.integerIndices)
        node._bound_lp()
        # infeasible problems should come back as neither lp nor mip feasible
        self.assertFalse(node.lp_feasible)
        self.assertFalse(node.mip_feasible)
        self.assertFalse(node.unbounded)
        self.assertTrue(node.solution is None)
        self.assertTrue(node.objective_value == float('inf'))

    def test_bound_lp_unbounded(self):
        node = BaseNode(unbounded.lp, unbounded.integerIndices)
        node._bound_lp()

        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.unbounded)

    def test_cut_generation_iteration_fails_asserts(self):
        node = BaseNode(negative.lp, negative.integerIndices)
        node._bound_lp()
        self.assertRaisesRegex(AssertionError, 'we must have x >= 0',
                               node._cut_generation_iteration)

    def test_cut_generation_iteration(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node._bound_lp()
        obj = node.objective_value
        cuts = {'cut_gomory_0_1_0': (CyLPArray([0, -1, 0]), -2)}

        def patch_bound_lp():
            node.objective_value -= .00001

        # check function calls and check attribute changes
        with patch.object(node, '_bound_lp', new=patch_bound_lp) as bl, \
                patch.object(node, '_remove_slack_cuts') as rsc, \
                patch.object(node, '_generate_cuts',) as gc, \
                patch.object(node, '_select_cuts') as sc:

            gc.return_value = cuts
            node._cut_generation_iteration()

            self.assertTrue(node.cut_generation_iterations == 1)
            self.assertTrue(rsc.called)
            self.assertTrue(gc.called)
            self.assertTrue(sc.called)
            self.assertTrue(obj == node.objective_value + .00001)  # bound_lp called if true
            self.assertTrue(node.cut_generation_stalled)

        # do a normal run just to make sure
        node = BaseNode(self.cut2_std.lp, self.cut2_std.integerIndices)
        node._bound_lp()
        obj = node.objective_value
        constrs = node.lp.nConstraints
        node._cut_generation_iteration(gomory_cuts=True)
        self.assertFalse(node.cut_generation_stalled)
        self.assertTrue(-1.6 > obj - node.objective_value > -1.61)
        self.assertTrue(node.lp.nConstraints > constrs)

    def test_remove_slack_cuts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node.lp.addConstraint(CyLPArray([0, -1, 0]) * node.lp.getVarByName('x') >= -2,
                              'cut_gomory_0_1_0')
        node.lp.addConstraint(CyLPArray([0, -1, 0]) * node.lp.getVarByName('x') >= -1,
                              'cut_gomory_0_2_0')
        node._bound_lp()
        with patch.object(node, '_update_gmic_counts') as ugc:
            node._remove_slack_cuts()
            self.assertRaisesRegex(Exception, 'Constraint "cut_gomory_0_1_0" does not exist',
                                   node.lp.removeConstraint, 'cut_gomory_0_1_0')
            node.lp.removeConstraint('cut_gomory_0_2_0')  # checks second constraint still there
            self.assertTrue(ugc.called)
            self.assertTrue(ugc.call_args.kwargs['operation'] == 'removed')
            self.assertTrue(ugc.call_args.kwargs['cut_idxs'] == ['cut_gomory_0_1_0'])

    def test_udpate_gmic_counts_fails_asserts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        self.assertRaisesRegex(AssertionError, 'not a single string itself',
                               node._update_gmic_counts, cut_idxs='cut_gmic_1',
                               operation='added')
        self.assertRaisesRegex(AssertionError, 'should be str',
                               node._update_gmic_counts, cut_idxs=[5],
                               operation='added')
        self.assertRaisesRegex(AssertionError, 'operation must be "added"',
                               node._update_gmic_counts, cut_idxs=['cut_gmic_1'],
                               operation='add')

    def test_udpate_gmic_counts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        cut_idxs = ['cut_gomory_1_1_1', 'cut_gomory_1_1_2', 'cut_cglp_1_1']
        for operation in ['added', 'created', 'removed']:
            node._update_gmic_counts(cut_idxs=cut_idxs, operation=operation)
            self.assertTrue(getattr(node, f'iterations_gmic_{operation}') == 1)
            self.assertTrue(getattr(node, f'number_gmic_{operation}') == 2)

    def test_generate_cuts_fails_asserts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node._bound_lp()
        self.assertRaisesRegex(AssertionError, 'gomory_cuts is boolean',
                               node._generate_cuts, gomory_cuts='False')

    def test_generate_cuts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices, idx=0)
        node._bound_lp()

        with patch.object(node, '_find_gomory_cuts') as fgc, \
                patch('simple_mip_solver.nodes.base_node.numerically_safe_cut') as nsc, \
                patch.object(node, '_update_gmic_counts') as ugc:
            fgc.return_value = {0: (CyLPArray([0, -1, 0]), -2)}
            nsc.return_value = (CyLPArray([0, -1, 0]), -2)

            cut_pool = node._generate_cuts(gomory_cuts=True)
            self.assertTrue(fgc.called)
            self.assertTrue(all(nsc.call_args.kwargs['pi'] == CyLPArray([0, -1, 0])))
            self.assertTrue(nsc.call_args.kwargs['pi0'] == -2)
            self.assertTrue(nsc.call_args.kwargs['estimate'] == 'over')
            self.assertTrue(all(cut_pool['cut_gomory_0_0_0'][0] == CyLPArray([0, -1, 0])))
            self.assertTrue(cut_pool['cut_gomory_0_0_0'][1] == -2)
            self.assertTrue(ugc.called)
            self.assertTrue(ugc.call_args.kwargs['operation'] == 'created')
            self.assertTrue(ugc.call_args.kwargs['cut_idxs'] == cut_pool)

        self.assertFalse(node._generate_cuts(gomory_cuts=False))

    def test_select_cuts_fails_asserts(self):
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node._bound_lp()
        self.assertRaisesRegex(AssertionError, 'max_nonzero_coefs must be positive int',
                               node._select_cuts, max_nonzero_coefs=0)
        self.assertRaisesRegex(AssertionError, 'parallel_cut_tolerance must be number in \(0, 90\]',
                               node._select_cuts, parallel_cut_tolerance=100)

    def test_select_cuts(self):
        cuts = {
            'cut_1': (CyLPArray([-1, -1, -1]), -2),  # option to pick off for too many nonzero
            'cut_2': (CyLPArray([-1, 0, -1]), -1),  # keep
            'cut_3': (CyLPArray([0, -1, 0]), -1),  # keep
            'cut_4': (CyLPArray([0, 0, 0]), 0),  # pick off for all zero
            'cut_5': (CyLPArray([-99, 0, -101]), -110),  # option to pick off for too parallel
            'cut_6': (CyLPArray([-1, 0, 0]), -2)  # pick off for not enough depth
        }
        correct_cuts = {'cut_1', 'cut_2', 'cut_3'}
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node._bound_lp()

        node.cut_pool = cuts
        with patch.object(node, '_update_gmic_counts') as ugc:
            added_cuts = node._select_cuts()
            self.assertTrue(set(added_cuts.keys()) == correct_cuts)
            for idx in correct_cuts:
                # check each cut was added - will fail if not
                node.lp.removeConstraint(idx)
            self.assertTrue(set(node.cut_pool.keys()) == {'cut_4', 'cut_5', 'cut_6'})
            self.assertTrue(ugc.called)
            self.assertTrue(ugc.call_args.kwargs['operation'] == 'added')
            self.assertTrue(ugc.call_args.kwargs['cut_idxs'] == added_cuts)

    def test_select_cuts_different_tolerances(self):
        cuts = {
            'cut_1': (CyLPArray([-1, -1, -1]), -2),  # option to pick off for too many nonzero
            'cut_2': (CyLPArray([-1, 0, -1]), -1),  # keep
            'cut_3': (CyLPArray([0, -1, 0]), -1),  # keep
            'cut_4': (CyLPArray([0, 0, 0]), 0),  # pick off for all zero
            'cut_5': (CyLPArray([-99, 0, -101]), -110),  # option to pick off for too parallel
            'cut_6': (CyLPArray([-1, 0, 0]), -2)  # pick off for not enough depth
        }
        correct_cuts = {'cut_2', 'cut_3', 'cut_5'}
        node = BaseNode(self.small_branch_std.lp, self.small_branch_std.integerIndices)
        node._bound_lp()
        node.cut_pool = cuts
        with patch.object(node, '_update_gmic_counts') as ugc:
            added_cuts = node._select_cuts(max_nonzero_coefs=2, parallel_cut_tolerance=.0001)
            self.assertTrue(set(added_cuts.keys()) == correct_cuts)
            self.assertTrue(set(node.cut_pool.keys()) == {'cut_1', 'cut_4', 'cut_6'})
            for idx in correct_cuts:
                node.lp.removeConstraint(idx)  # will fail if the constraint not present
            self.assertTrue(ugc.called)
            self.assertTrue(ugc.call_args.kwargs['operation'] == 'added')
            self.assertTrue(ugc.call_args.kwargs['cut_idxs'] == added_cuts)

    def test_find_gomory_cuts(self):
        node = BaseNode(lp=self.cut3_std.lp, integer_indices=self.cut3_std.integerIndices)
        node._bound_lp()
        cuts = node._find_gomory_cuts()
        self.assertTrue(len(cuts) == 1)
        self.assertTrue(np.max(np.abs(cuts[0][0] - np.array([-5, -10]))) < .0001)
        self.assertTrue(isclose(cuts[0][1], -5, abs_tol=.01))

        mock_pth = 'simple_mip_solver.nodes.base_node.BaseNode.basic_variable_indices'
        with patch(mock_pth, new_callable=PropertyMock) as bvi:
            bvi.return_value = [0, 1]
            cuts = node._find_gomory_cuts()
            self.assertFalse(cuts)

    def test_tableau(self):
        node = BaseNode(lp=self.cut3_std.lp, integer_indices=self.cut3_std.integerIndices)
        node._bound_lp()
        expected_tableau = np.array([[1, 2, 0, 0, 1],
                                     [0, -2, 1, 0, -3],
                                     [0, 0, 0, 1, -5]])
        self.assertTrue(np.max(abs(expected_tableau - node.tableau)) < .0001)

        mock_pth = 'simple_mip_solver.nodes.base_node.BaseNode.basic_variable_indices'
        with patch(mock_pth, new_callable=PropertyMock) as bvi:
            bvi.return_value = [0, 1]
            self.assertFalse(node.tableau)

    def test_basic_variable_indices(self):
        node = BaseNode(lp=self.cut3_std.lp, integer_indices=self.cut3_std.integerIndices)
        node._bound_lp()
        self.assertTrue(all(node.basic_variable_indices == [0, 2, 3]))

    def test_base_branch_fails_asserts(self):
        # branching with multiple named variables in lp should fail
        node = self.make_multivariable_node()
        self.assertRaisesRegex(AssertionError, 'x must be our only variable',
                               node._base_branch, branch_idx=1)

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
                self.assertTrue(n.dual_bound == node.objective_value)
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
        self.assertFalse(node._is_fractional(5.999999999999))
        self.assertFalse(node._is_fractional(5.000000000001))

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
        self.assertTrue(q.get().dual_bound < 0)
        self.assertTrue(q.get().dual_bound == 0)

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

    def test_x_only_variable(self):
        node = BaseNode(small_branch.lp, small_branch.integerIndices, 0)
        self.assertTrue(node._x_only_variable)
        node = self.make_multivariable_node()
        self.assertFalse(node._x_only_variable)
        
    def make_multivariable_node(self):
        s = self.cut2_std.lp.addVariable('s', 1)
        self.cut2_std.lp += s >= CyLPArray([0])
        return BaseNode(self.cut2_std.lp, self.cut2_std.integerIndices, 0)

    Node = BaseNode  # node type to use in base_test_models

    def test_models(self):
        self.base_test_models()


if __name__ == '__main__':
    unittest.main()
