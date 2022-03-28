from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray
import inspect
import os
from scipy.sparse import csc_matrix

# so pulp and pyomo don't believe reading in flat files is a worthwhile feature
# so we don't have much of an option here but to use some sort of commercial solver

try:  # if you don't have gurobipy installed, all tests except those using gurobi will run
    import gurobipy as gu
except ImportError:
    gu = None
from itertools import product
from math import isclose
import numpy as np
import unittest
from unittest.mock import patch

from simple_mip_solver import DisjunctiveCutBoundNode, BaseNode, BranchAndBound
from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from test_simple_mip_solver.example_models import cut1, infeasible, no_branch, \
    cut2, lift_project, generate_random_variety
from test_simple_mip_solver.helpers import TestModels


class TestNode(TestModels):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'cut1_std': cut1, 'cut2_std': cut2,
                        'infeasible_std': infeasible, 'no_branch_std': no_branch,
                        'lift_project_std': lift_project}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = BaseAlgorithm._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    def test_init_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'must have Ax >= b',
                               DisjunctiveCutBoundNode, lp=cut1.lp, integer_indices=cut1.integerIndices)
        l = self.cut1_std.lp.variablesLower.copy()
        l[0] = -10
        self.cut1_std.lp.variablesLower = l
        self.assertRaisesRegex(AssertionError, 'must have x >= 0 for all variables',
                               DisjunctiveCutBoundNode, lp=self.cut1_std.lp,
                               integer_indices=self.cut1_std.integerIndices)

    def test_init_fails_asserts2(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        self.assertRaisesRegex(AssertionError, 'cglp must be CutGeneratingLP instance',
                               DisjunctiveCutBoundNode, cglp=cglp.lp, cut_generating_lp=True,
                               cglp_cumulative_constraints=True, lp=bb.root_node.lp,
                               integer_indices=self.cut1_std.integerIndices)

    def test_init(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        n = DisjunctiveCutBoundNode(lp=bb.root_node.lp, integer_indices=self.cut1_std.integerIndices,
                                    cglp=cglp)
        self.assertTrue(cglp is n.cglp)
        self.assertTrue(n.prev_cglp_basis is None)
        self.assertFalse(n.cglp_cut_added)
        self.assertTrue(n.previous_cglp_added)
        self.assertTrue(n.cglp_name_pattern)
        self.assertFalse(n.sharable_cuts)
        self.assertFalse(n.cglp_calls)

    def test_bound(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)

        # check function calls and returns
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp,
                                       integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp)

        with patch.object(BaseNode, 'bound') as b:

            # no cuts sharable
            b.return_value = {}
            rtn = node.bound()
            self.assertTrue(b.called)
            self.assertFalse(rtn)

            # some cut sharable
            node.sharable_cuts = {'cut_cglp_1_1': 'some cut'}
            rtn = node.bound()
            self.assertTrue(b.called)
            self.assertTrue(rtn['cuts'] == node.sharable_cuts)

    def test_generate_cuts_fails_asserts(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        self.assertRaisesRegex(AssertionError, 'max_cglp_calls is a nonnegative integer',
                               node._generate_cuts, max_cglp_calls=-1)

    def test_generate_cuts(self):
        bb = BranchAndBound(self.cut1_std, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp,
                                       integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp, idx=0)

        # check function calls
        with patch('simple_mip_solver.nodes.base_node.BaseNode._generate_cuts') as gc, \
                patch.object(node.cglp, 'solve') as s, \
                patch('simple_mip_solver.nodes.bound.disjunctive_cut.numerically_safe_cut') as nsc, \
                patch.object(node, '_get_cglp_starting_basis') as gcsb:
            gc.return_value = {}
            s.return_value = (CyLPArray([0, 1]), 1)
            nsc.return_value = (CyLPArray([0, 1]), 1)
            gcsb.return_value = None

            # previous cglp added and max_cglp_calls more
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 1)
            self.assertTrue(s.call_count == 1)
            self.assertTrue(s.call_args.kwargs['starting_basis'] is None)
            self.assertTrue(gcsb.call_count == 1)
            self.assertTrue(nsc.call_count == 1)
            self.assertTrue({cut for cut in cut_pool} == {'cut_cglp_0_0'})

        with patch('simple_mip_solver.nodes.base_node.BaseNode._generate_cuts') as gc, \
                patch.object(node.cglp, 'solve') as s, \
                patch('simple_mip_solver.nodes.bound.disjunctive_cut.numerically_safe_cut') as nsc, \
                patch.object(node, '_get_cglp_starting_basis') as gcsb:
            gc.return_value = {}

            # previous cglp not added but max_cglp_calls more
            node.previous_cglp_added = False
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 1)
            self.assertFalse(s.called)
            self.assertFalse(gcsb.called)
            self.assertFalse(nsc.called)
            self.assertFalse(cut_pool)

            # previous cglp added but max_cglp_calls less
            node.cut_generation_iterations += 1
            node.previous_cglp_added = True
            cut_pool = node._generate_cuts(max_cglp_calls=0)
            self.assertFalse(s.called)
            self.assertFalse(gcsb.called)
            self.assertFalse(nsc.called)
            self.assertFalse(cut_pool)

    def test_generate_cuts_gets_warm_start_right(self):
        bb = BranchAndBound(self.cut1_std, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, cglp=cglp, idx=0,
                                       integer_indices=self.cut1_std.integerIndices)

        # check function calls
        with patch('simple_mip_solver.nodes.base_node.BaseNode._generate_cuts') as gc, \
                patch.object(node.cglp, 'solve') as s, \
                patch('simple_mip_solver.nodes.bound.disjunctive_cut.numerically_safe_cut') as nsc, \
                patch.object(node, '_get_cglp_starting_basis') as gcsb:
            gc.return_value = {}
            s.return_value = (CyLPArray([0, 1]), 1)
            nsc.return_value = (CyLPArray([0, 1]), 1)
            gcsb.return_value = None

            # both true and max_cglp_calls more
            node._generate_cuts(cut_generating_lp=True, warm_start_cglp=True)
            self.assertTrue(gcsb.call_args.kwargs['warm_start_cglp'])

            node._generate_cuts(cut_generating_lp=True, warm_start_cglp=False)
            self.assertFalse(gcsb.call_args.kwargs['warm_start_cglp'])

    def test_get_cglp_starting_basis_fails_asserts(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        self.assertRaisesRegex(AssertionError, 'warm_start_cglp is boolean',
                               node._get_cglp_starting_basis, warm_start_cglp=None)

    def test_get_cglp_starting_basis(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp, prev_cglp_basis=(np.array([5]), np.array([5])))

        # don't warm start
        basis = node._get_cglp_starting_basis(warm_start_cglp=False)
        self.assertTrue((basis[0] == [3, 3]).all())
        self.assertTrue((basis[1] == [1]*5).all())

        # warm start the cold start
        self.assertFalse(node._get_cglp_starting_basis(warm_start_cglp=True))

        # warm start subsequent start
        node.cut_generation_iterations = 1
        basis = node._get_cglp_starting_basis(warm_start_cglp=True)
        self.assertTrue(basis[0] == 5)
        self.assertTrue(basis[1] == 5)

    def test_select_cuts_fails_asserts(self):
        bb = BranchAndBound(self.cut1_std, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)

        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_constraints is bool',
                               node._select_cuts, cglp_cumulative_constraints=0)
        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_bounds is bool',
                               node._select_cuts, cglp_cumulative_bounds=0)

    def test_select_cuts(self):
        bb = BranchAndBound(self.cut1_std, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp,
                                       integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp, idx=0)

        # check function calls
        with patch('simple_mip_solver.nodes.base_node.BaseNode._select_cuts') as sc:
            # no cglp cut
            sc.return_value = {'cut_gomory_0_0_0': (CyLPArray([1, 0]), 1)}
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 1)
            self.assertFalse(node.cglp_cut_added)
            self.assertFalse(node.previous_cglp_added)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_gomory_0_0_0'})

            # yes cglp cut
            sc.return_value = {'cut_cglp_0_0': (CyLPArray([0, 1]), 1),
                               'cut_gomory_0_0_0': (CyLPArray([1, 0]), 1)}

            # both true
            added_cuts = node._select_cuts(cglp_cumulative_constraints=True,
                                           cglp_cumulative_bounds=True)
            self.assertTrue(sc.call_count == 2)
            self.assertTrue(node.cglp_cut_added)
            self.assertTrue(node.previous_cglp_added)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            # one false
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=True)
            self.assertTrue(sc.call_count == 3)
            self.assertTrue(node.cglp_cut_added)
            self.assertTrue(node.previous_cglp_added)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            added_cuts = node._select_cuts(cglp_cumulative_constraints=True,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 4)
            self.assertTrue(node.cglp_cut_added)
            self.assertTrue(node.previous_cglp_added)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            # both false
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 5)
            self.assertTrue(node.cglp_cut_added)
            self.assertTrue(node.previous_cglp_added)
            self.assertTrue(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

    def test_branch_fails_asserts(self):
        n = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        n.bound()
        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_constraints is bool',
                               n.branch, cglp_cumulative_constraints=0)
        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_bounds is bool',
                               n.branch, cglp_cumulative_bounds=0)

    def test_branch(self):
        bb = BranchAndBound(self.cut1_std, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)

        # no cglp
        n = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        n.bound()
        with patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            rtn = n.branch()
            self.assertTrue(bm.called)
            self.assertFalse(bm.call_args.args)
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

        # cglp cut not added
        n = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices,
                                    cglp=cglp)
        with patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            rtn = n.branch()
            self.assertTrue(bm.called)
            self.assertFalse(bm.call_args.args)
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

        # growing cglp
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        n = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices,
                                    cglp=cglp)
        n._bound_lp()
        n.cglp_cut_added = True
        with patch('simple_mip_solver.nodes.bound.disjunctive_cut.CutGeneratingLP',
                   spec=CutGeneratingLP) as cm, patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            rtn = n.branch(cglp_cumulative_constraints=True, cglp_cumulative_bounds=True)

            # check cglp instantiation
            self.assertTrue(cm.called)
            args, kwargs = cm.call_args
            self.assertTrue((kwargs['A'] == n.lp.coefMatrix).toarray().all())
            self.assertTrue(isinstance(kwargs['A'], csc_matrix))
            self.assertTrue((kwargs['b'] == n.lp.constraintsLower).all())
            self.assertTrue(isinstance(kwargs['b'], CyLPArray))
            self.assertTrue((kwargs['var_lb'] == n.lp.variablesLower).all())
            self.assertTrue(isinstance(kwargs['var_lb'], CyLPArray))
            self.assertTrue((kwargs['var_ub'] == n.lp.variablesUpper).all())
            self.assertTrue(isinstance(kwargs['var_ub'], CyLPArray))

            # check branch call
            self.assertTrue(bm.called)
            args, kwargs = bm.call_args
            self.assertTrue(isinstance(kwargs['cglp'], CutGeneratingLP))
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

        # static cglp
        n = DisjunctiveCutBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices,
                                    cglp=cglp)
        n._bound_lp()
        n.cglp_cut_added = True
        with patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            n.branch()
            # check branch call
            self.assertTrue(bm.called)
            args, kwargs = bm.call_args
            self.assertTrue(kwargs['cglp'] is cglp)
            self.assertTrue((kwargs['prev_cglp_basis'][0] == cglp.lp.getBasisStatus()[0]).all())
            self.assertTrue((kwargs['prev_cglp_basis'][1] == cglp.lp.getBasisStatus()[1]).all())
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

    # test after fixing branch and bound tests
    def test_models(self):
        ratio_run = .05
        count_different = 0
        dif = {}
        self.assertTrue(gu, 'gurobipy needed for this test')
        fldr = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(generate_random_variety))),
            'example_models'
        )
        kwarg_values = product(*([[True, False] for _ in range(5)] + [[1, None]]))
        kwargs_list = [
            {'cut_generating_lp': cglp_bool, 'cglp_cumulative_constraints': cc_bool,
            'cglp_cumulative_bounds': cb_bool, 'gomory_cuts': gc_bool} for
            (cglp_bool, cc_bool, cb_bool, gc_bool, ws_cglp, max_cglp) in kwarg_values
        ]
        num_kwargs = len(kwargs_list)
        num_fldrs = len(os.listdir(fldr))
        for j, kwargs in enumerate(kwargs_list):
            for i, file in enumerate(os.listdir(fldr)):
                if np.random.uniform() > ratio_run:
                    continue
                print(f'running test {(i + 1) + j * num_fldrs} of {num_kwargs * num_fldrs}')
                pth = os.path.join(fldr, file)

                # check gurobi
                gu_mdl = gu.read(pth)
                gu_mdl.setParam(gu.GRB.Param.LogToConsole, 0)
                gu_mdl.optimize()

                # check ours
                model = MILPInstance(file_name=pth)
                cglp_bb = BranchAndBound(model, node_limit=8, gomory_cuts=kwargs['gomory_cuts'])
                cglp_bb.solve()
                cglp = CutGeneratingLP(cglp_bb, cglp_bb.root_node.idx)
                bb = BranchAndBound(model, DisjunctiveCutBoundNode, cglp=cglp, **kwargs)
                bb.solve()

                if not isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01):
                    print(f'different for {file}')
                    print(f'mine: {bb.objective_value}')
                    print(f'gurobi: {gu_mdl.objVal}')
                    dif[i, j] = {'mine': bb.objective_value, 'gurobi': gu_mdl.objVal}
                    count_different += 1
                # self.assertTrue(isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01),
                #                 f'different for {file}')
        self.assertTrue(count_different/(num_kwargs*num_fldrs) < .02*ratio_run,
                        'try to get less than 2% failure or less')
        print()


if __name__ == '__main__':
    unittest.main()
