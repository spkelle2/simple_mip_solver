from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy as np
from scipy.sparse import csc_matrix
import unittest
from unittest.mock import patch

from simple_mip_solver import DisjunctiveCutBoundNode, BaseNode, BranchAndBound
from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from test_simple_mip_solver.example_models import cut1, infeasible, no_branch, \
    cut2, lift_project
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
        self.assertFalse(n.current_node_added_cglp)
        self.assertTrue(n.previous_cglp_added)
        self.assertTrue(n.cglp_name_pattern)
        self.assertFalse(n.sharable_cuts)
        self.assertFalse(n.number_cglp_created)
        self.assertFalse(n.number_cglp_added)
        self.assertFalse(n.number_cglp_removed)

    def test_bound_fails_asserts(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = DisjunctiveCutBoundNode(lp=bb.root_node.lp, integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node.bound, total_number_cglp_created=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node.bound, total_number_cglp_added=-1)
        self.assertRaisesRegex(AssertionError, 'is nonnegative integer',
                               node.bound, total_number_cglp_removed=-1)

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
            self.assertFalse(rtn['total_number_cglp_created'])
            self.assertFalse(rtn['total_number_cglp_added'])
            self.assertFalse(rtn['total_number_cglp_removed'])

            # some cut sharable
            # check to make sure previous values added to
            for key in rtn:
                rtn[key] += 1
            node.sharable_cuts = {'cut_cglp_1_1': 'some cut'}
            rtn = node.bound(**rtn)
            self.assertTrue(b.called)
            self.assertTrue(rtn['cuts'] == node.sharable_cuts)
            self.assertTrue(rtn['total_number_cglp_created'])
            self.assertTrue(rtn['total_number_cglp_added'])
            self.assertTrue(rtn['total_number_cglp_removed'])

    def test_remove_slack_cuts(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)

        # check function calls and returns
        node = DisjunctiveCutBoundNode(lp=self.cut1_std.lp,
                                       integer_indices=self.cut1_std.integerIndices,
                                       cglp=cglp)
        with patch('simple_mip_solver.nodes.base_node.BaseNode._remove_slack_cuts') as rsc:
            rsc.return_value = ['cut_gomory_0_2_0', 'cut_cglp_0_0']
            idxs = node._remove_slack_cuts()
            self.assertTrue(idxs == ['cut_gomory_0_2_0', 'cut_cglp_0_0'])
            self.assertTrue(rsc.called)
            self.assertTrue(node.number_cglp_removed == 1)

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
            s.side_effect = [(None, None), (CyLPArray([1e-12, 1e-8]), 1e-10),
                             (CyLPArray([0, 1]), 1)]
            nsc.return_value = (CyLPArray([0, 1]), 1)
            gcsb.return_value = None

            # previous cglp added and max_cglp_calls more on no cut
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 1)
            self.assertTrue(s.call_count == 1)
            self.assertTrue(s.call_args.kwargs['starting_basis'] is None)
            self.assertTrue(gcsb.call_count == 1)
            self.assertTrue(nsc.call_count == 0)
            self.assertFalse(cut_pool)
            self.assertFalse(node.number_cglp_created)

            # previous cglp added and max_cglp_calls more on small cut
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 2)
            self.assertTrue(s.call_count == 2)
            self.assertTrue(s.call_args.kwargs['starting_basis'] is None)
            self.assertTrue(gcsb.call_count == 2)
            self.assertTrue(nsc.call_count == 0)
            self.assertFalse(cut_pool)
            self.assertFalse(node.number_cglp_created)

            # previous cglp added and max_cglp_calls more on good cut
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 3)
            self.assertTrue(s.call_count == 3)
            self.assertTrue(s.call_args.kwargs['starting_basis'] is None)
            self.assertTrue(gcsb.call_count == 3)
            self.assertTrue(nsc.call_count == 1)
            self.assertTrue({cut for cut in cut_pool} == {'cut_cglp_0_0'})
            self.assertTrue(node.number_cglp_created == 1)

        with patch('simple_mip_solver.nodes.base_node.BaseNode._generate_cuts') as gc, \
                patch.object(node.cglp, 'solve') as s, \
                patch('simple_mip_solver.nodes.bound.disjunctive_cut.numerically_safe_cut') as nsc, \
                patch.object(node, '_get_cglp_starting_basis') as gcsb:
            gc.return_value = {}

            # previous cglp not added but max_cglp_calls more
            node.previous_cglp_added = False
            cut_pool = node._generate_cuts()
            self.assertTrue(gc.call_count == 1)  # new patch object thats why
            self.assertFalse(s.called)
            self.assertFalse(gcsb.called)
            self.assertFalse(nsc.called)
            self.assertFalse(cut_pool)
            self.assertTrue(node.number_cglp_created == 1)

            # previous cglp added but max_cglp_calls less
            node.cut_generation_iterations += 1
            node.previous_cglp_added = True
            cut_pool = node._generate_cuts(max_cglp_calls=0)
            self.assertFalse(s.called)
            self.assertFalse(gcsb.called)
            self.assertFalse(nsc.called)
            self.assertFalse(cut_pool)
            self.assertTrue(node.number_cglp_created == 1)

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
        self.assertTrue((basis[0] == [3]*12).all())
        self.assertTrue((basis[1] == [1]*4).all())

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
            # no cglp cut from this node
            sc.return_value = {'cut_gomory_0_0_0': (CyLPArray([1, 0]), 1),
                               'cut_cglp_1_0': (CyLPArray([0, 1]), 1)}
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 1)
            self.assertFalse(node.current_node_added_cglp)
            self.assertFalse(node.previous_cglp_added)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue(node.number_cglp_added == 1)
            self.assertTrue({c for c in added_cuts} == {'cut_gomory_0_0_0', 'cut_cglp_1_0'})

            # yes cglp cut
            sc.return_value = {'cut_cglp_0_0': (CyLPArray([0, 1]), 1),
                               'cut_gomory_0_0_0': (CyLPArray([1, 0]), 1)}

            # both true
            added_cuts = node._select_cuts(cglp_cumulative_constraints=True,
                                           cglp_cumulative_bounds=True)
            self.assertTrue(sc.call_count == 2)
            self.assertTrue(node.current_node_added_cglp)
            self.assertTrue(node.previous_cglp_added)
            self.assertTrue(node.number_cglp_added == 2)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            # one false
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=True)
            self.assertTrue(sc.call_count == 3)
            self.assertTrue(node.current_node_added_cglp)
            self.assertTrue(node.previous_cglp_added)
            self.assertTrue(node.number_cglp_added == 3)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            added_cuts = node._select_cuts(cglp_cumulative_constraints=True,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 4)
            self.assertTrue(node.current_node_added_cglp)
            self.assertTrue(node.previous_cglp_added)
            self.assertTrue(node.number_cglp_added == 4)
            self.assertFalse(node.sharable_cuts)
            self.assertTrue({c for c in added_cuts} == {'cut_cglp_0_0', 'cut_gomory_0_0_0'})

            # both false
            added_cuts = node._select_cuts(cglp_cumulative_constraints=False,
                                           cglp_cumulative_bounds=False)
            self.assertTrue(sc.call_count == 5)
            self.assertTrue(node.current_node_added_cglp)
            self.assertTrue(node.previous_cglp_added)
            self.assertTrue(node.number_cglp_added == 5)
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
        n.current_node_added_cglp = True
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
        n.current_node_added_cglp = True
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

    Node = DisjunctiveCutBoundNode

    # test after fixing branch and bound tests
    def test_models(self):
        self.disjunctive_cut_test_models()


if __name__ == '__main__':
    unittest.main()
