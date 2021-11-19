from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray, CyLPExpr
import inspect
import os

# so pulp and pyomo don't believe reading in flat files is a worthwhile feature
# so we don't have much of an option here but to use some sort of commercial solver
import simple_mip_solver.nodes.bound.cutting_plane

try:  # if you don't have gurobipy installed, all tests except those using gurobi will run
    import gurobipy as gu
except ImportError:
    gu = None
from math import isclose
import numpy as np
import unittest
from unittest.mock import patch

from simple_mip_solver import CuttingPlaneBoundNode, BaseNode, BranchAndBound
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
                               CuttingPlaneBoundNode, lp=cut1.lp, integer_indices=cut1.integerIndices)
        l = self.cut1_std.lp.variablesLower.copy()
        l[0] = -10
        self.cut1_std.lp.variablesLower = l
        self.assertRaisesRegex(AssertionError, 'must have x >= 0 for all variables',
                               CuttingPlaneBoundNode, lp=self.cut1_std.lp,
                               integer_indices=self.cut1_std.integerIndices)
        s = self.cut2_std.lp.addVariable('s', 1)
        self.cut2_std.lp += s >= CyLPArray([0])
        self.assertRaisesRegex(AssertionError, 'x must be our only variable',
                               CuttingPlaneBoundNode, lp=self.cut2_std.lp,
                               integer_indices=self.cut2_std.integerIndices)

    def test_init_fails_asserts2(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        self.assertRaisesRegex(AssertionError, 'cglp must be CutGeneratingLP instance',
                               CuttingPlaneBoundNode, cglp=cglp.lp, cut_generating_lp=True,
                               cglp_cumulative_constraints=True, lp=bb.root_node.lp,
                               integer_indices=self.cut1_std.integerIndices)

    def test_init(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        n = CuttingPlaneBoundNode(lp=bb.root_node.lp, integer_indices=self.cut1_std.integerIndices,
                                  cglp=cglp)
        self.assertTrue(cglp is n.cglp)
        self.assertTrue(n.cglp_starting_basis is None)

    def test_branch_fails_asserts(self):
        n = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        n.bound()
        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_constraints is bool',
                               n.branch, cglp_cumulative_constraints=0)
        self.assertRaisesRegex(AssertionError, 'cglp_cumulative_bounds is bool',
                               n.branch, cglp_cumulative_bounds=0)

    def test_branch(self):

        # no cglp
        n = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        n.bound()
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
        n = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        n.bound(cut_generating_lp=True)
        with patch('simple_mip_solver.nodes.bound.cutting_plane.CutGeneratingLP',
                   spec=CutGeneratingLP) as cm, patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            rtn = n.branch(cglp_cumulative_constraints=True, cglp_cumulative_bounds=True)

            # check cglp instantiation
            self.assertTrue(cm.called)
            args, kwargs = cm.call_args
            self.assertTrue((kwargs['A'] == n.lp.coefMatrix).toarray().all())
            self.assertTrue((kwargs['b'] == n.lp.constraintsLower).all())
            self.assertTrue((kwargs['var_lb'] == n.lp.variablesLower).all())
            self.assertTrue((kwargs['var_ub'] == n.lp.variablesUpper).all())

            # check branch call
            self.assertTrue(bm.called)
            args, kwargs = bm.call_args
            self.assertTrue(isinstance(kwargs['cglp'], CutGeneratingLP))
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

        # static cglp
        n = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        n.bound(cut_generating_lp=True)
        with patch.object(BaseNode, 'branch') as bm:
            bm.return_value = 'rtn'
            n.branch()
            # check branch call
            self.assertTrue(bm.called)
            args, kwargs = bm.call_args
            self.assertTrue(kwargs['cglp'] is cglp)
            self.assertTrue((kwargs['cglp_starting_basis'][0] == cglp.lp.getBasisStatus()[0]).all())
            self.assertTrue((kwargs['cglp_starting_basis'][1] == cglp.lp.getBasisStatus()[1]).all())
            self.assertTrue(rtn == 'rtn', 'should just return what parent branch does')

    # todo: resume here
    def test_bound_fails_asserts(self):
        n = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        self.assertRaisesRegex(AssertionError, 'optimized_gomory_cuts is bool',
                               n.branch, optimized_gomory_cuts=0)
        self.assertRaisesRegex(AssertionError, 'cut_generating_lp is bool',
                               n.branch, cut_generating_lp=0)
        self.assertRaisesRegex(AssertionError, 'cglp attribute must be defined',
                               n.branch, cut_generating_lp=True)

    def test_bound(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)

        # check function calls
        node = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices,
                                     cglp=cglp)
        node_inf = CuttingPlaneBoundNode(self.infeasible_std.lp, self.infeasible_std.integerIndices,
                                         cglp=cglp)
        node_opt = CuttingPlaneBoundNode(self.no_branch_std.lp, self.no_branch_std.integerIndices,
                                         cglp=cglp)
        with patch.object(BaseNode, 'bound') as b:
            node.bound()
            self.assertTrue(b.called)

        with patch.object(node, '_add_optimized_gomory_cuts') as aogc, \
                patch.object(node, '_add_cglp_cut') as acc:
            node_inf.bound()
            self.assertFalse(aogc.called, 'infeasible shouldnt call')
            self.assertFalse(acc.called, 'infeasible shouldnt call')
            node_opt.bound()
            self.assertFalse(aogc.called, 'mip optimal shouldnt call')
            self.assertFalse(acc.called, 'mip optimal shouldnt call')
            node.bound()
            self.assertFalse(aogc.called, 'false shouldnt call')
            self.assertFalse(acc.called, 'false shouldnt call')
            node.bound(optimized_gomory_cuts=True, cut_generating_lp=True)
            self.assertTrue(aogc.called)
            self.assertTrue(acc.called)

        # check return
        node = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        rtn = node.bound(optimized_gomory_cuts=True, cut_generating_lp=True)
        self.assertTrue(isinstance(rtn, dict), 'should return dict')
        self.assertTrue(len(rtn['cuts'] == 1))
        self.assertTrue(isinstance(rtn['cuts'][0], CyLPExpr), 'should return CyLPExpr')

    def test_add_optimized_gomory_cuts(self):
        node = CuttingPlaneBoundNode(self.cut2_std.lp, self.cut2_std.integerIndices)
        # check function calls
        with patch.object(node, '_find_gomory_cuts') as fgb, \
                patch.object(node, '_optimize_cut') as oc:
            fgb.return_value = [(CyLPArray([-3.3, -1.2]), -24.1),
                                (CyLPArray([-1.2, -1.8]), -15.4)]
            oc.return_value = -50
            node._add_optimized_gomory_cuts()
            self.assertTrue(fgb.called)
            self.assertTrue(oc.call_count == 2)
            self.assertTrue(len(node.lp.constraints) == 3)

        # check returns
        rtn = super(CuttingPlaneBoundNode, node).bound()
        self.assertTrue(isinstance(rtn, dict), 'should return dict')

    def test_find_gomory_cuts(self):
        node = CuttingPlaneBoundNode(self.cut2_std.lp, self.cut2_std.integerIndices)
        super(CuttingPlaneBoundNode, node).bound()
        cuts = node._find_gomory_cuts()
        self.assertTrue(len(cuts) == 2)
        self.assertTrue(np.max(np.abs(cuts[0][0] - np.array([-3.3, -1.2]))) < node._epsilon)
        self.assertTrue(isclose(cuts[0][1], -24.1, abs_tol=.01))
        self.assertTrue(np.max(np.abs(cuts[1][0] - np.array([-1.2, -1.8]))) < node._epsilon)
        self.assertTrue(isclose(cuts[1][1], -15.4, abs_tol=.01))

    def test_optimize_cut(self):
        node = CuttingPlaneBoundNode(self.cut2_std.lp, self.cut2_std.integerIndices)
        super(CuttingPlaneBoundNode, node).bound()
        cuts = node._find_gomory_cuts()
        pi0 = node._optimize_cut(cuts[0][0])
        self.assertTrue(isclose(pi0, -22.5, abs_tol=.01))
        pi0 = node._optimize_cut(cuts[1][0])
        self.assertTrue(isclose(pi0, -15, abs_tol=.01))

    def test_add_cglp_cut_fails_asserts(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = CuttingPlaneBoundNode(lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices, cglp=cglp)
        super(CuttingPlaneBoundNode, node).bound()

        self.assertRaisesRegex(AssertionError, 'optimized_gomory_cuts is bool',
                               node.branch, optimized_gomory_cuts=0)
        self.assertRaisesRegex(AssertionError, 'cut_generating_lp is bool',
                               node.branch, cut_generating_lp=0)

        # check function calls
        with patch.object(node.cglp, 'solve') as s:
            s.return_value = (None, None)
            self.assertRaisesRegex(AssertionError, 'should get solution', node._add_cglp_cut)

    def test_add_cglp_cut(self):
        bb = BranchAndBound(self.cut1_std)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        node = CuttingPlaneBoundNode(cglp=cglp, lp=self.cut1_std.lp, integer_indices=self.cut1_std.integerIndices)
        super(CuttingPlaneBoundNode, node).bound()
        pi, pi0 = cglp.solve(x_star=CyLPArray(node.lp.solution),
                             starting_basis=node.cglp_starting_basis)

        # check function calls
        with patch.object(node.lp, 'addConstraint') as ac, \
                patch.object(node.cglp, 'solve') as s:
            s.return_value = (pi, pi0)
            node._add_cglp_cut()
            self.assertTrue(s.called)
            self.assertTrue(ac.called)
            self.assertTrue(ac.call_args.args == [pi * node.lp.getVarByName('x') >= pi0])

        # check returns
        cut = node._add_cglp_cut(cglp_cumulative_constraints=True,
                                 cglp_cumulative_bounds=True)
        self.assertTrue(isinstance(cut, CyLPExpr), 'should return CyLPExpr')
        cut = node._add_cglp_cut()
        self.assertFalse(cut)

    # test after fixing branch and bound tests
    def test_models(self):
        self.assertTrue(gu, 'gurobipy needed for this test')
        fldr = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(generate_random_variety))),
            'example_models'
        )

        # todo: fix gomory cuts and add their check here
        kwargs_list = [
            {'cut_generating_lp': True},
            {'cut_generating_lp': True, 'cglp_cumulative_constraints': True},
            {'cut_generating_lp': True, 'cglp_cumulative_bounds': True},
            {'cut_generating_lp': True, 'cglp_cumulative_constraints': True,
             'cglp_cumulative_bounds': True},
        ]
        for j, kwargs in enumerate(kwargs_list):
            for i, file in enumerate(os.listdir(fldr)):
                print(f'running test {(i + 1)*(j + 1)}')
                pth = os.path.join(fldr, file)
                model = MILPInstance(file_name=pth)
                cglp_bb = BranchAndBound(model, node_limit=8)
                cglp_bb.solve()
                cglp = CutGeneratingLP(cglp_bb, cglp_bb.root_node.idx)
                bb = BranchAndBound(model, CuttingPlaneBoundNode, cglp=cglp, **kwargs)
                bb.solve()
                gu_mdl = gu.read(pth)
                gu_mdl.setParam(gu.GRB.Param.LogToConsole, 0)
                gu_mdl.optimize()
                if not isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01):
                    print(f'different for {file}')
                    print(f'mine: {bb.objective_value}')
                    print(f'gurobi: {gu_mdl.objVal}')
                self.assertTrue(isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01),
                                f'different for {file}')


if __name__ == '__main__':
    unittest.main()
