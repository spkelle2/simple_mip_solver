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
import unittest
from unittest.mock import patch

from simple_mip_solver import CuttingPlaneBoundNode, BaseNode
from simple_mip_solver.algorithms.utils import Utils
from test_simple_mip_solver.example_models import cut1, infeasible, no_branch, cut2
from test_simple_mip_solver.helpers import TestModels


class TestNode(TestModels):
    Node = CuttingPlaneBoundNode  # define this for the TestModels attribute

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'cut1_std': cut1, 'cut2_std': cut2,
                        'infeasible_std': infeasible, 'no_branch_std': no_branch}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = Utils._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    # get utils working then test with standardized model
    def test_init(self):
        CuttingPlaneBoundNode(self.cut1_std.lp, self.cut1_std.integerIndices)

    def test_init_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'must have Ax >= b',
                               CuttingPlaneBoundNode, cut1.lp, cut1.integerIndices)
        l = self.cut1_std.lp.variablesLower.copy()
        l[0] = -10
        self.cut1_std.lp.variablesLower = l
        self.assertRaisesRegex(AssertionError, 'must have x >= 0 for all variables',
                               CuttingPlaneBoundNode, self.cut1_std.lp,
                               self.cut1_std.integerIndices)

    def test_bound(self):
        # check function calls
        node = CuttingPlaneBoundNode(self.cut1_std.lp, self.cut1_std.integerIndices)
        node_inf = CuttingPlaneBoundNode(self.infeasible_std.lp, self.infeasible_std.integerIndices)
        node_opt = CuttingPlaneBoundNode(self.no_branch_std.lp, self.no_branch_std.integerIndices)
        with patch.object(BaseNode, 'bound') as b:
            node.bound()
            self.assertTrue(b.called)

        with patch.object(node, '_add_optimized_gomory_cuts') as aogc:
            node_inf.bound(optimized_gomory_cuts=False)
            self.assertFalse(aogc.called, 'infeasible shouldnt call')
            node_opt.bound(optimized_gomory_cuts=False)
            self.assertFalse(aogc.called, 'mip optimal shouldnt call')
            node.bound(optimized_gomory_cuts=False)
            self.assertFalse(aogc.called, 'false shouldnt call')
            node.bound()
            self.assertTrue(aogc.called)

        # check return
        node = CuttingPlaneBoundNode(self.cut1_std.lp, self.cut1_std.integerIndices)
        rtn = node.bound()  # come back and fix after done rest
        self.assertTrue(isinstance(rtn, dict), 'should return dict')

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

    def test_models(self):
        self.base_test_models(standardize_model=True)
        # pass


if __name__ == '__main__':
    unittest.main()