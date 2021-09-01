from coinor.cuppy.milpInstance import MILPInstance

from math import isclose
import numpy as np
import unittest

from simple_mip_solver import CuttingPlane
from simple_mip_solver.algorithms.utils import Utils
from test_simple_mip_solver.example_models import cut1, infeasible, unbounded, cut2


class TestNode(unittest.TestCase):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'cut1_std': cut1, 'cut2_std': cut2,
                        'infeasible_std': infeasible, 'unbounded': unbounded}.items():
            new_m = MILPInstance(A=m.A, b=m.b, c=m.lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(m.lp.objective))
            new_m = Utils._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    def test_init(self):
        cp = CuttingPlane(self.cut1_std)
        self.assertTrue(isinstance(cp, Utils))
        self.assertTrue(cp._max_iters)
        self.assertFalse(cp._iterations)
        self.assertFalse(cp.solution)
        self.assertTrue(cp.status == 'unsolved')
        self.assertFalse(cp.objective_value)

    def test_init_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'max_iter is positive', CuttingPlane,
                               self.cut1_std, max_iters=0)

    def test_solve_optimal(self):
        cp = CuttingPlane(self.cut1_std)
        cp.solve()
        self.assertTrue(cp.status == 'optimal')
        self.assertTrue(np.max(np.abs(cp.solution - np.array([7, 5]))) < cp._root_node._epsilon)
        self.assertTrue(isclose(cp.objective_value, -5, abs_tol=.01))

