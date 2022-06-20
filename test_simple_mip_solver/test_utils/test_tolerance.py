from math import isclose
import unittest

from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from simple_mip_solver.algorithms.branch_and_bound import BranchAndBound
from test_simple_mip_solver.example_models import generate_random_MILPInstance


class TestTolerance(unittest.TestCase):

    def test_tolerances(self):
        milp = generate_random_MILPInstance(numVars=32, numCons=32)
        bb = BranchAndBound(model=milp, Node=PseudoCostBranchNode, pseudo_costs={},
                            min_cut_depth=1e-4, )
        tight_bb = BranchAndBound(model=milp, Node=PseudoCostBranchNode, pseudo_costs={},
                                  max_term=1e16, max_cut_generation_iterations=1000,
                                  cutting_plane_progress_tolerance=1e-8, parallel_cut_tolerance=1)
        bb.solve()
        tight_bb.solve()
        self.assertTrue(isclose(bb.objective_value, tight_bb.objective_value, rel_tol=.01))
        self.assertTrue(tight_bb.evaluated_nodes < bb.evaluated_nodes)
        self.assertTrue(tight_bb.cut_generation_time > bb.cut_generation_time)
