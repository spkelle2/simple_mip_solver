from cylp.cy import CyClpSimplex
import unittest

from simple_mip_solver import Node
from example_models import no_branch, small_branch, infeasible


class TestNode(unittest.TestCase):

    def test_init(self):
        node = Node(small_branch)
        self.assertTrue(node.model, 'should get a model on proper instantiation')
        self.assertTrue(node.lower_bound == -float('inf'))
        self.assertFalse(node.objective_value, 'should have obj but empty')
        self.assertFalse(node.solution, 'should have solution but empty')
        self.assertFalse(node.lp_feasible, 'should have lp_feasible but empty')
        self.assertFalse(node.mip_feasible, 'should have mip_feasible but empty')

    def test_init_fails_asserts(self):
        self.assertRaises(AssertionError, Node, CyClpSimplex())
        self.assertRaises(AssertionError, Node, small_branch, 'five')

    def test_solve_integer(self):
        node = Node(no_branch)
        node.solve()
        self.assertTrue(node.objective_value == -2)
        self.assertTrue(all(node.solution == [1, 1, 0]))
        # integer solutions should come back as both lp and mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.mip_feasible)

    def test_solve_fractional(self):
        node = Node(small_branch)
        node.solve()
        self.assertTrue(node.objective_value == -3)
        self.assertTrue(all(node.solution == [0, 1.5, 1.5]))
        # fractional solutions should come back as lp but not mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertFalse(node.mip_feasible)

    def test_solve_infeasible(self):
        node = Node(infeasible)
        node.solve()
        # infeasible problems should come back as neither lp nor mip feasible
        self.assertFalse(node.lp_feasible)
        self.assertFalse(node.mip_feasible)

    def test_most_fractional_index(self):
        node = Node(no_branch)
        node.solve()
        self.assertFalse(node.most_fractional_index,
                         'int solution should have no fractional index')

        node = Node(small_branch)
        node.solve()
        self.assertTrue(node.most_fractional_index == 2)


if __name__ == '__main__':
    unittest.main()