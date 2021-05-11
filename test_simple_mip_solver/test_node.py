import unittest

from simple_mip_solver import Node
from test_simple_mip_solver.example_models import no_branch, small_branch, infeasible


class TestNode(unittest.TestCase):

    def test_init(self):
        node = Node(small_branch.lp, small_branch.integerIndices)
        self.assertTrue(node.lp, 'should get a model on proper instantiation')
        self.assertTrue(node.lower_bound == -float('inf'))
        self.assertFalse(node.objective_value, 'should have obj but empty')
        self.assertFalse(node.solution, 'should have solution but empty')
        self.assertFalse(node.lp_feasible, 'should have lp_feasible but empty')
        self.assertFalse(node.mip_feasible, 'should have mip_feasible but empty')

    def test_init_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'lp must be CyClpSimplex instance',
                               Node, small_branch, small_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'indices must match variables',
                               Node, small_branch.lp, [4])
        self.assertRaisesRegex(AssertionError, 'indices must be distinct',
                               Node, small_branch.lp, [0, 1, 1])
        self.assertRaisesRegex(AssertionError, 'lower bound must be a float or an int',
                               Node, small_branch.lp, small_branch.integerIndices,
                               'five')

    def test_bound_integer(self):
        node = Node(no_branch.lp, no_branch.integerIndices)
        node.bound()
        self.assertTrue(node.objective_value == -2)
        self.assertTrue(all(node.solution == [1, 1, 0]))
        # integer solutions should come back as both lp and mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertTrue(node.mip_feasible)

    def test_bound_fractional(self):
        node = Node(small_branch.lp, small_branch.integerIndices)
        node.bound()
        self.assertTrue(node.objective_value == -2.75)
        self.assertTrue(all(node.solution == [0, 1.25, 1.5]))
        # fractional solutions should come back as lp but not mip feasible
        self.assertTrue(node.lp_feasible)
        self.assertFalse(node.mip_feasible)

    def test_bound_infeasible(self):
        node = Node(infeasible.lp, infeasible.integerIndices)
        node.bound()
        # infeasible problems should come back as neither lp nor mip feasible
        self.assertFalse(node.lp_feasible)
        self.assertFalse(node.mip_feasible)

    def test_most_fractional_index(self):
        node = Node(no_branch.lp, no_branch.integerIndices)
        node.bound()
        self.assertFalse(node.most_fractional_index,
                         'int solution should have no fractional index')

        node = Node(small_branch.lp, small_branch.integerIndices)
        node.bound()
        self.assertTrue(node.most_fractional_index == 2)

    def test_branch_fails_asserts(self):
        # branching before solving should fail
        node = Node(no_branch.lp, no_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'must solve before branching',
                               node.branch)

        # branching on integer feasible node should fail
        node.bound()
        self.assertRaisesRegex(AssertionError, 'must have fractional index to branch',
                               node.branch)

    def test_branch(self):
        node = Node(small_branch.lp, small_branch.integerIndices, -float('inf'))
        node.bound()
        idx = node.most_fractional_index
        ln, rn = node.branch()

        # check each node
        for name, n in {'left': ln, 'right': rn}.items():
            self.assertTrue(all(n.lp.matrix.elements == node.lp.matrix.elements))
            self.assertTrue(all(n.lp.objective == node.lp.objective))
            self.assertTrue(all(n.lp.constraintsLower == node.lp.constraintsLower))
            self.assertTrue(all(n.lp.constraintsUpper == node.lp.constraintsUpper))
            if name == 'left':
                self.assertTrue(all(n.lp.variablesUpper >= [1e10, 1e10, 1]))
                self.assertTrue(n.lp.variablesUpper[idx] == 1)
                self.assertTrue(all(n.lp.variablesLower == node.lp.variablesLower))
            else:
                self.assertTrue(all(n.lp.variablesUpper == node.lp.variablesUpper))
                self.assertTrue(all(n.lp.variablesLower == [0, 0, 2]))

    def test_lt(self):
        node1 = Node(small_branch.lp, small_branch.integerIndices, -float('inf'))
        node2 = Node(small_branch.lp, small_branch.integerIndices, 0)

        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)
        self.assertRaises(TypeError, node1.__lt__, 5)

    def test_eq(self):
        node1 = Node(small_branch.lp, small_branch.integerIndices, -float('inf'))
        node2 = Node(small_branch.lp, small_branch.integerIndices, 0)
        node3 = Node(small_branch.lp, small_branch.integerIndices, 0)

        self.assertTrue(node3 == node2)
        self.assertFalse(node1 == node2)
        self.assertRaises(TypeError, node1.__eq__, 5)


if __name__ == '__main__':
    unittest.main()