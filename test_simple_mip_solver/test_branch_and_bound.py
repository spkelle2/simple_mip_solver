from cylp.cy import CyClpSimplex
import unittest

from simple_mip_solver import Node, BranchAndBound
from test_simple_mip_solver.example_models import no_branch, small_branch, infeasible


class TestBranchAndBound(unittest.TestCase):

    def test_init(self):
        bb = BranchAndBound(small_branch)
        self.assertTrue(bb.model)
        self.assertTrue(bb._global_upper_bound == float('inf'))
        self.assertTrue(bb._global_lower_bound == -float('inf'))
        self.assertFalse(bb._best_solution)
        self.assertFalse(bb._nodes)
        self.assertFalse(bb.solution)
        self.assertTrue(bb.status == 'unsolved')
        self.assertFalse(bb.objective_value)
        self.assertFalse(bb._current_node)

    def test_init_fails_asserts(self):
        lp = CyClpSimplex()
        self.assertRaisesRegex(AssertionError, 'model must be cuppy MILPInstance',
                               BranchAndBound, lp)

    def test_solve_optimal(self):
        bb = BranchAndBound(small_branch)
        bb.solve()
        self.assertTrue(bb.status == 'optimal')
        self.assertTrue(all(s.is_integer for s in bb.solution))
        self.assertTrue(bb.objective_value == -2)

    def test_solve_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb.solve()
        self.assertTrue(bb.status == 'infeasible')
        self.assertFalse(bb.solution)
        self.assertTrue(bb.objective_value == float('inf'))

    def test_evaluate_next_node_infeasible(self):
        bb = BranchAndBound(infeasible)
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices,
                              bb._global_lower_bound))
        bb._evaluate_next_node()

        self.assertFalse(bb._nodes, 'inf model should create no nodes')
        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(bb._global_lower_bound == -float('inf'), 'shouldnt change')

    def test_evaluate_next_node_fractional(self):
        bb = BranchAndBound(small_branch)
        current_node = Node(bb.model.lp, bb.model.integerIndices,
                            bb._global_lower_bound)
        bb._nodes.append(current_node)
        bb._evaluate_next_node()

        self.assertFalse(bb._best_solution, 'best solution should not change')
        self.assertTrue(bb._global_upper_bound == float('inf'), 'shouldnt change')
        self.assertTrue(len(bb._nodes) == 2, 'should branch and add two nodes')
        current_node.solve()
        self.assertTrue(bb._global_lower_bound == current_node.objective_value,
                        'shouldnt change')

    def test_evaluate_next_node_integer(self):
        bb = BranchAndBound(no_branch)
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices,
                              bb._global_lower_bound))
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices, -4))
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices, 0))
        bb._evaluate_next_node()

        self.assertTrue(all(bb._best_solution == [1, 1, 0]))
        self.assertTrue(bb._global_upper_bound == -2)
        self.assertTrue(len(bb._nodes) == 1,
                        'integer solution should rid highest bound node')
        self.assertTrue(bb._global_lower_bound == -float('inf'), 'shouldnt change')

    def test_branch_fails_asserts(self):
        bb = BranchAndBound(small_branch)
        bb._current_node = Node(bb.model.lp, bb.model.integerIndices,
                                bb._global_lower_bound)
        # should fail assert if not solved yet
        self.assertRaises(AssertionError, bb._branch, 1)
        bb._current_node.solve()
        # should fail if not int
        self.assertRaises(AssertionError, bb._branch, '1')
        # should fail if out of range
        self.assertRaises(AssertionError, bb._branch, 4)

    def test_branch(self):
        idx = 1
        bb = BranchAndBound(small_branch)
        bb._current_node = Node(bb.model.lp, bb.model.integerIndices,
                                bb._global_lower_bound)
        bb._current_node.solve()
        ln, rn = bb._branch(idx)
        # check each node
        for name, n in {'left': ln, 'right': rn}.items():
            self.assertTrue(all(n.lp.matrix.elements ==
                                bb._current_node.lp.matrix.elements))
            self.assertTrue(all(n.lp.objective == bb._current_node.lp.objective))
            self.assertTrue(all(n.lp.constraintsLower == bb._current_node.lp.constraintsLower))
            self.assertTrue(all(n.lp.constraintsUpper == bb._current_node.lp.constraintsUpper))
            if name == 'left':
                self.assertTrue(all(n.lp.variablesUpper >= [1e10, 1, 1e10]))
                self.assertTrue(n.lp.variablesUpper[idx] == 1)
                self.assertTrue(all(n.lp.variablesLower == bb._current_node.lp.variablesLower))
            else:
                self.assertTrue(all(n.lp.variablesUpper == bb._current_node.lp.variablesUpper))
                self.assertTrue(all(n.lp.variablesLower == [0, 2, 0]))

    def test_least_lower_bound(self):
        bb = BranchAndBound(no_branch)
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices, 4))
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices, -4))
        bb._nodes.append(Node(bb.model.lp, bb.model.integerIndices, 0))
        node = bb._least_lower_bound()
        self.assertTrue(node.lower_bound == -4)


if __name__ == '__main__':
    unittest.main()
