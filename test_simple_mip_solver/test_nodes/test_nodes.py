import unittest

from simple_mip_solver import PseudoCostBranchDepthFirstSearchNode, \
    DisjunctiveCutBoundPseudoCostBranchNode
from test_simple_mip_solver.helpers import TestModels


class TestPseudoCostBranchDepthFirstSearchNode(TestModels):

    Node = PseudoCostBranchDepthFirstSearchNode  # node type to use in base_test_models

    def test_models(self):
        self.base_test_models()


class TestDisjunctiveCutBoundPseudoCostBranchNode(TestModels):

    Node = DisjunctiveCutBoundPseudoCostBranchNode  # node type to use in disjunctive_cut_test_models

    def test_models(self):
        self.disjunctive_cut_test_models()


if __name__ == '__main__':
    unittest.main()
