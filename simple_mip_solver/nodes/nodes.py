'''This file serves as a place to create nodes that multiply inherit from other
nodes. I.e. create new classes here for nodes with both custom search and branch'''

from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from simple_mip_solver.nodes.bound.disjunctive_cut import DisjunctiveCutBoundNode
from simple_mip_solver.nodes.search.depth_first import DepthFirstSearchNode


class PseudoCostBranchDepthFirstSearchNode(PseudoCostBranchNode, DepthFirstSearchNode):
    pass


class DisjunctiveCutBoundPseudoCostBranchNode(DisjunctiveCutBoundNode, PseudoCostBranchNode):
    pass
