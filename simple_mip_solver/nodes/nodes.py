from simple_mip_solver.nodes.branch import PseudoCostBranch, MostFractionalBranch
from simple_mip_solver.nodes.search import BestFirstSearch


class PseudoCostBranchBestFirstSearchNode(PseudoCostBranch, BestFirstSearch):
    pass


class MostFractionalBranchBestFirstSearchNode(MostFractionalBranch, BestFirstSearch):
    pass
