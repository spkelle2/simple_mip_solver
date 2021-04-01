from typing import Dict, Tuple

from branch import branch
from node import Node
from node_selection import least_lower_bound


class BranchAndBound:
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    def __init__(self, model):
        self.global_upper_bound = float('inf')
        self.global_lower_bound = -float('inf')
        self.best_solution = None
        self.nodes = []
        self.model = model

    def solve(self) -> Tuple[int, Dict[int: float], float]:
        """Solves the Branch and Bound algorithm

        :return: status code, the variable values in the best solution, and the
        best objective value
        """
        self.nodes.append(Node(self.model, None, self.global_lower_bound))

        while self.nodes:
            self._evaluate_next_node()

        return 0 if self.best_solution else -1, self.best_solution, \
            self.global_upper_bound

    def _evaluate_next_node(self) -> None:
        """Solves one iteration of the branch and bound algorithm, which is as
        follows

        :return:
        """
        # select best first
        current_node = least_lower_bound(self.nodes)
        self.nodes.remove(current_node)
        current_node.solve()

        # do nothing if node infeasible or worse than the existing bound
        if current_node.lp_feasible and current_node.obj < self.global_upper_bound:
            if current_node.mip_feasible:
                self.best_solution = current_node.solution
                self.global_upper_bound = current_node.obj
                # prune nodes that won't yield better solutions
                self.nodes = [n for n in self.nodes if n.lower_bound <
                              self.global_upper_bound]
            else:
                self.nodes.extend(branch(self.model, current_node))
                self.global_lower_bound = min(n.lower_bound for n in self.nodes)
