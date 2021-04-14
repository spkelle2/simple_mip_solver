from coinor.cuppy.milpInstance import MILPInstance
from math import floor
from operator import attrgetter
from typing import List

from simple_mip_solver import Node


class BranchAndBound:
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    def __init__(self, model: MILPInstance):
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'
        self._global_upper_bound = float('inf')
        self._global_lower_bound = -float('inf')
        self._best_solution = None
        self._nodes = []
        self.model = model
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None
        self._current_node = None

    def solve(self) -> None:
        """Solves the Branch and Bound algorithm

        :return:
        """
        self._nodes.append(Node(self.model, self._global_lower_bound))

        while self._nodes:
            self._evaluate_next_node()

        self.status = 'optimal' if self._best_solution else 'infeasible'
        self.solution = self._best_solution
        self.objective_value = self._global_upper_bound

    def _evaluate_next_node(self) -> None:
        """Solves one node of the branch and bound algorithm

        :return:
        """
        # select best first
        self._current_node = self._least_lower_bound()
        self._nodes.remove(self._current_node)
        self._current_node.solve()

        # do nothing if node infeasible or worse than the existing bound
        if self._current_node.lp_feasible and self._current_node.objective_value < self._global_upper_bound:
            if self._current_node.mip_feasible:
                self._best_solution = self._current_node.solution
                self._global_upper_bound = self._current_node.objective_value
                # only keep nodes that give us a chance to find better solutions
                self._nodes = [n for n in self._nodes if n.lower_bound <
                               self._global_upper_bound]
            else:
                idx = self._current_node.most_fractional_index
                self._nodes.extend(self._branch(idx))
                self._global_lower_bound = min(n.lower_bound for n in self._nodes)

    def _branch(self, index: int) -> List[Node]:
        """ Creates two new copies of the node with new bounds placed on the variable
        with given index, one with the variable's lower bound set to the next integer
        above its current value and another with the variable's upper bound set to
        the integer immediately below its current value.

        :param index: index of variable to branch on
        :return: list of Nodes with the new bounds
        """
        assert isinstance(index, int), 'index must be an integer'
        assert 0 <= index < self._current_node.model.numVars, 'index must match a variable'
        assert self._current_node.lp_feasible, 'must have solved to set bounds'

        int_value = floor(self._current_node.solution[index])
        # in one branch set upper bound for index as floor
        u = self._current_node.model.lp.variablesUpper.copy()
        u[index] = int_value
        left_model = MILPInstance(
            A=self._current_node.model.A, b=self._current_node.model.b,
            c=self._current_node.model.c, l=self._current_node.model.l, u=u,
            integerIndices=self._current_node.model.integerIndices)

        # in other branch set lower bound for same index as ceiling
        l = self._current_node.model.lp.variablesLower.copy()
        l[index] = int_value + 1
        right_model = MILPInstance(
            A=self._current_node.model.A, b=self._current_node.model.b,
            c=self._current_node.model.c, l=l, u=self._current_node.model.u,
            integerIndices=self._current_node.model.integerIndices)

        return [Node(left_model, self._current_node.objective_value),
                Node(right_model, self._current_node.objective_value)]

    def _least_lower_bound(self) -> Node:
        """ Finds the node with the least lower bound

        :param nodes: list of nodes representing where in the branch and bound tree we are
        :return: node with the least lower bound
        """
        return min(self._nodes, key=attrgetter('lower_bound'))
