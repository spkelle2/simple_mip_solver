from __future__ import annotations

from cylp.cy import CyClpSimplex
from math import floor
import numpy as np
from typing import Union, List

epsilon = .0001


class Node:

    def __init__(self, lp: CyClpSimplex, integerIndices: List[int],
                 lower_bound: Union[float, int] = -float('inf')):
        assert isinstance(lp, CyClpSimplex), 'lp must be CyClpSimplex instance'
        assert all(0 <= idx < lp.nVariables for idx in integerIndices), \
            'indices must match variables'
        assert len(set(integerIndices)) == len(integerIndices), \
            'indices must be distinct'
        assert isinstance(lower_bound, float) or isinstance(lower_bound, int), \
            'lower bound must be a float or an int'
        self.lp = lp
        self.integerIndices = integerIndices
        self.lower_bound = lower_bound
        self.objective_value = None
        self.solution = None
        self.lp_feasible = None
        self.mip_feasible = None

    def bound(self) -> None:
        """Solve the current node with simplex. Save the run solution.

        :return:
        """
        self.lp.primal(startFinishOptions='x')
        self.objective_value = self.lp.objectiveValue
        # first cyclpsimplex has variables keyed, rest are list
        sol = self.lp.primalVariableSolution
        self.solution = sol['x'] if type(sol) == dict else sol
        self.lp_feasible = self.lp.getStatusCode() == 0
        int_var_vals = self.solution[self.integerIndices]
        self.mip_feasible = np.max(np.abs(np.round(int_var_vals) - int_var_vals)) \
            < epsilon and self.lp_feasible

    @property
    def most_fractional_index(self) -> int:
        """ Returns the index of the integer variable with current value furthest from
        being integer. If one does not exist, returns None.

        :return furthest_index: index corresponding to variable with most fractional
        value
        """
        furthest_index = None
        furthest_dist = epsilon
        for idx in self.integerIndices:
            dist = min(self.solution[idx] - floor(self.solution[idx]),
                       floor(self.solution[idx]) + 1 - self.solution[idx])
            if dist > furthest_dist:
                furthest_dist = dist
                furthest_index = idx
        return furthest_index

    # inherit and overwrite to make your own branching rule
    def branch(self) -> List[Node]:
        """ Creates two new copies of the node with new bounds placed on the variable
        with given index, one with the variable's lower bound set to the next integer
        above its current value and another with the variable's upper bound set to
        the integer immediately below its current value.

        :return: list of Nodes with the new bounds
        """
        assert self.lp_feasible, 'must solve before branching'
        index = self.most_fractional_index
        assert index is not None, 'must have fractional index to branch'
        int_value = floor(self.solution[index])

        # in one branch set upper bound for index as floor
        u = self.lp.variablesUpper.copy()
        u[index] = int_value
        left_lp = CyClpSimplex()
        left_lp.loadProblem(
            self.lp.matrix, self.lp.variablesLower,
            u, self.lp.objective, self.lp.constraintsLower,
            self.lp.constraintsUpper
        )

        # in other branch set lower bound for same index as ceiling
        l = self.lp.variablesLower.copy()
        l[index] = int_value + 1
        right_lp = CyClpSimplex()
        right_lp.loadProblem(
            self.lp.matrix, l, self.lp.variablesUpper,
            self.lp.objective, self.lp.constraintsLower,
            self.lp.constraintsUpper
        )

        return [Node(left_lp, self.integerIndices,
                     self.objective_value),
                Node(right_lp, self.integerIndices,
                     self.objective_value)]

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.lower_bound == other.lower_bound
        else:
            raise TypeError('A Node can only be compared with another Node')

    # inherit and overwrite to make your own priority rule
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.lower_bound < other.lower_bound
        else:
            raise TypeError('A Node can only be compared with another Node')

    def __repr__(self):
        return f'node with bound {self.lower_bound}'
