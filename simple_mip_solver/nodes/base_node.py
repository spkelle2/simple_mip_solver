from __future__ import annotations

from cylp.cy import CyClpSimplex
from math import floor, ceil
import numpy as np
from typing import Union, List, TypeVar

T = TypeVar('T', bound='BaseNode')


class BaseNode:

    def __init__(self: T, lp: CyClpSimplex, integerIndices: List[int],
                 lower_bound: Union[float, int] = -float('inf'),
                 b_idx: int = None, b_dir: str = None, b_val: float = None):
        """
        :param lp: model object simplex is run against
        :param integerIndices: indices of variables we aim to find integer solutions
        :param lower_bound: starting lower bound on optimal objective value
        for the minimization problem in this node
        :param b_idx: index of the branching variable
        :param b_dir: direction of branching
        :param b_val: initial value of the branching variable
        """
        assert isinstance(lp, CyClpSimplex), 'lp must be CyClpSimplex instance'
        assert all(0 <= idx < lp.nVariables and isinstance(idx, int) for idx in
                   integerIndices), 'indices must match variables'
        assert len(set(integerIndices)) == len(integerIndices), \
            'indices must be distinct'
        assert isinstance(lower_bound, float) or isinstance(lower_bound, int), \
            'lower bound must be a float or an int'
        assert (b_dir is None) == (b_idx is None) == (b_val is None), \
            'none are none or all are none'
        assert b_idx in integerIndices or b_idx is None, \
            'branch index corresponds to integer variable if it exists'
        assert b_dir in ['up', 'down'] or b_dir is None, \
            'we can only round a variable up or down when branching'
        if b_val is not None:
            good_down = 0 < b_val - lp.variablesUpper[b_idx] < 1
            good_up = 0 < lp.variablesLower[b_idx] - b_val < 1
            assert (b_dir == 'down' and good_down) or \
                   (b_dir == 'up' and good_up)
        self.lp = lp
        self.integerIndices = integerIndices
        self.lower_bound = lower_bound
        self.objective_value = None
        self.solution = None
        self.lp_feasible = None
        self.mip_feasible = None
        self._epsilon = .0001
        self.b_dir = b_dir
        self.b_idx = b_idx
        self.b_val = b_val

    def bound(self: T) -> None:
        """Solve the current node with simplex to generate a bound on objective
        values of integer feasible solutions of descendent nodes. Save the run solution.

        :return:
        """
        self.lp.dual(startFinishOptions='x')  # Todo check this doesnt hinder warm starts
        self.objective_value = self.lp.objectiveValue
        # first cyclpsimplex has variables keyed, rest are list
        sol = self.lp.primalVariableSolution
        self.solution = sol['x'] if type(sol) == dict else sol
        self.lp_feasible = self.lp.getStatusCode() == 0
        int_var_vals = self.solution[self.integerIndices]
        self.mip_feasible = np.max(np.abs(np.round(int_var_vals) - int_var_vals)) \
                            < self._epsilon and self.lp_feasible

    def base_branch(self: T, idx: int) -> List[T]:
        """ Creates two new copies of the node with new bounds placed on the variable
        with index <idx>, one with the variable's lower bound set to the ceiling
        of its current value and another with the variable's upper bound set to
        the floor of its current value.

        :param idx: index of variable to branch on

        :return: list of Nodes with the new bounds
        """
        assert self.lp_feasible, 'must solve before branching'
        assert idx in self.integerIndices, 'must branch on integer index'
        b_val = self.solution[idx]
        assert min(b_val - floor(b_val), ceil(b_val) - b_val) > self._epsilon, \
            "index branched on must be fractional"

        # get end basis to warm start the kiddos
        basis = self.lp.getBasisStatus()

        # when branching down set floor as upper bound for given index
        u = self.lp.variablesUpper.copy()
        u[idx] = floor(b_val)
        down_lp = CyClpSimplex()
        down_lp.loadProblem(
            self.lp.matrix, self.lp.variablesLower,
            u, self.lp.objective, self.lp.constraintsLower,
            self.lp.constraintsUpper
        )
        down_lp.setBasisStatus(basis)

        # when branching up set ceiling as lower bound for given index
        l = self.lp.variablesLower.copy()
        l[idx] = ceil(b_val)
        up_lp = CyClpSimplex()
        up_lp.loadProblem(
            self.lp.matrix, l, self.lp.variablesUpper,
            self.lp.objective, self.lp.constraintsLower,
            self.lp.constraintsUpper
        )
        up_lp.setBasisStatus(basis)

        # return instances of the subclass that calls this function
        return [type(self)(down_lp, self.integerIndices, self.objective_value,
                           idx, 'down', b_val),
                type(self)(up_lp, self.integerIndices, self.objective_value,
                           idx, 'up', b_val)]

    def is_fractional(self: T, idx: int) -> bool:
        """Returns true if variable <idx> is fractional, False if not.

        :param idx: which variable to check is fractional
        :return: boolean of if variable <idx> is fractional
        """

        assert 0 <= idx < self.lp.nVariables and isinstance(idx, int), \
            'idx must belong to an index of a variable'
        assert self.lp_feasible, 'must be feasible to have a solution to check'

        return min(self.solution[idx] - floor(self.solution[idx]),
                   ceil(self.solution[idx]) - self.solution[idx]) > self._epsilon
