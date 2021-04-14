from coinor.cuppy.milpInstance import MILPInstance
from math import floor
import numpy as np
from typing import Union

epsilon = .0001


class Node:
    def __init__(self, model: MILPInstance,
                 lower_bound: Union[float, int]=-float('inf')):
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'
        assert isinstance(lower_bound, float) or isinstance(lower_bound, int), \
            'lower bound must be a float or an int'
        self.model = model
        self.lower_bound = lower_bound
        self.objective_value = None
        self.solution = None
        self.lp_feasible = None
        self.mip_feasible = None

    def solve(self) -> None:
        """Solve the current node with simplex. Save the run solution.

        :return:
        """
        self.model.lp.primal(startFinishOptions='x')
        self.objective_value = self.model.lp.objectiveValue
        # all variables in MILPInstance are keyed by 'x
        self.solution = self.model.lp.primalVariableSolution['x']
        self.lp_feasible = self.model.lp.getStatusCode() == 0
        int_var_vals = self.solution[self.model.integerIndices]
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
        furthest_dist = 0
        for idx in self.model.integerIndices:
            dist = abs(self.solution[idx] - (floor(self.solution[idx]) + 0.5))
            if dist > furthest_dist:
                furthest_dist = dist
                furthest_index = idx
        return furthest_index
