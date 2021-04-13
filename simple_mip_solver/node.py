from coinor.cuppy.milpInstance import MILPInstance
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
        self.obj = None
        self.solution = None
        self.lp_feasible = None
        self.mip_feasible = None

    def solve(self) -> None:
        """Solve the current node with simplex. Save the run solution.

        :return:
        """
        self.model.lp.primal(startFinishOptions='x')
        self.obj = self.model.lp.objectiveValue
        # all variables in MILPInstance are keyed by 'x
        self.solution = self.model.lp.primalVariableSolution['x']
        self.lp_feasible = self.model.lp.getStatusCode() == 0
        int_var_vals = self.solution[self.model.integerIndices]
        self.mip_feasible = np.max(np.abs(np.round(int_var_vals) - int_var_vals)) \
            < epsilon and self.lp_feasible
