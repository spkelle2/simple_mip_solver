import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


class Node:
    def __init__(self, model, cylp_model=None, lower_bound=-float('inf')):
        # get asserts on model and primal_bound
        # make sure optimization direction is always min
        # cylp_model.optimizationDirection = 'min'
        self.model = model
        self.lower_bound = lower_bound
        if not cylp_model:
            cylp_model = CyClpSimplex()
            x = cylp_model.addVariable('x', len(model.c))
            b = CyLPArray(model.b)
            l = CyLPArray(model.l)
            u = CyLPArray(model.u)
            c = CyLPArray(model.c)
            cylp_model += model.A * x <= b
            cylp_model += l <= x <= u
            cylp_model.objective = c * x
        self.cylp = cylp_model

        self.obj = None
        self.solution = None
        self.lp_feasible = None
        self.mip_feasible = None

    def solve(self) -> None:
        """Solve the current node with simplex. Save the run solution.

        :return:
        """
        self.cylp.primal(startFinishOptions='x')
        self.obj = self.cylp.objectiveValue
        # TODO see how variables come out in a list
        self.solution = self.cylp.primalVariableSolution
        self.lp_feasible = self.cylp.getStatusCode() == 0
        int_var_vals = self.solution[self.model.int_index_array]
        self.mip_feasible = np.max(np.abs(np.round(int_var_vals) - int_var_vals)) \
            < self.model.epsilon and self.lp_feasible
