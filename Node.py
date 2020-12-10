import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


class Node():
    def __init__(self, model, cylp=None):
        self.cylp = cylp
        self.model = model

    def cylp_init(self, A, b_l, b_u, c, l, u):
        model = CyClpSimplex()
        x = model.addVariable('x', len(c))
        l = CyLPArray(l)
        u = CyLPArray(u)
        c = CyLPArray(c)
        model += (-10000 <= A * x <= b_u)
        model += l <= x <= u
        model.objective = c * x
        self.cylp = model

    def primal_solve(self):
        self.cylp.primal(startFinishOptions='x')

    def dual_solve(self):
        self.cylp.dual()

    def value(self):
        return self.cylp.objectiveValue

    def get_primalVariableSolution(self):
        solution = self.cylp.primalVariableSolution
        if type(solution) == dict:
            return solution['x']
        else:
            return solution

    def check_feasility(self):
        return self.cylp.getStatusCode() == 0

    def check_int(self):
        primal_value = self.get_primalVariableSolution()
        int_vars = primal_value[self.model.int_index_array]
        if np.max(np.abs(np.round(int_vars) - int_vars)) < self.model.episilon:
            return True
        return False
