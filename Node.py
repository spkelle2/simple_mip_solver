from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy as np


class Node():
    def __init__(self, cylp=None):
        self.cylp = cylp

    def cylp_init(self, A, b, c, l, u):
        model = CyClpSimplex()
        x = model.addVariable('x', len(c))
        A = np.matrix(A)
        b = np.matrix(b)
        l = CyLPArray(l)
        u = CyLPArray(u)
        c = CyLPArray(c)
        model += (A * x <= b)
        model += l <= x <= u
        model.objective = c * x
        self.cylp = model

    def primal_solve(self):
        self.cylp.primal()

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
