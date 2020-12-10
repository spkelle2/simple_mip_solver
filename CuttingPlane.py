import numpy as np
from cylp.py.modeling.CyLPModel import CyLPArray

from Cutting import Cuttings
from Node import Node


class CuttingPlane():
    def __init__(self, model, option):
        self.model = model

        self.cutting = Cuttings[option['Cutting']]()

        self.variable_type = self.model.type
        self.int_index = [i for i in range(len(self.model.c)) if self.model.type[i] == 1]

        self.episilon = 0.000001
        self.max_iter = 1000

        self.eps = 5

    def solve(self):
        subproblem = Node(self.model)
        subproblem.cylp_init(self.model.A_, self.model.b_l_, self.model.b_u_, self.model.c_, self.model.l_,
                             self.model.u_)
        prev_sol = np.inf
        for i in range(self.max_iter):
            subproblem.cylp.primal(startFinishOptions='x')
            print('Current bound:', subproblem.cylp.objectiveValue)
            sol = subproblem.cylp.primalVariableSolution['x']
            print('Current solution: ', sol)

            if (sol - prev_sol).any():
                prev_sol = sol
            else:
                print("Solution repeated, stalling detected")
                print("Exiting")
                break

            if subproblem.check_int():
                print('Integer solution found!')
                return 0, prev_sol, subproblem.cylp.objectiveValue

            cuts, disj = self.cutting.cutting(self.model, subproblem)
            if cuts == []:
                print('No cuts found and terminating!')
                break
            for (coeff, r) in cuts:
                subproblem.cylp += CyLPArray(coeff) * subproblem.cylp.variables[0] <= r
