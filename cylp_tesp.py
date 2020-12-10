import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

A = [[50, 31], [-3, 2, ]]
b_l = [-10000, -10000]
b_u = [250, 4]
c = [-1, -0.64]
l = [1, 0]
u = [10000, 10000]

model = CyClpSimplex()
x = model.addVariable('x', len(c))
A = np.matrix(A)
b_l = np.matrix(b_l)
b_l = -1000
b_u = np.matrix(b_u)
l = CyLPArray(l)
u = CyLPArray(u)
c = CyLPArray(c)
model += (b_l <= A * x <= b_u)
model += l <= x <= u
model.objective = c * x

model.primal()
# print(model.getBasisStatus())
print(model.primalConstraintSolution)  # row value
print(model.primalVariableSolutionAll)  # row slack

print(model.dualConstraintSolution)  # dual variable
print(model.dualVariableSolution)  # dual slack

# from cylp.py.modeling.CyLPModel import CyLPArray
#
# asd = np.array([1, 2, 3])
# b = CyLPArray(asd)
# print(b)
