import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

A = [[50, 31], [-3, 2, ]]
b = [250, 4]
c = [-1, -0.64]
l = [1, 0]
u = [10000, 10000]

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

model.primal()
#print(model.getBasisStatus())
print(model.primalConstraintSolution)   #row value
print(model.primalVariableSolutionAll)  #row slack

print(model.dualConstraintSolution)     #dual variable
print(model.dualVariableSolution)       #dual slack
