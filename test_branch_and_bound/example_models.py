from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


# a model with MIP feasible LP relaxation
no_branch = CyClpSimplex()

x = no_branch.addVariable('x', 1)
y = no_branch.addVariable('y', 1)

c_i = CyLPArray([-1])
l = CyLPArray([0])
u = CyLPArray([1])

for var in [x, y]:
    no_branch += l <= var <= u

no_branch += c_i*x + c_i*y


# a small model that needs to branch to solve the MIP
small_branch = CyClpSimplex()
x = small_branch.addVariable('x', 1)
y = small_branch.addVariable('y', 1)
z = small_branch.addVariable('z', 1)

b = CyLPArray([1])
c_i = CyLPArray([-1])
l = CyLPArray([0])

small_branch += x + z <= b
small_branch += y <= b
for var in [x, y, z]:
    small_branch += l <= var

small_branch += c_i*x + c_i*y + c_i*z


# a model that is infeasible
infeasible = CyClpSimplex()
x = infeasible.addVariable('x', 1)
y = infeasible.addVariable('y', 1)

b = CyLPArray([-1])
c_i = CyLPArray([1])
l = CyLPArray([0])

infeasible += x + y <= b
for var in [x, y]:
    infeasible += l <= var

infeasible += c_i*x + c_i*y
