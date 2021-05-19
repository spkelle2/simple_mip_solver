from coinor.cuppy.milpInstance import MILPInstance
from coinor.grumpy.BranchAndBound import GenerateRandomMIP
from cylp.py.modeling.CyLPModel import CyLPArray
import pandas as pd
import numpy as np


# ----------------- a model with MIP feasible LP relaxation -----------------
A = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
b = CyLPArray([1, 1, 1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

no_branch = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                         integerIndices=[0, 1], numVars=3)


# ----------- a small model that needs to branch to solve the MIP -----------
A = np.matrix([[1, 0, 1],
               [0, 1, 0]])
b = CyLPArray([1.5, 1.25])
c = CyLPArray([1, 1, 1])
l = CyLPArray([0, 0, 0])

small_branch = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                            integerIndices=[0, 1, 2], numVars=3)


# ------------------------ a model that is infeasible ------------------------
A = np.matrix([[1, 1, 0]])
b = CyLPArray([-1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

infeasible = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                          integerIndices=[0, 1], numVars=3)

# ----------------------- a larger model that is random ----------------------
cs, vs, objective, A, b = GenerateRandomMIP()
A = np.asmatrix(pd.DataFrame.from_dict(A).to_numpy())
objective = CyLPArray(list(objective.values()))
b = CyLPArray(b)
l = CyLPArray([0] * len(vs))
random = MILPInstance(A=A, b=b, c=objective, l=l, sense=['Max', '<='],
                      integerIndices=list(range(len(vs))), numVars=len(vs))
