from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray
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
