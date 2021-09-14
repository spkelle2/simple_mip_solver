from coinor.cuppy.milpInstance import MILPInstance
from coinor.grumpy.BranchAndBound import GenerateRandomMIP
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from itertools import product
import pandas as pd
import numpy as np


# helpers to generate random models
def generate_random_MILPInstance(numVars=40, numCons=20, density=0.2,
                                 maxObjCoeff=10, maxConsCoeff=10,
                                 tightness=2, rand_seed=2):
    cs, vs, objective, A, b = GenerateRandomMIP(
        numVars=numVars, numCons=numCons, density=density, maxObjCoeff=maxObjCoeff,
        maxConsCoeff=maxConsCoeff, tightness=tightness, rand_seed=rand_seed
    )
    A = np.asmatrix(pd.DataFrame.from_dict(A).to_numpy())
    objective = CyLPArray(list(objective.values()))
    b = CyLPArray(b)
    l = CyLPArray([0] * len(vs))
    u = CyLPArray([maxObjCoeff] * len(vs))
    return MILPInstance(A=A, b=b, c=objective, l=l, u=u, sense=['Max', '<='],
                        integerIndices=list(range(len(vs))), numVars=len(vs))


def generate_random_variety():
    constraints = {2: 'low', 4: 'high'}
    variables = {2: 'low', 4: 'high'}
    densities = {.2: 'low', .8: 'high'}
    max_obj_coeffs = {10: 'low', 100: 'high'}
    max_cons_coeffs = {10: 'low', 100: 'high'}
    tightnesses = {2: 'low', 8: 'high'}

    for constraint, variable, density, max_obj_coeff, max_cons_coeff, \
        tightness in product(constraints, variables, densities, max_obj_coeffs,
                             max_cons_coeffs, tightnesses):
        m = generate_random_MILPInstance(constraint, variable, density,
                                         max_obj_coeff, max_cons_coeff, tightness)
        for i in m.integerIndices:
            m.lp.setInteger(i)
        m.lp.writeMps(f'constraints_{constraints[constraint]}'
                      f'_variables_{variables[variable]}'
                      f'_density_{densities[density]}'
                      f'_max_obj_coeff_{max_obj_coeffs[max_obj_coeff]}'
                      f'_max_cons_coeff_{max_cons_coeffs[max_cons_coeff]}'
                      f'_tightness_{tightnesses[tightness]}.mps')


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
u = CyLPArray([10, 10, 10])

small_branch = MILPInstance(A=A, b=b, c=c, l=l, u=u, sense=['Max', '<='],
                            integerIndices=[0, 1, 2], numVars=3)

# a copy of the above to avoid test_pseudo_cost failing when running all tests in series
# and using this model after its attributes have been changed
A = np.matrix([[1, 0, 1],
               [0, 1, 0]])
b = CyLPArray([1.5, 1.25])
c = CyLPArray([1, 1, 1])
l = CyLPArray([0, 0, 0])
u = CyLPArray([10, 10, 10])

small_branch_copy = MILPInstance(A=A, b=b, c=c, l=l, u=u, sense=['Max', '<='],
                                 integerIndices=[0, 1, 2], numVars=3)

# ------------------------ a model that is infeasible ------------------------
A = np.matrix([[1, 1, 0]])
b = CyLPArray([-1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

infeasible = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                          integerIndices=[0, 1], numVars=3)

# --------------------- another model that is infeasible ---------------------
A = np.matrix([[1, 1, 0],
               [0, 0, 1]])
b = CyLPArray([-1, 1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

infeasible2 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                           integerIndices=[0, 1], numVars=3)

# ------------------------ a model that is unbounded -------------------------
A = np.matrix([[1, -1],
               [-1, 1]])
b = CyLPArray([1/2],
              [1/2])
c = CyLPArray([1, 1])
l = CyLPArray([0, 0])

unbounded = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                         integerIndices=[0, 1], numVars=2)

# ----------------------- a larger model that is random ----------------------
random = generate_random_MILPInstance(numVars=20, numCons=10)

# ---------------- a model for testing cutting plane methods -----------------
A = np.matrix([[-8, 30],
               [-14, 8],
               [10, 10]])
b = CyLPArray([[115],
               [1],
               [127]])
c = CyLPArray([0, 1])
l = CyLPArray([0, 0])

cut1 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                    integerIndices=[0, 1], numVars=2)

# ---------------- a model for testing cutting plane methods -----------------
A = np.matrix([[4, 1],
               [1, 4],
               [1, -1]])
b = CyLPArray([[28],
               [27],
               [1]])
c = CyLPArray([2, 5])
l = CyLPArray([0, 0])

cut2 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Max', '<='],
                    integerIndices=[0, 1], numVars=2)

# --------- instance from ISE 418 HW 3 problem 1 to test dual bound ----------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[3.5],
               [-3.5]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                    integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[0],
               [0]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_0 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[1],
               [-1]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_1 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[2],
               [-2]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_2 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[3],
               [-3]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_3 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[4],
               [-4]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_4 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

# ------ instance from ISE 418 HW 3 problem 1 to create value function -------
A = np.matrix([[2, 5, -2, -2, 5, 5],
               [-2, -5, 2, 2, -5, -5]])
b = CyLPArray([[5],
               [-5]])
c = CyLPArray([1, 4, 6, 4, 5, 7])
l = CyLPArray([0, 0, 0, 0, 0, 0])

h3p1_5 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1, 3], numVars=6)

if __name__ == '__main__':
    generate_random_variety()
