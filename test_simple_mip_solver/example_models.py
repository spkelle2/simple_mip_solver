from coinor.cuppy.milpInstance import MILPInstance
from coinor.grumpy.BranchAndBound import GenerateRandomMIP
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from itertools import product
import os
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
    return MILPInstance(A=-A, b=-b, c=-objective, l=l, u=u, sense=['Min', '>='],
                        integerIndices=list(range(len(vs))), numVars=len(vs))


def generate_random_variety(scale=1):
    constraints = {int(2*scale): 'low', int(4*scale): 'high'}
    variables = {int(2*scale): 'low', int(4*scale): 'high'}
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


def generate_random_value_functions(num_probs=40, num_evals=40):
    for prob_num in range(num_probs):
        num_vars = 4
        num_constrs = 2
        density = np.random.uniform(.2, .8)
        max_obj_coef = np.random.randint(10, 100)
        max_const_coef = np.random.randint(10, 100)
        tightness = 15

        # get a random A and objective
        cs, vs, objective, A, b = GenerateRandomMIP(
            numVars=num_vars, numCons=num_constrs, density=density, maxObjCoeff=max_obj_coef,
            maxConsCoeff=max_const_coef, tightness=15
        )
        A = np.asmatrix(pd.DataFrame.from_dict(A).to_numpy())
        objective = CyLPArray(list(objective.values()))
        l = CyLPArray([0] * len(vs))
        u = CyLPArray([max_obj_coef] * len(vs))

        # make a new directory to save all these instances
        fldr = f'example_value_functions/instance_{prob_num}'
        os.mkdir(fldr)

        for eval_num in range(num_evals):
            # make a random b for each evaluation of this instance
            b = CyLPArray(
                [np.random.randint(int(num_vars * density * max_const_coef / tightness),
                                   int(num_vars * density * max_const_coef / 1.5))
                 for i in range(num_constrs)]
            )
            instance = MILPInstance(A=A, b=b, c=objective, l=l, u=u, sense=['Max', '<='],
                                    integerIndices=list(range(len(vs))), numVars=len(vs))
            for i in instance.integerIndices:
                instance.lp.setInteger(i)
            file = f'evaluation_{eval_num}'
            instance.lp.writeMps(f'{os.path.join(fldr, file)}.mps')


# ----------------- a model with MIP feasible LP relaxation -----------------
A = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
b = CyLPArray([1, 1, 1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

no_branch = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
                         integerIndices=[0, 1], numVars=3)


# ----------- a small model that needs to branch to solve the MIP -----------
A = np.matrix([[1, 0, 1],
               [0, 1, 0]])
b = CyLPArray([1.5, 1.25])
c = CyLPArray([1, 1, 1])
l = CyLPArray([0, 0, 0])
u = CyLPArray([10, 10, 10])

small_branch = MILPInstance(A=-A, b=-b, c=-c, l=l, u=u, sense=['Min', '>='],
                            integerIndices=[0, 1, 2], numVars=3)

# a copy of the above to avoid test_pseudo_cost failing when running all tests in series
# and using this model after its attributes have been changed
A = np.matrix([[1, 0, 1],
               [0, 1, 0]])
b = CyLPArray([1.5, 1.25])
c = CyLPArray([1, 1, 1])
l = CyLPArray([0, 0, 0])
u = CyLPArray([10, 10, 10])

small_branch_copy = MILPInstance(A=-A, b=-b, c=-c, l=l, u=u, sense=['Min', '>='],
                                 integerIndices=[0, 1, 2], numVars=3)

# a copy of the above to have constraints in <= form for base algorithm tests
A = np.matrix([[1, 0, 1],
               [0, 1, 0]])
b = CyLPArray([1.5, 1.25])
c = CyLPArray([1, 1, 1])
l = CyLPArray([0, 0, 0])
u = CyLPArray([10, 10, 10])

small_branch_max = MILPInstance(A=A, b=b, c=c, l=l, u=u, sense=['Max', '<='],
                                integerIndices=[0, 1, 2], numVars=3)

# ------------------------ a model that is infeasible ------------------------
A = np.matrix([[1, 1, 0]])
b = CyLPArray([-1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

infeasible = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
                          integerIndices=[0, 1], numVars=3)

# --------------------- another model that is infeasible ---------------------
A = np.matrix([[1, 1, 0],
               [0, 0, 1]])
b = CyLPArray([-1, 1])
c = CyLPArray([1, 1, 0])
l = CyLPArray([0, 0, 0])

infeasible2 = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
                           integerIndices=[0, 1], numVars=3)

# ------------------------ a model that is unbounded -------------------------
A = np.matrix([[1, -1],
               [-1, 1]])
b = CyLPArray([1/2],
              [1/2])
c = CyLPArray([1, 1])
l = CyLPArray([0, 0])

unbounded = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
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

cut1 = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
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

cut2 = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
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

# -------------------- instance to test lift and project ---------------------
A = np.matrix([[-1, 1],
               [1, 1]])
b = CyLPArray([[-1],
               [2]])
c = CyLPArray([1, 2])
l = CyLPArray([0, 0])
lift_project = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                            integerIndices=[0, 1], numVars=2)

# ----------------------------- square instance ------------------------------
A = np.matrix([[1, 0],
               [0, 1]])
b = CyLPArray([[1.5],
               [1.5]])
c = CyLPArray([1, 1])
l = CyLPArray([0, 0])
square = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
                      integerIndices=[0, 1], numVars=2)

# ----------------------------- negative instance ------------------------------
A = np.matrix([[1, 0],
               [0, 1]])
b = CyLPArray([[-.5],
               [.5]])
c = CyLPArray([1, 1])
l = CyLPArray([-1, -1])
negative = MILPInstance(A=-A, b=-b, c=-c, l=l, sense=['Min', '>='],
                        integerIndices=[0, 1], numVars=2)

# ------------------------- model to test gomory cuts --------------------------
A = np.matrix([[-3, -4],
               [-5, -10],
               [-1, -2]])
b = CyLPArray([[-10],
               [-8],
               [-1.2]])
c = CyLPArray([-8, -12])
l = CyLPArray([0, 0])
cut3 = MILPInstance(A=A, b=b, c=c, l=l, sense=['Min', '>='],
                    integerIndices=[0, 1], numVars=2)

if __name__ == '__main__':
    generate_random_variety(scale=1)
    # generate_random_value_functions()
