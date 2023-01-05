from cylp.cy.CyClpSimplex import CyClpSimplex
from itertools import product
import numpy as np

from simple_mip_solver.utils.tolerance import cut_tolerance


def check_solution(sol: list, lp: CyClpSimplex):
    """ raise an exception if sol does not belong to the feasible region of the
    given LP so we can debug here and figure out what is going on

    To be used in BaseNode._bound_lp
    """
    assert all((lp.variablesLower <= sol) * (sol <= lp.variablesUpper)) and \
           all(lp.coefMatrix * np.vstack(sol) >= lp.constraintsLower.reshape(-1, 1) - 1e-6)


def check_cut(sol: list, lp: CyClpSimplex, pi: np.ndarray, pi0: float):
    """ raise an exception if sol belongs to this disjunctive term's LP (first two checks)
    relaxation and the cut violates it (third check) so we can debug here and figure out what
    is going on

    To be used in BaseNode._select_cuts"""
    assert not(all((lp.variablesLower <= sol) * (sol <= lp.variablesUpper)) and
               all((lp.constraintsLower.reshape(-1, 1) <= lp.coefMatrix * np.vstack(sol)) *
                   (lp.coefMatrix * np.vstack(sol) <= lp.constraintsUpper.reshape(-1, 1))) and
               np.dot(pi, sol) < pi0 - cut_tolerance)


def check_cut_against_grid(lp: CyClpSimplex, pi: np.array, pi0: float, max_val: float):
    """ Check all possible integer solutions to find a violated point"""
    for sol in product(*[list(range(max_val)) for _ in pi]):
        check_cut(sol=sol, lp=lp, pi=pi, pi0=pi0)


def check_dual_feasibility(lp):
    assert all(lp.coefMatrix.T * np.concatenate([a for a in lp.dualConstraintSolution.values()])
               <= lp.objectiveCoefficients)


def check_cglp(lp: CyClpSimplex, cglp):
    """if cglp says problem is infeasible, prove solution exists and cylp is wrong"""

    # create solution
    u = np.array([np.ones(lp.nConstraints) / (lp.nConstraints * 2)])
    pi = u @ lp.coefMatrix.toarray()
    pi0 = (u @ lp.constraintsLower.reshape(-1, 1))
    sol = np.concatenate([pi[0], pi0[0], u[0], u[0], np.zeros(lp.nVariables * 4)]).reshape(-1, 1)

    # test constraint validity
    assert (cglp.lp.constraintsLower.reshape(-1, 1) <= np.round(cglp.lp.coefMatrix @ sol, 11)) * \
           (np.round(cglp.lp.coefMatrix @ sol, 11) <= cglp.lp.constraintsUpper.reshape(-1, 1))

    # test variable validity
    assert (cglp.lp.variablesLower.reshape(-1, 1) <= np.round(sol, 11)) * \
           (np.round(sol, 11) <= cglp.lp.variablesUpper.reshape(-1, 1))