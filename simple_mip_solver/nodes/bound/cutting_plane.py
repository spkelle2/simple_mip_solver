from __future__ import annotations
from coinor.cuppy.milpInstance import MILPInstance
from cylp.cy.CyClpSimplex import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray, CyLPExpr
from typing import Dict, Any, TypeVar, Tuple, List
from math import floor, ceil
import numpy as np

from simple_mip_solver import BaseNode, BranchAndBound, PseudoCostBranchNode

G = TypeVar('G', bound='CuttingPlaneBoundNode')


class CuttingPlaneBoundNode(BaseNode):
    """ An extension of the BaseNode class to allow for cutting plane methods.
    Types of cutting planes available are as follows:
        * Optimized Gomory Cuts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._sense == '>=', 'must have Ax >= b'
        assert self._variables_nonnegative, 'must have x >= 0 for all variables'
        assert len(self.lp.variables) == 1 and self.lp.variables[0].name == 'x', \
            'x must be our only variable'

    # def branch(self: G, **kwargs: Any) -> Dict[str, Any]:
    #     return super().branch(**kwargs)

    def bound(self: G, optimized_gomory_cuts: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """ Extends BaseNode's bound by finding a variety of cuts to add after
        solving the LP relaxation

        :param optimized_gomory_cuts: if True, add optimized gomory cuts to LP
        relaxation after bounding
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: rtn, the dictionary returned by the bound method of classes
        with lower priority in method order resolution
        """
        # inherit this class before others for combos to work
        # https://stackoverflow.com/questions/27954695/what-is-a-sibling-class-in-python
        rtn = super().bound(**kwargs)
        if self.lp_feasible and not self.mip_feasible:
            if optimized_gomory_cuts:
                self._add_optimized_gomory_cuts(**kwargs)
        return rtn

    def _add_optimized_gomory_cuts(self: G, **kwargs: Any):
        """ Calculates all possible gomory cuts for current tableau, finds
        the tightest bound for each, and then adds the cut to the model

        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return:
        """
        for pi, pi0 in self._find_gomory_cuts():
            # max gives most restrictive lower bound, which we want b/c >= constraints
            pi0 = max(self._optimize_cut(pi, **kwargs), pi0)
            cut = pi * self.lp.getVarByName('x') >= pi0
            self.lp.addConstraint(cut)

    def _find_gomory_cuts(self: G) -> List[Tuple[CyLPArray, float]]:
        """Find Gomory Mixed Integer Cuts (GMICs) for this node's solution.
        Defined in Lehigh University ISE 418 lecture 14 slide 18 and 5.31
        in Conforti et al Integer Programming. Assumes Ax >= b and x >= 0.

        Warning: this method does not currently work. It creates some cuts that
        are invalid. Stuck on debugging because I can't spot the difference between
        this and what is referenced above.

        :return: cuts, a list of tuples (pi, pi0) that represent the cut pi*x >= pi0
        """
        cuts = []
        for row_idx in self._row_indices:
            basic_idx = self.lp.basicVariables[row_idx]
            if basic_idx in self._integer_indices and \
                    self._is_fractional(self.solution[basic_idx]):
                f0 = self._get_fraction(self.solution[basic_idx])
                # 0 for basic variables avoids getting small numbers that should be zero
                f = {i: 0 if i in self.lp.basicVariables else
                     self._get_fraction(self.lp.tableau[row_idx, i]) for i in
                     self._var_indices}
                a = {i: 0 if i in self.lp.basicVariables else self.lp.tableau[row_idx, i]
                     for i in self._var_indices}
                # primary variable coefficients in GMI cut
                pi = CyLPArray(
                    [f[j]/f0 if f[j] <= f0 and j in self._integer_indices else
                     (1 - f[j])/(1 - f0) if j in self._integer_indices else
                     a[j]/f0 if a[j] > 0 else -a[j]/(1 - f0) for j in self._var_indices]
                )
                # slack variable coefficients in GMI cut
                pi_slacks = np.array([x/f0 if x > 0 else -x/(1 - f0) for x in
                                      self.lp.tableau[row_idx, self.lp.nVariables:]])
                # sub out slack variables for primary variables. Ax >= b =>'s
                # Ax - s = b => s = Ax - b. gomory is pi^T * x + pi_s^T * s >= 1, thus
                # pi^T * x + pi_s^T * (Ax - b) >= 1 => (pi + A^T * pi_s)^T * x >= 1 + pi_s^T * b
                pi += self.lp.coefMatrix.T * pi_slacks
                pi0 = 1 + np.dot(pi_slacks, self.lp.constraintsLower)
                # append pi >= pi0
                cuts.append((pi, pi0))
        return cuts

    def _optimize_cut(self: G, pi: np.ndarray, cut_optimization_node_limit: int = 10,
                      **kwargs: Any) -> float:
        """ Given the valid inequality pi >= pi0, try to find a smaller RHS such
        that pi >= smaller RHS is still a valid inequality

        :param pi: coefficients of the vector to optimize
        :param cut_optimization_node_limit: maximimum number of nodes to evaluate
        before terminating
        :param kwargs: catch all for unused passed kwargs
        :return: the objective value of the best milp feasible solution found
        """
        A = self.lp.coefMatrix.toarray()
        assert A.shape[1] == pi.shape[0], 'number of columns of A and length of c should match'

        # make new model where we minimize the cut
        model = MILPInstance(A=A, b=self.lp.constraintsLower.copy(), c=pi,
                             sense=['Min', '>='], integerIndices=self._integer_indices,
                             numVars=pi.shape[0])

        # find a tighter bound with branch and bound
        bb = BranchAndBound(model=model, Node=BaseNode,
                            node_limit=cut_optimization_node_limit, pseudo_costs={})
        bb.solve()
        return bb.objective_value if bb.status == 'optimal' else bb._global_lower_bound
        # need lower bound because upper bound is furthest feasible point intersecting
        # the cut. lower bound then ensures no feasible points cut

    def _add_split_inequality(self: G, **kwargs: Any):
        pass
        # pi, pi0 = self._find_lift_and_project_cut()
        # cut = pi * self.lp.getVarByName('x') >= pi0  # which direction in derivation
        # self.lp.addConstraint(cut)

    def _find_split_inequality(self: G, idx: int, **kwargs: Any):
        assert idx in self._integer_indices, 'must lift and project on integer index'
        x_idx = self.solution[idx]
        assert self._is_fractional(x_idx), "must lift and project on index with fractional value"

        # build the CGLP model from ISE 418 Lecture 15 Slide 7 but for LP with >= constraints
        lp = CyClpSimplex()

        # declare variables
        pi = lp.addVariable('pi', self.lp.nVariables)
        pi0 = lp.addVariable('pi0', 1)
        u1 = lp.addVariable('u1', self.lp.nConstraints)
        u2 = lp.addVariable('u2', self.lp.nConstraints)
        w1 = lp.addVariable('w1', self.lp.nVariables)
        w2 = lp.addVariable('w2', self.lp.nVariables)

        # set bounds
        lp += u1 >= CyLPArray(np.zeros(self.lp.nConstraints))
        lp += u2 >= CyLPArray(np.zeros(self.lp.nConstraints))

        w_ub = CyLPArray(np.zeros(self.lp.nVariables))
        w_ub[idx] = float('inf')
        lp += w_ub >= w1 >= CyLPArray(np.zeros(self.lp.nVariables))
        lp += w_ub >= w2 >= CyLPArray(np.zeros(self.lp.nVariables))

        # set constraints
        # (pi, pi0) must be valid for both parts of the disjunction
        lp += 0 >= -pi + self.lp.coefMatrix.T * u1 - w1
        lp += 0 >= -pi + self.lp.coefMatrix.T * u2 + w2
        lp += 0 <= -pi0 + CyLPArray(self.lp.constraintsLower) * u1 - floor(x_idx) * w1.sum()
        lp += 0 <= -pi0 + CyLPArray(self.lp.constraintsLower) * u2 + ceil(x_idx) * w2.sum()
        # normalize variables
        lp += u1.sum() + u2.sum() + w1.sum() + w2.sum() == 1

        # set objective: find the deepest cut
        # since pi * x >= pi0 for all x in disjunction, we want min pi * x_star - pi0
        lp.objective = CyLPArray(self.solution) * pi - pi0

        # solve
        lp.primal(startFinishOptions='x')
        assert lp.getStatusCode() == 0, 'we should get optimal solution'
        assert lp.objectiveValue <= 0, 'pi * x >= pi -> pi * x - pi >= 0 -> ' \
                                       'negative objective at x^* since it gets cut off'

        # get solution
        return lp.primalVariableSolution['pi'], lp.primalVariableSolution['pi0']

