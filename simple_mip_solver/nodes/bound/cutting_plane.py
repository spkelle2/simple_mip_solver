from __future__ import annotations
from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray, CyLPExpr
from typing import Dict, Any, TypeVar, Tuple, List
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
        assert self._variables_nonnegative, 'must have x >= 0 in constraints'
        self.cuts = []
        self.x = np.array([0,0,1,0])
        self.b = None

    def branch(self: G, **kwargs: Any) -> Dict[str, Any]:
        return super().branch(**kwargs)

    def bound(self: G, cuts: List[CyLPExpr] = None, optimized_gomory_cuts: bool = True,
              **kwargs: Any) -> Dict[str, Any]:
        """ Extends BaseNode's bound by finding a variety of cuts to add after
        solving the LP relaxation

        :param optimized_gomory_cuts: if True, add optimized gomory cuts to LP
        relaxation after bounding
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: rtn, the dictionary returned by the bound method of classes
        with lower priority in method order resolution
        """
        if cuts:
            self.cuts.extend(cuts)
        self.b = kwargs['b']
        # inherit this class before others for combos to work
        # https://stackoverflow.com/questions/27954695/what-is-a-sibling-class-in-python
        rtn = super().bound(**kwargs)
        if self.lp_feasible and not self.mip_feasible:
            if optimized_gomory_cuts:
                self._add_optimized_gomory_cuts(**kwargs)
        rtn['cuts'] = self.cuts
        return rtn

    def _add_optimized_gomory_cuts(self: G, **kwargs: Any):
        """ Calculates all possible gomory cuts for current tableau, finds
        the tightest bound for each, and then adds the cut to the model

        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return:
        """
        A = self._lp.coefMatrix.toarray()
        b = self._lp.constraintsLower
        # for debugging
        # if len(self.cuts) == 3:
        #     print()
        for pi, pi0 in self._find_gomory_cuts():
            # max gives most restrictive lower bound, which we want b/c >= constraints
            pi0 = max(self._optimize_cut(pi, **kwargs), pi0)
            cut = pi * self._lp.getVarByName('x') >= pi0
            self._lp.addConstraint(cut)
            # for debugging
            # if this cut violates optimal solution but optimal solution would be feasible otherwise
            # if sum(cut.left.left * self.x) < cut.right and (A.dot(self.x) >= b).all():
            #     print()
            self.cuts.append(cut)


    def _find_gomory_cuts(self: G) -> List[Tuple[np.ndarray, float]]:
        """Find Gomory Mixed Integer Cuts (GMICs) for this node's solution.
        Defined in Lehigh University ISE 418 lecture 14 slide 18 and 5.31
        in Conforti et al Integer Programming. Assumes Ax >= b and the bounds
        x >= 0 exist and have been added to constraints.

        :return: cuts, a list of tuples (pi, pi0) that represent the cut pi*x >= pi0
        """
        cuts = []
        for row_idx in self._row_indices:
            basic_idx = self._lp.basicVariables[row_idx]
            if basic_idx in self._integer_indices and \
                    self._is_fractional(self.solution[basic_idx]):
                f0 = self._get_fraction(self.solution[basic_idx])
                # 0 for basic variables avoids getting small numbers that should be zero
                f = {i: 0 if i in self._lp.basicVariables else
                     self._get_fraction(self._lp.tableau[row_idx, i]) for i in
                     self._var_indices}
                a = {i: 0 if i in self._lp.basicVariables else self._lp.tableau[row_idx, i]
                     for i in self._var_indices}
                # primary variable coefficients in GMI cut
                pi = CyLPArray(
                    [f[j]/f0 if f[j] <= f0 and j in self._integer_indices else
                     (1 - f[j])/(1 - f0) if j in self._integer_indices else
                     a[j]/f0 if a[j] > 0 else -a[j]/(1 - f0) for j in self._var_indices]
                )
                # slack variable (including bounds) coefficients in GMI cut
                pi_slacks = np.array([x/f0 if x > 0 else -x/(1 - f0) for x in
                                      self._lp.tableau[row_idx, self._lp.nVariables:]])
                # sub out slack variables for primary variables. Ax >= b =>'s
                # Ax - s = b => s = Ax - b. gomory is pi*x + pi_slacks*s >= 1, thus
                # pi*x + pi_slacks*(Ax - b) >= 1 => (pi + pi_slacks*A)*x >= 1 + pi_slacks*b
                pi += pi_slacks * self._lp.coefMatrix
                pi0 = 1 + np.dot(pi_slacks, self._lp.constraintsLower)
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
        A = self._lp.coefMatrix.toarray()
        assert A.shape[1] == pi.shape[0], 'number of columns of A and length of c should match'

        # make new model where we minimize the cut
        model = MILPInstance(A=A, b=self._lp.constraintsLower.copy(), c=pi,
                             sense=['Min', '>='], integerIndices=self._integer_indices,
                             numVars=pi.shape[0])

        # find a tighter bound with branch and bound
        bb = BranchAndBound(model=model, Node=BaseNode,
                            node_limit=cut_optimization_node_limit, pseudo_costs={})
        bb.solve()
        return bb._global_lower_bound
