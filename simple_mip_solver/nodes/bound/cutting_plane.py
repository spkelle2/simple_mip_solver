from __future__ import annotations
from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.modeling.CyLPModel import CyLPArray, CyLPExpr
from typing import Dict, Any, TypeVar, Tuple, List, Union
import numpy as np

from simple_mip_solver import BaseNode, BranchAndBound
from simple_mip_solver.utils import constraint_epsilon
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP

G = TypeVar('G', bound='CuttingPlaneBoundNode')


class CuttingPlaneBoundNode(BaseNode):
    """ An extension of the BaseNode class to allow for cutting plane methods.
    Types of cutting planes available (with their respective key word arguments) are as follows:
        * Optimized Gomory Cuts
        * Cut Generating LP Cuts

    Instantiate an algorithm, e.g. Branch and Bound, with this node type as follows:
        BranchAndBound(MILPInstance, CuttingPlaneBoundNode, cglp=cglp, cut_generating_lp=True,
                       cglp_cumulative_constraints=True)
    """

    def __init__(self, cglp: CutGeneratingLP = None,
                 cglp_starting_basis: Tuple[np.ndarray, np.ndarray] = None,
                 *args, **kwargs):
        """

        :param cglp: The CGLP instance to use for creating disjunctive cuts, off which
        children's CGLP's will be built
        :param cglp_starting_basis: The starting basis status for each variable in the CGLP instance
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        assert self._sense == '>=', 'must have Ax >= b'
        assert self._variables_nonnegative, 'must have x >= 0 for all variables'
        if cglp is not None:
            assert isinstance(cglp, CutGeneratingLP), 'cglp must be CutGeneratingLP instance'

        self.cglp = cglp
        self.cglp_starting_basis = cglp_starting_basis

    def branch(self: G, cglp_cumulative_constraints: bool = False,
               cglp_cumulative_bounds: bool = False, cglp: CutGeneratingLP = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Before calling parent and sibling class branch methods, create the
        cglp instance for each child node. Note, if the user sets either parameter to
        True, cuts generated from the CGLP can no longer be added to other nodes

        :param cglp_cumulative_constraints: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with cuts added to this node
        :param cglp_cumulative_bounds: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with the bounds placed on each
        variable in this node
        :param cglp: Pulls 'cglp' from kwargs as to not interfere with cglp assignment
        in called subroutines
        :param kwargs:
        :return:
        """
        assert isinstance(cglp_cumulative_constraints, bool), 'cglp_cumulative_constraints is bool'
        assert isinstance(cglp_cumulative_bounds, bool), 'cglp_cumulative_bounds is bool'

        if self.cglp is None:
            return super().branch(**kwargs)

        elif cglp_cumulative_constraints or cglp_cumulative_bounds:
            A = None if not cglp_cumulative_constraints else self.lp.coefMatrix.copy()
            b = None if not cglp_cumulative_constraints else CyLPArray(self.lp.constraintsLower.copy())
            var_lb = None if not cglp_cumulative_bounds else CyLPArray(self.lp.variablesLower.copy())
            var_ub = None if not cglp_cumulative_bounds else CyLPArray(self.lp.variablesUpper.copy())

            cglp = CutGeneratingLP(bb=self.cglp.bb, root_id=self.cglp.root_id,
                                   A=A, b=b, var_lb=var_lb, var_ub=var_ub)
            # todo: warm start - set all new slack variables as basic (1) and
            # todo: pivot to find feasible basis when bounds remove constraints from CGLP
            return super().branch(cglp=cglp, **kwargs)

        else:
            # if recycling CGLP, pass on basis because children solutions wont be far off
            return super().branch(cglp=self.cglp, cglp_starting_basis=self.cglp.lp.getBasisStatus(),
                                  **kwargs)

    def bound(self: G, optimized_gomory_cuts: bool = False, cut_generating_lp: bool = False,
              **kwargs: Any) -> Dict[str, Any]:
        """ Extends BaseNode's bound by finding a variety of cuts to add after
        solving the LP relaxation. Any cuts generated that are valid for all other
        Node instances can be added to the `cuts` list and added as a key to the
        return dictionary

        :param optimized_gomory_cuts: if True, add optimized gomory cuts to LP
        relaxation after bounding
        :param cut_generating_lp: If True, add disjunctive cuts generated from the
        CGLP to the LP relaxation after bounding
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: rtn, the dictionary returned by the bound method of classes
        with lower priority in method order resolution
        """
        assert isinstance(optimized_gomory_cuts, bool), 'optimized_gomory_cuts is boolean'
        assert isinstance(cut_generating_lp, bool), 'cut_generating_lp is boolean'
        if cut_generating_lp:
            assert self.cglp is not None, 'cglp attribute must be defined if using Cut Generating LP'

        # inherit this class first before inheriting others for combos to work
        # https://stackoverflow.com/questions/27954695/what-is-a-sibling-class-in-python
        rtn = super().bound(**kwargs)
        if self.lp_feasible and not self.mip_feasible:
            cuts = {}
            if optimized_gomory_cuts:
                self._add_optimized_gomory_cuts(**kwargs)
            if cut_generating_lp:
                cglp_cut = self._add_cglp_cut(**kwargs)
                if cglp_cut is not None:
                    cuts[f'node_{self.idx}_cglp_cut'] = cglp_cut
            rtn['cuts'] = cuts
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
                # todo: maybe this needs a constriant epsilon too
                # todo: check out if we can use actual constraints instead of coefMatrix and lower
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
        return bb.objective_value if bb.status == 'optimal' else bb.global_lower_bound
        # need lower bound because upper bound is furthest feasible point intersecting
        # the cut. lower bound then ensures no feasible points cut

    def _add_cglp_cut(self: G, cglp_cumulative_constraints: bool = False,
                      cglp_cumulative_bounds: bool = False, **kwargs: Any) -> Union[CyLPExpr, None]:
        """ Add the disjunctive cut resulting from the CGLP back to the current node.
        If both cglp_cumulative_constraints and cglp_cumulative_bounds are False,
        then return the added cut so the algorithm can add it to other nodes

        :param optimized_gomory_cuts: if True, add optimized gomory cuts to LP
        relaxation after bounding
        :param cut_generating_lp: If True, add disjunctive cuts generated from the
        CGLP to the LP relaxation after bounding
        :param kwargs:
        :return:
        """
        assert isinstance(cglp_cumulative_constraints, bool), 'cglp_cumulative_constraints is bool'
        assert isinstance(cglp_cumulative_bounds, bool), 'cglp_cumulative_bounds is bool'

        pi, pi0 = self.cglp.solve(x_star=CyLPArray(self.solution),
                                  starting_basis=self.cglp_starting_basis)
        assert pi is not None and pi0 is not None, 'should get solution if not in the interior'

        # don't return anything if constraint coefficients are effectively 0
        if np.linalg.norm(pi) > constraint_epsilon:
            cut = pi * self.lp.getVarByName('x') >= pi0
            self.lp.addConstraint(cut, name=f'node_{self.idx}_cglp_cut')

            # if not running against node specific constraints or bounds, add to other nodes in queue
            if not cglp_cumulative_constraints and not cglp_cumulative_bounds:
                return cut

        return None
