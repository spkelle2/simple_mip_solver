from __future__ import annotations

from cylp.cy.CyClpSimplex import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from math import floor, ceil, degrees, acos
import numpy as np
import re
from statistics import median
from typing import Union, List, TypeVar, Dict, Any, Tuple

from simple_mip_solver.utils.floating_point import numerically_safe_cut
from simple_mip_solver.utils.tolerance import variable_epsilon,\
    good_coefficient_approximation_epsilon, max_nonzero_coefs, parallel_cut_tolerance, \
    cutting_plane_progress_tolerance, max_cut_generation_iterations
from test_simple_mip_solver.test_utils.test_utils import check_cut_against_grid

T = TypeVar('T', bound='BaseNode')


class BaseNode:
    """ A node off of which all other types of nodes can be built for running
    against objects defined in algorithms. This default implementation includes
    best-first search and most fractional branching.
    """

    def __init__(self: T, lp: CyClpSimplex, integer_indices: List[int], idx: int = None,
                 dual_bound: Union[float, int] = -float('inf'), b_idx: int = None,
                 b_dir: str = None, b_val: float = None, depth: int = 0,
                 ancestors: tuple = None, *args, **kwargs):
        """
        :param lp: model object simplex is run against. Assumed Ax >= b
        :param idx: index of this node (e.g. in the branch and bound tree)
        :param integer_indices: indices of variables we aim to find integer solutions
        :param dual_bound: starting lower bound on optimal objective value
        assuming minimization problem in this node
        :param b_idx: index of the branching variable
        :param b_dir: direction of branching
        :param b_val: initial value of the branching variable
        :param depth: how deep in the tree this node is
        :param ancestors: tuple of nodes that preceded this node (e.g. were branched
        on to create this node)
        :param args: spillover for extra arguments passed by the API not needed for instantiation
        :param kwargs: spillover for extra arguments passed by the API not needed for instantiation
        """
        assert isinstance(lp, CyClpSimplex), 'lp must be CyClpSimplex instance'
        assert all(0 <= idx < lp.nVariables and isinstance(idx, int) for idx in
                   integer_indices), 'indices must match variables'
        assert idx is None or isinstance(idx, int), 'node idx must be integer if provided'
        assert len(set(integer_indices)) == len(integer_indices), \
            'indices must be distinct'
        assert isinstance(dual_bound, float) or isinstance(dual_bound, int), \
            'dual bound must be a float or an int'
        assert (b_dir is None) == (b_idx is None) == (b_val is None), \
            'none are none or all are none'
        assert b_idx in integer_indices or b_idx is None, \
            'branch index corresponds to integer variable if it exists'
        assert b_dir in ['right', 'left'] or b_dir is None, \
            'we can only branch right or left'
        if b_val is not None:
            good_left = 0 < b_val - lp.variablesUpper[b_idx] < 1
            good_right = 0 < lp.variablesLower[b_idx] - b_val < 1
            assert (b_dir == 'left' and good_left) or \
                   (b_dir == 'right' and good_right), 'branch val should be within 1 of both bounds'
        assert isinstance(depth, int) and depth >= 0, 'depth is a positive integer'
        if ancestors is not None:
            assert isinstance(ancestors, tuple), 'ancestors must be a tuple if provided'
            assert idx not in ancestors, 'idx cannot be an ancestor of itself'

        lp.logLevel = 0
        self.lp = lp
        self._integer_indices = integer_indices
        self.idx = idx
        self.dual_bound = dual_bound
        self.objective_value = None
        self.solution = None
        self.lp_feasible = None
        self.unbounded = None
        self.mip_feasible = None
        self._b_dir = b_dir
        self._b_idx = b_idx
        self._b_val = b_val
        self.depth = depth
        self.search_method = 'best first'
        self.branch_method = 'most fractional'
        self.is_leaf = True
        ancestors = ancestors or tuple()
        idx_tuple = (idx,) if idx is not None else tuple()
        self.lineage = ancestors + idx_tuple or None  # test all 4 ways this can pan out
        self.cut_generation_iterations = 0
        self.cut_name_pattern = re.compile('^cut_')
        self.cut_generation_stalled = False

    def bound(self: T, **kwargs: Any) -> Dict[str, Any]:
        """ Wrapper function for calling the base bound subroutine. This function
        is here to match the bound routine in all sub classes so that they are
        called the same

        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return:
        """
        self._base_bound(**kwargs)
        return {}

    def _base_bound(self: T, max_cut_generation_iterations: int = max_cut_generation_iterations,
                    **kwargs) -> None:
        """ bound subroutine to be shared by all superclasses. Bounds the LP
        relaxation then calls the cut generation subroutine

        :param max_cut_generation_iterations: max number of times to call
        cut generation
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return:
        """
        assert isinstance(max_cut_generation_iterations, int) and max_cut_generation_iterations > 0, \
            'max_cut_generation_iterations must be a positive integer'

        self._bound_lp()
        while self.lp_feasible and not self.mip_feasible and not self.cut_generation_stalled and \
                self.cut_generation_iterations < max_cut_generation_iterations:
            self._cut_generation_iteration(**kwargs)

    def _bound_lp(self: T) -> None:
        """Solve the current node with simplex to generate a bound on objective
        values of integer feasible solutions of descendent nodes. If feasible,
        save the run solution.

        :return: a placeholder dictionary for return that the branch and bound
        algorithm expects
        """
        assert self._x_only_variable, 'x must be our only variable'
        self.lp.dual()
        self.lp_feasible = self.lp.getStatusCode() in [0, 2]  # optimal or dual infeasible
        self.unbounded = self.lp.getStatusCode() == 2
        self.objective_value = self.lp.objectiveValue if self.lp_feasible else float('inf')
        # first cyclpsimplex has variables keyed, rest are list
        sol = self.lp.primalVariableSolution
        self.solution = None if not self.lp_feasible else sol['x'] if \
            type(sol) == dict else sol
        int_var_vals = None if not self.lp_feasible else self.solution[self._integer_indices]
        self.mip_feasible = self.lp_feasible and \
                            np.max(np.abs(np.round(int_var_vals) - int_var_vals)) <= variable_epsilon

    def _cut_generation_iteration(self: T, **kwargs: Any) -> None:
        """ Generate cuts to refine the current LP relaxation.

        :param max_cut_generation_iterations: number of rounds of cut generation to perform
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of cuts that can be added to other instances
        """
        assert all(self.solution > -variable_epsilon), 'we must have x >= 0'

        # bring up anything close to 0 to avoid numerical errors
        self.solution = np.maximum(self.solution, 0)
        self.cut_generation_iterations += 1
        prev_objective_value = self.objective_value

        self._remove_slack_cuts(**kwargs)
        cut_pool = self._generate_cuts(**kwargs)
        self._select_cuts(cut_pool=cut_pool, **kwargs)
        self._bound_lp()
        if abs(prev_objective_value - self.objective_value)/abs(prev_objective_value) < \
                cutting_plane_progress_tolerance:
            self.cut_generation_stalled = True

    def _remove_slack_cuts(self: T, **kwargs) -> None:
        """ Removes all previously added cutting planes with 0 dual value. I.e.
        removes all cutting planes that wont change the optimal objective

        :param kwargs:
        :return:
        """
        removable_idxs = [constr_name for constr_name, dual_values in
                          self.lp.dualConstraintSolution.items() if all(dual_values == 0)
                          and self.cut_name_pattern.match(constr_name)]
        for idx in removable_idxs:
            self.lp.removeConstraint(idx)

    def _generate_cuts(self: T, gomory_cuts: bool = True, **kwargs) -> \
            Dict[str: Union[CyLPArray, float]]:
        """ Generates one round of cuts

        :param gomory_cuts: if True, add gomory cuts to LP relaxation
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of cuts that can be added to the LP relaxation
        """
        assert isinstance(gomory_cuts, bool), 'gomory_cuts is boolean'

        cut_pool = {}

        if gomory_cuts:
            for row_idx, (pi, pi0) in self._find_gomory_cuts().items():
                idx = f'cut_gomory_{self.idx}_{self.cut_generation_iterations}_{row_idx}'
                safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='over')
                cut_pool[idx] = safe_pi, safe_pi0

        return cut_pool

    def _select_cuts(self, cut_pool: Dict[str: Union[CyLPArray, float]] = None,
                     max_nonzero_coefs: int = max_nonzero_coefs,
                     parallel_cut_tolerance: float = parallel_cut_tolerance,
                     **kwargs) -> Dict[str, Union[CyLPArray, float]]:
        """ Pick the best subset of cuts from the cut pool. Best is defined by
        deepest cuts that are not too parallel to one another.

        :param cut_pool: dictionary of cuts that can be added
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of cuts chosen to be added
        """
        cut_pool = cut_pool or {}
        for idx, (pi, pi0) in cut_pool.items():
            assert isinstance(pi, CyLPArray), 'pi must be CyLPArray'
            assert isinstance(pi0, (int, float)), 'pi0 must be number'
        assert isinstance(max_nonzero_coefs, int) and 0 < max_nonzero_coefs, \
            'max_nonzero_coefs must be positive int'
        assert 0 < parallel_cut_tolerance <= 90, \
            'parallel_cut_tolerance must be number in (0, 90]'

        def nonzero_coefs(pi):
            return sum((pi > good_coefficient_approximation_epsilon) +
                       (pi < -good_coefficient_approximation_epsilon))

        # use euclidean distance on cut to normalize for different coef scales
        # to avoid numerical errors ensure reasonable amount of nonzero coefs
        cut_depths = {idx: (np.dot(pi, self.solution) - pi0) / np.linalg.norm(pi)
                      for idx, (pi, pi0) in cut_pool.items() if
                      0 < nonzero_coefs(pi) <= max_nonzero_coefs}
        added_cuts = {}

        # add cuts in order of depth of violation
        for idx in sorted(cut_depths, key=cut_depths.get):
            # if cuts are no longer violated by current optimal solution, break
            if cut_depths[idx] >= 0:
                break
            (pi, pi0) = cut_pool[idx]
            parallel_cut = False
            # select most useful cuts by ensuring >10 degrees between this cut and others added
            for pi_prime, pi0_prime in added_cuts.values():
                # take the median to avoid floating point error causing arc cos to fail
                cos_theta = median(
                    [-1, np.dot(pi, pi_prime) / (np.linalg.norm(pi) * np.linalg.norm(pi_prime)), 1]
                )
                if degrees(acos(cos_theta)) < parallel_cut_tolerance:
                    parallel_cut = True
                    break
            if not parallel_cut:
                # uncomment to check if cut is valid
                # check_cut_against_grid(lp=self.lp, pi=pi, pi0=pi0, max_val=5)
                cut = pi * self.lp.getVarByName('x') >= pi0
                self.lp.addConstraint(cut, idx)
                added_cuts[idx] = (pi, pi0)

        # not needed in cut generation routine but helpful for testing
        return added_cuts

    def _find_gomory_cuts(self: T) -> Dict[int: Tuple[CyLPArray, float]]:
        """Find Gomory Mixed Integer Cuts (GMICs) for this node's solution.
        Defined in Lehigh University ISE 418 lecture 14 slide 18 and 5.31
        in Conforti et al Integer Programming. Assumes Ax >= b and x >= 0.

        Warning: this method does not currently work. It creates some cuts that
        are invalid. Stuck on debugging because I can't spot the difference between
        this and what is referenced above.

        :return: a dict of tuples (pi, pi0) that represent the cut pi*x >= pi0.
        Each tuple is indexed by the row in the LP tableau that generated the cut
        """
        cuts = {}
        tableau = self.tableau
        if tableau is None:
            return cuts
        for row_idx, basic_idx in enumerate(self.basic_variable_indices):
            if basic_idx in self._integer_indices and \
                    self._is_fractional(self.solution[basic_idx]):
                f0 = self._get_fraction(self.solution[basic_idx])
                # if whole number or close, skip to avoid numerical issues from division
                if f0 < good_coefficient_approximation_epsilon or \
                        f0 + good_coefficient_approximation_epsilon > 1:
                    continue
                # 0 for basic variables avoids getting small numbers that should be zero
                f = {i: 0 if i in self.basic_variable_indices else
                     self._get_fraction(tableau[row_idx, i]) for i in
                     range(self.lp.nVariables)}
                # values for continuous variables
                a = {i: 0 if i in self.basic_variable_indices else tableau[row_idx, i]
                     for i in range(self.lp.nVariables)}
                # primary variable coefficients in GMI cut
                pi = CyLPArray(
                    [f[j]/f0 if f[j] <= f0 and j in self._integer_indices else
                     (1 - f[j])/(1 - f0) if j in self._integer_indices else
                     a[j]/f0 if a[j] > 0 else -a[j]/(1 - f0) for j in range(self.lp.nVariables)]
                )
                # slack variable coefficients in GMI cut
                pi_slacks = np.array([x/f0 if x > 0 else -x/(1 - f0) for x in
                                      tableau[row_idx, self.lp.nVariables:]])
                # sub out slack variables for primary variables. Ax >= b =>
                # Ax - s = b => s = Ax - b. gomory is pi^T * x + pi_s^T * s >= 1, thus
                # pi^T * x + pi_s^T * (Ax - b) >= 1 => (pi + A^T * pi_s)^T * x >= 1 + pi_s^T * b
                coefs = pi + self.lp.coefMatrix.T * pi_slacks
                rhs = 1 + np.dot(pi_slacks, self.lp.constraintsLower)
                # append coefs^T * x >= rhs
                cuts[row_idx] = (coefs, rhs)
        return cuts

    @property
    def tableau(self):
        """CyLP builds the tableau incorrectly, so building from scratch. Assumes Ax >= b"""
        B = self.basic_variable_indices
        # There should be nConstrs basic variables but sometimes CyLP confuses itself
        if len(B) != self.lp.nConstraints:
            return None
        A = np.concatenate((self.lp.coefMatrix.toarray(), -np.identity(self.lp.nConstraints)), axis=1)
        A_B = A[:, B]
        A_B_inv = np.linalg.inv(A_B)
        return A_B_inv @ A
    
    @property
    def basic_variable_indices(self):
        return np.where(np.concatenate(self.lp.getBasisStatus()) == 1)[0]

    def branch(self: T, **kwargs: Any) -> Dict[str, T]:
        """ Creates two new nodes which are branched on the most fractional index
        of this node's LP relaxation solution

        :param kwargs: a dictionary to hold unneeded arguments sent by a general
        branch and bound method
        :return: list of Nodes with the new bounds
        """
        branch_idx = self._most_fractional_index
        return self._base_branch(branch_idx, **kwargs)

    # implementation of most fractional branch
    @property
    def _most_fractional_index(self: T) -> int:
        """ Returns the index of the integer variable with current value furthest from
        being integer. If one does not exist or the problem has not yet been solved,
        returns None.

        :return furthest_index: index corresponding to variable with most fractional
        value
        """
        furthest_index = None
        furthest_dist = variable_epsilon
        if self.lp_feasible:
            for idx in self._integer_indices:
                dist = min(self.solution[idx] - floor(self.solution[idx]),
                           ceil(self.solution[idx]) - self.solution[idx])
                if dist > furthest_dist:
                    furthest_dist = dist
                    furthest_index = idx
        return furthest_index

    def _base_branch(self: T, branch_idx: int, next_node_idx: int = None,
                     **kwargs: Any) -> Dict[str, T]:
        """ Creates two new copies of the node with new bounds placed on the variable
        with index <idx>, one with the variable's lower bound set to the ceiling
        of its current value and another with the variable's upper bound set to
        the floor of its current value.

        :param branch_idx: index of variable to branch on
        :param next_node_idx: index that should be assigned to the next node created.
        If left None, assign no indices to both child nodes.

        :return: dict of Nodes with the new bounds keyed by direction they branched
        """
        assert self._x_only_variable, 'x must be our only variable'
        assert next_node_idx is None or isinstance(next_node_idx, int), \
            'next node index should be integer if provided'
        assert self.lp_feasible, 'must solve before branching'
        assert branch_idx in self._integer_indices, 'must branch on integer index'
        b_val = self.solution[branch_idx]
        assert self._is_fractional(b_val), "index branched on must be fractional"

        self.is_leaf = False

        # get end basis to warm start the children
        # appears to be tuple  (variable statuses, slack statuses)
        basis = self.lp.getBasisStatus()

        # create new lp's for each direction
        children = {'right': CyClpSimplex(), 'left': CyClpSimplex()}
        for direction, lp in children.items():
            x = lp.addVariable('x', self.lp.nCols)
            l = CyLPArray(self.lp.variablesLower.copy())
            u = CyLPArray(self.lp.variablesUpper.copy())
            if direction == 'left':
                u[branch_idx] = floor(b_val)
            else:
                l[branch_idx] = ceil(b_val)
            lp += l <= x <= u
            for constr in self.lp.constraints:
                lp.addConstraint(
                    CyLPArray(constr.lower.copy()) <= constr.varCoefs[constr.variables[0]] * x
                    <= CyLPArray(constr.upper.copy()), name=constr.name
                )
            lp.objective = self.lp.objective.copy()
            lp.setBasisStatus(*basis)  # warm start

        # return instances of the subclass that calls this function
        return {
            'left': type(self)(
                lp=children['left'], integer_indices=self._integer_indices,
                idx=next_node_idx, dual_bound=self.objective_value, b_idx=branch_idx,
                b_dir='left', b_val=b_val, depth=self.depth + 1, ancestors=self.lineage,
                **kwargs
            ),
            'right': type(self)(
                lp=children['right'], integer_indices=self._integer_indices,
                idx=next_node_idx + 1 if next_node_idx is not None else next_node_idx,
                dual_bound=self.objective_value, b_idx=branch_idx, b_dir='right',
                b_val=b_val, depth=self.depth + 1, ancestors=self.lineage, **kwargs
            ),
            'next_node_idx': next_node_idx + 2 if next_node_idx is not None else next_node_idx
        }

    def _strong_branch(self: T, idx: int, iterations: int = 5) -> Dict[str, T]:
        """ Run <iterations> iterations of dual simplex starting from the
        optimal solution of this node after branching on index <idx>. Returns
        bounds of both branches if feasible, None if false

        :param idx: which index to branch on
        :param iterations: how many iterations of dual simplex to perform
        :return: dict of nodes with attributes showing changes in bounds for
        feasible branches. Looks like: {'left': <node from branching down/left>,
        'right': <node from branching up/right>}
        """
        assert isinstance(iterations, int) and iterations > 0, \
            'iterations must be positive integer'
        nodes = {k: v for k, v in self._base_branch(idx).items()
                 if k in ['left', 'right']}
        for n in nodes.values():
            n.lp.maxNumIteration = iterations
            n.lp.dual()
        return nodes

    def _is_fractional(self: T, value: Union[int, float]) -> bool:
        """Returns True if value fractional, False if not.

        :param value: value to check if fractional
        :return: boolean of value is fractional
        """
        assert isinstance(value, (int, float)), 'value should be a number'
        return min(value - floor(value), ceil(value) - value) > variable_epsilon

    @staticmethod
    def _get_fraction(value: Union[int, float]) -> Union[int, float]:
        """Returns fractional part of value

        :param value: value to return decimal part from
        :return: decimal part of value
        """
        assert isinstance(value, (int, float)), 'value should be a number'
        return value - floor(value)

    # implementation of best first search
    def __eq__(self: T, other):
        if isinstance(other, BaseNode):
            return self.dual_bound == other.dual_bound
        else:
            raise TypeError('A Node can only be compared with another Node')

    # self < other means self gets better priority in priority queue
    # want priority to go to node with lowest lower_bound
    def __lt__(self: T, other):
        if isinstance(other, BaseNode):
            return self.dual_bound < other.dual_bound
        else:
            raise TypeError('A Node can only be compared with another Node')

    @property
    def _sense(self: T):
        inf = self.lp.getCoinInfinity()
        lower_bounded = self.lp.constraintsLower.max() > -inf
        upper_bounded = self.lp.constraintsUpper.min() < inf
        assert not (lower_bounded and upper_bounded),\
            "all constraints should be bounded same way"
        return '<=' if upper_bounded else '>='

    @property
    def _variables_nonnegative(self: T):
        """ Determines if all variables in the lp model are bound to be nonnegative

        :return:
        """
        return (self.lp.variablesLower >= 0).all()

    @property
    def _x_only_variable(self: T):
        """ Determine if x is the only variable in our model

        :return:
        """
        return len(self.lp.variables) == 1 and self.lp.variables[0].name == 'x'
