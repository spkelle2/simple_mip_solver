from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
import numpy as np
from scipy.sparse import csc_matrix
import time
from typing import Tuple, TypeVar, Iterable, Union

from simple_mip_solver.utils.branch_and_bound_tree import BranchAndBoundTree

CGLP = TypeVar('CGLP', bound='CutGeneratingLP')


class CutGeneratingLP:

    def __init__(self: CGLP, tree: BranchAndBoundTree, root_id: int, A: np.matrix = None,
                 b: CyLPArray = None, var_lb: CyLPArray = None, var_ub: CyLPArray = None,
                 depth: int = None):
        """ Creates an object that can generate a strong cut valid for the
        disjunction encoded in the subtree of <bb> rooted at node <root_id> by
        solving what is called the Cut Generating LP (CGLP).
        This cut by default is optimized to maximize the violation of the LP
        relaxation solution at node <root_id>, although this can be overwritten
        in the solve subroutine.

        :param bb: branch and bound tree from which we recover the disjunction
        :param root_id: id of the node off which we will base the disjunction
        :param A: The coefficient matrix to use for each disjunctive term's LP
        relaxation. If None, each disjunctive term will use its existing LP relaxation's
        coefficient matrix
        :param b: The RHS to be used for the constraints (assumed to be Ax >= b)
        in each disjunctive term's LP relaxation. If None, each disjunctive term
        will use its existing LP relaxation's RHS
        :param var_lb: Lower bound to place on the variables in each disjunctive term.
        If provided, the lower bound on each disjunctive term's variables is updated
        to be max(var_lb, <current lb>). If not, lower bounds are unchanged.
        :param var_ub: Upper bound to place on the variables in each disjunctive term.
        If provided, the upper bound on each disjunctive term's variables is updated
        to be min(var_ub, <current ub>). If not, upper bounds are unchanged.
        """
        init_start = time.time()

        # sanity checks
        assert isinstance(tree, BranchAndBoundTree), 'tree must be a BranchAndBoundTree instance'
        assert root_id in tree, 'root node of the disjunction must be present in B & B tree'
        if depth is not None:
            assert isinstance(depth, int) and depth > 0, 'depth is postive integer'

        self.tree = tree
        self.root_id = root_id
        self.depth = depth
        self.lp = self._create_cglp(A, b, var_lb, var_ub)
        self.cylp_failure = False
        self.init_time = time.time() - init_start

    # test by making sure we get the coef matrix, bounds, and objective we would expect for the given inputs
    # for passing nothing, coef matrix, and bounds
    def _create_cglp(self, A: np.matrix = None, b: CyLPArray = None,
                     var_lb: CyLPArray = None, var_ub: CyLPArray = None) -> CyClpSimplex:
        """ Create the cut generating LP, optionally overriding the constraints
        or variable bounds of the disjunctive terms' LP relaxations.

        See ISE 418 Lecture 13 slide 3, Lecture 14 slide 9, and Lecture 15 slides
        6-7 for derivation of LP in the model object constructed below.

        :param A: The coefficient matrix to use for each disjunctive term's LP
        relaxation. If None, each disjunctive term will use its existing LP relaxation's
        coefficient matrix
        :param b: The RHS to be used for the constraints (assumed to be Ax >= b)
        in each disjunctive term's LP relaxation. If None, each disjunctive term
        will use its existing LP relaxation's RHS
        :param var_lb: Lower bound to place on the variables in each disjunctive term.
        If provided, the lower bound on each disjunctive term's variables is updated
        to be max(var_lb, <current lb>). If not, lower bounds are unchanged.
        :param var_ub: Upper bound to place on the variables in each disjunctive term.
        If provided, the upper bound on each disjunctive term's variables is updated
        to be min(var_ub, <current ub>). If not, upper bounds are unchanged.
        :return: CyClpSimplex instance representing the CGLP
        """

        # get each disjunctive term (fathomed nodes do not expand disjunction, so remove them)
        disjunctive_nodes = {
            n.idx: n for n in self.tree.get_leaves(self.root_id, depth=self.depth,
                                                   keep='not infeasible')
        }
        var_dicts = [{v.name: v.dim for v in n.lp.variables} for n in disjunctive_nodes.values()]
        assert all(var_dicts[0] == d for d in var_dicts), \
            'Each disjunctive term should have the same variables. The feature allowing' \
            ' otherwise remains to be developed.'
        assert all(n._sense == '>=' for n in disjunctive_nodes.values()), \
            "all nodes assumed to be of form Ax >= b"

        # useful constants
        num_vars = sum(var_dim for var_dim in var_dicts[0].values())
        root = self.tree.get_node_instances(self.root_id)
        # assert root.solution is not None, 'root must be solved to create CGLP'
        inf = root.lp.getCoinInfinity()

        # sanity checks
        assert (A is None and b is None) or (A is not None and b is not None), \
            "A and b must both have values or must both be None"
        if A is not None:
            assert isinstance(A, np.matrix) or isinstance(A, csc_matrix), \
                "A must be a numpy or sparse csc matrix"
            assert A.shape[1] == num_vars, \
                "A must have same number of columns as each disjunctive term has variables"
        if b is not None:
            assert isinstance(b, CyLPArray), "b must be a CyLPArray"
            assert b.shape == (A.shape[0],), "A must have the same number of rows " \
                                             "as b has entries"
        if var_lb is not None:
            assert isinstance(var_lb, CyLPArray), "var_lb must be a CyLPArray"
            assert var_lb.shape == (num_vars,), "Must have same number of lower bounds as variables"
        else:
            var_lb = CyLPArray([-float('inf')] * num_vars)
        if var_ub is not None:
            assert isinstance(var_ub, CyLPArray), "var_ub must be a CyLPArray"
            assert var_ub.shape == (num_vars,), "Must have same number of upper bounds as variables"
        else:
            var_ub = CyLPArray([float('inf')] * num_vars)

        # set coefficients and bounds on variables in CGLP related to variable bounds in disjunction
        lb, ub, wb, vb = {}, {}, {}, {}
        for idx, n in list(disjunctive_nodes.items()):
            l = np.maximum(n.lp.variablesLower, var_lb)
            u = np.minimum(n.lp.variablesUpper, var_ub)
            if any(l > u):
                # this disjunctive term will be infeasible with new bounds, so remove it
                del disjunctive_nodes[idx]
            else:
                # collect the coefficients for columns of constraints arising from variable bounds
                # set coefficients to 0 when the variable is unbounded to avoid numerical errors in solver
                # adjusted lower bound from disjunction
                lb[idx] = CyLPArray([val if val > -inf else 0 for val in l])
                # adjusted upper bound from disjunction
                ub[idx] = CyLPArray([val if val < inf else 0 for val in u])

                # set corresponding variables in cglp to 0 to reflect there is no bound
                # i.e. this variable should not exist in cglp
                # bound on w - variable pertaining to lb constraints in disjunction
                wb[idx] = CyLPArray([inf if val > -inf else 0 for val in l])
                # bound on v - variable pertaining to ub constraints in disjunction
                vb[idx] = CyLPArray([inf if val < inf else 0 for val in u])

        # create dicts to hold coef matrices and rhs's
        constrs = {idx: {'A': A if A is not None else n.lp.coefMatrix,
                         'b': b if b is not None else CyLPArray(n.lp.constraintsLower)}
                   for idx, n in disjunctive_nodes.items()}

        # instantiate LP
        lp = CyClpSimplex()
        lp.logLevel = 0  # quiet output when resolving

        # declare variables
        pi = lp.addVariable('pi', num_vars)
        pi0 = lp.addVariable('pi0', 1)
        u = {idx: lp.addVariable(f'u_{idx}', constr['b'].size) for idx, constr in
             constrs.items()}  # one variable for each constraint in each disjunctive term
        w = {idx: lp.addVariable(f'w_{idx}', n.lp.nVariables) for idx, n in
             disjunctive_nodes.items()}
        v = {idx: lp.addVariable(f'v_{idx}', n.lp.nVariables) for idx, n in
             disjunctive_nodes.items()}

        # bound them
        for idx in disjunctive_nodes:
            lp += u[idx] >= 0
            lp += 0 <= w[idx] <= wb[idx]
            lp += 0 <= v[idx] <= vb[idx]

        # add constraints
        for i, constr in constrs.items():
            # (pi, pi0) must be valid for each disjunctive term's LP relaxation
            lp.addConstraint(0 >= -pi + constr['A'].T * u[i] + np.matrix(np.eye(num_vars)) * w[i] -
                             np.matrix(np.eye(num_vars)) * v[i], name=f'Au_{i} + Iw_{i} - Iv{i} <= pi')
            lp.addConstraint(0 <= -pi0 + constr['b'] * u[i] + lb[i] * w[i] - ub[i] * v[i],
                             name=f'bu_{i} + lbw_{i} - ubv{i} >= pi0')
        # normalize variables so they don't grow arbitrarily
        lp.addConstraint(sum(var.sum() for var_dict in [u, w, v] for var in var_dict.values()) == 1,
                         name='normalize')

        # set objective: find the deepest cut
        # since pi * x >= pi0 for all x in disjunction, we want min pi * x_star - pi0
        # lp.objective = CyLPArray(root.solution) * pi - pi0

        return lp

    def solve(self: CGLP, x_star: CyLPArray = None,
              starting_basis: Tuple[np.ndarray, np.ndarray] = None) -> \
            Tuple[Union[CyLPArray, None], Union[float, None]]:
        """ Finds the valid inequality that maximally separates x_star from the convex
        hull of the LP relaxations of each disjunctive term of the branch and bound
        tree within <self.bb> (If one exists and CyLP can avoid numerical errors in finding it).

        :param x_star: The point we want to cut off. If None, the CGLP will find a valid
        inequality that maximizes the separation of the solution of the LP relaxation
        of the node which had its ID provided at initialization.
        :return: a valid inequality (pi, pi0), i.e. pi^T x >= pi0 for all x in
        the convex hull of the disjunctive terms' LP relaxations
        """

        if x_star is not None:
            pi, pi0 = self.lp.getVarByName('pi'), self.lp.getVarByName('pi0')
            assert isinstance(x_star, CyLPArray), 'x_star must be a CyLPArray'
            assert x_star.shape == (pi.dim,), \
                'x_star must have the same number of variables as the LP relaxations ' \
                'in the branch and bound tree this instance was created with'
            self.lp.objective = x_star * pi - pi0
        if starting_basis is not None:
            assert isinstance(starting_basis, Iterable) and not isinstance(starting_basis, str) \
                and len(starting_basis) == 2, 'starting basis must be an iterable with two elements'
            for status_array in starting_basis:
                assert isinstance(status_array, np.ndarray), \
                    'elements of starting basis must be np.ndarrays'
            assert starting_basis[0].shape == (self.lp.nVariables,), \
                'first starting_basis element should give status for exactly each decision variable in CGLP'
            assert starting_basis[1].shape == (self.lp.nConstraints,), \
                'second starting_basis element should give status for exactly each slack variable in CGLP'
            self.lp.setBasisStatus(*starting_basis)

        # solve - use primal since objective is main thing changing in cutting plane method
        try:
            self.lp.primal()
        except:
            print('CLP tripped on its own shoelaces')
            self.cylp_failure = True
            return None, None

        if self.lp.getStatusCode() in [0, 2]:
            return CyLPArray(self.lp.primalVariableSolution['pi']), \
                   self.lp.primalVariableSolution['pi0'][0]
        else:
            # CGLP always has a solution. CyLP has a floating point issue in this case.
            self.cylp_failure = True
            return None, None
