import numpy as np
from coinor.cuppy.milpInstance import MILPInstance
from coinor.gimpy.tree import BinaryTree
from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
from queue import PriorityQueue
import time
from typing import Any, Dict, TypeVar, List, Tuple, Union, Iterable

from simple_mip_solver.algorithms.utils import Utils
from simple_mip_solver.nodes.base_node import BaseNode, T
from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

B = TypeVar('B', bound='BranchAndBound')

BT = TypeVar('BT', bound='BranchAndBoundTree')


class BranchAndBoundTree(BinaryTree):
    """Class used to represent the underlying tree structure of branch and bound"""

    def get_leaves(self: BT, subtree_root_id: int) -> List[BaseNode]:
        """ Gather all leaves of a subtree rooted at node with id <subtree_root_id>

        :param subtree_root_id: The id of the node that roots our subtree
        :return: the leaves of the subtree
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        return [n.attr['node'] for n in self.nodes.values() if n.attr['node'].is_leaf
                and subtree_root_id in n.attr['node'].lineage]

    # make this work with just one node passed
    def get_node_instances(self: BT, node_ids: Union[int, Iterable[int]]) -> List[BaseNode]:
        is_int = False
        if isinstance(node_ids, int):
            is_int = True
            node_ids = [node_ids]
        else:
            assert isinstance(node_ids, Iterable) and not isinstance(node_ids, str), \
                'node_ids must be an integer or iterable (that is not a string)'
            node_ids = list(node_ids)
        missing_ids = set(node_ids) - set(self.nodes)
        assert not missing_ids, f'the following node_ids are not in the tree: {missing_ids}'
        instances = [self.nodes[idx].attr.get('node') for idx in node_ids]
        assert all(instance is not None for instance in instances), \
            'each vertex in the branch and bound tree must have an attribute for a node instance'
        return instances if not is_int else instances[0]


class BranchAndBound(Utils):
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    _node_attributes = ['lower_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible', 'search_method',
                        'branch_method', 'idx', 'lp']  # move lp to public
    _node_funcs = ['bound', 'branch', '__lt__', '__eq__']
    _queue_funcs = ['put', 'get', 'empty']

    # these kwargs get passed to the branch and bound functions
    # so **kwargs = {'strong_branch_iters': 5, 'pseudo_costs': {}}
    def __init__(self: B, model: MILPInstance, Node: Any = BaseNode,
                 node_queue: Any = None, node_limit: int = float('inf'),
                 mip_gap: float = .0001, **kwargs: Any):
        f""" Instantiates a Branch and Bound instance.
        
        :param model: A MILPInstance object that defines the MILP we solve
        :param Node: A class containing attributes {self._node_attributes} and methods
        {self._node_funcs}. Represents a single node in the branch and bound tree.
        :param node_queue: An object containing methods {self._queue_funcs}.
        This object is what holds and prioritizes nodes to be solved in branch and
        bound.
        :param node_limit: if provided, max number of nodes to explore
        :param standardize_model: if True, converts model to Ax >= b with bounds
        moved to constraints to enable display and cutting plane methods
        :param kwargs: dictionary passed to the branch and bound functions as
        key worded arguments and which adds keys and updates values based on
        what is returned
        """
        node_queue = node_queue or PriorityQueue()

        # call super
        super().__init__(model=model, Node=Node, node_attributes=self._node_attributes,
                         node_funcs=self._node_funcs, **kwargs)

        # node_queue asserts
        for func in self._queue_funcs:
            c = getattr(node_queue, func, None)
            assert callable(c), f'node_queue needs a {func} function'

        # node_limit assert
        assert node_limit == float('inf') or (isinstance(node_limit, int) and node_limit > 0), \
            "node limit must be positive integer or infinity"

        # mip_gap assert
        assert 0 <= mip_gap < 1, 'mip_gap is a ratio between 0 and 1'

        # kwargs assert
        assert set(kwargs.keys()).isdisjoint({'right', 'left'}), \
            'keys "right" and "left" are saved for later use'
        assert all(isinstance(k, str) for k in kwargs), 'kwargs keys must be strings'

        # instantiate
        self._node_queue = node_queue
        self._unbounded = None
        self._best_solution = None
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None
        self.global_upper_bound = float('inf')
        self.global_lower_bound = -float('inf')
        self.node_limit = node_limit
        self.tree = BranchAndBoundTree()
        self.tree.add_root(self.root_node.idx, node=self.root_node)
        self.solve_time = 0
        self.mip_gap = mip_gap

    @property
    def current_gap(self):
        if self.global_upper_bound == self.global_lower_bound == 0:
            gap = 0
        elif self.global_upper_bound == 0:
            gap = float('inf')
        elif self.global_upper_bound == float('inf'):
            gap = None
        else:
            gap = abs(self.global_upper_bound - self.global_lower_bound)/abs(self.global_upper_bound)
        return gap

    def solve(self: B) -> None:
        """Solves the Branch and Bound algorithm using the bound, search, and
        branching methods provided by the Node class and node_queue we instatiated
        the BranchAndBound instance with.

        :return:
        """
        start = time.process_time()
        if self.status == 'unsolved':
            self._node_queue.put(self.root_node)

        while not (self._node_queue.empty() or self._unbounded or
                   self.evaluated_nodes >= self.node_limit or
                   (self.current_gap is not None and self.current_gap <= self.mip_gap)):
            self._evaluate_node(self._node_queue.get())

        run_time = time.process_time() - start
        self.solve_time += run_time
        self.status = 'unbounded' if self._unbounded else 'infeasible' if \
            self._node_queue.empty() and self.global_upper_bound == float('inf') else \
            'optimal' if self.global_upper_bound < float('inf') and self.current_gap <= self.mip_gap \
            else 'stopped on iterations'
        self.solution = self._best_solution
        self.objective_value = self.global_upper_bound

    def _evaluate_node(self: B, node: T) -> None:
        """Bounds and optionally branches on the given node. Updates any attributes
        with the values keyed in the rtn's for bound and branch methods.

        :param node: the object that is bounded and potentially branched on.
        :return:
        """
        if node.lower_bound < self.global_upper_bound:
            self.evaluated_nodes += 1

            self._process_rtn(node.bound(**self._kwargs))

            # this solver is not designed to handle unboundedness accurately
            # need feasible milp solution, but may never find one, so assumes we do
            if node.unbounded:
                self._unbounded = True

            if node.lp_feasible and node.objective_value < self.global_upper_bound:
                if node.mip_feasible:
                    self._best_solution = node.solution
                    self.global_upper_bound = node.objective_value
                else:
                    self._process_branch_rtn(node.idx, node.branch(**self._kwargs))

        self.global_lower_bound = min([n.lower_bound for n in self._node_queue.queue] +
                                      [self.global_upper_bound])

    def _process_branch_rtn(self: B, parent_id: int, rtn: Dict[str, Any]):
        """ Pull the nodes returned from branching out of the rtn dict and into
        the node queue and branch and bound tree before updating the rest of the
        key value pairs in _kwargs

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        assert isinstance(parent_id, int), 'parent_id must be integer'
        assert parent_id in self.tree, 'parent must already exist in tree'
        # left is down right is up
        for direction in ['left', 'right']:
            assert direction in rtn, f'{direction} must be in the returned dict'
            assert isinstance(rtn[direction], self._Node), \
                f'{direction} value must be type {type(self._Node)}'
            assert rtn[direction].idx not in self.tree, 'please give unique node ID'
            self._node_queue.put(rtn[direction])
            getattr(self.tree, f'add_{direction}_child')(rtn[direction].idx, parent_id,
                                                         node=rtn[direction])
            del rtn[direction]
        self._process_rtn(rtn)

    # todo: refactor for multiple constraints and get rid of change in b
    # entrust the user to better know what they're doing using this function with cuts added
    # enforce above that branching is done by bounding variables and not adding constraints
    # i don't even think we need that enforcement because it would be cpatured in a constraint
    def dual_bound(self, b: CyLPArray) -> float:
        """ Calculates a lower bound on the optimal objective value of the current
        MIP at a new RHS b by evaluating the dual function (BB.D from ISE 418 Lecture 8
        Slide 29) at b. Assumes the underlying LP relaxations all have a single
        constraint object.

        :param b: new RHS to evaluate
        :return: a lower bound for the value function of the MIP evaluated at b
        """
        assert isinstance(b, CyLPArray), 'this function only works with CyLP arrays'
        assert self.status != 'unsolved', 'must solve this instance before using this method'
        terminal_nodes = self.tree.get_leaves(self.root_node.idx)
        multi_const_nodes = [n.idx for n in terminal_nodes if len(n.lp.constraints) != 1]
        assert not multi_const_nodes, \
            f'This feature expects the root node to have a single constraint object and ' \
            f'all nodes to branch by bounding variables instead of by adding constraints. ' \
            f'It does not currently handle cuts being added after bounding. The following ' \
            f'IDs belong to nodes that do not conform to these rules: {multi_const_nodes}'
        assert all(b.shape == n.lp.constraints[0].lower.shape for n in terminal_nodes), \
            'the shape of the RHS being added should match that of each node'
        if self._swapped_constraint_direction:
            b = -b
            print('WARNING: your rhs was made negative to reflect constraints'
                  ' flipping direction at instantiation')
        # note: after first call, this routine will not bound duals again since they already are
        # also does not update other attributes of the node to reflect its values from solve
        infeasible_nodes = [n for n in terminal_nodes if n.lp.getStatusCode() == 1]
        for n in infeasible_nodes:
            n.lp = self._bound_dual(n.lp)

        assert all(n.lp.getStatusCode() in [-1, 0] for n in terminal_nodes)

        # We take a max in each lineage because when minimizing b/c ancestors' LP relaxes
        # are LB on terminal node objective value. Since ancestor dual evals at b are
        # LB for for each ancestor node, they are also LB's for terminal node
        bounds = {}
        for node in terminal_nodes:
            bounds[node.idx] = max(
                np.inner(n.lp.dualConstraintSolution[n.lp.constraints[0].name], b) +
                np.inner(np.maximum(np.concatenate([sol for sol in n.lp.dualVariableSolution.values()]), np.zeros(n.lp.nVariables)), n.lp.variablesLower) +
                np.inner(np.minimum(np.concatenate([sol for sol in n.lp.dualVariableSolution.values()]), np.zeros(n.lp.nVariables)), n.lp.variablesUpper)
                for n in self.tree.get_node_instances(node.lineage)
            )
        return min(bounds.values())

    def _bound_dual(self, cur_lp: CyClpSimplex) -> CyClpSimplex:
        """ Place a bound on each index of the dual variable associated with the
        constraints of this node's LP relaxation and resolve. We do this by adding
        to each constraint i the slack variable 's_i' in the node's LP relaxation,
        and we give each new variable a large, positive coefficient in the objective.
        By duality, we get the desired dual LP constraints. Therefore, for nodes
        with infeasible primal LP relaxations and unbounded dual LP relaxations,
        resolving gives us a finite (albeit very large) dual solution, which can
        be used to parametrically lower bound the objective value of this node as
        we change its right hand side.

        :param cur_lp: the CyClpSimplex instance for which we want to bound its
        dual solution and resolve
        :return: A CyClpSimplex instance representing the same model as input,
        with the additions prescribed in the method description
        """

        # we add slack at end so user does not have to worry about adding to each node they create
        # and so we don't have to go back and update needlessly
        assert isinstance(cur_lp, CyClpSimplex), 'must give CyClpSimplex instance'
        for i, constr in enumerate(cur_lp.constraints):
            assert f's_{i}' not in [v.name for v in cur_lp.variables], \
                f"variable 's_{i}' is a reserved name. please name your variable something else"

        # cylp lacks (a documented) way to add a column, so rebuild the LP :[
        new_lp = CyClpSimplex()
        new_lp.logLevel = 0  # quiet output when resolving

        # recreate variables
        var_map = {v: new_lp.addVariable(v.name, v.dim) for v in cur_lp.variables}

        # bound them
        for orig_v, new_v in var_map.items():
            new_lp += CyLPArray(orig_v.lower) <= new_v <= CyLPArray(orig_v.upper)

        # re-add constraints, with slacks this time
        s = {}
        for i, constr in enumerate(cur_lp.constraints):
            s[i] = new_lp.addVariable(f's_{i}', constr.nRows)
            new_lp += s[i] >= CyLPArray(np.zeros(constr.nRows))
            new_lp += CyLPArray(constr.lower) <= \
                sum(constr.varCoefs[v] * var_map[v] for v in constr.variables) \
                + np.matrix(np.identity(constr.nRows))*s[i] <= CyLPArray(constr.upper)

        # set objective
        new_lp.objective = sum(
            CyLPArray(cur_lp.objectiveCoefficients[orig_v.indices]) * new_v for
            orig_v, new_v in var_map.items()
        ) + sum(self._M * v.sum() for v in s.values())

        # warm start
        orig_var_status, orig_slack_status = cur_lp.getBasisStatus()
        # each s_i at lower bound of 0 when added - status 3
        np.concatenate((orig_var_status, np.ones(sum(v.dim for v in s.values()))*3))
        new_lp.setBasisStatus(orig_var_status, orig_slack_status)

        # rerun and reassign
        new_lp.dual(startFinishOptions='x')
        return new_lp

    def find_strong_disjunctive_cut(self, root_id: int) -> Tuple[CyLPArray, float]:
        """ Generate a strong cut valid for the disjunction encoded in the subtree
        rooted at node <root_id>. This cut is optimized to maximize the violation
        of the LP relaxation solution at node <root_id>

        see ISE 418 Lecture 13 slide 3, Lecture 14 slide 9, and Lecture 15 slides
        6-7 for derivation

        :param root_id: id of the node off which we will base the disjunction
        :return: a valid inequality (pi, pi0), i.e. pi^T x >= pi0 for all x in
        the convex hull of the disjunctive terms' LP relaxations
        """
        # sanity checks
        assert root_id in self.tree, 'parent must already exist in tree'
        root = self.tree.get_node_instances(root_id)
        # get each disjunctive term
        terminal_nodes = self.tree.get_leaves(root_id)
        # terminal nodes pruned for infeasibility do not expand disjunction, so remove them
        disjunctive_nodes = {n.idx: n for n in terminal_nodes if n.lp_feasible is not False}
        var_dicts = [{v.name: v.dim for v in n.lp.variables} for n in disjunctive_nodes.values()]
        assert all(var_dicts[0] == d for d in var_dicts), \
            'Each disjunctive term should have the same variables. The feature allowing' \
            ' otherwise remains to be developed.'

        # useful constants
        num_vars = sum(var_dim for var_dim in var_dicts[0].values())
        inf = root.lp.getCoinInfinity()

        # set infinite lower/upper bounds to 0 so they don't create numerical issues in constraints
        lb = {idx: CyLPArray([val if val > -inf else 0 for val in n.lp.variablesLower])
              for idx, n in disjunctive_nodes.items()}  # adjusted lower bound
        ub = {idx: CyLPArray([val if val < inf else 0 for val in n.lp.variablesUpper])
              for idx, n in disjunctive_nodes.items()}  # adjusted upper bound

        # set corresponding variables in cglp to 0 to reflect there is no bound
        # i.e. this variable should not exist in cglp
        wb = {idx: CyLPArray([inf if val > -inf else 0 for val in n.lp.variablesLower])
              for idx, n in disjunctive_nodes.items()}  # w bounds - variable on lb constraints
        vb = {idx: CyLPArray([inf if val < inf else 0 for val in n.lp.variablesUpper])
              for idx, n in disjunctive_nodes.items()}  # v bounds - variable on ub constraints

        # instantiate LP
        cglp = CyClpSimplex()
        cglp.logLevel = 0  # quiet output when resolving

        # declare variables (what to do with case when we have degenerate constraint)
        pi = cglp.addVariable('pi', num_vars)
        pi0 = cglp.addVariable('pi0', 1)
        u = {idx: cglp.addVariable(f'u_{idx}', n.lp.nConstraints) for idx, n in
             disjunctive_nodes.items()}
        w = {idx: cglp.addVariable(f'w_{idx}', n.lp.nVariables) for idx, n in
             disjunctive_nodes.items()}
        v = {idx: cglp.addVariable(f'v_{idx}', n.lp.nVariables) for idx, n in
             disjunctive_nodes.items()}

        # bound them
        for idx in disjunctive_nodes:
            cglp += u[idx] >= 0
            cglp += 0 <= w[idx] <= wb[idx]
            cglp += 0 <= v[idx] <= vb[idx]

        # add constraints
        for i, n in disjunctive_nodes.items():
            # (pi, pi0) must be valid for each disjunctive term's LP relaxation
            cglp += 0 >= -pi + n.lp.coefMatrix.T * u[i] + \
                np.matrix(np.eye(num_vars)) * w[i] - np.matrix(np.eye(num_vars)) * v[i]
            cglp += 0 <= -pi0 + CyLPArray(n.lp.constraintsLower) * u[i] + \
                lb[i] * w[i] - ub[i] * v[i]
        # normalize variables so they don't grow arbitrarily
        cglp += sum(var.sum() for var_dict in [u, w, v] for var in var_dict.values()) == 1

        # set objective: find the deepest cut
        # since pi * x >= pi0 for all x in disjunction, we want min pi * x_star - pi0
        cglp.objective = CyLPArray(root.solution) * pi - pi0

        # solve
        cglp.primal(startFinishOptions='x')
        assert cglp.getStatusCode() == 0, 'we should get optimal solution'
        assert cglp.objectiveValue <= 0, 'pi * x >= pi0 -> pi * x - pi0 >= 0 -> ' \
            'negative objective at x^* since it gets cut off'

        # get solution
        return cglp.primalVariableSolution['pi'], cglp.primalVariableSolution['pi0']


if __name__ == '__main__':
    bb = BranchAndBound(small_branch, Node=PseudoCostBranchNode)
    bb.solve()
    print(f"objective: {bb.objective_value}")
    print(f"solution: {bb.solution}")
