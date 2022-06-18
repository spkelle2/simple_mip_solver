import numpy as np
from coinor.cuppy.milpInstance import MILPInstance
from coinor.gimpy.tree import BinaryTree
from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
from queue import PriorityQueue
import time
from typing import Any, Dict, TypeVar, List, Union, Iterable, Type, Tuple

from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from simple_mip_solver.nodes.base_node import BaseNode
from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

B = TypeVar('B', bound='BranchAndBound')

BT = TypeVar('BT', bound='BranchAndBoundTree')


class BranchAndBoundTree(BinaryTree):
    """Class used to represent the underlying tree structure of branch and bound"""

    def get_leaves(self: BT, subtree_root_id: int, depth: int = None,
                   keep: str = 'all') -> List[BaseNode]:
        """ If depth is None, gather all leaves for a subtree rooted at node with
        id <subtree_root_id>. Otherwise, gather all leaves for a subtree rooted at
        node with id <subtree_root_id> after descendents more than <depth> edges
        away have been removed.

        Caution: Could be very slow when used repeatedly on large trees with depth > 1

        :param subtree_root_id: The id of the node that roots our subtree
        :param depth: Depth beyond which nodes are excluded from the subtree
        :param keep: Specifies if returned leaves should keep 'all' of those found, only
        those that are LP 'feasible', or only those with LP's that are 'not infeasible'.
        :return: the desired leaves of the subtree
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        assert keep in ['all', 'feasible', 'not infeasible'], \
            "keep is one of 'all', 'feasible', or 'not infeasible'"
        if depth is not None:
            assert isinstance(depth, int) and depth >= 0, 'depth is a nonnegative integer'
            if depth == 0:
                rtn = self.get_node_instances([subtree_root_id])
            elif depth == 1:
                rtn = self.get_node_instances(self.get_children(subtree_root_id))
            else:
                # leaves less than <depth> levels away
                leaves_within_depth = [
                    n.attr['node'] for n in self.nodes.values() if n.attr['node'].is_leaf
                    and subtree_root_id in n.attr['node'].lineage[-depth:]
                ]
                # nodes <depth> levels away
                depth_descendents = [
                    n.attr['node'] for n in self.nodes.values() if
                    len(n.attr['node'].lineage) >= depth + 1 and
                    subtree_root_id == n.attr['node'].lineage[-(depth + 1)]
                ]
                rtn = leaves_within_depth + depth_descendents
        else:
            rtn = [n.attr['node'] for n in self.nodes.values() if n.attr['node'].is_leaf
                   and subtree_root_id in n.attr['node'].lineage]
        return rtn if keep == 'all' else [n for n in rtn if n.lp_feasible] if \
            keep == 'feasible' else [n for n in rtn if n.lp_feasible is not False]

    def get_disjunction(self: BT, subtree_root_id: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """ Return the disjunction encoded in the terminal leaves of the branch
        and bound subtree rooted at node with id <subtree_root_id>

        :param subtree_root_id: The id of the node that roots our subtree
        :return: a dictionary keyed by indices of nodes with values as pairs of
        arrays, (lb, ub), representing the variable bounds on the node with the
        given index. For x to be a feasible solution, there must be a (lb, ub) value
        in the dict such that lb <= x <= ub.
        """
        return {n.idx: (n.lp.variablesLower.copy(), n.lp.variablesUpper.copy())
                for n in self.get_leaves(subtree_root_id, keep='not infeasible')}

    def get_node_instances(self: BT, node_ids: Union[int, Iterable[int]]) -> \
            Union[BaseNode, List[BaseNode]]:
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

    def subtree_dual_bound(self: BT, subtree_root_id: int, depth: int = None) -> \
            Union[float, int]:
        """ Finds the dual bound for the branch and bound subtree rooted at node
        <subtree_root_id> with maximum depth <depth>

        :param subtree_root_id: The id of the node that roots our subtree
        :param depth: depth beyond the subtree root which nodes are excluded
        for calculating dual bound
        :return: the dual bound for the branch and bound subtree rooted at node
        <subtree_root_id> with maximum depth <depth>
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        return min(n.objective_value if n.objective_value is not None else n.dual_bound
                   for n in self.get_leaves(subtree_root_id, depth=depth))


class BranchAndBound(BaseAlgorithm):
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    _node_attributes = ['dual_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible', 'search_method',
                        'branch_method', 'idx', 'lp', 'is_leaf', 'lineage']
    _node_funcs = ['bound', 'branch', '__lt__', '__eq__']
    _queue_funcs = ['put', 'get', 'empty']

    # these kwargs get passed to the branch and bound functions
    # so kwargs = {'strong_branch_iters': 5, 'pseudo_costs': {}}
    def __init__(self: B, model: MILPInstance, Node: Type[BaseNode] = BaseNode,
                 node_queue: Any = None, node_limit: int = float('inf'),
                 mip_gap: float = .0001, logging: bool = False, max_run_time: float = float('inf'),
                 initial_primal_bound: float = float('inf'), **kwargs: Any):
        f""" Instantiates a Branch and Bound instance.
        
        CAUTION: During instantiation, all problems are converted to minimization
        with constraints of the form Ax >= b
        
        :param model: A MILPInstance object that defines the MILP we solve
        :param Node: A class containing attributes {self._node_attributes} and methods
        {self._node_funcs}. Represents a single node in the branch and bound tree.
        :param node_queue: An object containing methods {self._queue_funcs}.
        This object is what holds and prioritizes nodes to be solved in branch and
        bound.
        :param node_limit: if provided, max number of nodes to explore
        :param mip_gap: How close 1 minus the ratio of dual to primal bound must be 
        for the solver to terminate
        :param logging: Whether or not the solver prints status updates
        :param max_run_time: Maximum amount of time (in seconds) the solver will run
        before terminating
        :param initial_primal_bound: Best known objective value for feasible solutions
        to the MIP. NOTE: the MIP will only return a solution if a better one is found.
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

        # logging assert
        assert isinstance(logging, bool), 'logging is boolean'

        # run time assert
        assert max_run_time > 0, 'max_run_time is positive value'

        # initial primal bound assert
        assert initial_primal_bound > -float('inf'), 'initial_primal_bound is real or infinite'

        # kwargs assert
        special_keys = {'right', 'left', 'cuts'}
        assert set(kwargs.keys()).isdisjoint(special_keys), \
            f'keys {special_keys} are saved for later use'
        assert all(isinstance(k, str) for k in kwargs), 'kwargs keys must be strings'

        # instantiate
        self._node_queue = node_queue
        self._unbounded = None
        self._best_solution = None
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None
        self.primal_bound = initial_primal_bound
        self.node_limit = node_limit
        self.tree = BranchAndBoundTree()
        self.tree.add_root(self.root_node.idx, node=self.root_node)
        self.solve_time = 0
        self.mip_gap = mip_gap
        self.logging = logging
        self.max_run_time = max_run_time

    @property
    def dual_bound(self):
        return self.tree.subtree_dual_bound(self.root_node.idx)

    @property
    def current_gap(self):
        if self.primal_bound == self.dual_bound == 0:
            gap = 0
        elif self.primal_bound == 0:
            gap = float('inf')
        elif self.primal_bound == float('inf'):
            gap = None
        else:
            gap = abs(self.primal_bound - self.dual_bound) / abs(self.primal_bound)
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
                   (self.current_gap is not None and self.current_gap <= self.mip_gap) or
                   time.process_time() - start > self.max_run_time):
            if self.evaluated_nodes % 100 == 0 and self.logging:
                print(f'{self.evaluated_nodes} nodes evaluated gap: {self.current_gap}')
            self._evaluate_node(self._node_queue.get())

        run_time = time.process_time() - start
        self.solve_time += run_time
        self.status = 'unbounded' if self._unbounded else 'infeasible' if \
            self._node_queue.empty() and self.primal_bound == float('inf') else \
            'optimal' if self.primal_bound < float('inf') and self.current_gap <= self.mip_gap \
            else 'stopped on iterations or time'
        self.solution = self._best_solution
        self.objective_value = self.primal_bound

    def _evaluate_node(self: B, node: BaseNode) -> None:
        """Bounds and optionally branches on the given node. Updates any attributes
        with the values keyed in the rtn's for bound and branch methods. Updates
        the global lower bound

        :param node: the object that is bounded and potentially branched on.
        :return:
        """
        if node.dual_bound < self.primal_bound:
            self.evaluated_nodes += 1

            self._process_bound_rtn(node.bound(**self._kwargs))

            # this solver is not designed to handle unboundedness accurately
            # need feasible milp solution, but may never find one, so assumes we do
            if node.unbounded:
                self._unbounded = True

            if node.lp_feasible and node.objective_value < self.primal_bound:
                if node.mip_feasible:
                    self._best_solution = node.solution
                    self.primal_bound = node.objective_value
                else:
                    self._process_branch_rtn(node.idx, node.branch(**self._kwargs))

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

    def _process_bound_rtn(self: B, rtn: Dict[str, Any]):
        """ Pull the cuts returned from bounding out of the rtn dict and add
        to all nodes in the node queue before updating the rest of the key value
        pairs in _kwargs

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        cuts = rtn.get('cuts')
        if cuts:
            for name, (pi, pi0) in cuts.items():
                for node in self._node_queue.queue:
                    node.cut_pool[name] = (pi, pi0)
            del rtn['cuts']
        self._process_rtn(rtn)

    # todo: refactor for multiple constraints and get rid of change in b
    # todo: accomplish b work by just requiring all models entered in min c^T x : Ax >= b
    # todo: then just flip all the example models
    # entrust the user to better know what they're doing using this function with cuts added
    # enforce above that branching is done by bounding variables and not adding constraints
    # i don't even think we need that enforcement because it would be cpatured in a constraint
    def find_parameterized_dual_bound(self, b: CyLPArray) -> float:
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
            n.lp = self._bound_parameterized_dual(n.lp)

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

    def _bound_parameterized_dual(self, cur_lp: CyClpSimplex) -> CyClpSimplex:
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
        new_lp.dual()
        return new_lp


if __name__ == '__main__':
    bb = BranchAndBound(small_branch, Node=PseudoCostBranchNode)
    bb.solve()
    print(f"objective: {bb.objective_value}")
    print(f"solution: {bb.solution}")
