import numpy as np
from coinor.cuppy.milpInstance import MILPInstance
from coinor.gimpy.tree import BinaryTree
from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
from queue import PriorityQueue
from typing import Any, Dict, TypeVar, List

from simple_mip_solver.algorithms.utils import Utils
from simple_mip_solver.nodes.base_node import BaseNode, T
from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

B = TypeVar('B', bound='BranchAndBound')

BT = TypeVar('BT', bound='BranchAndBoundTree')


class BranchAndBoundTree(BinaryTree):
    """Class used to represent the underlying tree structure of branch and bound"""

    def get_leaves(self: BT, subtree_root_id: int) -> List[int]:
        """ Gather all leaves of a subtree rooted at node with id <subtree_root_id>

        :param subtree_root_id: The id of the node that roots our subtree
        :return: the leaves of the subtree
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        leaves = []
        if not self.neighbors[subtree_root_id]:
            leaves.append(subtree_root_id)
        else:
            for child_id in self.get_children(subtree_root_id):
                leaves.extend(self.get_leaves(child_id))
        return leaves

    def get_ancestors(self: BT, node_id: int) -> List[int]:
        """ Gather all ancestors for a given node with id <node_id>

        :param node_id: node to gather ancestors for
        :return: list of node's ancestors
        """
        assert node_id in self, 'node_id must belong to the tree'
        ancestors = []
        parent_id = self.get_parent(node_id)
        if parent_id is not None:
            ancestors.append(parent_id)
            ancestors.extend(self.get_ancestors(parent_id))
        return ancestors

    def get_node_instances(self: BT, node_ids: List[int]) -> List[B]:
        assert isinstance(node_ids, list), 'node_ids must be a list'
        missing_ids = set(node_ids) - set(self.nodes)
        assert not missing_ids, f'the following node_ids are not in the tree: {missing_ids}'
        instances = [self.nodes[idx].attr.get('node') for idx in node_ids]
        assert all(instance is not None for instance in instances), \
            'each vertex in the branch and bound tree must have an attribute for a node instance'
        return instances


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
                 node_queue: Any = None, node_limit: int = float('inf'), **kwargs: Any):
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
        assert node_limit == float('inf') or isinstance(node_limit, int), \
            "node limit must be integer or infinity"

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
        self._global_upper_bound = float('inf')
        self._global_lower_bound = -float('inf')
        self._node_limit = node_limit
        self._tree = BranchAndBoundTree()
        self._tree.add_root(self._root_node.idx, node=self._root_node)

    def solve(self: B) -> None:
        """Solves the Branch and Bound algorithm using the bound, search, and
        branching methods provided by the Node class and node_queue we instatiated
        the BranchAndBound instance with.

        :return:
        """
        assert self.status == 'unsolved', "This instance is solved. Please " \
                                          "create a new instance to resolve"
        self._node_queue.put(self._root_node)

        while not (self._node_queue.empty() or self._unbounded or
                   self._evaluated_nodes >= self._node_limit):
            self._evaluate_node(self._node_queue.get())

        self.status = 'unbounded' if self._unbounded else 'optimal' if \
            self._best_solution is not None else 'infeasible'
        self.solution = self._best_solution
        self.objective_value = self._global_upper_bound

    def _evaluate_node(self: B, node: T) -> None:
        """Bounds and optionally branches on the given node. Updates any attributes
        with the values keyed in the rtn's for bound and branch methods.

        :param node: the object that is bounded and potentially branched on.
        :return:
        """
        if node.lower_bound >= self._global_upper_bound:
            return
        self._evaluated_nodes += 1

        self._process_rtn(node.bound(**self._kwargs))

        # this solver is not designed to handle unboundedness accurately
        # need feasible milp solution, but may never find one, so assumes we do
        if node.unbounded:
            self._unbounded = True

        if node.lp_feasible and node.objective_value < self._global_upper_bound:
            if node.mip_feasible:
                self._best_solution = node.solution
                self._global_upper_bound = node.objective_value
            else:
                self._process_branch_rtn(node.idx, node.branch(**self._kwargs))
                self._global_lower_bound = min(n.lower_bound for n in self._node_queue.queue)

    def _process_branch_rtn(self: B, parent_id: int, rtn: Dict[str, Any]):
        """ Pull the nodes returned from branching out of the rtn dict and into
        the node queue and branch and bound tree before updating the rest of the
        key value pairs in _kwargs

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        assert isinstance(parent_id, int), 'parent_id must be integer'
        assert parent_id in self._tree, 'parent must already exist in tree'
        # left is down right is up
        for direction in ['left', 'right']:
            assert direction in rtn, f'{direction} must be in the returned dict'
            assert isinstance(rtn[direction], self._Node), \
                f'{direction} value must be type {type(self._Node)}'
            assert rtn[direction].idx not in self._tree, 'please give unique node ID'
            self._node_queue.put(rtn[direction])
            getattr(self._tree, f'add_{direction}_child')(rtn[direction].idx, parent_id,
                                                          node=rtn[direction])
            del rtn[direction]
        self._process_rtn(rtn)

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
        terminal_nodes = self._tree.get_node_instances(self._tree.get_leaves(self._root_node.idx))
        multi_const_nodes = [n.idx for n in terminal_nodes if len(n.lp.constraints) != 1]
        assert not multi_const_nodes, \
            f'This feature expects the root node to have a single constraint object and ' \
            f'all nodes to branch by bounding variables instead of by adding constraints. ' \
            f'It does not currently handle cuts being added after bounding. The following ' \
            f'IDs belong to nodes that do not conform to these rules: {multi_const_nodes}'
        assert all(b.shape == n.lp.constraints[0].lower.shape for n in terminal_nodes), \
            'the shape of the RHS being added should match that of each node'
        
        b = -b if self._swapped_constraint_direction else b
        # nb: after first call, this routine will not bound duals again
        # also does not update other attributes of the node to reflect its values from solve
        infeasible_nodes = [n for n in terminal_nodes if n.lp.getStatusCode() == 1]
        for n in infeasible_nodes:
            n.lp = self._bound_dual(n.lp)
        return min(self._calculate_dual_bounds(terminal_nodes, b).values())
        # for testing, assert strong at b and leq for all other b
        # assert dual comes out to value we expect for some problem
        # assert that any duplicate call does not result in bound_dual being called
        # but first call hits all the infeasible nodes

    def _calculate_dual_bounds(self, nodes: List[BaseNode], b: CyLPArray) -> Dict[int, float]:
        """ Calculate the lower bound on each of a list of nodes evaluated at a new
        RHS b based on their current dual solution

        todo: Why again can we take a max here in each lineage? Well, when minimizing,
        ancestors' LP relaxes are LB on terminal node objective value.
        Since ancestor dual evals at b are LB for for each ancestor node, they are
        also LB's for terminal node

        :param nodes: list of node instances
        :param b:
        :return:
        """

        assert all(isinstance(n, self._Node) for n in nodes), 'all nodes should be instance of given Node type'
        assert all(n.lp.getStatusCode() in [-1, 0] for n in nodes), \
            'all nodes must have finite optimal solutions'

        bounds = {}
        for node in nodes:
            lineage = [node] + self._tree.get_node_instances(self._tree.get_ancestors(node.idx))
            bounds[node.idx] = max(
                np.inner(n.lp.dualConstraintSolution[n.lp.constraints[0].name], b) +
                np.inner(np.maximum(n.lp.dualVariableSolution['x'], np.zeros(n.lp.nVariables)), n.lp.variablesLower) +
                np.inner(np.minimum(n.lp.dualVariableSolution['x'], np.zeros(n.lp.nVariables)), n.lp.variablesUpper)
                for n in lineage
            )
        return bounds

    def _bound_dual(self, cur_lp: CyClpSimplex) -> CyClpSimplex:
        """ Place a bound on each index of the dual variable associated with the
        constraints of this node's LP relaxation and resolve. We do this by adding
        to each constraint i the slack variable 's_i' in the node's LP relaxation,
        and we give each new variable a large, positive coefficient in the objective.
        By duality, we get the desired dual LP constraints. Therefore, for nodes
        with infeasible primal LP relaxations and unbounded dual LP relaxations,
        resolving gives us a finite dual solution, which can be used to parametrically
        lower bound the objective value of this node as we change its right hand side.

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

        # recreate variables
        var_map = {v: new_lp.addVariable(v.name, v.dim) for v in cur_lp.variables}

        # bound them
        for orig_v, new_v in var_map.items():
            new_lp += CyLPArray(orig_v.lower) <= new_v <= CyLPArray(orig_v.upper)

        # re-add constraints, with slacks this time todo: just use lp.coef matrix here
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


if __name__ == '__main__':
    bb = BranchAndBound(small_branch, Node=PseudoCostBranchNode)
    bb.solve()
    print(f"objective: {bb.objective_value}")
    print(f"solution: {bb.solution}")
