from coinor.cuppy.milpInstance import MILPInstance
from queue import PriorityQueue
from typing import Any, Dict, TypeVar

from simple_mip_solver.algorithms.utils import Utils
from simple_mip_solver.nodes.base_node import BaseNode, T
from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

B = TypeVar('B', bound='BranchAndBound')


class BranchAndBound(Utils):
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    _node_attributes = ['lower_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible', 'search_method',
                        'branch_method']
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
        assert set(kwargs.keys()).isdisjoint({'up', 'down'}), \
            'keys "up" and "down" are saved for later use'
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
        self._node_count = 0
        self._node_limit = node_limit

    def solve(self: B) -> None:
        """Solves the Branch and Bound algorithm using the bound, search, and
        branching methods provided by the Node class and node_queue we instatiated
        the BranchAndBound instance with.

        :return:
        """
        self._node_queue.put(self._root_node)

        while not (self._node_queue.empty() or self._unbounded or
                   self._node_count >= self._node_limit):
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
        self._node_count += 1

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
                self._process_branch_rtn(node.branch(**self._kwargs))
                self._global_lower_bound = min(n.lower_bound for n in self._node_queue.queue)

    def _process_branch_rtn(self: B, rtn: Dict[str, Any]):
        """ Pull the nodes returned from branching out of the rtn dict and into
        the node queue before updating the rest of the key value pairs in _kwargs

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        for direction in ['up', 'down']:
            assert direction in rtn, f'{direction} must be in the returned dict'
            assert isinstance(rtn[direction], self._Node), \
                f'{direction} value must be type {type(self._Node)}'
            self._node_queue.put(rtn[direction])
            del rtn[direction]
        self._process_rtn(rtn)


if __name__ == '__main__':
    bb = BranchAndBound(small_branch, Node=PseudoCostBranchNode)
    bb.solve()
    print(f"objective: {bb.objective_value}")
    print(f"solution: {bb.solution}")
