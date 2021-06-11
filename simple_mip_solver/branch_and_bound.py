from coinor.cuppy.milpInstance import MILPInstance
import inspect
from queue import PriorityQueue
from typing import Any, Dict, TypeVar

from simple_mip_solver.nodes.base_node import BaseNode, T
from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

B = TypeVar('B', bound='BranchAndBound')


class BranchAndBound:
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    _node_attributes = ['lower_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible', 'search_method',
                        'branch_method']
    _node_funcs = ['bound', 'branch', '__lt__', '__eq__']
    _queue_funcs = ['put', 'get', 'empty']

    def __init__(self: B, model: MILPInstance, Node: Any = BaseNode,
                 node_queue: Any = None, strong_branch_iters: int = 5):
        f""" Instantiates a Branch and Bound instance
        
        :param model: A MILPInstance object that defines the MILP we solve
        :param Node: A class containing attributes {self._node_attributes} and methods
        {self._node_funcs}. Represents a single node in the branch and bound tree.
        :param node_queue: An object containing methods {self._queue_funcs}.
        This object is what holds and prioritizes nodes to be solved in branch and
        bound.
        :param strong_branch_iters: How many iterations of strong branching to run
        if the provided node uses strong branching. 
        """
        node_queue = node_queue or PriorityQueue()
        self._global_upper_bound = float('inf')

        # model asserts
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'

        # Node asserts
        assert inspect.isclass(Node), 'Node must be a class'
        # ensures Node constructor has the args we need and no other required ones
        root_node = Node(lp=model.lp, integerIndices=model.integerIndices,
                         lower_bound=-float('inf'))
        for attribute in self._node_attributes:
            assert hasattr(root_node, attribute), f'Node needs a {attribute} attribute'
        for func in self._node_funcs:
            c = getattr(root_node, func, None)
            assert callable(c), f'Node needs a {func} function'

        # node_queue asserts
        for func in self._queue_funcs:
            c = getattr(node_queue, func, None)
            assert callable(c), f'node_queue needs a {func} function'

        # strong branch iters assert
        assert isinstance(strong_branch_iters, int) and strong_branch_iters > 0, \
            'strong branching iterations must be positive integer'

        self._Node = Node
        self._root_node = root_node
        self._node_queue = node_queue
        self.model = model
        self._best_solution = None
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None
        self._pseudo_costs = {}
        self._strong_branch_iters = strong_branch_iters

    def solve(self: B) -> None:
        """Solves the Branch and Bound algorithm using the bound, search, and
        branching methods provided by the Node class and node_queue we instatiated
        the BranchAndBound instance with.

        :return:
        """
        self._node_queue.put(self._root_node)

        while not self._node_queue.empty():
            self._evaluate_node(self._node_queue.get())

        self.status = 'optimal' if self._best_solution is not None else 'infeasible'
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
        # todo generalize API returns into single decorator
        rtn = node.bound(pseudo_costs=self._pseudo_costs,
                         strong_branch_iters=self._strong_branch_iters)
        self._process_rtn(rtn)

        if node.lp_feasible and node.objective_value < self._global_upper_bound:
            if node.mip_feasible:
                self._best_solution = node.solution
                self._global_upper_bound = node.objective_value
            else:
                rtn = node.branch(pseudo_costs=self._pseudo_costs)
                self._process_branch_rtn(rtn)

    def _process_rtn(self: B, rtn: Dict):
        """ Assign the values of <rtn> to their keyed attributes

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        for key, val in rtn.items():
            setattr(self, key, val)

    def _process_branch_rtn(self: B, rtn: Dict):
        """ Pull the nodes returned from branching out of the rtn dict and into
        the node queue before assigning the rest of the values to their keyed
        attributes

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
