from coinor.cuppy.milpInstance import MILPInstance
import inspect
from typing import Any

from simple_mip_solver import Node
from queue import PriorityQueue


class BranchAndBound:
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    def __init__(self, model: MILPInstance, Node: Any = Node,
                 node_queue: Any = None):
        node_queue = node_queue or PriorityQueue()
        self._global_upper_bound = float('inf')
        self._node_attributes = ['lower_bound', 'objective_value', 'solution',
                                 'lp_feasible', 'mip_feasible']
        self._node_funcs = ['bound', 'branch', '__lt__', '__eq__']
        self._queue_funcs = ['put', 'get', 'empty']

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
            # make sure no additional required args from priority queue class
            assert callable(c), f'node_queue needs a {func} function'

        self._root_node = root_node
        self._node_queue = node_queue
        self.model = model
        self._best_solution = None
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None
        self._current_node = None

    def solve(self) -> None:
        """Solves the Branch and Bound algorithm

        :return:
        """
        self._node_queue.put(self._root_node)

        while not self._node_queue.empty():
            self._evaluate_next_node()

        self.status = 'optimal' if self._best_solution is not None else 'infeasible'
        self.solution = self._best_solution
        self.objective_value = self._global_upper_bound

    def _evaluate_next_node(self) -> None:
        """Solves one node of the branch and bound algorithm

        :return:
        """
        self._current_node = self._node_queue.get()
        if self._current_node.lower_bound >= self._global_upper_bound:
            return
        self._current_node.bound()

        # do nothing if node infeasible or worse than the existing bound
        if self._current_node.lp_feasible and self._current_node.objective_value \
                < self._global_upper_bound:
            if self._current_node.mip_feasible:
                self._best_solution = self._current_node.solution
                self._global_upper_bound = self._current_node.objective_value
            else:
                ln, rn = self._current_node.branch()
                self._node_queue.put(ln)
                self._node_queue.put(rn)
