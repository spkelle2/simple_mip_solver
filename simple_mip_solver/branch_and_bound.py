from coinor.cuppy.milpInstance import MILPInstance
from coinor.gimpy.tree import BinaryTree
import inspect
from typing import Any

from simple_mip_solver import Node, PriorityQueue


class BranchAndBound(BinaryTree):
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    def __init__(self, model: MILPInstance, node_class: Any = Node,
                 node_queue: Any = PriorityQueue('lower_bound'), **attrs):
        super().__init__(**attrs)
        self._global_upper_bound = float('inf')
        self._global_lower_bound = -float('inf')

        # model asserts
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'

        # node_class asserts
        assert inspect.isclass(node_class), 'node_class must be a class'
        args = {'lp', 'integerIndices', 'lower_bound'}
        assert not args - set(inspect.signature(node_class).parameters), \
            f'node_class constructor should contain the following arguments: {args}'
        root = node_class(lp=model.lp, integerIndices=model.integerIndices,
                          lower_bound=self._global_lower_bound)
        attributes = ['lp', 'integerIndices', 'lower_bound', 'objective_value',
                      'solution', 'lp_feasible', 'mip_feasible']
        for attribute in attributes:
            assert hasattr(root, attribute), f'node_class needs an {attribute} attribute'
        for func in ['solve', 'branch']:
            c = getattr(root, func, None)
            assert callable(c), f'node_class needs a {func} function'

        # node_queue asserts
        for func in ['__bool__', 'push', 'min', 'pop', 'bound']:
            c = getattr(node_queue, func, None)
            assert callable(c), f'node_queue needs a {func} function'

        self._root = root
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
        self._node_queue.push(self._root)

        while self._node_queue:
            self._evaluate_next_node()

        self.status = 'optimal' if self._best_solution is not None else 'infeasible'
        self.solution = self._best_solution
        self.objective_value = self._global_upper_bound

    def _evaluate_next_node(self) -> None:
        """Solves one node of the branch and bound algorithm

        :return:
        """
        self._current_node = self._node_queue.pop()
        self._current_node.solve()

        # do nothing if node infeasible or worse than the existing bound
        if self._current_node.lp_feasible and self._current_node.objective_value \
                < self._global_upper_bound:
            if self._current_node.mip_feasible:
                self._best_solution = self._current_node.solution
                self._global_upper_bound = self._current_node.objective_value
                self._node_queue.bound(self._global_upper_bound)
            else:
                ln, rn = self._current_node.branch()
                self._node_queue.push(ln)
                self._node_queue.push(rn)
                self._global_lower_bound = self._node_queue.min().lower_bound
