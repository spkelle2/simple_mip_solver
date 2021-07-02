from coinor.cuppy.milpInstance import MILPInstance
import numpy as np
from typing import Any, TypeVar

from simple_mip_solver.nodes.bound.cutting_plane import CuttingPlaneBoundNode
from simple_mip_solver.algorithms.utils import Utils

C = TypeVar('C', bound='CuttingPlane')


class CuttingPlane(Utils):
    """Class used to solve Mixed Integer Linear Programs with cutting plane methods"""

    _node_attributes = ['lower_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible']
    _node_funcs = ['bound']

    def __init__(self: C, model: MILPInstance, Node: Any = CuttingPlaneBoundNode,
                 max_iters: int = 100, **kwargs: Any):
        f""" Instantiates a Branch and Bound instance.

        :param model: A MILPInstance object that defines the MILP we solve
        :param Node: A class containing attributes {self._node_attributes} and methods
        {self._node_funcs}. Represents the LP relaxation to which we add cuts.
        :param max_iters: if provided, max number of iterations to complete
        :param kwargs: dictionary passed to the bound function as key worded
        arguments and which adds keys and updates values based on what is returned
        """
        # call super
        super().__init__(model=model, Node=Node, node_attributes=self._node_attributes,
                         node_funcs=self._node_funcs, **kwargs)

        # max_iter asserts
        assert isinstance(max_iters, int) and max_iters > 0, 'max_iter is positive integer'

        self._max_iters = max_iters
        self._iterations = 0
        self.solution = None
        self.status = 'unsolved'
        self.objective_value = None

    def solve(self):

        prev_objective_value = -float('inf')
        while self._iterations < self._max_iters:
            self._iterations += 1
            print('Iteration: ', self._iterations)
            self._process_rtn(self._root_node.bound())
            print('Current bound:', self._root_node.objective_value)
            if self._root_node.mip_feasible:
                print('Integer solution found!')
                self.status = 'optimal'
                self.solution = self._root_node.solution
                self.objective_value = self._root_node.objective_value
                break
            # if we have change in objective
            if np.abs(self._root_node.objective_value - prev_objective_value) \
                    >= self._root_node._epsilon:
                prev_objective_value = self._root_node.objective_value
            else:
                print("Solution repeated, stalling detected. Exiting")
                self.status = 'stalled'
                break
        if self.status == 'unsolved':
            self.status = 'max iterations'
