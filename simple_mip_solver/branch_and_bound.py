from coinor.cuppy.milpInstance import MILPInstance
from math import floor
from operator import attrgetter
from typing import Dict, Tuple, List

from simple_mip_solver import Node


class BranchAndBound:
    """Class used to solve Mixed Integer Linear Programs with the Branch and
    Bound algorithm"""

    def __init__(self, model: MILPInstance):
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'
        self.global_upper_bound = float('inf')
        self.global_lower_bound = -float('inf')
        self.best_solution = None
        self.nodes = []
        self.model = model

    def solve(self) -> Tuple[int, Dict[int, float], float]:
        """Solves the Branch and Bound algorithm

        :return: status code, the variable keys and values in the best solution,
        and the best objective value
        """
        self.nodes.append(Node(self.model, self.global_lower_bound))

        while self.nodes:
            self._evaluate_next_node()

        return 0 if self.best_solution else -1, self.best_solution, \
            self.global_upper_bound

    def _evaluate_next_node(self) -> None:
        """Solves one iteration of the branch and bound algorithm

        :return:
        """
        # select best first
        current_node = self._least_lower_bound()
        self.nodes.remove(current_node)
        current_node.solve()

        # do nothing if node infeasible or worse than the existing bound
        if current_node.lp_feasible and current_node.obj < self.global_upper_bound:
            if current_node.mip_feasible:
                self.best_solution = current_node.solution
                self.global_upper_bound = current_node.obj
                # only keep nodes that give us a chance to find better solutions
                self.nodes = [n for n in self.nodes if n.lower_bound <
                              self.global_upper_bound]
            else:
                self.nodes.extend(branch(current_node))
                self.global_lower_bound = min(n.lower_bound for n in self.nodes)

    def _branch(self, node: Node) -> List[Node]:
        """ Returns two nodes that are branched by the current node's most fractional
        value. Returns nothing if the current node is a feasible mip. Right now
        this function is kinda unnecessary but it provides the structure for
        adding multiple branching strategies in the future.

        :param model: object containing initial parameters for the problem being solved
        :param node: object representing where in the branch and bound tree we are
        :return: list of Nodes with the new bounds
        """
        assert isinstance(node, Node), 'node must be a Node instance'
        idx = _find_most_fractional_index(model, node)
        if idx:
            return _create_nodes(model, node, idx)
        else:
            return []

    def _find_most_fractional_index(self, node: Node) -> int:
        """ Returns the index of the integer variable with current value furthest from
        being integer. If one does not exist, returns None.

        :param model: object containing initial parameters for the problem being solved
        :param node: object representing where in the branch and bound tree we are
        :return furthest_index: index corresponding to variable with most fractional
        value
        """
        assert isinstance(node, Node), 'node must be a Node instance'
        furthest_index = None
        furthest_dist = 0
        for idx in model.integerIndices:
            dist = abs(node.solution[idx] - (floor(node.solution[idx]) + 0.5))
            if dist > furthest_dist:
                furthest_dist = dist
                furthest_index = idx
        return furthest_index

    def _create_nodes(self, node: Node, index: int) -> List[Node]:
        """ Creates two new copies of the node with new bounds placed on the variable
        with given index, one with the variable's lower bound set to the next integer
        above its current value and another with the variable's upper bound set to
        the integer immediately below its current value.

        :param model: object containing initial parameters for the problem being solved
        :param node: object representing where in the branch and bound tree we are
        :param index: index of variable to branch on
        :return: list of Nodes with the new bounds
        """
        assert isinstance(node, Node), 'node must be a Node instance'
        assert isinstance(index, int), 'index must be an integer'
        int_value = floor(node.solution[index])

        # in one branch set upper bound for index as floor
        u = node.model.lp.variablesUpper.copy()
        u[index] = int_value
        left_model = MILPInstance(A=node.model.A, b=node.model.b, c=node.model.c,
                                  l=node.model.l, u=u,
                                  integerIndices=node.model.integerIndices)

        # in other branch set lower bound for same index as ceiling
        l = node.model.lp.variablesLower.copy()
        l[index] = int_value + 1
        right_model = MILPInstance(A=node.model.A, b=node.model.b, c=node.model.c,
                                   l=l, u=node.model.u,
                                   integerIndices=node.model.integerIndices)

        return [Node(left_model, node.obj), Node(right_model, node.obj)]

    def _least_lower_bound(nodes: List[Node]) -> Node:
        """ Finds the node with the least lower bound

        :param nodes: list of nodes representing where in the branch and bound tree we are
        :return: node with the least lower bound
        """
        assert all(isinstance(n, Node) for n in nodes), 'must have a list of nodes'
        return min(nodes, key=attrgetter('lower_bound'))
