from cylp.cy.CyClpSimplex import CyClpSimplex
from math import floor
from typing import List

from model import Model
from node import Node


def _find_most_fractional_index(model: Model, node: Node) -> int:
    """ Returns the index of the integer variable with current value furthest from
    being integer. If one does not exist, returns None.

    :param model: object containing initial parameters for the problem being solved
    :param node: object representing where in the branch and bound tree we are
    :return furthest_index: index corresponding to variable with most fractional
    value
    """
    furthest_index = None
    furthest_dist = model.epsilon
    for idx in model.int_index:
        dist = abs(node.solution[idx] - (floor(node.solution[idx]) + 0.5))
        if dist > furthest_dist:
            furthest_dist = dist
            furthest_index = idx
    return furthest_index


def _create_nodes(model: Model, node: Node, index: int) -> List[Node]:
    """ Creates two new copies of the node with new bounds placed on the variable
    with given index, one with the variable's lower bound set to the next integer
    above its current value and another with the variable's upper bound set to
    the integer immediately below its current value.

    :param model: object containing initial parameters for the problem being solved
    :param node: object representing where in the branch and bound tree we are
    :param index: index of variable to branch on
    :return: list of Nodes with the new bounds
    """
    int_value = floor(node.solution[index])

    # in one branch set upper bound for index as floor
    left_cylp = CyClpSimplex()
    upper = node.cylp.variablesUpper.copy()
    upper[index] = int_value
    left_cylp.loadProblem(node.cylp.matrix, node.cylp.variablesLower, upper,
                          node.cylp.objective, node.cylp.constraintsLower,
                          node.cylp.constraintsUpper)

    # in other branch set lower bound for same index as ceiling
    # TODO just make a copy of the node and update the one bound
    # http://coin-or.github.io/CyLP/modules/CyClpSimplex.html#cylp.cy.CyClpSimplex.CyClpSimplex.setColumnUpper
    right_cylp = CyClpSimplex()
    lower = node.cylp.variablesLower.copy()
    lower[index] = int_value + 1
    right_cylp.loadProblem(node.cylp.matrix, lower, node.cylp.variablesUpper,
                           node.cylp.objective, node.cylp.constraintsLower,
                           node.cylp.constraintsUpper)

    return [Node(model, left_cylp, node.obj), Node(model, right_cylp, node.obj)]


def branch(model: Model, node: Node) -> List[Node]:
    """ Returns two nodes that are branched by the current node's most fractional
    value. Returns nothing if the current node is a feasible mip.

    :param model: object containing initial parameters for the problem being solved
    :param node: object representing where in the branch and bound tree we are
    :return: list of Nodes with the new bounds
    """
    assert isinstance(model, Model)
    assert isinstance(node, Node)
    idx = _find_most_fractional_index(model, node)
    if idx:
        return _create_nodes(model, node, idx)
    else:
        return []

