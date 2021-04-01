from operator import attrgetter
from typing import List

from node import Node


def least_lower_bound(nodes: List[Node]) -> Node:
    """ Finds the node with the least lower bound

    :param nodes: list of nodes representing where in the branch and bound tree we are
    :return: node with the least lower bound
    """
    assert all(isinstance(n, Node) for n in nodes)
    return min(nodes, key=attrgetter('lower_bound'))

