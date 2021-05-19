from typing import Any, TypeVar

from simple_mip_solver.nodes.base_node import BaseNode

T = TypeVar('T', bound='PseudoCostBranchNode')


class DepthFirstSearchNode(BaseNode):
    """ An extension of the BaseNode class to allow for depth first search
    when nodes are stored in a priority queue"""

    def __init__(self: T, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.search_method = 'depth first'

    def __eq__(self, other: T):
        if isinstance(other, DepthFirstSearchNode):
            return self.depth == other.depth
        else:
            raise TypeError('A DFS Node can only be compared with another DFS Node')

    # self < other means self gets better priority in priority queue
    # want priority to go to node with deepest depth (i.e. highest depth value)
    def __lt__(self, other: T):
        if isinstance(other, DepthFirstSearchNode):
            return self.depth > other.depth
        else:
            raise TypeError('A DFS Node can only be compared with another DFS Node')

