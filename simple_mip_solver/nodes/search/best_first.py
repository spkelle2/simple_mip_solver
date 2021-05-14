from simple_mip_solver.nodes.base_node import BaseNode


class BestFirstSearch(BaseNode):
    """ An extension of the BaseNode class to allow for best first search
    when nodes are stored in a priority queue"""

    def __eq__(self, other):
        if isinstance(other, BaseNode):
            return self.lower_bound == other.lower_bound
        else:
            raise TypeError('A Node can only be compared with another Node')

    def __lt__(self, other):
        if isinstance(other, BaseNode):
            return self.lower_bound < other.lower_bound
        else:
            raise TypeError('A Node can only be compared with another Node')