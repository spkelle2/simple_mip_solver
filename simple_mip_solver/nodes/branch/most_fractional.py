from __future__ import annotations
from math import floor, ceil
from typing import List, Dict, Any, TypeVar

from simple_mip_solver.nodes.base_node import BaseNode

T = TypeVar('T', bound='MostFractionalBranch')


class MostFractionalBranch(BaseNode):
    """ An extension of the BaseNode class to allow for most fractional branching
    """

    @property
    def most_fractional_index(self) -> int:
        """ Returns the index of the integer variable with current value furthest from
        being integer. If one does not exist or the problem has not yet been solved,
        returns None.

        :return furthest_index: index corresponding to variable with most fractional
        value
        """
        furthest_index = None
        furthest_dist = self._epsilon
        if self.lp_feasible:
            for idx in self.integerIndices:
                dist = min(self.solution[idx] - floor(self.solution[idx]),
                           ceil(self.solution[idx]) - self.solution[idx])
                if dist > furthest_dist:
                    furthest_dist = dist
                    furthest_index = idx
        return furthest_index

    def branch(self, **kwargs: Dict[Any, Any]) -> List[MostFractionalBranch]:
        """ Creates two new nodes which are branched on the most fractional index
        of this node's LP relaxation solution

        :param kwargs: a dictionary to hold unneeded arguments sent by a general
        branch and bound method
        :return: list of Nodes with the new bounds
        """
        idx = self.most_fractional_index
        return self.base_branch(idx)


