from __future__ import annotations
from math import floor, ceil
from typing import List, Dict, Union, Any, TypeVar

from simple_mip_solver.nodes.base_node import BaseNode

T = TypeVar('T', bound='PseudoCostBranch')


class PseudoCostBranch(BaseNode):
    """ An extension of the BaseNode class to allow for pseudo cost branching
    """

    def branch(self: T, pseudo_up: Dict[int, Dict[str, Union[float, int]]],
               pseudo_down: Dict[int, Dict[str, Union[float, int]]],
               **kwargs: Dict[Any, Any]) -> List[T]:
        """ Branch via pseudo costs. Select the index that appears to give us
        the best combination of unit reduced cost and distance from integrality

        :param pseudo_up: up pseudo costs
        :param pseudo_down: down pseudo costs
        :param kwargs: a dictionary to hold unneeded arguments sent by a general
        branch and bound method
        :return:
        """

        for d, pseudo_dict in {'down': pseudo_down, 'up': pseudo_up}.items():
            problems = self.check_pseudo_costs(pseudo_dict)
            assert not problems, f'{d} pseudo cost dict has following errors: {problems}'

        scores = {
            i: min(pseudo_up[i]['cost'] * (ceil(self.solution[i]) - self.solution[i]),
                   pseudo_down[i]['cost'] * (self.solution[i] - floor(self.solution[i])))
            for i in self.integerIndices if self.is_fractional(i)
        }
        b_idx = sorted(scores, key=scores.get, reverse=True)[0]

        return self.base_branch(b_idx)

    def check_pseudo_costs(self: T, pseudo_dict:
                           Dict[int, Dict[str, Union[float, int]]]) -> List[str]:
        problems = []
        for idx in self.integerIndices:
            if idx not in pseudo_dict:
                problems.append(f'missing index {idx}')
                continue
            if 'cost' not in pseudo_dict[idx]:
                problems.append(f'index {idx} missing cost')
            elif not isinstance(pseudo_dict[idx]['cost'], (int, float)):
                problems.append(f'index {idx} cost is not a number')
            if 'times' not in pseudo_dict[idx]:
                problems.append(f'index {idx} missing times')
            elif not (isinstance(pseudo_dict[idx]['times'], int) and
                      pseudo_dict[idx]['times'] >= 0):
                problems.append(f'index {idx} times is not a nonnegative int')
        return problems
