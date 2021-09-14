from __future__ import annotations
from math import floor, ceil
from typing import List, Dict, Union, Any, TypeVar

from simple_mip_solver.nodes.base_node import BaseNode

T = TypeVar('T', bound='PseudoCostBranchNode')
pseudo_costs_hint = Dict[int, Dict[str, Dict[str, Union[float, int]]]]


class PseudoCostBranchNode(BaseNode):
    """ An extension of the BaseNode class to allow for pseudo cost branching
    """

    def __init__(self: T, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.branch_method = 'pseudo cost'
        self.pseudo_costs = None
        self.strong_branch_iters = None

    def bound(self: T, pseudo_costs: pseudo_costs_hint, strong_branch_iters: int = 5,
              **kwargs: Any) -> Dict[str, Any]:
        """ Extends BaseNode's bound by updating psuedocosts if the underlying
        relaxation is feasible

        :param pseudo_costs: dictionary holding expected change in objective
        per unit change in variable value
        :param strong_branch_iters: how many iterations to do during strong
        branching when initializing pseudo costs
        :param kwargs: a dictionary to hold unneeded arguments sent by a general
        branch and bound method
        :return: a dictionary mapping the pseudo costs to the '_pseudo_costs'
        parameter in the branch and bound object
        """
        problems = self._check_pseudo_costs(pseudo_costs)
        assert not problems, f'pseudo cost dict has following errors: {problems}'
        self.pseudo_costs = pseudo_costs
        self.strong_branch_iters = strong_branch_iters
        self._base_bound()
        if self.lp_feasible:
            self._update_pseudo_costs()
        return {'pseudo_costs': self.pseudo_costs}

    def _update_pseudo_costs(self: T) -> None:
        """ Update the pseudo costs for the variable branched on in the current
        node and instantiate the pseudo costs via strong branching for any
        integer variable that takes on a fractional value for the first time.

        Follows the scheme detailed in Integer Programming by Conforti, et al.
        Page 367.

        :return:
        """
        # strong branch all fractional indices that have not been assigned pseudocost
        sb_indices = [idx for idx in self._integer_indices if
                      self._is_fractional(self.solution[idx])
                      and idx not in self.pseudo_costs]
        for idx in sb_indices:
            for strong_branch_node in self._strong_branch(
                    idx, self.strong_branch_iters).values():
                self._calculate_costs(strong_branch_node)

        # calculate pseudo_cost[self.b_idx][self.b_dir] if we didn't just above
        if self._b_idx is not None and self._b_idx not in sb_indices:
            self._calculate_costs(self)

    def _calculate_costs(self: T, node: T) -> None:
        """ Calculate and save the pseudocost for the index and direction
        branched on in <node>. This is done by finding the rate of change of the
        bound with respect to the change in branched on variable's value and
        averaging that with rates of change from branching on this variable previously.

        Based on: https://www.scipopt.org/download/slides/SCIP-branching.ppt

        :param node: a PseudoCostBranchNode object or subclass instance
        :return:
        """
        idx = node._b_idx
        direction = node._b_dir

        # set defaults for pseudo costs if they dont exist
        self.pseudo_costs[idx] = self.pseudo_costs.get(idx, {})
        self.pseudo_costs[idx][direction] = self.pseudo_costs[idx].get(
            direction, {'cost': 0, 'times': 0})
        if node.lp.getStatusCode() in [0, 3]:  # optimal or hit max iters
            bound_change = node.lp.objectiveValue - node.lower_bound
            variable_change = node._b_val - node.lp.variablesUpper[idx] if \
                direction == 'left' else node.lp.variablesLower[idx] - node._b_val
            cost = self.pseudo_costs[idx][direction]['cost']
            times = self.pseudo_costs[idx][direction]['times']
            self.pseudo_costs[idx][direction]['cost'] = (cost * times + bound_change /
                                                         variable_change) / (times + 1)
        # cost stays 0 if strong branching was infeasible
        self.pseudo_costs[idx][direction]['times'] += 1

    def branch(self: T, pseudo_costs: pseudo_costs_hint, **kwargs: Any) -> Dict[str, T]:
        """ Branch via pseudo costs

        :param pseudo_costs: dictionary holding expected change in objective
        per unit change in variable value
        :param kwargs: a dictionary to hold unneeded arguments sent by a general
        branch and bound method
        :return: Two children nodes branched from this one on the index with the
        best pseudo cost
        """
        assert not self.mip_feasible, 'must have fractional value to branch'
        problems = self._check_pseudo_costs(pseudo_costs)
        assert not problems, f'pseudo cost dict has following errors: {problems}'
        b_idx = self._best_pseudo_costs_index(pseudo_costs)
        return self._base_branch(b_idx, **kwargs)

    def _best_pseudo_costs_index(self: T, pseudo_costs: pseudo_costs_hint) -> int:
        """ Select the index that appears to give us the best combination of
        unit reduced cost and distance from integrality.

        :param pseudo_costs: dictionary holding expected change in objective
        per unit change in variable value
        :return: index with the best pseudo cost
        """
        scores = {
            i: min(pseudo_costs[i]['right']['cost'] *
                   (ceil(self.solution[i]) - self.solution[i]),
                   pseudo_costs[i]['left']['cost'] *
                   (self.solution[i] - floor(self.solution[i])))
            for i in self._integer_indices if self._is_fractional(self.solution[i])
        }
        return sorted(scores, key=scores.get, reverse=True)[0]

    def _check_pseudo_costs(self: T, pseudo_costs: pseudo_costs_hint) -> List[str]:
        """ Ensure the passed pseudo costs dictionary is of proper form

        :param pseudo_costs: dictionary suspected of holding expected change in
        objective per unit change in variable value
        :return: list of any problems found with the pseudo cost dictionary
        """
        problems = []
        for idx in pseudo_costs:
            if idx not in self._integer_indices:
                problems.append(f'index {idx} not integer index')
                continue
            for direction in ['right', 'left']:
                if direction not in pseudo_costs[idx]:
                    problems.append(f'index {idx} missing direction {direction}')
                    continue
                if 'cost' not in pseudo_costs[idx][direction]:
                    problems.append(f'index {idx} direction {direction} missing cost')
                elif not (isinstance(pseudo_costs[idx][direction]['cost'], (int, float)) and
                          pseudo_costs[idx][direction]['cost'] >= 0):
                    problems.append(f'index {idx} direction {direction} cost must'
                                    ' be nonnegative number')
                if 'times' not in pseudo_costs[idx][direction]:
                    problems.append(f'index {idx} direction {direction} missing times')
                elif not (isinstance(pseudo_costs[idx][direction]['times'], int) and
                          pseudo_costs[idx][direction]['times'] >= 0):
                    problems.append(f'index {idx} direction {direction} times must'
                                    ' be nonnegative int')
        return problems
