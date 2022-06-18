from __future__ import annotations
from cylp.py.modeling.CyLPModel import CyLPArray
from typing import Dict, Any, TypeVar, Tuple, Union, List
import numpy as np
import re

from simple_mip_solver import BaseNode
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from simple_mip_solver.utils.floating_point import numerically_safe_cut
from simple_mip_solver.utils.tolerance import min_cglp_norm

G = TypeVar('G', bound='CuttingPlaneBoundNode')


class DisjunctiveCutBoundNode(BaseNode):
    """ An extension of the BaseNode class to allow for disjunctive cuts. This
    class adds the following keyword arguments to Branch and Bound instantiation:

    max_cglp_calls (int): The max number of cutting plane generation iterations
        to create disjunctive cuts. Defaults to None.
    warm_start_cglp (bool): Whether or not to allow the CGLP to be warm started
        from the previous call's optimal basis. Defaults to True.
    cglp_cumulative_constraints (bool): Whether or not to apply the disjunction
        in the CGLP to this node's constraints for all children nodes. If False,
        the same constraints used to generate this node's CGLP will be used for
        generating its children. Defaults to True.
    cglp_cumulative_bounds (bool): Whether or not to intersect the disjunction
        in the CGLP with this node's variable bounds for all children nodes. If
        False, the same variable bounds used to generate this node's CGLP will
        be used for generating its children. Defaults to True.
    """

    def __init__(self: G, cglp: CutGeneratingLP = None,
                 prev_cglp_basis: Tuple[np.ndarray, np.ndarray] = None,
                 force_create_cglp: bool = False, *args, **kwargs):
        """

        :param cglp: The CGLP instance to use for creating disjunctive cuts, off which
        children's CGLP's will be built
        :param prev_cglp_basis: The starting basis status for each variable in the CGLP instance
        :param force_create_cglp: Create the CGLP even if previous nodes or
        iterations failed to make one that generated a tightening cut
        :param args: Arguments to pass on to super class instantiation
        :param kwargs: Key word arguments to pass on to super class instantiation
        """
        super().__init__(*args, **kwargs)
        assert isinstance(force_create_cglp, bool), 'force_create_cglp is bool'
        if cglp is not None:
            assert isinstance(cglp, CutGeneratingLP), 'cglp must be CutGeneratingLP instance'
        else:
            assert not force_create_cglp, 'cannot force creation of CGLP that does not exist'

        self.cglp = cglp
        self.prev_cglp_basis = prev_cglp_basis
        self.current_node_added_cglp = force_create_cglp  # override as true if force creating
        # flag tracking if current node or previous cut generation iteration added cglp
        self.previous_cglp_added = self.cglp is not None
        self.cglp_name_pattern = re.compile('^cut_cglp_')
        self.current_cglp_name_pattern = re.compile(f'^cut_cglp_{self.idx}_')
        self.sharable_cuts = {}
        self.number_cglp_created = 0
        self.number_cglp_added = 0
        self.number_cglp_removed = 0
        self.force_create_cglp = force_create_cglp

    def bound(self: G, total_number_cglp_created: int = 0, total_number_cglp_added: int = 0,
              total_number_cglp_removed: int = 0, **kwargs: Any) -> Dict[str, Any]:
        """ Extends super's bound by returning disjunctive cuts valid for all
        other nodes

        :param total_number_cglp_created: Running total of disjunctive cuts created across
        all branch and bound subproblems
        :param total_number_cglp_added: Running total of disjunctive cuts added to the
        underlying LP relaxation across all branch and bound subproblems
        :param total_number_cglp_removed: Running total of disjunctive cuts removed from the
        underlying LP relaxation across all branch and bound subproblems
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary returned from super's bound()
        calls
        """
        assert isinstance(total_number_cglp_added, int) and total_number_cglp_added >= 0, \
            "total_number_cglp_added is nonnegative integer"
        assert isinstance(total_number_cglp_created, int) and total_number_cglp_created >= 0, \
            "total_number_gmic_created is nonnegative integer"
        assert isinstance(total_number_cglp_removed, int) and total_number_cglp_removed >= 0, \
            "total_number_cglp_removed is nonnegative integer"

        rtn = super().bound(**kwargs)
        rtn['total_number_cglp_created'] = total_number_cglp_created + self.number_cglp_created
        rtn['total_number_cglp_added'] = total_number_cglp_added + self.number_cglp_added
        rtn['total_number_cglp_removed'] = total_number_cglp_removed + self.number_cglp_removed
        if self.sharable_cuts:
            rtn['cuts'] = self.sharable_cuts
        return rtn

    def _remove_slack_cuts(self: G, **kwargs) -> List[str]:
        """ calls super()'s method then counts removal of disjunctive cuts

        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: list of indices corresponding to removed constraints
        """
        removable_idxs = super()._remove_slack_cuts(**kwargs)
        self.number_cglp_removed += sum(bool(self.cglp_name_pattern.match(idx))
                                        for idx in removable_idxs)
        return removable_idxs

    def _generate_cuts(self: G, max_cglp_calls: int = None,
                       min_cglp_norm: float = min_cglp_norm, **kwargs) -> \
            Dict[str: Union[CyLPArray, float]]:
        """ Extend super's cut generation by making CGLP cuts if possible

        Caution: not limiting the number of cut generation iterations or not warm
        starting the CGLP from the previous cut generation iteration can lead
        CyLP to find incorrect optimal solutions on rare occasions

        :param max_cglp_calls: Number of cut generation iterations in which CGLP
        will try to add cuts
        :param min_cglp_norm: smallest acceptable norm for disjunctive cut
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of cuts that can be added to the LP relaxation
        """
        if max_cglp_calls is not None:
            assert isinstance(max_cglp_calls, int) and max_cglp_calls >= 0, \
                'max_cglp_calls is a nonnegative integer'
        assert isinstance(min_cglp_norm, (float, int)) and min_cglp_norm > 0, \
            'min_cglp_norm is a positive number'

        max_cglp_calls = float('inf') if max_cglp_calls is None else max_cglp_calls
        cut_pool = super()._generate_cuts(**kwargs)

        # dont do if parent or previous iteration's cglp failed to yield good cut
        if self.previous_cglp_added and self.cut_generation_iterations <= max_cglp_calls:
            pi, pi0 = self.cglp.solve(x_star=CyLPArray(self.solution),
                                      starting_basis=self._get_cglp_starting_basis(**kwargs))
            if pi is not None and pi0 is not None and np.linalg.norm(pi) > min_cglp_norm:
                idx = f'cut_cglp_{self.idx}_{self.cut_generation_iterations}'
                pi, pi0 = (numerically_safe_cut(pi=pi, pi0=pi0, estimate='over'))
                cut_pool[idx] = (pi, pi0)
                self.number_cglp_created += 1

        return cut_pool

    def _get_cglp_starting_basis(self, warm_start_cglp: bool = True, **kwargs) -> \
            Union[None, Tuple[np.ndarray, np.ndarray]]:
        """ Determine the starting basis for the CGLP. Note, when warm_start_cglp == True
        and we're past the first iteration of cut generation, the CGLP object
        will already be at the previous solve's optimal basis, which is why no
        basis is returned

        :param warm_start_cglp: Override starting basis to reset to initial tableau
        when False. When True, use the previous CGLP's starting basis.
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: Starting basis for CGLP
        """
        assert isinstance(warm_start_cglp, bool), 'warm_start_cglp is boolean'
        if not warm_start_cglp:
            return (np.array([3]*self.cglp.lp.nVariables, dtype=np.int32),
                    np.array([1]*self.cglp.lp.nConstraints, dtype=np.int32))
        elif self.cut_generation_iterations == 1:
            return self.prev_cglp_basis
        else:
            return None

    def _select_cuts(self, cglp_cumulative_constraints: bool = True,
                     cglp_cumulative_bounds: bool = True, **kwargs) -> \
            Dict[str, Union[CyLPArray, float]]:
        """ Extends super()._select_cuts by marking down if this cut generation
        iteration added the disjunctive cut created from the CGLP. Additionally,
        if the initial disjunction and feasible region in each disjunctive term
        are being used, the CGLP cut generated is valid for all other nodes, so
        add it to sharable_cuts

        :param cglp_cumulative_constraints: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with cuts added to this node
        :param cglp_cumulative_bounds: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with the bounds placed on each
        variable in this node
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of cuts chosen to be added
        """

        assert isinstance(cglp_cumulative_constraints, bool), 'cglp_cumulative_constraints is bool'
        assert isinstance(cglp_cumulative_bounds, bool), 'cglp_cumulative_bounds is bool'

        # previous CGLP is always "added" if we're forcing the creation of CGLP in each iteration
        self.previous_cglp_added = self.force_create_cglp
        added_cuts = super()._select_cuts(**kwargs)
        for idx, (pi, pi0) in added_cuts.items():
            if self.cglp_name_pattern.match(idx):  # this node or previous node's cglps
                self.number_cglp_added += 1
                if self.current_cglp_name_pattern.match(idx):  # only this node's cglp
                    self.current_node_added_cglp = True  # marker for branching
                    self.previous_cglp_added = True  # marker for next cut generation iteration
                    # if using original bounds and constraints, cut valid for other subproblems
                    if not cglp_cumulative_bounds and not cglp_cumulative_constraints:
                        self.sharable_cuts[idx] = (pi, pi0)

        return added_cuts

    def branch(self: G, cglp_cumulative_constraints: bool = False,
               cglp_cumulative_bounds: bool = False, cglp: CutGeneratingLP = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Before calling parent and sibling class branch methods, create the
        cglp instance for each child node. Note, if the user sets either
        cglp_cumulative_constraints or cglp_cumulative_bounds to True, cuts
        generated from the CGLP can no longer be added to other nodes.

        :param cglp_cumulative_constraints: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with cuts added to this node
        :param cglp_cumulative_bounds: Whether or not to refine the feasible
        region of each disjunctive term in the CGLP with the bounds placed on each
        variable in this node
        :param cglp: Pulls 'cglp' from kwargs used to create root DisjunctiveCutBoundNode
        as to not interfere with cglp assignment in called subroutines
        :param kwargs: dictionary of arguments to pass on to selected subroutines
        :return: dictionary of children nodes
        """
        assert isinstance(cglp_cumulative_constraints, bool), 'cglp_cumulative_constraints is bool'
        assert isinstance(cglp_cumulative_bounds, bool), 'cglp_cumulative_bounds is bool'

        # don't pass on cglp if this node didn't use the cut
        if self.cglp is None or not self.current_node_added_cglp:
            return super().branch(force_create_cglp=self.force_create_cglp, **kwargs)

        elif cglp_cumulative_constraints or cglp_cumulative_bounds:
            A = None if not cglp_cumulative_constraints else self.lp.coefMatrix.copy()
            b = None if not cglp_cumulative_constraints else CyLPArray(self.lp.constraintsLower.copy())
            var_lb = None if not cglp_cumulative_bounds else CyLPArray(self.lp.variablesLower.copy())
            var_ub = None if not cglp_cumulative_bounds else CyLPArray(self.lp.variablesUpper.copy())

            cglp = CutGeneratingLP(bb=self.cglp.bb, root_id=self.cglp.root_id,
                                   A=A, b=b, var_lb=var_lb, var_ub=var_ub)
            return super().branch(cglp=cglp, force_create_cglp=self.force_create_cglp, **kwargs)

        else:
            # if recycling CGLP, pass on basis because children solutions wont be far off
            return super().branch(cglp=self.cglp, prev_cglp_basis=self.cglp.lp.getBasisStatus(),
                                  force_create_cglp=self.force_create_cglp, **kwargs)
