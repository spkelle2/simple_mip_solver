from coinor.gimpy.tree import BinaryTree
from cylp.cy.CyCbcModel import CyCbcModel
from cylp.cy.CyClpSimplex import CyClpSimplex
import itertools
import numpy as np
from typing import Any, Dict, TypeVar, List, Union, Iterable, Type, Tuple

from simple_mip_solver.nodes.base_node import BaseNode


BT = TypeVar('BT', bound='BranchAndBoundTree')
# todo: cuts include integer variable bounds edited beyond branching and continuous variable bounds
# todo: for identifying cuts, make sure to count ranges from initial LP as two constraints
# todo: mark feasible, fathomed, or infeasible for CGLP building


class BranchAndBoundTree(BinaryTree):
    """Class used to represent the underlying tree structure of branch and bound"""

    # assuming the user is smart enough here to match bnb with appropriate root_lp
    def __init__(self, bnb: CyCbcModel = None, root_lp: CyClpSimplex = None, **kwargs):
        """

        :param bnb: The CyCbcModel off which we'll rebuild the branch and bound tree
        :param root_lp: The CyClpSimplex used to create the bnb CyCbcModel instance
        :param kwargs: options to pass to super classes' inits
        """

        super().__init__(**kwargs)

        if bnb is not None and root_lp is not None:
            assert isinstance(bnb, CyCbcModel)
            assert isinstance(root_lp, CyClpSimplex)
        else:
            assert bnb is None and root_lp is None, 'bnb and root_lp should both or neither be None'

        self.bnb = bnb
        self.root_lp = root_lp

        if bnb is not None and root_lp is not None:
            self._build_tree()

    def _build_tree(self) -> None:
        """ rebuilds the tree from the nodes gathered in CyCbcModel. This won't
        be productionalized, so the user needs to make sure lp is the LP from which
        bnb was made

        :return:
        """
        assert self.bnb.persistNodes, 'must persist nodes to rebuild tree'

        # get nodes from CBC
        node_map = {
            n.index: BaseNode(
                lp=lp, integer_indices=lp.integerIndices, idx=n.index,
                ancestors=tuple(n.lineage[:-1]), b_idx=n.branchVariable,
                b_dir=n.branchWay, is_leaf=n.isLeaf, lp_feasible=n.lpFeasible
            )
            for n, lp in self.bnb.nodeMap.items()
        }
        root_nodes = {n.idx: n for n in node_map.values() if len(n.lineage) == 1}
        assert len(root_nodes) == 1, 'there should only be one root'

        for n in node_map.values():
            parent_node = None if len(n.lineage) == 1 else node_map[n.lineage[-2]]
            if parent_node is None:
                self.add_root(n.idx, node=n)
            else:
                assert parent_node.idx in self.nodes, 'parent node should already be added'
                assert n.is_child(parent_node), 'CBC branching requirements not met'
                getattr(self, f'add_{n._b_dir}_child')(n.idx, parent_node.idx, node=n)

        # for tree_node_idx in self.nodes:
        #     if self.get_children(tree_node_idx) and tree_node_idx in node_map:
        #         node_map[tree_node_idx].is_leaf = False

    def get_leaves(self: BT, subtree_root_id: int, depth: int = None,
                   keep: str = 'all') -> List[BaseNode]:
        """ If depth is None, gather all leaves for a subtree rooted at node with
        id <subtree_root_id>. Otherwise, gather all leaves for a subtree rooted at
        node with id <subtree_root_id> after descendents more than <depth> edges
        away have been removed.

        Caution: Could be very slow when used repeatedly on large trees with <depth> > 1

        :param subtree_root_id: The id of the node that roots our subtree
        :param depth: Depth beyond which nodes are excluded from the subtree
        :param keep: Specifies if returned leaves should keep 'all' of those found, only
        those that are LP 'feasible', or only those with LP's that are 'not infeasible'.
        :return: the desired leaves of the subtree
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        assert keep in ['all', 'feasible', 'not infeasible'], \
            "keep is one of 'all', 'feasible', or 'not infeasible'"
        if depth is not None:
            assert isinstance(depth, int) and depth >= 0, 'depth is a nonnegative integer'
            if depth == 0:
                rtn = self.get_node_instances([subtree_root_id])
            elif depth == 1:
                rtn = self.get_node_instances(self.get_children(subtree_root_id))
            else:
                # leaves less than <depth> levels away
                leaves_within_depth = [
                    n.attr['node'] for n in self.nodes.values() if n.attr['node'].is_leaf
                    and subtree_root_id in n.attr['node'].lineage[-depth:]
                ]
                # nodes <depth> levels away
                depth_descendents = [
                    n.attr['node'] for n in self.nodes.values() if
                    len(n.attr['node'].lineage) >= depth + 1 and
                    subtree_root_id == n.attr['node'].lineage[-(depth + 1)]
                ]
                rtn = leaves_within_depth + depth_descendents
        else:
            rtn = [n.attr['node'] for n in self.nodes.values() if n.attr['node'].is_leaf
                   and subtree_root_id in n.attr['node'].lineage]
        return rtn if keep == 'all' else [n for n in rtn if n.lp_feasible] if \
            keep == 'feasible' else [n for n in rtn if n.lp_feasible is not False]

    def get_disjunction(self: BT, subtree_root_id: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """ Return the disjunction encoded in the terminal leaves of the branch
        and bound subtree rooted at node with id <subtree_root_id>

        :param subtree_root_id: The id of the node that roots our subtree
        :return: a dictionary keyed by indices of nodes with values as pairs of
        arrays, (lb, ub), representing the variable bounds on the node with the
        given index. For x to be a feasible solution, there must be a (lb, ub) value
        in the dict such that lb <= x <= ub.
        """
        return {n.idx: (n.lp.variablesLower.copy(), n.lp.variablesUpper.copy())
                for n in self.get_leaves(subtree_root_id, keep='not infeasible')}

    def get_node_instances(self: BT, node_ids: Union[int, Iterable[int]]) -> \
            Union[BaseNode, List[BaseNode]]:
        is_int = False
        if isinstance(node_ids, int):
            is_int = True
            node_ids = [node_ids]
        else:
            assert isinstance(node_ids, Iterable) and not isinstance(node_ids, str), \
                'node_ids must be an integer or iterable (that is not a string)'
            node_ids = list(node_ids)
        missing_ids = set(node_ids) - set(self.nodes)
        assert not missing_ids, f'the following node_ids are not in the tree: {missing_ids}'
        instances = [self.nodes[idx].attr.get('node') for idx in node_ids]
        assert all(instance is not None for instance in instances), \
            'each vertex in the branch and bound tree must have an attribute for a node instance'
        return instances if not is_int else instances[0]

    def subtree_dual_bound(self: BT, subtree_root_id: int, depth: int = None) -> \
            Union[float, int]:
        """ Finds the dual bound for the branch and bound subtree rooted at node
        <subtree_root_id> with maximum depth <depth>

        :param subtree_root_id: The id of the node that roots our subtree
        :param depth: depth beyond the subtree root which nodes are excluded
        for calculating dual bound
        :return: the dual bound for the branch and bound subtree rooted at node
        <subtree_root_id> with maximum depth <depth>
        """
        assert subtree_root_id in self, 'subtree_root_id must belong to the tree'
        return min(n.objective_value if n.objective_value is not None else n.dual_bound
                   for n in self.get_leaves(subtree_root_id, depth=depth))

    def _tie_node_into_tree(self, input_node: BaseNode) -> None:
        assert isinstance(input_node, BaseNode), 'input_node must be a BaseNode instance'

        nearest_node = self.nearest_node(input_node)  # closest ancestor already in tree
        diff_lbs, diff_ubs = nearest_node.different_bounds(input_node)

        # and all other nodes in between the root and n
        for idx in diff_ubs:
            child = self.branch_node(nearest_node, idx, input_node.lp.variablesUpper[idx] + .5)
            nearest_node = child['left']

        for idx in diff_lbs:
            child = self.branch_node(nearest_node, idx, input_node.lp.variablesLower[idx] - .5)
            nearest_node = child['right']

        self.swap_node(nearest_node, input_node)

    def branch_node(self, node: BaseNode, idx: int, b_val: float) -> dict[str: BaseNode]:
        child = node._base_branch(branch_idx=idx, next_node_idx=min(self.nodes.keys()) - 2,
                                  b_val=b_val)
        self.add_left_child(child['left'].idx, node.idx, node=child['left'])
        self.add_right_child(child['right'].idx, node.idx, node=child['right'])
        return child

    def swap_node(self, out_node: BaseNode, in_node: BaseNode) -> None:
        """ Replace node <out_node> in the branch and bound tree with node
        <in_node>

        :param out_node: node to be replaced
        :param in_node: replacement node
        :return:
        """
        assert not out_node.number_different_bounds(in_node), \
               "in_node replaces out_node, so they should have the same integer variable bounds"
        assert not self.get_children(out_node.idx), 'not set up to replace internal nodes'
        b_dir = 'right' if self.get_node_attr(out_node.idx, 'direction') == 'R' else 'left'
        self.del_node(out_node.idx)
        getattr(self, f'add_{b_dir}_child')(in_node.idx, out_node.lineage[-2], node=in_node)
        in_node.lineage = out_node.lineage[:-1] + in_node.lineage

    def nearest_node(self: BT, n: BaseNode) -> BaseNode:
        """ finds the closest node in the tree to n in terms of least number of
        different branching decisions

        :param n: node for which we find the closest node in the branch and bound
        tree
        :return: the closest node in the branch and bound tree
        """
        assert isinstance(n, BaseNode), 'input_node must be a BaseNode instance'
        # want to find fewest different bounds while also being an outside approximation
        different_bounds = {
            tree_node.name: tree_node.attr['node'].number_different_bounds(n) for tree_node
            in self.nodes.values() if tree_node.attr['node'].relaxed_disjunction(n)
        }
        assert different_bounds, 'no ancestor of n exists in the tree'
        nearest_node = self.nodes[min(different_bounds, key=different_bounds.get)].attr['node']
        return nearest_node
