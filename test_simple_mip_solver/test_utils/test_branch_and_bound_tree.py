from coinor.cuppy.milpInstance import MILPInstance
from cylp.cy.CyCbcModel import CyCbcModel
from cylp.cy.CyClpSimplex import CyClpSimplex
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode, BranchAndBound
from simple_mip_solver.algorithms.base_algorithm import BaseAlgorithm
from test_simple_mip_solver.example_models import no_branch, small_branch, infeasible, \
    small_branch_copy

from simple_mip_solver.utils.branch_and_bound_tree import BranchAndBoundTree


class TestBranchAndBoundTree(unittest.TestCase):

    def setUp(self) -> None:
        # reset models each test so lps dont keep added constraints
        for name, m in {'small_branch_std': small_branch, 'infeasible_std': infeasible,
                        'no_branch_std': no_branch}.items():
            lp = m.lp
            new_m = MILPInstance(A=m.A, b=m.b, c=lp.objective, l=m.l, sense=['Min', m.sense],
                                 integerIndices=m.integerIndices, numVars=len(lp.objective))
            new_m = BaseAlgorithm._convert_constraints_to_greq(new_m)
            setattr(self, name, new_m)

    def create_prelims(self):
        # create bnb
        self.lp = CyClpSimplex()
        self.lp.extractCyLPModel('/Users/sean/coin-or/Data/Sample/p0201.mps')
        self.bnb = CyCbcModel(self.lp)
        self.bnb.persistNodes = True
        self.bnb.solve(arguments=["-preprocess", "off", "-presolve", "off"])

    def test_get_leaves_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, "subtree_root_id must belong to the tree",
                               bb.tree.get_leaves, 20)
        self.assertRaisesRegex(AssertionError, "depth is a nonnegative integer",
                               bb.tree.get_leaves, subtree_root_id=0, depth=1.5)
        self.assertRaisesRegex(AssertionError, "keep is one of 'all', 'feasible', or 'not infeasible'",
                               bb.tree.get_leaves, subtree_root_id=0, keep=False)

    def test_get_leaves(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False, node_limit=1)
        bb.solve()

        # feasible vs not infeasible
        leaves = bb.tree.get_leaves(0, keep='not infeasible')
        self.assertTrue(len(leaves) == 2)
        leaves = bb.tree.get_leaves(0, keep='feasible')
        self.assertFalse(leaves)

        bb.node_limit = float('inf')
        bb.solve()

        # all leaves
        leaves = bb.tree.get_leaves(0)
        for node_id, node in bb.tree.nodes.items():
            if node_id in [n.idx for n in leaves]:
                self.assertFalse(bb.tree.get_children(node_id))
            else:
                self.assertTrue(len(bb.tree.get_children(node_id)) == 2)

        # all feasible leaves
        leaves = bb.tree.get_leaves(0, keep='feasible')
        for node_id, node in bb.tree.nodes.items():
            if node_id in [n.idx for n in leaves]:
                self.assertFalse(bb.tree.get_children(node_id))
                self.assertTrue(node.attr['node'].lp_feasible)
            else:
                self.assertTrue(len(bb.tree.get_children(node_id)) == 2 or not
                                node.attr['node'].lp_feasible)

        # depth 0 subtree
        leaves = bb.tree.get_leaves(2, depth=0)
        self.assertTrue(len(leaves) == 1)
        self.assertTrue(leaves[0].idx == 2)

        leaves = bb.tree.get_leaves(2, depth=0, keep='feasible')
        self.assertFalse(leaves)

        # depth 1 subtree
        leaves = bb.tree.get_leaves(0, depth=1)
        self.assertTrue(len(leaves) == 2)
        self.assertTrue(set(n.idx for n in leaves) == {1, 2})

        leaves = bb.tree.get_leaves(0, depth=1, keep='feasible')
        self.assertTrue(len(leaves) == 1)
        self.assertTrue(leaves[0].idx == 1)

        # depth 2 subtree
        leaves = bb.tree.get_leaves(1, depth=2)
        self.assertTrue({n.idx for n in leaves} == {5, 6, 7, 8})
        for node in leaves:
            self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)

        leaves = bb.tree.get_leaves(1, depth=2, keep='feasible')
        self.assertTrue({n.idx for n in leaves} == {5, 7})
        for node in leaves:
            self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)

        # depth 3 subtree
        leaves = bb.tree.get_leaves(1, depth=3)
        self.assertTrue({n.idx for n in leaves} == {5, 6, 8, 9, 10})
        for node in leaves:
            if node.idx <= 8:
                self.assertTrue(bb.tree.get_parent(bb.tree.get_parent(node.idx)) == 1)
            else:
                self.assertTrue(
                    bb.tree.get_parent(bb.tree.get_parent(bb.tree.get_parent(node.idx))) == 1
                )

        leaves = bb.tree.get_leaves(1, depth=3, keep='feasible')
        self.assertTrue({n.idx for n in leaves} == {5, 9})

    def test_get_disjunction_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, "subtree_root_id must belong to the tree",
                               bb.tree.get_disjunction, 20)

    def test_get_disjunction(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        disjunction = bb.tree.get_disjunction(0)
        self.assertTrue(all(disjunction[5][0] == [0, 0, 0]))
        self.assertTrue(all(disjunction[5][1] == [0, 1, 1]))
        self.assertTrue(all(disjunction[11][0] == [1, 0, 0]))
        self.assertTrue(all(disjunction[11][1] == [1, 1, 0]))

    def test_get_node_instances_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, 'must be an integer or iterable',
                               bb.tree.get_node_instances, '1')
        self.assertRaisesRegex(AssertionError, 'node_ids are not in the tree',
                               bb.tree.get_node_instances, [20])
        del bb.tree.nodes[0].attr['node']
        self.assertRaisesRegex(AssertionError, 'must have an attribute for a node instance',
                               bb.tree.get_node_instances, [0])

    def test_get_node_instances(self):
        bb = BranchAndBound(small_branch_copy, gomory_cuts=False)
        bb.solve()

        # test list
        node1, node2 = bb.tree.get_node_instances([1, 2])
        self.assertTrue(node1.idx == 1, 'we should get node with matching id')
        self.assertTrue(isinstance(node1, BaseNode), 'we should get a node')
        self.assertTrue(node2.idx == 2, 'we should get node with matching id')
        self.assertTrue(isinstance(node2, BaseNode), 'we should get a node')

        # test singleton
        node1 = bb.tree.get_node_instances(1)
        self.assertTrue(node1.idx == 1, 'we should get node with matching id')
        self.assertTrue(isinstance(node1, BaseNode), 'we should get a node')

    def test_subtree_dual_bound_fails_asserts(self):
        bb = BranchAndBound(self.small_branch_std, gomory_cuts=False)
        self.assertRaisesRegex(AssertionError, 'subtree_root_id must belong to the tree',
                               bb.tree.subtree_dual_bound, subtree_root_id=1)

    def test_subtree_dual_bound(self):
        bb = BranchAndBound(small_branch_copy, gomory_cuts=False, node_limit=1)

        # 0 nodes
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -float('inf'))

        # 1 node
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2.75)

        # 2 nodes
        bb.node_limit = 2
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2.75)

        # all nodes
        bb.node_limit = float('inf')
        bb.solve()
        self.assertTrue(bb.tree.subtree_dual_bound(0) == -2)
        self.assertTrue(bb.tree.subtree_dual_bound(2) == float('inf'))
        self.assertTrue(bb.tree.subtree_dual_bound(0, depth=1) == -2.75)

    def test_build_tree(self):
        self.create_prelims()

        tree = BranchAndBoundTree()
        tree.bnb, tree.root_lp = self.bnb, self.lp
        tree._build_tree()

        # check for each node that parents follow lineage
        for tree_node in tree.nodes.values():
            n = tree_node.attr['node']
            current_ancestor = tree_node
            for next_ancestor_idx in reversed(n.lineage[:-1]):
                next_ancestor = tree.nodes[next_ancestor_idx]
                self.assertTrue(tree.get_parent(current_ancestor.name) ==
                                next_ancestor.name)
                current_ancestor = next_ancestor

        # check if leaf status assigned correctly
        for tree_node in tree.nodes.values():
            if tree_node.attr['node'].is_leaf:
                self.assertFalse(tree.get_children(tree_node.attr['node'].idx))
            else:
                self.assertTrue(tree.get_children(tree_node.attr['node'].idx))

    def test_tie_node_into_tree(self):

        # # create tree
        # tree = BranchAndBoundTree()
        # tree.bnb, tree.root_lp = self.bnb, self.lp
        # node_map = self.bnb.nodeMap
        # root_idxs = {n.lineage[0] for n in node_map}
        # root_node_map = {n: root_lp for n, root_lp in node_map.items() if n.index in root_idxs}
        #
        # # test different cases
        # root_node = list(root_node_map.keys())[0]
        # root_lp = list(root_node_map.values())[0]
        # integer_indices = [i for i, integer in enumerate(root_lp.integerInformation) if integer]
        # n = BaseNode(lp=root_lp, integer_indices=integer_indices, idx=root_node.index,
        #              ancestors=tuple(root_node.lineage[:-1]), is_leaf=root_node.isLeaf)
        # tree._tie_node_into_tree(n)
        pass

    def test_branch_node(self):
        # root_node = BaseNode(lp=self.lp, idx=-1, integer_indices=self.lp.integerIndices)
        # tree = BranchAndBoundTree()
        # tree.add_root(root_node.idx, node=root_node)
        # child = tree.branch_node(root_node, 0, .5)
        #
        # self.assertTrue(len(tree.nodes) == 3)
        # self.assertTrue(tree.get_left_child(-1).attr['node'] is child['left'])
        # self.assertTrue(tree.get_right_child(-1).attr['node'] is child['right'])
        pass

    def test_swap_node_fails_asserts(self):
        # root_node = BaseNode(lp=self.lp, idx=-1, integer_indices=self.lp.integerIndices)
        # tree = BranchAndBoundTree()
        # tree.add_root(root_node.idx, node=root_node)
        # child = tree.branch_node(root_node, 0, .5)
        #
        # self.assertRaisesRegex(AssertionError, 'should have the same integer variable bounds',
        #                        tree.swap_node, child['left'], child['right'])
        # self.assertRaisesRegex(AssertionError, 'not set up to replace internal nodes',
        #                        tree.swap_node, root_node, root_node)
        pass

    def test_swap_node(self):
        # root_node = BaseNode(lp=self.lp, idx=-1, integer_indices=self.lp.integerIndices)
        # tree = BranchAndBoundTree()
        # tree.add_root(root_node.idx, node=root_node)
        # child = tree.branch_node(root_node, 0, .5)
        # other_child = root_node._base_branch(0, 0, .5)
        # tree.swap_node(child['left'], other_child['left'])
        # self.assertTrue({0, -1} == {n.idx for n in tree.get_leaves(-1)})
        pass

    def test_nearest_node_fails_asserts(self):
        # root_node = BaseNode(lp=self.lp, idx=-1, integer_indices=self.lp.integerIndices)
        # tree = BranchAndBoundTree()
        # child = root_node._base_branch(branch_idx=97, next_node_idx=-3, b_val=.5)
        # tree.add_root(child['left'].idx, node=child['left'])
        #
        # self.assertRaisesRegex(AssertionError, 'must be a BaseNode instance',
        #                        tree.nearest_node, self.lp)
        # self.assertRaisesRegex(AssertionError, 'no ancestor of n exists',
        #                        tree.nearest_node, root_node)
        pass

    def test_nearest_node(self):

        # check if the most number of same branching decisions ignored for ancestor
        # root_node = BaseNode(lp=self.lp, idx=-1, integer_indices=self.lp.integerIndices)
        # tree = BranchAndBoundTree()
        # tree.add_root(root_node.idx, node=root_node)
        #
        # first = tree.branch_node(root_node, 0, .5)
        # second_left = tree.branch_node(first['left'], 1, .5)
        # third_left = tree.branch_node(second_left['left'], 2, .5)
        #
        # second_right = first['right']._base_branch(branch_idx=1,
        #                                            next_node_idx=min(tree.nodes.keys()) - 2,
        #                                            b_val=.5)
        # third_right = second_right['left']._base_branch(branch_idx=2,
        #                                                 next_node_idx=min(tree.nodes.keys()) - 2,
        #                                                 b_val=.5)
        # self.assertTrue(tree.nearest_node(third_right['left']) is first['right'])
        pass


if __name__ == '__main__':
    unittest.main()
