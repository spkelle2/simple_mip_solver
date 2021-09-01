from itertools import product
import unittest
from unittest.mock import patch

from simple_mip_solver.nodes.branch.pseudo_cost import PseudoCostBranchNode
from test_simple_mip_solver.example_models import small_branch

from test_simple_mip_solver.helpers import TestModels


class TestNode(TestModels):

    Node = PseudoCostBranchNode

    def setUp(self) -> None:
        self.kwargs = {'pseudo_costs': {}}

    def test_init(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        self.assertTrue(node.branch_method == 'pseudo cost')
        self.assertFalse(node.pseudo_costs, 'should exist but be none')
        self.assertFalse(node.strong_branch_iters, 'should exist but be none')

    def test_bound_fails_assertions(self):
        pc = {1: 'hi'}
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'pseudo cost dict has following errors:',
                               node.bound, pc)

    def test_bound(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        rtn = node.bound({})

        # check assignments
        for idx, direction in product([1, 2], ['right', 'left']):
            self.assertTrue(node.pseudo_costs[idx][direction]['times'] == 1)
            if idx == 1 and direction == 'left':
                self.assertTrue(node.pseudo_costs[idx][direction]['cost'] == 1)
            else:
                self.assertTrue(node.pseudo_costs[idx][direction]['cost'] == 0)
        self.assertTrue(node.strong_branch_iters == 5)

        # check returns
        for idx, direction in product([1, 2], ['right', 'left']):
            self.assertTrue(rtn['pseudo_costs'][idx][direction]['times'] == 1)
            if idx == 1 and direction == 'left':
                self.assertTrue(rtn['pseudo_costs'][idx][direction]['cost'] == 1)
            else:
                self.assertTrue(rtn['pseudo_costs'][idx][direction]['cost'] == 0)

        # check function calls
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        node.lp_feasible = False
        with patch.object(node, '_check_pseudo_costs') as cpc, \
                patch.object(node, '_base_bound') as bb, \
                patch.object(node, '_update_pseudo_costs') as upc:
            cpc.return_value = []
            node.bound({})
            self.assertTrue(cpc.call_count == 1)
            self.assertTrue(bb.call_count == 1)
            self.assertTrue(upc.call_count == 0)

        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        node.lp_feasible = True
        with patch.object(node, '_check_pseudo_costs') as cpc, \
                patch.object(node, '_base_bound') as bb, \
                patch.object(node, '_update_pseudo_costs') as upc:
            cpc.return_value = []
            node.bound({})
            self.assertTrue(cpc.call_count == 1)
            self.assertTrue(bb.call_count == 1)
            self.assertTrue(upc.call_count == 1)

    def test_update_pseudo_costs(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        node.pseudo_costs = {}
        node._base_bound()

        # for root node (sb_index = 2)
        # check we call strong branch once and calculate costs twice for each sb_index
        with patch.object(node, '_strong_branch') as sb, \
                patch.object(node, '_calculate_costs') as cc:
            sb.return_value = {'right': PseudoCostBranchNode(small_branch.lp,
                                                          small_branch.integerIndices),
                               'left': PseudoCostBranchNode(small_branch.lp,
                                                            small_branch.integerIndices)}
            node._update_pseudo_costs()
            self.assertTrue(sb.call_count == 2)
            self.assertTrue(cc.call_count == 4)

        # branch on 2 (len(sb_index) = 1)
        # check we call strong branch len(sb_index) times and calc costs 2*len(sb_index) + 1
        rtn = node._base_branch(2)  # force x0 to go from int to fractional
        left_node = rtn['left']  # just do left bc right infeasible
        left_node.pseudo_costs = {
            1: {'right': {'cost': 0, 'times': 1}, 'left': {'cost': 1, 'times': 1}},
            2: {'right': {'cost': 0, 'times': 1}, 'left': {'cost': 0, 'times': 1}}}
        left_node._base_bound()
        with patch.object(left_node, '_strong_branch') as sb, \
                patch.object(left_node, '_calculate_costs') as cc:
            sb.return_value = {'right': PseudoCostBranchNode(small_branch.lp,
                                                          small_branch.integerIndices),
                               'left': PseudoCostBranchNode(small_branch.lp,
                                                            small_branch.integerIndices)}
            left_node._update_pseudo_costs()
            self.assertTrue(sb.call_count == 1)
            self.assertTrue(cc.call_count == 3)

    def test_calculate_costs(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        node.pseudo_costs = {}
        node._base_bound()

        # check that each fractional index gets proper pseudo cost instantiated
        # check that infeasible strong branch direction gets (0, 1) ie right and x >= 2
        for idx in [1, 2]:
            for strong_branch_node in node._strong_branch(idx).values():
                node._calculate_costs(strong_branch_node)
        for idx, direction in product([1, 2], ['right', 'left']):
            self.assertTrue(node.pseudo_costs[idx][direction]['times'] == 1)
            if idx == 1 and direction == 'left':
                self.assertTrue(node.pseudo_costs[idx][direction]['cost'] == 1)
            else:
                self.assertTrue(node.pseudo_costs[idx][direction]['cost'] == 0)

        # check that branched on index updates the instantiated value correctly
        rtn = {k: v for k, v in node._base_branch(1).items() if k in ['left', 'right']}
        for direction, child_node in rtn.items():
            child_node.pseudo_costs = node.pseudo_costs
            child_node._base_bound()
            child_node._calculate_costs(child_node)
        for direction in ['right', 'left']:
            self.assertTrue(node.pseudo_costs[1][direction]['times'] == 2)
            if direction == 'left':
                self.assertTrue(node.pseudo_costs[1][direction]['cost'] == 1)
            else:
                self.assertTrue(node.pseudo_costs[1][direction]['cost'] == 0)

    def test_branch_fails_assertions(self):
        pc = {1: 'hi'}
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        self.assertRaisesRegex(AssertionError, 'pseudo cost dict has following errors:',
                               node.branch, pc)
        node.mip_feasible = True
        self.assertRaisesRegex(AssertionError, 'must have fractional value to branch',
                               node.branch, pc)

    def test_branch(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        rtn = node.bound({})

        # check function calls
        with patch.object(node, '_check_pseudo_costs') as cpc, \
                patch.object(node, '_best_pseudo_costs_index') as bpci, \
                patch.object(node, '_base_branch') as bb:
            cpc.return_value = []
            bpci.return_value = 2
            node.branch(rtn['pseudo_costs'], )
            self.assertTrue(cpc.called)
            self.assertTrue(bpci.called)
            self.assertTrue(bb.called)

        # check return
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        rtn = node.bound({})
        rtn = node.branch(rtn['pseudo_costs'], )
        for direction in ['right', 'left']:
            self.assertTrue(direction in rtn,
                            f'{direction} must be in the returned dict')
            self.assertTrue(isinstance(rtn[direction], PseudoCostBranchNode))

    def test_best_pseudo_cost_index(self):
        pc = {1: {'right': {'cost': 1, 'times': 1}, 'left': {'cost': 1, 'times': 1}},
              2: {'right': {'cost': 1, 'times': 1}, 'left': {'cost': 1, 'times': 1}}}
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)
        node.solution = [0, 1.25, 2.5]
        self.assertTrue(node._best_pseudo_costs_index(pc) == 2)
        pc[1] = {'right': {'cost': 10, 'times': 1}, 'left': {'cost': 1, 'times': 1}}
        self.assertTrue(node._best_pseudo_costs_index(pc) == 2)
        pc[1] = {'right': {'cost': 10, 'times': 1}, 'left': {'cost': 10, 'times': 1}}
        self.assertTrue(node._best_pseudo_costs_index(pc) == 1)

    def test_check_pseudo_costs(self):
        node = PseudoCostBranchNode(small_branch.lp, small_branch.integerIndices)

        # check good
        self.assertFalse(node._check_pseudo_costs({}), 'empty should be fine')
        pc = {1: {'right': {'cost': 0, 'times': 0}, 'left': {'cost': 0, 'times': 0}}}
        self.assertFalse(node._check_pseudo_costs(pc), 'good dict should be fine')

        # check bad times
        pc[1]['left']['times'] = -1
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 1 direction left times must be nonnegative int',
                        "check pc[1]['left']['times']")
        del pc[1]['left']['times']
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 1 direction left missing times',
                        "check pc[1]['left']")

        # check bad costs
        pc[1]['left']['cost'] = -1
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 1 direction left cost must be nonnegative number',
                        "check pc[1]['left']['cost']")
        del pc[1]['left']['cost']
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 1 direction left missing cost',
                        "check pc[1]['left']")

        # check missing direction
        del pc[1]['left']
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 1 missing direction left',
                        "check pc[1]")

        # check bad integers
        pc[15] = 'hi'
        del pc[1]
        err = node._check_pseudo_costs(pc)[0]
        self.assertTrue(err == 'index 15 not integer index',
                        "pc[15] should error")

    def test_models(self):
        self.base_test_models()


if __name__ == '__main__':
    unittest.main()
