from cylp.cy import CyClpSimplex
import inspect
import unittest
from unittest.mock import patch

from simple_mip_solver import BaseNode
from simple_mip_solver.algorithms.utils import Utils
from test_simple_mip_solver.example_models import small_branch


class TestUtils(unittest.TestCase):
    _node_attributes = ['lower_bound', 'objective_value', 'solution',
                        'lp_feasible', 'mip_feasible', 'search_method',
                        'branch_method']
    _node_funcs = ['bound', 'branch', '__lt__', '__eq__']

    def test_init(self):

        alg = Utils(small_branch, BaseNode, self._node_attributes, self._node_funcs)

        # check attributes
        self.assertTrue(inspect.isclass(alg._Node))
        self.assertTrue(alg._root_node)
        self.assertTrue(alg.model)
        self.assertTrue(alg._kwargs == {'next_node_idx': 1})

        # check function calls
        with patch.object(Utils, '_convert_constraints_to_greq') as cctg:
            cctg.return_value = small_branch
            alg = Utils(small_branch, BaseNode, self._node_attributes,
                        self._node_funcs, standardize_model=True)
            self.assertTrue(cctg.called)

    def test_init_fails_asserts(self):
        lp = CyClpSimplex()

        # model asserts
        self.assertRaisesRegex(AssertionError, 'model must be cuppy MILPInstance',
                               Utils, lp, BaseNode, self._node_funcs,
                               self._node_attributes)

        # Node asserts
        self.assertRaisesRegex(AssertionError, 'Node must be a class',
                               Utils, small_branch, 'Node', self._node_funcs,
                               self._node_attributes)
        for attribute in self._node_attributes:

            class BadNode(BaseNode):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    delattr(self, attribute)

            self.assertRaisesRegex(AssertionError, f'Node needs a {attribute}',
                                   Utils, small_branch, BadNode, self._node_attributes,
                                   self._node_funcs)

        for func in self._node_funcs:

            class BadNode(BaseNode):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.__dict__[func] = 5

            self.assertRaisesRegex(AssertionError, f'Node needs a {func}',
                                   Utils, small_branch, BadNode, self._node_attributes,
                                   self._node_funcs)

        # kwarg asserts
        self.assertRaisesRegex(AssertionError, 'next_node_idx is reserved',
                               Utils, small_branch, BaseNode, self._node_attributes,
                               self._node_funcs, next_node_idx=1)

    def test_convert_constraints_to_greq(self):
        # check one that needs changed
        m = Utils._convert_constraints_to_greq(small_branch)
        self.assertTrue(m.sense == '>=')
        self.assertTrue((-m.A == small_branch.A).all())
        self.assertTrue((-m.b == small_branch.b).all())

        # check one that doesnt
        m2 = Utils._convert_constraints_to_greq(m)
        self.assertTrue(m2.sense == '>=')
        self.assertTrue((m2.A == m.A).all())
        self.assertTrue((m2.b == m.b).all())

    def test_process_rtn_fails_asserts(self):
        alg = Utils(small_branch, BaseNode, self._node_attributes, self._node_funcs)
        self.assertRaisesRegex(AssertionError, 'rtn must be a dictionary',
                               alg._process_rtn, 'fish')

    def test_process_rtn(self):
        alg = Utils(small_branch, BaseNode, self._node_attributes, self._node_funcs)
        alg._process_rtn({'pseudo_costs': 5})
        self.assertTrue(alg._kwargs['pseudo_costs'] == 5)


if __name__ == '__main__':
    unittest.main()
