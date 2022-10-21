from coinor.cuppy.milpInstance import MILPInstance
# so pulp and pyomo don't believe reading in flat files is a worthwhile feature
# so we don't have much of an option here but to use some sort of commercial solver
try:  # if you don't have gurobipy installed, all tests except those using gurobi will run
    import gurobipy as gu
except ImportError:
    gu = None
import inspect
from itertools import product
from math import isclose
import numpy as np
import os
import unittest

from simple_mip_solver import DisjunctiveCutBoundNode, BranchAndBound, PseudoCostBranchNode
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from test_simple_mip_solver.example_models import generate_random_variety


class TestModels(unittest.TestCase):
    """ A unit testing base class that can be inherited from to test a BaseNode
    subclass against the suite of random milps in the example_models directory.

    Make sure when using that the child class overwrites the Node attribute with
    the class of the node being tested
    """

    Node = None

    def base_test_models(self, **kwargs):
        self.assertTrue(gu, 'gurobipy needed for this test')
        fldr = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(generate_random_variety))),
            'example_models'
        )
        for i, file in enumerate(os.listdir(fldr)):
            print(f'running test {i + 1}')
            pth = os.path.join(fldr, file)
            gu_mdl = gu.read(pth)
            gu_mdl.setParam(gu.GRB.Param.LogToConsole, 0)
            gu_mdl.optimize()
            model = MILPInstance(file_name=pth)
            bb = BranchAndBound(model, self.Node, pseudo_costs={}, **kwargs)
            bb.solve()
            if not isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01):
                print(f'different for {file}')
                print(f'mine: {bb.objective_value}')
                print(f'gurobi: {gu_mdl.objVal}')
            self.assertTrue(isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01),
                            f'different for {file}')
            self.check_pseudo_costs(bb)
            self.check_gmics(bb)

    def check_pseudo_costs(self, bb):
        if bb.evaluated_nodes >= 4 and isinstance(bb.root_node, PseudoCostBranchNode):
            p = bb._kwargs['pseudo_costs']
            # again just some rough numbers here to make sure nothing is super weird
            self.assertTrue(len(p.keys()) <= bb.root_node.lp.nVariables)
            self.assertTrue(sum(sum(b['times'] for b in idx_dict.values()) for idx_dict in p.values())
                            <= 2*(bb.evaluated_nodes + bb.root_node.lp.nVariables))

    def check_gmics(self, bb):
        if bb.evaluated_nodes >= 2 and bb._kwargs.get('gomory_cuts', True):
            # these are just rough values - if broken just check to make sure ok
            self.assertTrue(bb._kwargs['total_iterations_gmic_created'] /
                            bb._kwargs['total_cut_generation_iterations'] >= .25)
            self.assertTrue(bb._kwargs['total_number_gmic_created'] /
                            bb._kwargs['total_cut_generation_iterations'] >= .5)
            # you can end up with more being removed because of branching
            self.assertTrue(bb._kwargs['total_number_gmic_added'] <=
                            bb._kwargs['total_number_gmic_created'])
            # you can end up with more iters added from cuts left over in pool
            self.assertTrue(bb._kwargs['total_iterations_gmic_added'] <=
                            1.5*bb._kwargs['total_iterations_gmic_created'])

    def disjunctive_cut_test_models(self, ratio_run=.1, **input_kwargs):
        count_different = 0
        dif = {}
        self.assertTrue(gu, 'gurobipy needed for this test')
        fldr = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(generate_random_variety))),
            'example_models'
        )
        kwarg_values = product(*([[True, False] for _ in range(4)] + [[1, None]]))
        kwargs_list = [
            {'cglp_cumulative_constraints': cc_bool, 'cglp_cumulative_bounds': cb_bool,
             'gomory_cuts': gc_bool, 'warm_start_cglp': ws_cglp, 'max_cglp_calls': max_cglp}
            for (cc_bool, cb_bool, gc_bool, ws_cglp, max_cglp) in kwarg_values
        ]
        num_kwargs = len(kwargs_list)
        num_fldrs = len(os.listdir(fldr))
        for j, kwargs in enumerate(kwargs_list):
            for i, file in enumerate(os.listdir(fldr)):
                # todo: test_disjunctive_cut.test_models generates a bad GMIC for i == 3
                if np.random.uniform() > ratio_run or i == 3:
                    continue
                print(f'running test {(i + 1) + j * num_fldrs} of {num_kwargs * num_fldrs}')
                pth = os.path.join(fldr, file)

                # check gurobi
                gu_mdl = gu.read(pth)
                gu_mdl.setParam(gu.GRB.Param.LogToConsole, 0)
                gu_mdl.optimize()

                # check ours
                model = MILPInstance(file_name=pth)
                cglp_bb = BranchAndBound(model, node_limit=8, gomory_cuts=kwargs['gomory_cuts'])
                cglp_bb.solve()
                cglp = CutGeneratingLP(cglp_bb.tree, cglp_bb.root_node.idx)
                bb = BranchAndBound(model, self.Node, cglp=cglp, pseudo_costs={},
                                    **input_kwargs, **kwargs)
                bb.solve()

                if not isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01):
                    print(f'different for {file}')
                    print(f'mine: {bb.objective_value}')
                    print(f'gurobi: {gu_mdl.objVal}')
                    dif[i, j] = {'mine': bb.objective_value, 'gurobi': gu_mdl.objVal,
                                 'file': file, **kwargs}
                    count_different += 1

                # check cuts and pseudo costs
                self.check_pseudo_costs(bb)
                self.check_gmics(bb)
        print(dif)
        print(f"count_different: {count_different}")
        self.assertFalse(count_different, 'Check the above runs. They should be same.')

    def check_disjunctive_cuts(self, bb):
        if bb.evaluated_nodes >= 2:
            # these are just rough values - if broken just check to make sure ok
            self.assertTrue(bb._kwargs['total_number_cglp_created'] /
                            bb.evaluated_nodes >= .1)
