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

from simple_mip_solver import DisjunctiveCutBoundNode, BranchAndBound
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

    def disjunctive_cut_test_models(self):
        ratio_run = .05
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
                if np.random.uniform() > ratio_run:
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
                cglp = CutGeneratingLP(cglp_bb, cglp_bb.root_node.idx)
                bb = BranchAndBound(model, self.Node, cglp=cglp, pseudo_costs={}, **kwargs)
                bb.solve()

                if not isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01):
                    print(f'different for {file}')
                    print(f'mine: {bb.objective_value}')
                    print(f'gurobi: {gu_mdl.objVal}')
                    dif[i, j] = {'mine': bb.objective_value, 'gurobi': gu_mdl.objVal}
                    count_different += 1
                # self.assertTrue(isclose(bb.objective_value, gu_mdl.objVal, abs_tol=.01),
                #                 f'different for {file}')
        self.assertTrue(count_different / (num_kwargs * num_fldrs) < .02 * ratio_run,
                        'try to get less than 2% failure or less')
        print()
