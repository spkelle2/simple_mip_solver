from coinor.cuppy.milpInstance import MILPInstance
from cylp.cy.CyClpSimplex import CyClpSimplex, CyLPArray
import inspect
from math import isclose
import numpy as np
from numpy.testing import assert_allclose
import os
import unittest
from unittest.mock import patch

from simple_mip_solver import BranchAndBound, BaseNode
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP
from test_simple_mip_solver.example_models import small_branch, square, generate_random_variety


class TestCutGeneratingLP(unittest.TestCase):

    def test_init_fails_asserts(self):
        bb = BranchAndBound(small_branch, BaseNode, gomory_cuts=False)
        bb.solve()
        self.assertRaisesRegex(AssertionError, 'bb must be a BranchAndBound',
                               CutGeneratingLP, bb=5, root_id=0)
        self.assertRaisesRegex(AssertionError, 'root node of the disjunction must be present',
                               CutGeneratingLP, bb=bb, root_id=100)

    def test_init(self):
        bb = BranchAndBound(small_branch, BaseNode, gomory_cuts=False)
        bb.solve()
        with patch('simple_mip_solver.utils.cut_generating_lp.CutGeneratingLP._create_cglp') as cp:
            cglp = CutGeneratingLP(bb, bb.root_node.idx)
            self.assertTrue(cglp.bb is bb)
            self.assertTrue(cglp.root_id == 0)
            self.assertTrue(cp.called)

    def test_create_cglp_fails_asserts(self):
        bb = BranchAndBound(small_branch, gomory_cuts=False)
        bb.solve()
        terminal_nodes = bb.tree.get_leaves(0)
        disjunctive_nodes = [n for n in terminal_nodes if n.lp_feasible is not False]
        n = disjunctive_nodes[0]
        n.lp.addVariable('d', 3)
        with patch('simple_mip_solver.utils.cut_generating_lp.CutGeneratingLP._create_cglp') as cp:
            cglp = CutGeneratingLP(bb, bb.root_node.idx)
        self.assertRaisesRegex(AssertionError, 'Each disjunctive term should have the same variables',
                               cglp._create_cglp)

        bb = BranchAndBound(small_branch, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        A = np.append(-small_branch.A.copy(), np.matrix([[-1, -1, -1]]), axis=0)
        A_prime = np.append(-small_branch.A.copy(), np.matrix([[-1], [-1]]), axis=1)
        b = CyLPArray(np.append(-small_branch.b.copy(), np.array([-3])))
        self.assertRaisesRegex(AssertionError, "A and b must both", cglp._create_cglp, A=A)
        self.assertRaisesRegex(AssertionError, "A and b must both", cglp._create_cglp, b=b)
        self.assertRaisesRegex(AssertionError, 'A must be a numpy', cglp._create_cglp,
                               A=np.array(A), b=b)
        self.assertRaisesRegex(AssertionError, 'A must have same number of columns',
                               cglp._create_cglp, A=A_prime, b=b)
        self.assertRaisesRegex(AssertionError, 'b must be a CyLPArray',
                               cglp._create_cglp, A=A, b=np.append(-small_branch.b.copy(), np.array([-3])))
        self.assertRaisesRegex(AssertionError, 'A must have the same number of rows',
                               cglp._create_cglp, A=A, b=CyLPArray(-small_branch.b.copy()))
        self.assertRaisesRegex(AssertionError, 'var_lb must be a CyLPArray',
                               cglp._create_cglp, var_lb=[1, 0, 0])
        self.assertRaisesRegex(AssertionError, 'Must have same number of lower bounds as variables',
                               cglp._create_cglp, var_lb=CyLPArray([1, 0]))
        self.assertRaisesRegex(AssertionError, 'var_ub must be a CyLPArray',
                               cglp._create_cglp, var_ub=[1, 0, 0])
        self.assertRaisesRegex(AssertionError, 'Must have same number of upper bounds as variables',
                               cglp._create_cglp, var_ub=CyLPArray([1, 0]))

    def test_create_cglp_standard(self):
        inf = small_branch.lp.getCoinInfinity()
        bb = BranchAndBound(small_branch, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        lp = cglp._create_cglp()
        terminal_nodes = bb.tree.get_leaves(bb.root_node.idx)
        dn = {n.idx: n for n in terminal_nodes if n.lp_feasible is not False}

        pi = lp.getVarByName('pi')
        pi0 = lp.getVarByName('pi0')
        u_5 = lp.getVarByName('u_5')
        w_5 = lp.getVarByName('w_5')
        v_5 = lp.getVarByName('v_5')
        u_11 = lp.getVarByName('u_11')
        w_11 = lp.getVarByName('w_11')
        v_11 = lp.getVarByName('v_11')

        # check variables are what we expect
        self.assertTrue(len(lp.variables) == 8)
        for v in [pi, pi0, u_5, w_5, v_5, u_11, w_11, v_11]:
            if v.name in ['pi', 'pi0']:
                assert_allclose(v.lower/-inf, 1)
            else:
                self.assertTrue((v.lower == 0).all())
            assert_allclose(v.upper / inf, 1)

        # check objective is what we expect
        obj = np.concatenate((bb.root_node.solution, [-1], np.zeros(16)), axis=None)
        self.assertTrue((obj == lp.objective).all())

        # check each constraint is what we expect (in simple case)
        self.assertTrue(len(lp.constraints) == 5)

        self.assertTrue(len(lp.constraints[0].varCoefs) == 4)
        self.assertTrue((lp.constraints[0].varCoefs[pi] == -np.eye(3)).all())
        self.assertTrue((lp.constraints[0].varCoefs[u_5] == dn[5].lp.coefMatrix.T).toarray().all())
        self.assertTrue((lp.constraints[0].varCoefs[w_5] == np.eye(3)).all())
        self.assertTrue((lp.constraints[0].varCoefs[v_5] == -np.eye(3)).all())

        self.assertTrue(len(lp.constraints[1].varCoefs) == 4)
        self.assertTrue((lp.constraints[1].varCoefs[pi0] == -1).toarray().all())
        self.assertTrue((lp.constraints[1].varCoefs[u_5] == np.array([-1.5, -1.25])).all())
        self.assertTrue((lp.constraints[1].varCoefs[w_5] == np.zeros(3)).all())
        self.assertTrue((lp.constraints[1].varCoefs[v_5] == np.array([0, -1, -1])).all())

        self.assertTrue(len(lp.constraints[2].varCoefs) == 4)
        self.assertTrue((lp.constraints[2].varCoefs[pi] == -np.eye(3)).all())
        self.assertTrue((lp.constraints[2].varCoefs[u_11] == dn[11].lp.coefMatrix.T).toarray().all())
        self.assertTrue((lp.constraints[2].varCoefs[w_11] == np.eye(3)).all())
        self.assertTrue((lp.constraints[2].varCoefs[v_11] == -np.eye(3)).all())

        self.assertTrue(len(lp.constraints[3].varCoefs) == 4)
        self.assertTrue((lp.constraints[3].varCoefs[pi0] == -1).toarray().all())
        self.assertTrue((lp.constraints[3].varCoefs[u_11] == np.array([-1.5, -1.25])).all())
        self.assertTrue((lp.constraints[3].varCoefs[w_11] == np.array([1, 0, 0])).all())
        self.assertTrue((lp.constraints[3].varCoefs[v_11] == np.array([-1, -1, 0])).all())

        self.assertTrue(len(lp.constraints[4].varCoefs) == 6)
        self.assertTrue((lp.constraints[4].varCoefs[u_5] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[u_11] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[w_5] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[w_11] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[v_5] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[v_11] == 1).toarray().all())

    def test_create_cglp_infinite_bounds(self):
        # check that infinite var bounds are 0
        inf = square.lp.getCoinInfinity()
        bb = BranchAndBound(square, node_limit=1, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        lp = cglp._create_cglp()
        self.infinite = cglp
        terminal_nodes = bb.tree.get_leaves(bb.root_node.idx)
        dn = {n.idx: n for n in terminal_nodes if n.lp_feasible is not False}

        # need to set inf to 1e308
        pi = lp.getVarByName('pi')
        pi0 = lp.getVarByName('pi0')
        u_1 = lp.getVarByName('u_1')
        w_1 = lp.getVarByName('w_1')
        v_1 = lp.getVarByName('v_1')
        u_2 = lp.getVarByName('u_2')
        w_2 = lp.getVarByName('w_2')
        v_2 = lp.getVarByName('v_2')

        self.assertTrue(len(lp.variables) == 8)
        for v in [pi, pi0, u_1, w_1, v_1, u_2, w_2, v_2]:
            if v.name in ['pi', 'pi0']:
                assert_allclose(v.lower/-inf, 1)
            else:
                self.assertTrue((v.lower == 0).all())
            if v.name == 'v_1':
                assert_allclose(v.upper, np.array([inf, 0]))
            elif v.name == 'v_2':
                self.assertTrue((v.upper == np.zeros(2)).all())
            else:
                assert_allclose(v.upper / inf, 1)

        # check objective is what we expect
        obj = np.concatenate((bb.root_node.solution, [-1], np.zeros(12)), axis=None)
        self.assertTrue((obj == lp.objective).all())

        # check constraints are what we expect
        # terminal node with non-overlapping bounds on variables should be removed
        self.assertTrue(len(lp.constraints) == 5)

        self.assertTrue(len(lp.constraints[0].varCoefs) == 4)
        self.assertTrue((lp.constraints[0].varCoefs[pi] == -np.eye(2)).all())
        self.assertTrue((lp.constraints[0].varCoefs[u_1] == dn[1].lp.coefMatrix.T).toarray().all())
        self.assertTrue((lp.constraints[0].varCoefs[w_1] == np.eye(2)).all())
        self.assertTrue((lp.constraints[0].varCoefs[v_1] == -np.eye(2)).all())

        self.assertTrue(len(lp.constraints[1].varCoefs) == 4)
        self.assertTrue((lp.constraints[1].varCoefs[pi0] == -1).toarray().all())
        self.assertTrue((lp.constraints[1].varCoefs[u_1] == np.array([-1.5, -1.5])).all())
        self.assertTrue((lp.constraints[1].varCoefs[w_1] == np.zeros(2)).all())
        self.assertTrue((lp.constraints[1].varCoefs[v_1] == np.array([-1, 0])).all())

        self.assertTrue(len(lp.constraints[2].varCoefs) == 4)
        self.assertTrue((lp.constraints[2].varCoefs[pi] == -np.eye(2)).all())
        self.assertTrue((lp.constraints[2].varCoefs[u_2] == dn[2].lp.coefMatrix.T).toarray().all())
        self.assertTrue((lp.constraints[2].varCoefs[w_2] == np.eye(2)).all())
        self.assertTrue((lp.constraints[2].varCoefs[v_2] == -np.eye(2)).all())

        self.assertTrue(len(lp.constraints[3].varCoefs) == 4)
        self.assertTrue((lp.constraints[3].varCoefs[pi0] == -1).toarray().all())
        self.assertTrue((lp.constraints[3].varCoefs[u_2] == np.array([-1.5, -1.5])).all())
        self.assertTrue((lp.constraints[3].varCoefs[w_2] == np.array([2, 0])).all())
        self.assertTrue((lp.constraints[3].varCoefs[v_2] == np.zeros(2)).all())

        self.assertTrue(len(lp.constraints[4].varCoefs) == 6)
        self.assertTrue((lp.constraints[4].varCoefs[u_1] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[w_1] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[v_1] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[u_2] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[w_2] == 1).toarray().all())
        self.assertTrue((lp.constraints[4].varCoefs[v_2] == 1).toarray().all())

    def test_create_cglp_new_coef_matrix_and_var_bounds(self):
        # check we get correct constraints when updating them (new A)
        # check bounds coefficients update as expected for new lb/ub (new bounds)
        # ensure infeasible subproblems removed for updated bound coefs
        inf = small_branch.lp.getCoinInfinity()
        bb = BranchAndBound(small_branch, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        A = np.append(-small_branch.A.copy(), np.matrix([[-1, -1, -1]]), axis=0)
        b = CyLPArray(np.append(-small_branch.b.copy(), np.array([-3])))
        lb = CyLPArray([1, 0, 0])
        ub = CyLPArray([1, 1, 0])
        lp = cglp._create_cglp(A=A, b=b, var_lb=lb, var_ub=ub)
        self.new = CutGeneratingLP(bb, bb.root_node.idx, A=A, b=b, var_lb=lb, var_ub=ub)

        pi = lp.getVarByName('pi')
        pi0 = lp.getVarByName('pi0')
        u_11 = lp.getVarByName('u_11')
        w_11 = lp.getVarByName('w_11')
        v_11 = lp.getVarByName('v_11')

        # check variables are what we expect
        self.assertTrue(len(lp.variables) == 5)
        for v in [pi, pi0, u_11, w_11, v_11]:
            if v.name in ['pi', 'pi0']:
                assert_allclose(v.lower/-inf, 1)
            else:
                self.assertTrue((v.lower == 0).all())
            assert_allclose(v.upper/inf, 1)

        # check objective is what we expect
        obj = np.concatenate((bb.root_node.solution, [-1], np.zeros(9)), axis=None)
        self.assertTrue((obj == lp.objective).all())

        # check each constraint is what we expect (in simple case)
        self.assertTrue(len(lp.constraints) == 3)

        self.assertTrue(len(lp.constraints[0].varCoefs) == 4)
        self.assertTrue((lp.constraints[0].varCoefs[pi] == -np.eye(3)).all())
        self.assertTrue((lp.constraints[0].varCoefs[u_11] == A.T).all())
        self.assertTrue((lp.constraints[0].varCoefs[w_11] == np.eye(3)).all())
        self.assertTrue((lp.constraints[0].varCoefs[v_11] == -np.eye(3)).all())

        self.assertTrue(len(lp.constraints[1].varCoefs) == 4)
        self.assertTrue(all(lp.constraints[1].varCoefs[pi0] == -1))
        self.assertTrue((lp.constraints[1].varCoefs[u_11] == np.array([-1.5, -1.25, -3])).all())  # b
        self.assertTrue((lp.constraints[1].varCoefs[w_11] == np.array([1, 0, 0])).all())
        self.assertTrue((lp.constraints[1].varCoefs[v_11] == np.array([-1, -1, 0])).all())

        self.assertTrue(len(lp.constraints[2].varCoefs) == 3)
        self.assertTrue((lp.constraints[2].varCoefs[u_11].toarray() == 1).all())
        self.assertTrue((lp.constraints[2].varCoefs[w_11].toarray() == 1).all())
        self.assertTrue((lp.constraints[2].varCoefs[v_11].toarray() == 1).all())

    def test_solve_fails_asserts(self):
        bb = BranchAndBound(square, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)

        self.assertRaisesRegex(AssertionError, 'x_star must be a CyLPArray',
                               cglp.solve, x_star=[1.5, 2])
        self.assertRaisesRegex(AssertionError, 'x_star must have the same number of variables',
                               cglp.solve, x_star=CyLPArray([1.5, 2, 5]))

        pi, pi0 = cglp.solve()
        basis = cglp.lp.getBasisStatus()
        basis1 = (np.append(basis[0], np.array([1])), basis[1])
        self.assertRaisesRegex(AssertionError, 'first starting_basis element',
                               cglp.solve, starting_basis=basis1)
        basis2 = (basis[0], np.append(basis[1], np.array([1])))
        self.assertRaisesRegex(AssertionError, 'second starting_basis element',
                               cglp.solve, starting_basis=basis2)

    def test_solve(self):
        bb = BranchAndBound(square, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        pi, pi0 = cglp.solve()

        # check cut is what we expect, i.e. x1 <= 1 or x2 <= 1
        self.assertTrue(isinstance(pi, CyLPArray))
        if abs(pi[1]) > abs(pi[0]):
            assert_allclose(pi / pi0, np.array([0, 1]), atol=.01)
        else:
            assert_allclose(pi / pi0, np.array([1, 0]), atol=.01)
        self.assertTrue((pi - .01 < 0).all())
        self.assertTrue(pi0 - .01 < 0)

        # try another problem
        bb = BranchAndBound(small_branch, node_limit=10, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        pi, pi0 = cglp.solve()

        # check cut is what we expect, i.e. x3 <= 1
        self.assertTrue(isinstance(pi, CyLPArray))
        assert_allclose(pi / pi0, np.array([0, 0, 1]), atol=.01)
        self.assertTrue((pi - .01 < 0).all())
        self.assertTrue(pi0 - .01 < 0)

    def test_solve_doesnt_separate(self):
        bb = BranchAndBound(square, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        pi, pi0 = cglp.solve(x_star=CyLPArray([.5, .5]))
        # should still return something - let cut generation algorithm decide what to keeo
        self.assertTrue(pi is not None)
        self.assertTrue(pi0 is not None)

    def test_solve_different_x_star(self):
        bb = BranchAndBound(square, node_limit=1, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        # should shift to cutting off point from above
        pi, pi0 = cglp.solve(x_star=CyLPArray([1.5, 2]))
        self.assertTrue(isclose(pi0, -.75, abs_tol=.01))
        assert_allclose(pi, np.array([0, -.5]), atol=.01)
        print()

    def test_solve_starting_basis(self):
        bb = BranchAndBound(small_branch, node_limit=10, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        pi, pi0 = cglp.solve()
        basis = cglp.lp.getBasisStatus()

        bb = BranchAndBound(small_branch, node_limit=10, gomory_cuts=False)
        bb.solve()
        cglp = CutGeneratingLP(bb, bb.root_node.idx)
        pi, pi0 = cglp.solve(starting_basis=basis)
        self.assertTrue(cglp.lp.iteration == 0)

    def test_solve_many_times(self):
        fldr = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(generate_random_variety))),
            'example_models'
        )
        for i, file in enumerate(os.listdir(fldr)):
            if i >= 10:
                break
            print(f'running test {i + 1}')
            pth = os.path.join(fldr, file)
            model = MILPInstance(file_name=pth)
            bb = BranchAndBound(model, gomory_cuts=False)
            bb.solve()
            cglp = CutGeneratingLP(bb=bb, root_id=bb.root_node.idx)
            pi, pi0 = cglp.solve()

            # ensure we cut off the root solution
            self.assertTrue(sum(pi * bb.root_node.solution) <= pi0)

            # ensure we don't cut off disjunctive mins
            for n in bb.tree.get_leaves(0):
                if n.lp_feasible:
                    self.assertTrue(sum(pi * n.solution) >= pi0 - .01)


if __name__ == '__main__':
    unittest.main()
