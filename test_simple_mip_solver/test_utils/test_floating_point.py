from cylp.cy.CyClpSimplex import CyLPArray
from math import isclose
import numpy as np
import unittest
from unittest.mock import patch

from simple_mip_solver.utils.floating_point import scale_cut, numerically_safe_cut, \
    get_fraction, exact_coefficient_approximation_epsilon as eps


class TestFloatingPoint(unittest.TestCase):

    def test_scale_cut_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'pi is an nd.array', scale_cut,
                               pi=(1, 2, 3), pi0=4)
        self.assertRaisesRegex(AssertionError, 'pi0 is a number', scale_cut,
                               pi=np.array([1, 2, 3]), pi0='4')
        self.assertRaisesRegex(AssertionError, 'max_abs should be positive',
                               scale_cut, pi=np.array([1, 2, 3]), pi0=4, max_abs=-1)

    def test_scale_cut(self):
        pi = np.array([1, 0, 2, -4])
        pi0 = 3
        scaled_pi, scaled_pi0 = scale_cut(pi, pi0)
        self.assertTrue(all(scaled_pi == [.25, 0, .5, -1]))
        self.assertTrue(scaled_pi0 == .75)

    def test_numerically_safe_cut_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'pi is a CyLPArray', numerically_safe_cut,
                               pi=np.array([1, 2, 3]), pi0=4)
        self.assertRaisesRegex(AssertionError, 'pi0 is a number', numerically_safe_cut,
                               pi=CyLPArray([1, 2, 3]), pi0='4')
        self.assertRaisesRegex(AssertionError, 'estimate must be over or under',
                               numerically_safe_cut, pi=CyLPArray([1, 2, 3]),
                               pi0=4, estimate=None)

    def test_numerically_safe_cut(self):
        pi = CyLPArray([1, 2, 4])
        pi0 = 4

        # check function calls
        with patch('simple_mip_solver.utils.floating_point.scale_cut') as sc, \
                patch('simple_mip_solver.utils.floating_point.get_fraction') as gf:
            sc.return_value = (CyLPArray([.25, .5, 1]), 1)
            gf.side_effect = [(1, 4), (1, 2), (1, 1)]
            safe_pi, safe_pi0 = numerically_safe_cut(pi, pi0)

            self.assertTrue(sc.called)
            self.assertTrue(gf.call_count == 3)
            self.assertTrue(all(safe_pi == [1, 2, 4]))
            self.assertTrue(safe_pi0 == 4)

        # check returns
        runs = 100
        rand_pis = [CyLPArray(np.random.uniform(-10, 10, 4)) for i in range(runs)]
        rand_pi0s = [0] + [np.random.uniform(-10, 10) for i in range(runs - 1)]
        for pi, pi0 in zip(rand_pis, rand_pi0s):
            if abs(pi0) < .1:
                continue
            # check overs
            safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='over')
            self.assertTrue(all(np.equal(np.mod(safe_pi, 1), 0)), 'safe_pi must be integer')
            if pi0 > 0:
                # pi^T x >= pi0 <==> (pi/pi0)^T x >= 1
                # pi_safe^T x >= pi0_safe <==> (pi_safe/pi0_safe)^T x >= 1
                # safe is valid if pi_safe/pi0_safe >= pi/pi0 for pi0_safe, pi > 0, x >= 0
                self.assertTrue(all(safe_pi/safe_pi0 + eps >= pi/pi0))
            elif pi0 < 0:
                # swap sign when dividing by negative, i.e. pi0_safe, pi < 0, x >= 0
                self.assertTrue(all(safe_pi/safe_pi0 - eps <= pi/pi0))
            else:
                # since rhs is 0, back out the unscaled values
                self.assertTrue(all(safe_pi/max(abs(safe_pi)) >= pi*min(abs(1/pi))))
                self.assertTrue(safe_pi0 == pi0)

            # check unders - do the opposite of above
            safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='under')
            self.assertTrue(all(np.equal(np.mod(safe_pi, 1), 0)), 'safe_pi must be integer')
            if pi0 > 0:
                self.assertTrue(all(safe_pi / safe_pi0 - eps <= pi / pi0))
            elif pi0 < 0:
                self.assertTrue(all(safe_pi / safe_pi0 + eps >= pi / pi0))
            else:
                self.assertTrue(all(safe_pi/max(abs(safe_pi)) <= pi*min(abs(1/pi))))
                self.assertTrue(safe_pi0 == pi0)

    def test_numerically_safe_cut_really_close(self):
        # over-estimate on just slightly rounded over
        pi = CyLPArray([1.25, 2.375, 4]) + eps/10
        pi0 = 4
        safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='over')
        self.assertTrue(all(safe_pi == np.array([10, 19, 32])))
        self.assertTrue(isclose(safe_pi0, 32, abs_tol=eps))

        # under estimate on continued fractions
        # 16 digits will trick python into thinking equality but 15 won't
        pi = CyLPArray([1, .333333333333333, .666666666666667])
        pi0 = 1
        safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='under')
        self.assertTrue(all(safe_pi == np.array([3, 1, 2])))
        self.assertTrue(isclose(safe_pi0, 3, abs_tol=eps))

    # get 1's where we were close to 0 because too far apart
    def test_numerically_safe_cut_high_dynamism(self):
        pi = CyLPArray([1, 100, 10000])
        pi0 = 100
        safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='over')
        self.assertTrue(all(safe_pi == np.array([100, 1, 100])))
        self.assertTrue(isclose(safe_pi0, 1, abs_tol=eps))

        pi = CyLPArray([100, 9999, 10000])
        pi0 = 100
        safe_pi, safe_pi0 = numerically_safe_cut(pi=pi, pi0=pi0, estimate='under')
        self.assertTrue(all(safe_pi == np.array([1, 0, 100])))
        self.assertTrue(isclose(safe_pi0, 1, abs_tol=eps))

    def test_get_fraction_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'should be an int or float',
                               get_fraction, '5')
        self.assertRaisesRegex(AssertionError, 'should be a positive integer',
                               get_fraction, .3445345, max_term=-1)
        self.assertRaisesRegex(AssertionError, "should be 'over' or 'under'",
                               get_fraction, .3445345, estimate='over-estimate')

    def check_good_fraction(self, n, d):
        self.assertTrue(isinstance(n, int))
        self.assertTrue(isinstance(d, int))
        self.assertTrue(d > 0)

    def test_get_fraction_random_irrational(self):
        max_term = 1000
        rands = np.random.uniform(-1, 1, 1000)
        for x in rands:
            if x > 1/max_term:
                n, d = get_fraction(x, max_term=max_term)
                self.assertTrue(abs(x - n / d) <= .01, 'n/d should closely approximate x')
                self.check_good_fraction(n, d)
                n, d = get_fraction(x, max_term=max_term, estimate='under')
                self.assertTrue(n/d <= x, 'n/d should under approximate x')
                self.check_good_fraction(n, d)
                n, d = get_fraction(x, max_term=max_term, estimate='over')
                self.assertTrue(x <= n/d, 'n/d should over approximate x')
                self.check_good_fraction(n, d)

    def test_get_fraction_random_rational(self):
        max_term = 1000
        rands = np.random.randint(-256, 256, 1000)/256
        for x in rands:
            if x > 1/max_term:
                n, d = get_fraction(x, max_term=max_term)
                self.assertTrue(x == n / d, 'n/d should match x when easy rational')
                self.check_good_fraction(n, d)
                n, d = get_fraction(x, max_term=max_term, estimate='under')
                self.assertTrue(n / d <= x, 'n/d should under approximate x')
                self.check_good_fraction(n, d)
                n, d = get_fraction(x, max_term=max_term, estimate='over')
                self.assertTrue(x <= n / d, 'n/d should over approximate x')
                self.check_good_fraction(n, d)

    def test_get_fraction_dead_zones(self):
        # positives
        n, d = get_fraction(0.00000001, max_term=1000, estimate=None)
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(0.00000001, max_term=1000, estimate='under')
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(0.00000001, max_term=1000, estimate='over')
        self.assertTrue(n == 1 and d == 1)

        n, d = get_fraction(0.99999999, max_term=1000, estimate=None)
        self.assertTrue(n == 1 and d == 1)

        n, d = get_fraction(0.99999999, max_term=1000, estimate='under')
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(0.99999999, max_term=1000, estimate='over')
        self.assertTrue(n == 1 and d == 1)

        n, d = get_fraction(3141.59, max_term=1000, estimate=None)
        self.assertTrue(n == 3142 and d == 1)

        n, d = get_fraction(3141.59, max_term=1000, estimate='under')
        self.assertTrue(n == 3141 and d == 1)

        n, d = get_fraction(3141.59, max_term=1000, estimate='over')
        self.assertTrue(n == 3142 and d == 1)

        # negatives
        n, d = get_fraction(-0.00000001, max_term=1000, estimate=None)
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(-0.00000001, max_term=1000, estimate='under')
        self.assertTrue(n == -1 and d == 1)

        n, d = get_fraction(-0.00000001, max_term=1000, estimate='over')
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(-0.99999999, max_term=1000, estimate=None)
        self.assertTrue(n == -1 and d == 1)

        n, d = get_fraction(-0.99999999, max_term=1000, estimate='under')
        self.assertTrue(n == -1 and d == 1)

        n, d = get_fraction(-0.99999999, max_term=1000, estimate='over')
        self.assertTrue(n == 0 and d == 1)

        n, d = get_fraction(-3141.59, max_term=1000, estimate=None)
        self.assertTrue(n == -3142 and d == 1)

        n, d = get_fraction(-3141.59, max_term=1000, estimate='under')
        self.assertTrue(n == -3142 and d == 1)

        n, d = get_fraction(-3141.59, max_term=1000, estimate='over')
        self.assertTrue(n == -3141 and d == 1)


if __name__ == '__main__':
    unittest.main()
