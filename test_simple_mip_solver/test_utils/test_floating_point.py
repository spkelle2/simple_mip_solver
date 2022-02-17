import numpy as np
import unittest
from unittest.mock import patch

from simple_mip_solver.utils.floating_point import get_fraction


class TestFloatingPoint(unittest.TestCase):

    def test_get_fraction_fails_asserts(self):
        self.assertRaisesRegex(AssertionError, 'should be an int or float',
                               get_fraction, '5')
        self.assertRaisesRegex(AssertionError, 'should be a positive integer',
                               get_fraction, .3445345, max_term=-1)
        self.assertRaisesRegex(AssertionError, "should be 'over' or 'under'",
                               get_fraction, .3445345, estimate='over-estimate')

    def test_get_fraction_random_irrational(self):
        max_term = 1000
        rands = np.random.uniform(0, 1, 1000)
        for x in rands:
            if x > 1/max_term:
                n, d = get_fraction(x, max_term=max_term)
                self.assertTrue(abs(x - n / d) <= .01, 'n/d should closely approximate x')
                n, d = get_fraction(x, max_term=max_term, estimate='under')
                self.assertTrue(n/d <= x, 'n/d should under approximate x')
                n, d = get_fraction(x, max_term=max_term, estimate='over')
                self.assertTrue(x <= n/d, 'n/d should over approximate x')

    def test_get_fraction_random_rational(self):
        max_term = 1000
        rands = np.random.randint(0, 256, 1000)/256
        for x in rands:
            if x > 1/max_term:
                n, d = get_fraction(x, max_term=max_term)
                self.assertTrue(x == n / d, 'n/d should match x when easy rational')
                n, d = get_fraction(x, max_term=max_term, estimate='under')
                self.assertTrue(n / d <= x, 'n/d should under approximate x')
                n, d = get_fraction(x, max_term=max_term, estimate='over')
                self.assertTrue(x <= n / d, 'n/d should over approximate x')

    def test_get_fraction_dead_zones(self):
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


if __name__ == '__main__':
    unittest.main()
