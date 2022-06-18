from cylp.cy.CyClpSimplex import CyLPArray
from math import floor, ceil
import numpy as np
from typing import Tuple, Union
import warnings

from simple_mip_solver.utils.tolerance import good_coefficient_approximation_epsilon, \
    exact_coefficient_approximation_epsilon, max_term


def scale_cut(pi: np.ndarray, pi0: float, max_abs: float = 1, **kwargs) -> \
        Union[Tuple[np.ndarray, float], Tuple[None, None]]:
    """ Scale the input cut (pi, pi0) so that the largest absolute value of
    coefficients is max_abs

    :param pi: cut coefficients
    :param pi0: cut bound
    :param max_abs: the maximum absolute value cut coefficients should have after scaling
    :param kwargs: placeholder for extra arguments passed along to other subroutines
    :return: scaled cut
    """
    assert isinstance(pi, np.ndarray), 'pi is an nd.array'
    assert isinstance(pi0, float) or isinstance(pi0, int), 'pi0 is a number'
    assert (isinstance(max_abs, int) or isinstance(max_abs, float)) and max_abs > 0, \
        'max_abs should be positive'

    if min(pi) == max(pi) == 0:
        return None, None  # can't be scaled

    # scale the largest element to have max absolute value max_abs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        scale = min(abs(max_abs/pi))
    pi = pi * scale
    pi0 *= scale

    return pi, pi0


def numerically_safe_cut(pi: CyLPArray, pi0: float, estimate: str = 'over',
                         make_integer: bool = False, **kwargs) -> Tuple[CyLPArray, float]:
    """ Convert (pi, pi0) to a close, outer-approximation with integer coefficients.

    Note: For max_term used in get_fraction(), nonzero coefficients <
    (largest coefficient)/max_term will take on value equal to largest coefficient
    due to limitations in fractional approximation algorithm when over estimating.
    I consider this a design feature as this should simultaneously enforce the
    range of nonzero coefs from not getting too large. See
    numerically_safe_cut(pi=CyLPArray([1, 100, 10000]), pi0=100, estimate='over')
    for an example.

    Warning: For max_term used in get_fraction(), nonzero coefficients <
    largest coefficient but > largest coefficient - 1/max_term will take on value
    0 when under estimating due to limitations in fractional approximation algorithm.
    You'll still get a valid cut, but it may not be very tight. I don't consider
    this to be an issue as it will rarely happen and when it does we can just generate
    other cuts. See
    numerically_safe_cut(pi=CyLPArray([100, 9999, 10000]), pi0=100, estimate='under')
    for an example.

    :param pi: constraint coefficients to approximate
    :param pi0: constraint bound
    :param estimate: 'over' generates outer approximation for pi^T x >= pi0.
    'under' generates outer approximation for pi^T x <= pi0. Names based on whether
    cut coefficients are over or under approximated.
    :param make_integer: True multiplies all coefficients and the cut bound by the
    least common multiple of coefficient denominators so the coefficients are integer.
    False returns the coefficients and bound as fractions
    :param kwargs: placeholder for extra arguments passed along to subroutines
    :return: (safe_pi, safe_pi0), a close, outer-approximation of (pi, pi0) with
    integer coefficients
    """
    estimates = ['over', 'under']
    assert isinstance(pi, CyLPArray), 'pi is a CyLPArray'
    assert isinstance(pi0, float) or isinstance(pi0, int), 'pi0 is a number'
    assert estimate in estimates, 'estimate must be over or under to ensure safety'

    nums, dens = [], []

    pi_scaled, pi0_scaled = scale_cut(pi, pi0, **kwargs)
    if pi_scaled is None:
        return pi, pi0  # if pi == 0 then its already integer so just return

    for coef in pi_scaled:
        n, d = get_fraction(coef, estimate=estimate, **kwargs)
        # if the fraction is not a good approximation, see if the exact is really close
        if coef != 0 and abs(1 - ((n/d) / coef)) > good_coefficient_approximation_epsilon:
            n_prime, d_prime = get_fraction(coef, estimate=None, **kwargs)
            # I think this is small enough to avoid floating point error but I could be wrong...
            if abs(n_prime / d_prime - coef) < exact_coefficient_approximation_epsilon:
                n, d = n_prime, d_prime
        nums.append(n)
        dens.append(d)
    lcm = np.lcm.reduce(dens)
    # floating point error from doing this can make a tight approximation invalid
    # but I'm betting the floating point error is small enough CLP will tolerate it
    safe_pi = CyLPArray((lcm if make_integer else 1)*np.array(nums)/np.array(dens))
    # opposite estimate for other side of constraint
    n, d = get_fraction(x=pi0_scaled*lcm if make_integer else pi0_scaled,
                        estimate='under' if estimate == 'over' else 'over')
    safe_pi0 = n/d

    return safe_pi, safe_pi0


def get_fraction(x: float, max_term: int = max_term, estimate: str = None, **kwargs) -> Tuple[int, int]:
    """Get the nearest fraction to the input floating point number. The denominator
    will always come back positive. Implementation of
    https://en.wikipedia.org/wiki/Continued_fraction#Infinite_continued_fractions_and_convergents

    Warning: When using under/over estimate, you can get results far from expected
    when an early numerator or denominator surpasses the max term. Check
    get_fraction(.9991, max_term=1000, estimate='under') and
    get_fraction(.00000001, max_term=1000, estimate='over') for examples.

    :param x: the number to estimate as a fraction
    :param max_term: the largest numerator or denominator allowed
    :param estimate: 'over' ensures the returned fraction >= x, 'under' ensures
    the returned fraction <= x 
    :return: a tuple with the first element as the numerator and the second element
    as the denominator
    """
    assert isinstance(x, int) or isinstance(x, float), 'x should be an int or float'
    assert isinstance(max_term, (int, float)) and max_term > 0, 'max_term should be positive'
    if estimate is not None:
        assert estimate in ['over', 'under'], "estimate should be 'over' or 'under' when provided"

    # Just round if we're large
    if abs(x) > max_term:
        return ceil(x) if estimate == 'over' else floor(x) if estimate == 'under' else round(x), 1

    found_exact_match = False
    n, d = {}, {}

    # initialization conditions - n numerator, d denominator
    n[-2], d[-2] = 0, 1
    n[-1], d[-1] = 1, 0
    real_number = x
    i = -1

    while True:
        i += 1
        integer_part = floor(real_number)  # a_i in the wiki article
        n[i] = integer_part * n[i-1] + n[i-2]
        d[i] = integer_part * d[i-1] + d[i-2]
        # break if we exceed bound - even if this would be an exact solution
        if n[i] > max_term or d[i] > max_term:
            break
        fractional_part = real_number - integer_part
        if not fractional_part:
            found_exact_match = True
            break
        # set up for next iteration
        real_number = 1 / fractional_part

    # if we found an exact match, return the latest iterate
    if found_exact_match:
        return n[i], d[i]
    # else, the last iterate exceeded max_term, so take a previous one if available
    # odd iterations are overestimates and even number iterations are underestimates
    elif estimate == 'over':
        # return ceil(x) when x is juuussttt over a whole number - not close, but warned above
        return (n[i-1], d[i-1]) if (i-1) % 2 else (n[i-2], d[i-2]) if i-2 >= 0 else (ceil(x), 1)
    elif estimate == 'under':
        return (n[i-1], d[i-1]) if not (i-1) % 2 else (n[i-2], d[i-2])
    else:
        return n[i-1], d[i-1]

