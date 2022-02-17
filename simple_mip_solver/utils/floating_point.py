from math import floor, ceil
from typing import Tuple, Union

variable_epsilon = 1e-4
coefficient_epsilon = 1e-8
fraction_epsilon = 1e-12
min_constraint_depth = 1e-4
constraint_pad = 1e-8


def get_fraction(x: float, max_term: int = 1000, estimate: str = None) -> Tuple[int, int]:
    """Get the nearest fraction to the input floating point number. Implementation of
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
    # todo: scale before passing in?
    # todo: make sure we get a number close to the input when checking result - else toss the result or try not estimating
    # rest of this is just echoing above
    # todo: make sure we only have a certain number of sig figs coming in?
    # todo: so that .250000000001 can be overestimated to 1/4 (try not estimating and then checking that you're 1e-15 close)
    # todo: check if we're really close to a whole number - return rounded solution if so
    # todo: should be unnecessary if we cut off at many sig figs, but what about .33333333 or .99999999
    assert isinstance(x, int) or isinstance(x, float), 'x should be an int or float'
    assert isinstance(max_term, int) and max_term > 0, 'max_term should be a positive integer'
    if estimate is not None:
        assert estimate in ['over', 'under'], "estimate should be 'over' or 'under' when provided"

    # Just round if we're large
    if x > max_term:
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

