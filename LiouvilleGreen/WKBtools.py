from math import fmod, floor, fabs
from typing import Tuple

from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.range_reduce_mod_2pi import range_reduce_mod_2pi
from defaults import DEFAULT_ABS_TOLERANCE


# similar to LiouvilleGreen.range_reduce_mod_2pi.simple_mod_2pi, but with a specific
# convention for the (mod 2pi) component: it should always be negative
def WKB_mod_2pi(theta: float):
    theta_mod_2pi = fmod(theta, TWO_PI)
    theta_div_2pi = int(floor(fabs(theta) / TWO_PI))

    if theta < 0.0:
        theta_div_2pi = -theta_div_2pi

    # our convention is that theta mod 2pi is taken to be negative
    if theta_mod_2pi > 0:
        theta_div_2pi = theta_div_2pi + 1
        theta_mod_2pi = theta_mod_2pi - TWO_PI

    return theta_div_2pi, theta_mod_2pi


# similar to LiouvilleGreen.range_reduce_mod_2pi.range_reduce_mod_2pi, but with the
# same phase convention as WKB_mod_2pi above.
# Also allows an offset mod_2pi_init
def WKB_product_mod_2pi(big_number: float, small_number: float, mod_2pi_init: float):
    # use custom range reduction to (try to) preserve precision in the product big_number*small_number (mod 2pi)
    # theta_div_2pi and theta_mod_2pi should have the same sign as the product big_number*small_number
    theta_div_2pi, theta_mod_2pi = range_reduce_mod_2pi(big_number, small_number)

    # mod_2pi_init is an offset that should be added to mod_2pi. We then (possibly) have to range-reduce again.
    theta_mod_2pi = theta_mod_2pi + mod_2pi_init
    while theta_mod_2pi > 0.0:
        theta_mod_2pi = theta_mod_2pi - TWO_PI
        theta_div_2pi = theta_div_2pi + 1

    while theta_mod_2pi < -TWO_PI:
        theta_mod_2pi = theta_mod_2pi + TWO_PI
        theta_div_2pi = theta_div_2pi - 1

    if theta_mod_2pi > DEFAULT_ABS_TOLERANCE:
        raise RuntimeError(
            f"WKB_product_mod_2pi: big_number={big_number:.8g}, small_number={small_number:.8g}, product={big_number*small_number:.8g}, mod_2pi_init={mod_2pi_init:.8g}, theta_div_2pi={theta_div_2pi}, theta_mod_2pi={theta_mod_2pi:.8g}, theta={theta_div_2pi * TWO_PI + theta_mod_2pi:.8g}"
        )

    return theta_div_2pi, theta_mod_2pi


def wrap_theta(theta: float) -> Tuple[int, float]:
    # given a value of theta, range-reduce so that theta falls within (-TWO_PI, 0], and
    # work out what corresponding shift this produced in div 2pi

    # recall that by convention, all our mod 2pi values are negative

    # if theta is positive, reduce by 2pi until it is negative
    if theta > 0.0:
        shift = 0
        while theta > 0.0:
            shift = shift + 1
            theta = theta - TWO_PI

        return shift, theta

    # if theta < -2pi, increase by 2pi until theta >= -2pi
    if theta <= -TWO_PI:
        shift = 0
        while theta <= -TWO_PI:
            shift = shift - 1
            theta = theta + TWO_PI

        return shift, theta

    # otherwise nothing to do, theta is already in the required range, no shift required
    return 0, theta


def shift_theta_sample(div_2pi_sample, mod_2pi_sample, shift):
    # work out how the div_2pi, mod_2pi values should change when we add 'shift' to each
    # value in mod_2pi_sample

    # wrap_theta() returns a tuple: (div_2pi_shift, new_mod_2pi)
    #  - new_mod_2pi is the range-reduced value of mod 2pi and is guaranteed to fall in the range (-2pi, 0]
    #  - div_2pi_shift is the shift needed for mod 2pi to compensate
    theta_sample_shifts = [
        wrap_theta(theta_mod_2pi_sample + shift)
        for theta_mod_2pi_sample in mod_2pi_sample
    ]

    theta_div_2pi_shift, mod_2pi_sample = zip(*theta_sample_shifts)

    # because the phase is only defined mod 2pi, the absolute value of div 2pi doesn't matter, only its
    # relative value
    # To try to cut down unnecessary shifts, we rebase the div 2pi shift.
    theta_div_2pi_shift_base = theta_div_2pi_shift[0]
    div_2pi_sample = [
        d + shift - theta_div_2pi_shift_base
        for (d, shift) in zip(div_2pi_sample, theta_div_2pi_shift)
    ]

    return div_2pi_sample, mod_2pi_sample
