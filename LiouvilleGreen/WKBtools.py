from math import fmod, floor, fabs

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
