from math import fmod, fabs
from typing import List, Sequence, Tuple

from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.range_reduce_mod_2pi import simple_mod_2pi
from config.defaults import DEFAULT_ABS_TOLERANCE


# similar to LiouvilleGreen.range_reduce_mod_2pi.simple_mod_2pi, but with a specific
# convention for the (mod 2pi) component: it should always be negative.
#
# This is the (cycle count, remainder) representation the WKB producers store: the unreduced
# phase is formed once and reduced once, here, per sample (README §2 (e) of
# prompts/GkTk-remedial; the module docstring of LiouvilleGreen.range_reduce_mod_2pi).
def WKB_mod_2pi(theta: float):
    theta_mod_2pi = fmod(theta, TWO_PI)

    # The cycle count must be derived from the *remainder*, not from a second, independently
    # rounded division. fmod is exact, so |theta| - |theta_mod_2pi| is mathematically n*TWO_PI
    # for the integer n = floor(|theta| / TWO_PI) we want; whereas fabs(theta) / TWO_PI is a
    # correctly-rounded double division, and when the exact quotient sits within half an ulp
    # *below* an integer that division rounds up across it and floor() returns n+1. The pair
    # then reconstructs theta - 2*pi rather than theta
    # ([13-wkb-mod-2pi-cycle-count-inconsistent]; measured 1 of 77,975 production Gk samples at
    # k = 3e8/Mpc on LambdaCDM, docs/gktk-remedial-verification.md §3.7).
    #
    # Error budget for the expression below, with n ~ |theta| / (2 pi):
    #   * the subtraction rounds by at most ulp(|theta|)/2 <= |theta| * 2^-53, i.e. n * 2^-53
    #     cycles once divided by TWO_PI;
    #   * the division itself rounds by at most n * 2^-53.
    # So the argument of round() is within n * 2^-52 of the exact integer, and round-to-nearest
    # recovers n exactly while n * 2^-52 < 1/2, i.e. while n < 2^51 -- that is
    # |theta| < 2*pi*2^51 ~ 1.4e16, where ulp(theta) has itself grown to ~2 rad and the phase
    # has no fractional information left. Every |theta| this code sees is <= ~5e12 (n ~ 8e11),
    # where the bound is 1.8e-4 cycles.
    #
    # NOTE replacing floor() by round() on the OLD quotient does not fix this: it merely moves
    # the failure from quotients just below an integer to quotients just above one.
    theta_div_2pi = int(round((fabs(theta) - fabs(theta_mod_2pi)) / TWO_PI))

    if theta < 0.0:
        theta_div_2pi = -theta_div_2pi

    # our convention is that theta mod 2pi is taken to be negative
    if theta_mod_2pi > 0:
        theta_div_2pi = theta_div_2pi + 1
        theta_mod_2pi = theta_mod_2pi - TWO_PI

    return theta_div_2pi, theta_mod_2pi


# Decompose the product big_number*small_number into a cycle count plus a remainder, using the
# same phase convention as WKB_mod_2pi above. Also allows an offset mod_2pi_init.
#
# NOT USED BY PRODUCTION. Its only production consumer was stage 2 of the two-stage phase ODE
# (theta = theta_i + omega_i (1+u) Q), which prompts/GkTk-remedial prompt 06 removed from
# Quadrature/integrators/WKB_phase_function.py (review §3: the Q variable protected the wrong
# quantity and the stored phase was wrong by cycles). It is retained, unchanged, for the
# reproduction script docs/spec-code-audit/scripts/GK_05_phase_reassembly.py (README §7 D7).
#
# NOTE this used to call range_reduce_mod_2pi(), a prime-factorisation scheme that avoided forming
# the rounded product big_number*small_number. That scheme was measured to be no more accurate
# than the plain reduction used here -- worse in most cases -- and ~21x slower, so it has been
# removed; see the docstring of LiouvilleGreen.range_reduce_mod_2pi for the measurements.
#
# If you find yourself wanting the phase itself, do not reconstruct it as div*TWO_PI + mod and
# reduce again: pass the unreduced value to libm, which reduces more accurately than we can. See
# the module docstring of LiouvilleGreen.range_reduce_mod_2pi.
def WKB_product_mod_2pi(big_number: float, small_number: float, mod_2pi_init: float):
    # theta_div_2pi and theta_mod_2pi have the same sign as the product big_number*small_number
    theta_div_2pi, theta_mod_2pi = simple_mod_2pi(big_number * small_number)

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


def apply_phase_offset(
    div_2pi_sample: Sequence[int], mod_2pi_sample: Sequence[float], delta: float
) -> Tuple[List[int], List[float]]:
    """
    Add ``delta`` to every sample's phase, keeping the ``(div 2pi, mod 2pi)`` representation:
    ``wrap_theta(mod + delta)`` per sample, with the cycle shift it returns added to *that
    sample's* ``div``. Nothing is rebased across samples, so for every sample

        new_div * 2pi + new_mod == div * 2pi + mod + delta

    up to the rounding of the sum, and two objects that share a physical phase store the same
    cycle count (``prompts/GkTk-remedial/README.md`` §2 (e); ``RECONCILIATION.md`` §2 item 4).

    This is what ``GkWKBIntegration.store()`` uses to attach the initial-data offset
    ``deltaTheta`` to the phase returned by ``WKB_phase_function``. It replaces
    ``shift_theta_sample`` below, whose subtraction of the first sample's shift from every
    sample's ``div`` was constant within an object but differed between objects, which is what
    manufactured the +-1-cycle offsets between neighbouring objects that ``GkSource``'s
    rectifier then had to repair (review §8.1, §8.3).

    :param div_2pi_sample: the cycle counts, one per sample
    :param mod_2pi_sample: the remainders, one per sample, each in ``(-2pi, 0]``
    :param delta: the offset to add, in radians
    :return: ``(new_div_2pi_sample, new_mod_2pi_sample)`` as lists
    """
    delta = float(delta)
    new_div: List[int] = []
    new_mod: List[float] = []
    for div, mod in zip(div_2pi_sample, mod_2pi_sample):
        shift, wrapped = wrap_theta(float(mod) + delta)
        new_div.append(int(div) + shift)
        new_mod.append(wrapped)
    return new_div, new_mod


def shift_theta_sample(div_2pi_sample, mod_2pi_sample, shift):
    # NOT USED BY THE GREEN'S-FUNCTION PRODUCTION PATH. GkWKBIntegration.store() switched to
    # apply_phase_offset() above in prompts/GkTk-remedial prompt 06, because the rebase below
    # (subtracting the first sample's shift from every sample) differs between objects and is the
    # source of the +-1-cycle offsets between neighbouring GkWKBIntegration objects measured in
    # review §8.3. TkWKBIntegration.store() still calls it until prompt 07 makes the same switch;
    # after that it is retained only for the reproduction scripts
    # docs/gk-wkb-review-fable-2026-09-09/t6_sweep.py and
    # docs/spec-code-audit/scripts/GK_05_phase_reassembly.py (README §7 D7). Do not reintroduce
    # it into a producer.

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
