"""
Range reduction mod 2pi.

WHEN TO REDUCE, AND WHEN NOT TO
-------------------------------
There are two quite different reasons to want a phase mod 2pi, and they call for opposite
treatment. Confusing them costs accuracy.

(1) You want to evaluate sin(theta) or cos(theta). DO NOT REDUCE FIRST. Hand the argument to
    libm exactly as it stands, however large it is. A quality libm (glibc, Apple) performs its
    own argument reduction internally against a multi-hundred-bit representation of pi -- the
    Payne-Hanek algorithm -- and returns a correctly-rounded result for arguments right up to
    the top of the double range. Any reduction you perform yourself beforehand is done in
    double precision against a 53-bit value of 2*pi, and is therefore strictly worse.

    In particular, DO NOT use fmod(theta, TWO_PI) for this purpose. TWO_PI is 2*pi rounded to
    a double, so it differs from the true 2*pi by ~2.4e-16; fmod is exact for the arguments it
    is *given*, which means it faithfully computes the remainder with respect to the *wrong*
    modulus, and the error in the result grows in proportion to the number of cycles removed.
    At theta ~ 1e10 that is already several digits.

    Measured on (x*Q) mod 2pi with x from e^2 to e^16: passing the unreduced argument to
    libm sin() was 2-3x more accurate than any pre-reduction scheme tried, at every magnitude.

(2) You want a *representation* of the phase as an integer cycle count plus a bounded
    remainder -- for instance to spline the remainder as a smooth O(1) function, as
    bessel_phase does, where the raw phase reaches ~1e15 and could not be splined directly
    without losing every significant digit. Here a reduction is unavoidable, and
    simple_mod_2pi() below is the right tool: plain fmod plus a floor.

HISTORICAL NOTE
---------------
This module previously also provided range_reduce_mod_2pi(big, small), which computed
(big * small) mod 2pi via a prime-factorisation scheme designed to avoid ever forming the
rounded product big*small, on the theory that this would preserve precision that fmod of the
product must lose.

It has been removed, because it did not. Measured against the exact reduction of the exact
product of the given doubles (mpmath at 60 digits), for x = e^2 ... e^16 and O(1) multipliers,
range_reduce_mod_2pi was never more accurate than simple_mod_2pi(x*Q) and was worse in the
majority of cases, while costing ~21x more per call (it invoked sympy.factorint on every
evaluation). The reason the expected advantage does not materialise is that the error is
dominated by the sensitivity of the reduction to its *inputs*: x and Q are themselves doubles,
so the true product is uncertain at the eps*|x*Q| level regardless of how carefully the
multiplication is subsequently arranged, and the scheme's own multi-step arithmetic contributes
additional rounding on top of that.

Callers that need (big * small) mod 2pi should use simple_mod_2pi(big * small).
"""

from math import floor, fabs, fmod

from .constants import TWO_PI


def simple_mod_2pi(num):
    """
    Decompose num into an integer number of cycles plus a remainder in (-2pi, 2pi) carrying the
    sign of num, so that num = div_2pi * TWO_PI + mod_2pi.

    Use this when you need the (cycle count, remainder) *representation*. Do NOT use it merely
    to shrink an argument before calling sin/cos -- see the module docstring.
    """
    mod_2pi = fmod(fabs(num), TWO_PI)
    div_2pi = int(floor(fabs(num) / TWO_PI))

    if num < 0.0:
        div_2pi = -div_2pi
        mod_2pi = -mod_2pi

    return div_2pi, mod_2pi
