from math import modf, floor, fabs, fmod, prod

from sympy import factorint

from .constants import TWO_PI, TWO_PI_100


def simple_mod_2pi(num):
    mod_2pi = fmod(fabs(num), TWO_PI)
    div_2pi = int(floor(fabs(num) / TWO_PI))

    if num < 0.0:
        div_2pi = -div_2pi
        mod_2pi = -mod_2pi

    return div_2pi, mod_2pi


def range_reduce_mod_2pi(big_number, small_number):
    """
    Compute the product big_number * small_number mod 2pi, but with as high precision as we can.
    big_number and small_number are indicative names, but these quantities do not need to satisfy any
    specific conditions.
    :param big_number:
    :param small_number:
    :return:
    """

    # write big_number as N + b, where N is an integer
    b, N = modf(big_number)
    N = int(N)

    # write small_number as 2pi M + c, where M is an integer
    # the idea is to perform the mod step on small_number, because this will lose least precision
    M, c = simple_mod_2pi(small_number)

    # big * small = (N + b) * (2pi M + c) = 2pi M N + 2pi M b + N c + b c
    # = 2pi ( M N + floor(M b) ) + 2pi frac(M b) + N c + b c
    # so we can immediately pull out M N + floor(M b) as a contribution to div 2pi
    Mb_floor, Mb_frac = modf(M * b)
    Mb_floor = int(Mb_floor)

    div_2pi = N * M + Mb_floor

    # the remaining parts of 2 pi frac(M b) + N c + b c
    # of these, 2 pi frac(M b) and b c are not large and can be computed using fmod() without loss of precision
    # However, N is possibly a large integer, maybe as large as 1E7 or 1E8 or even a bit larger for our applications.

    # to handle this, our strategy is to move prime factors from N into c until it is modestly large,
    # but still small enough that using fmod() will not result in a drastic loss of precision

    mod_2pi = TWO_PI * Mb_frac + b * c

    factor_list = factorint(N, multiple=True)
    while len(factor_list) > 0:
        while fabs(c) < TWO_PI_100 and len(factor_list) > 0:
            # pop the smallest remaining prime factor
            factor = factor_list.pop(0)
            c = c * factor

        # write c = 2pi L + d
        L, d = simple_mod_2pi(c)

        # the product is now Nprime (2pi L + d) where Nprime is the product of remaining prime factors
        # from the original N. So we can move Nprime L into div_2pi
        # and continue with new c = d
        Nprime = prod(factor_list)
        div_2pi = div_2pi + Nprime * L
        c = d

    # when we get here, no more prime factors remain
    # add c to mod_2pi
    mod_2pi += c

    L, d = simple_mod_2pi(mod_2pi)
    div_2pi = div_2pi + L
    mod_2pi = d

    return div_2pi, mod_2pi
