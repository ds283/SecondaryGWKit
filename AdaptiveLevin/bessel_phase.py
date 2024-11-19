import numpy as np
from numpy.ma.core import logical_xor
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root_scalar
from scipy.special import yv, jv

from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_SAMPLE_DENSITY = 250

MINIMUM_X = 1e-12


def _weff_sq(nu: float, x: float):
    nu2 = nu * nu
    x2 = x * x

    return 1.0 + (1.0 / 4.0 - nu2) / x2


class XSplineWrapper:
    def __init__(self, spline, min_x, max_x):
        self._spline = spline

        self._min_x = min_x
        self._max_x = max_x

        self._log_min_x = np.log(min_x)
        self._log_max_x = np.log(max_x)

    def __call__(self, x, is_log=False):
        if is_log:
            raw_x = np.exp(x)
            log_x = x
        else:
            raw_x = x
            log_x = np.log(x)

        if raw_x < 0.99 * self._min_x:
            raise ValueError(
                f"x={raw_x:.5g} too small (minimum value={self._min_x:.5g})"
            )

        if raw_x > 1.01 * self._max_x:
            raise ValueError(
                f"x={raw_x:.5g} too large (maximum value={self._max_x:.5g})"
            )

        if raw_x < self._min_x:
            raw_x = self._min_x
            log_x = self._log_min_x

        elif raw_x > self._max_x:
            raw_x = self._max_x
            log_x = self._log_max_x

        return self._spline(log_x)


def bessel_phase(
    nu: float,
    max_x: float,
    sample_points: int = None,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
):
    """
    Build a Liouville-Green phase function for the Bessel functions on the interval [min_x, max_x]
    using the modulus method from Bremer (2022) http://arxiv.org/abs/2209.14561v2.
    We only provide the phase function back to the region where omega_eff = 0.
    :param nu:
    :param max_x:
    :return:
    """
    if _weff_sq(nu, max_x) < 0:
        raise RuntimeError(
            "weff_sq < 0 at the right-hand boundary, so WKB boundary conditions can not be applied. Please increase max_x."
        )

    # if nu > 1/2 then for small x the solution is not oscillatory, but instead exponential
    # We don't try to provide a phase function that is valid on this region
    if nu > 0.5:
        min_x = np.sqrt(nu * nu - 0.25)
    else:
        min_x = 1e-5

    norm_factor = 2.0 / np.pi

    def m(x):
        j = jv(nu, x)
        jsq = j * j

        y = yv(nu, x)
        ysq = y * y

        return jsq + ysq

    Q_INDEX = 0

    def RHS(log_x, state):
        x = np.exp(log_x)
        Q = state[Q_INDEX]

        dQ_dlogx = norm_factor / m(x) - Q
        return [dQ_dlogx]

    log_min_x = np.log(min_x)
    log_max_x = np.log(max_x)

    if sample_points is None or sample_points <= 0:
        sample_points = int(
            round(DEFAULT_SAMPLE_DENSITY * (log_max_x - log_min_x) + 0.5, 0)
        )

    sample_grid = np.linspace(log_min_x, log_max_x, sample_points)

    # try to build an accurate initial condition for Q
    init_jv = jv(0.0, min_x)
    init_sin = init_jv / m(min_x)
    init_phase = np.asin(init_sin)
    init_Q = init_phase / min_x

    init_state = [init_Q]

    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(log_min_x, log_max_x),
        y0=init_state,
        t_eval=sample_grid,
        atol=atol,
        rtol=rtol,
    )

    if not sol.success:
        if sol.status == -1:
            if sol.t[-1] > (1.01) * log_min_x:
                raise RuntimeError(
                    f"Final x-sample of phase function did not reach intended target (final point log(x)={sol.t[-1]:.5g}, expected log(x)={log_min_x:.5g})"
                )

    log_x_samples = sol.t
    Q_samples = sol.y[Q_INDEX]

    def map_phase_point(log_x, Q):
        x = np.exp(log_x)
        return x * Q

    def map_modulus(log_x):
        x = np.exp(log_x)
        return np.sqrt(m(x))

    phase_points = [
        (log_x, map_phase_point(log_x, Q))
        for (log_x, Q) in zip(log_x_samples, Q_samples)
    ]
    mod_points = [(log_x, map_modulus(log_x)) for (log_x, _) in phase_points]

    phase_points.sort(key=lambda x: x[0])
    mod_points.sort(key=lambda x: x[0])

    phase_x, phase_y = zip(*phase_points)
    mod_x, mod_y = zip(*mod_points)

    _phase_spline = InterpolatedUnivariateSpline(phase_x, phase_y, ext="raise")
    _mod_spline = InterpolatedUnivariateSpline(mod_x, mod_y, ext="raise")

    # determine where we will try to match the phase
    if log_min_x < 0.0:
        if log_max_x > 0.0:
            log_match_x = 0.0
        else:
            log_match_x = log_max_x
    else:
        log_match_x = log_min_x

    match_x = np.exp(log_match_x)

    stepsize = 0.1 * np.pi
    bracket_lo = _phase_spline(log_match_x) - np.pi / 2.0
    bracket_hi = bracket_lo + stepsize

    def match_f(phi):
        return _mod_spline(log_match_x) * np.sin(_phase_spline(log_match_x) - phi) - jv(
            nu, match_x
        )

    while (
        match_f(bracket_lo) * match_f(bracket_hi) >= 0.0
        and bracket_hi - bracket_lo < 2.0 * np.pi
    ):
        bracket_hi = bracket_hi + stepsize

    if match_f(bracket_lo) * match_f(bracket_hi) >= 0.0:
        raise RuntimeError(
            f"Could not bracket phase shift phi (x_cut={min_x:.5g}, max_x={max_x:.5g}, match_x={match_x:.5g}, log(match_x)={log_min_x:.5g}, jv({nu}, match_x)={jv(nu, match_x):.8g}, bracket_lo={bracket_lo:.5g}, bracket_hi={bracket_hi:.5g})"
        )

    root = root_scalar(
        match_f,
        bracket=(bracket_lo, bracket_hi),
        xtol=1e-6,
        rtol=1e-4,
    )

    if not root.converged:
        raise RuntimeError(
            f'root_scalar() did not converge to a solution: x_bracket=({bracket_lo:.5g}, {bracket_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    phi = root.root
    phase_points = [(log_x, alpha - phi) for (log_x, alpha) in phase_points]

    phase_x, phase_y = zip(*phase_points)
    _phase_spline = InterpolatedUnivariateSpline(phase_x, phase_y, ext="raise")

    phase_spline = XSplineWrapper(_phase_spline, min_x=min_x, max_x=max_x)
    mod_spline = XSplineWrapper(_mod_spline, min_x=min_x, max_x=max_x)

    def bessel_j(x: float, is_log=False) -> float:
        return mod_spline(x, is_log=is_log) * np.sin(phase_spline(x, is_log=is_log))

    def bessel_y(x: float, is_log=False) -> float:
        return -mod_spline(x, is_log=is_log) * np.cos(phase_spline(x, is_log=is_log))

    return {
        "phase": phase_spline,
        "mod": mod_spline,
        "phi": phi,
        "x_cut": min_x,
        "bessel_j": bessel_j,
        "bessel_y": bessel_y,
        "max_x": max_x,
    }
