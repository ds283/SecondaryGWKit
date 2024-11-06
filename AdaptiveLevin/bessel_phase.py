import numpy as np
from math import sqrt, fabs, pi, sin, cos
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline

from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_SAMPLE_DENSITY = 250

MINIMUM_X = 1e-12
SMALLEST_ACCEPTABLE_XMIN = 0.2


def _weff_sq(nu: float, x: float):
    nu2 = nu * nu
    x2 = x * x

    return 1.0 + (1.0 / 4.0 - nu2) / x2


def _d_weff(nu: float, x: float):
    nu2 = nu * nu
    x2 = x * x
    x3 = x * x2

    return (-1.0 + 4.0 * nu2) / (2.0 * x3 * sqrt(4.0 + (1.0 - 4.0 * nu2) / x2))


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
            raise ValueError(f"x too small (minimum value={self._min_x:.5g})")

        if raw_x > 1.01 * self._max_x:
            raise ValueError(f"x too large (maximum value={self._max_x:.5g})")

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
    Build a Liouville-Green phase function for the Bessel functions on the interval [0, max_x]
    :param nu:
    :param max_x:
    :return:
    """

    # to avoid the singularity at x=0, we start from the right-hand side of the interval (where the WKB
    # boundary condition is likely to be very good), and work backwards to the left.

    # alpha = theta''
    # beta = theta' (this is the normalization factor that appears in the Liouville-Green representation)
    # gamma = theta (this is the Liouville-Green phase we want)
    def RHS(log_x, state):
        # beta = exp(log_beta)
        # gamma = -exp(log_gamma) (so the name is slightly a misnomer)
        alpha, log_beta, log_gamma = state
        x = np.exp(log_x)

        beta = np.exp(log_beta)
        gamma = -np.exp(log_gamma)

        if max(fabs(alpha), fabs(beta), fabs(gamma)) > 1e100:
            print(
                f"** WARNING: very large values encountered @ x = {x:.5g}: (alpha, beta, gamma) = ({alpha:.5g}, {beta:.5g}, {gamma:.5g})"
            )

        alpha2 = alpha * alpha
        beta2 = beta * beta
        beta3 = beta * beta2

        dgamma_dx = beta
        dbeta_dx = alpha

        dlog_beta_dx = dbeta_dx / beta
        dlog_gamma_dx = dgamma_dx / gamma

        dalpha_dx = (
            (3.0 / 2.0) * alpha2 / beta - 2.0 * beta3 + 2.0 * beta * _weff_sq(nu, x)
        )

        if max(fabs(dalpha_dx), fabs(dbeta_dx), fabs(dgamma_dx)) > 1e100:
            print(
                f"** WARNING: very large values encountered @ x = {x:.5g}: (dalpha/dx, dbeta/dx, dgamma/dx) = ({dalpha_dx:.5g}, {dbeta_dx:.5g}, {dgamma_dx:.5g})"
            )

        return [x * dalpha_dx, x * dlog_beta_dx, x * dlog_gamma_dx]

    if _weff_sq(nu, max_x) < 0:
        raise RuntimeError(
            "weff_sq < 0 at the right-hand boundary, so WKB boundary conditions can not be applied. Consider extending max_x."
        )

    alpha_init = _d_weff(nu, max_x)
    beta_init = sqrt(_weff_sq(nu, max_x))
    log_beta_init = np.log(beta_init)
    log_gamma_init = 0

    init_state = [alpha_init, log_beta_init, log_gamma_init]

    log_min_x = np.log(MINIMUM_X)
    log_max_x = np.log(max_x)

    if sample_points is None or sample_points <= 0:
        sample_points = int(
            round(DEFAULT_SAMPLE_DENSITY * (log_max_x - log_min_x) + 0.5, 0)
        )

    sample_grid = np.linspace(log_min_x, log_max_x, sample_points)
    sample_grid = np.flip(sample_grid)

    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(log_max_x, log_min_x),
        y0=init_state,
        t_eval=sample_grid,
        atol=atol,
        rtol=rtol,
    )

    if not sol.success:
        if sol.status == -1:
            if sol.t[-1] > (1.01) * log_min_x:
                raise RuntimeError(
                    f"Final x-sample of phase function is too large (final point log(x)={sol.t[-1]})"
                )

    phase_points = []
    dphase_points = []
    ddphase_points = []

    t_samples = sol.t
    log_gamma_samples = sol.y[2]
    log_beta_samples = sol.y[1]
    alpha_samples = sol.y[0]

    min_phase = None

    for i in range(len(t_samples)):
        x = t_samples[i]
        phase = -np.exp(log_gamma_samples[i])
        dphase = np.exp(log_beta_samples[i])
        ddphase = alpha_samples[i]

        if min_phase is None or phase < min_phase:
            min_phase = phase

        phase_points.append((x, phase))
        dphase_points.append((x, dphase))
        ddphase_points.append((x, ddphase))

    phase_points.sort(key=lambda x: x[0])
    dphase_points.sort(key=lambda x: x[0])
    ddphase_points.sort(key=lambda x: x[0])

    # rebase the phase so it is zero at x=0
    phase_points = [(x, y - min_phase) for x, y in phase_points]

    init_dphase = dphase_points[0][1]
    init_ddphase = ddphase_points[0][1]

    phase_x, phase_y = zip(*phase_points)
    dphase_x, dphase_y = zip(*dphase_points)
    ddphase_x, ddphase_y = zip(*ddphase_points)

    _phase_spline = InterpolatedUnivariateSpline(phase_x, phase_y, ext="raise")
    _dphase_spline = InterpolatedUnivariateSpline(dphase_x, dphase_y, ext="raise")
    _ddphase_spline = InterpolatedUnivariateSpline(ddphase_x, ddphase_y, ext="raise")

    phase_spline = XSplineWrapper(_phase_spline, min_x=MINIMUM_X, max_x=max_x)
    dphase_spline = XSplineWrapper(_dphase_spline, min_x=MINIMUM_X, max_x=max_x)
    ddphase_spline = XSplineWrapper(_ddphase_spline, min_x=MINIMUM_X, max_x=max_x)

    init_dphase_pt5 = sqrt(init_dphase)
    init_dphase_1pt5 = init_dphase * init_dphase_pt5
    init_dphase_2pt5 = init_dphase * init_dphase_1pt5
    init_dphase_3pt5 = init_dphase * init_dphase_2pt5
    init_dphase_2 = init_dphase * init_dphase
    init_dphase_4 = init_dphase_2 * init_dphase_2
    init_ddphase_2 = init_ddphase * init_ddphase
    init_ddphase_3 = init_ddphase * init_ddphase_2

    def bessel_j(x: float) -> float:
        # for sufficiently small x, switch to a series expansion to ensure we do not have division by zero issues
        if x <= 1e-2:
            x_pt5 = sqrt(x)
            x_1pt5 = x * x_pt5
            x_2pt5 = x * x_1pt5

            x_pt5_term = sqrt(2.0 / pi) * init_dphase_pt5 * x_pt5
            x_2pt5_term = (
                (3.0 * init_ddphase_2 - 4.0 * init_dphase_4)
                * x_2pt5
                / (12.0 * init_dphase_1pt5 * sqrt(2.0 * pi))
            )

            return x_pt5_term + x_2pt5_term

        return sqrt(2.0 / (pi * x * dphase_spline(x))) * sin(phase_spline(x))

    def bessel_y(x: float) -> float:
        # likewise use a series expansion for small x, but now it is divergent
        # if x <= 1e-2:
        #     x_pt5 = sqrt(x)
        #     x_1pt5 = x * x_pt5
        #     x_2pt5 = x * x_1pt5
        #     x_3pt5 = x * x_2pt5
        #
        #     x_mpt5_term = sqrt(2.0 / pi) / init_dphase_pt5 / x_pt5
        #     x_pt5_term = -init_ddphase * x_pt5 / init_dphase_1pt5 / sqrt(2.0 * pi)
        #     x_1pt5_term = (
        #         (3.0 * init_ddphase_2 - 4.0 * init_dphase_4)
        #         * x_2pt5
        #         / (4.0 * init_dphase_2pt5 * sqrt(2.0 * pi))
        #     )
        #     x_2pt5_term = (
        #         -(5.0 * init_ddphase_3 + 4.0 * init_ddphase * init_dphase_4)
        #         * x_3pt5
        #         / (8.0 * init_dphase_3pt5 * sqrt(2.0 * pi))
        #     )
        #
        #     return x_mpt5_term + x_pt5_term + x_1pt5_term + x_2pt5_term

        return -sqrt(2.0 / (pi * x * dphase_spline(x))) * cos(phase_spline(x))

    return {
        "phase": phase_spline,
        "dphase": dphase_spline,
        "ddphase": ddphase_spline,
        "bessel_j": bessel_j,
        "bessel_y": bessel_y,
        "max_x": max_x,
    }
