import numpy as np
from math import sqrt, fabs, pi, sin, cos
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline

from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

DEFAULT_SAMPLE_DENSITY = 100

MINIMUM_X = 1e-8
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
    def RHS(x, state):
        alpha, beta, gamma = state

        if max(fabs(alpha), fabs(beta), fabs(gamma)) > 1e100:
            print(
                f"** WARNING: very large values encountered @ x = {x:.5g}: (alpha, beta, gamma) = ({alpha:.5g}, {beta:.5g}, {gamma:.5g})"
            )

        alpha2 = alpha * alpha
        beta2 = beta * beta
        beta3 = beta * beta2

        dgamma_dx = beta
        dbeta_dx = alpha
        dalpha_dx = (
            (3.0 / 2.0) * alpha2 / beta - 2.0 * beta3 + 2.0 * beta * _weff_sq(nu, x)
        )

        if max(fabs(dalpha_dx), fabs(dbeta_dx), fabs(dgamma_dx)) > 1e100:
            print(
                f"** WARNING: very large values encountered @ x = {x:.5g}: (dalpha/dx, dbeta/dx, dgamma/dx) = ({dalpha_dx:.5g}, {dbeta_dx:.5g}, {dgamma_dx:.5g})"
            )

        return [dalpha_dx, dbeta_dx, dgamma_dx]

    if _weff_sq(nu, max_x) < 0:
        raise RuntimeError(
            "weff_sq < 0 at the right-hand boundary, so WKB boundary conditions can not be applied. Consider extending max_x."
        )

    alpha_init = _d_weff(nu, max_x)
    beta_init = sqrt(_weff_sq(nu, max_x))
    gamma_init = 0

    init_state = [alpha_init, beta_init, gamma_init]

    if sample_points is None or sample_points <= 0:
        sample_points = int(round(DEFAULT_SAMPLE_DENSITY * max_x + 0.5, 0))

    sample_grid = np.linspace(MINIMUM_X, max_x, sample_points)
    sample_grid = np.flip(sample_grid)

    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(max_x, MINIMUM_X),
        y0=init_state,
        t_eval=sample_grid,
        atol=atol,
        rtol=rtol,
    )

    if not sol.success:
        if sol.status == -1:
            if sol.t[-1] > SMALLEST_ACCEPTABLE_XMIN:
                raise RuntimeError(
                    f"Final x-sample of phase function is too large (final point x={sol.t[-1]})"
                )

    phase_points = []
    dphase_points = []
    ddphase_points = []

    t_samples = sol.t
    phase_samples = sol.y[2]
    dphase_samples = sol.y[1]
    ddphase_samples = sol.y[0]

    for i in range(len(t_samples)):
        x = t_samples[i]
        phase = phase_samples[i]
        dphase = dphase_samples[i]
        ddphase = ddphase_samples[i]

        phase_points.append((x, phase))
        dphase_points.append((x, dphase))
        ddphase_points.append((x, ddphase))

    phase_points.sort(key=lambda x: x[0])
    dphase_points.sort(key=lambda x: x[0])
    ddphase_points.sort(key=lambda x: x[0])

    init_phase = phase_points[0][1]
    init_dphase = dphase_points[0][1]
    init_ddphase = ddphase_points[0][1]

    phase_points.insert(0, (0.0, init_phase))
    dphase_points.insert(0, (0.0, init_dphase))
    ddphase_points.insert(0, (0.0, init_ddphase))

    # rebase the phase so it is zero at x=0
    phase_points = [(x, y - init_phase) for x, y in phase_points]

    phase_x, phase_y = zip(*phase_points)
    dphase_x, dphase_y = zip(*dphase_points)
    ddphase_x, ddphase_y = zip(*ddphase_points)

    phase_spline = InterpolatedUnivariateSpline(phase_x, phase_y, ext="raise")
    dphase_spline = InterpolatedUnivariateSpline(dphase_x, dphase_y, ext="raise")
    ddphase_spline = InterpolatedUnivariateSpline(ddphase_x, ddphase_y, ext="raise")

    def bessel_j(x: float) -> float:
        # for sufficiently small x, switch to a series expansion to ensure we do not have division by zero issues
        # at the moment we use an expansion valid up to O(x)^(5/2). It would be nice to have a few more terms here.
        # But to do so, we would need higher derivative information about the phase function.
        if x <= 1e-3:
            return sqrt(2.0 / pi) * sqrt(init_dphase) * sqrt(x)

        return sqrt(2.0 / (pi * x * dphase_spline(x))) * sin(phase_spline(x))

    def bessel_y(x: float) -> float:
        # likewise use a series expansion for small x, but now it is divergent
        if x <= 1e-3:
            sqrt_x = sqrt(x)

            init_dphase_pt5 = sqrt(init_dphase)
            init_dphase_1pt5 = init_dphase * init_dphase_pt5
            init_dphase_2pt5 = init_dphase * init_dphase_1pt5
            init_dphase_2 = init_dphase * init_dphase
            init_dphase_4 = init_dphase_2 * init_dphase_2

            init_ddphase_2 = init_ddphase * init_ddphase

            return (
                sqrt(2.0 / pi) / (init_dphase_pt5 * sqrt_x)
                - (init_ddphase * sqrt_x) / (sqrt(2.0 * pi) * init_dphase_1pt5)
                + ((-4.0 * init_dphase_4 + 3.0 * init_ddphase_2) * pow(x, 1.5))
                / (4.0 * sqrt(2.0 * pi) * init_dphase_2pt5)
            )

        return -sqrt(2.0 / (pi * x * dphase_spline(x))) * cos(phase_spline(x))

    return {
        "phase": phase_spline,
        "dphase": dphase_spline,
        "ddphase": ddphase_spline,
        "bessel_j": bessel_j,
        "bessel_y": bessel_y,
    }
