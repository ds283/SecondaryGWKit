from math import pi, sqrt
from typing import Callable, Optional

from scipy.optimize import root_scalar

# fraction of a cycle advanced per search step when an effective frequency is available:
# the step is chosen so that the phase advances by 2 pi / PHASE_STEPS_PER_CYCLE
PHASE_STEPS_PER_CYCLE = 16

# relative-z step used when no effective frequency is available (or where omega^2 <= 0).
# This is the step the search used unconditionally before prompt 11 of prompts/GkTk-remedial.
DEFAULT_RELATIVE_STEP = 1e-3


def find_phase_extremum(
    sol,
    start_z: float,
    stop_z: float,
    value_index: int,
    deriv_index: int,
    omega_sq: Optional[Callable[[float], float]] = None,
):
    """
    Walk downwards in redshift from ``start_z`` until the derivative held in ``deriv_index``
    changes sign from negative to positive, then refine that sign change to a root.

    **This finds a maximum of the solution, not a minimum.** Stepping downwards in z the
    derivative dG/dz passes from negative to positive, so G itself passes through a *maximum*:
    every run measured in ``docs/gk-wkb-review-fable-2026-09-09.md`` §10.1 reports
    ``G_stop/envelope = +1.000000`` at the returned point. The function was called
    ``find_phase_minimum`` and its callers described the result as a minimum; the name survives
    as an alias, but the extremum is a maximum.

    Nothing downstream depends on *which* extremum is found: the WKB objects' ``store()``
    rotates arbitrary initial data (G, G') into a pure sine, so the historic motivation of
    "cutting at a minimum to avoid jitter in the phase" is obsolete. What matters is only that
    the cut is at a reproducible point of fixed phase.

    :param sol: a dense-output solution object, callable as ``sol(z) -> state vector``
    :param start_z: redshift at which the (downwards) search begins
    :param stop_z: redshift at which the search gives up
    :param value_index: index of the value within the state vector
    :param deriv_index: index of the derivative within the state vector
    :param omega_sq: optional callable ``z -> omega^2(z)`` giving the square of the effective
        WKB frequency. When supplied, and positive, the search steps by
        ``2 pi / (PHASE_STEPS_PER_CYCLE * omega)``, i.e. a fixed fraction of a cycle, rather
        than by a fixed fraction of z. Review §10.2: the fixed relative step of
        ``DEFAULT_RELATIVE_STEP`` gives 15 samples per cycle at x = 403 and fewer than one at
        x > 6283, so it is safe only inside the (z_e3, z_e6) search window. Stepping in phase
        removes that dependence on the width of the window. The window itself is **not**
        widened here; that is a hand-over decision.
    :return: dict with the redshift of the extremum and the value and derivative there
    """

    def step_from(z: float) -> float:
        """Downward (negative) step to take from redshift ``z``."""
        if omega_sq is not None:
            omega_sq_here = omega_sq(z)
            if omega_sq_here > 0.0:
                return -2.0 * pi / (PHASE_STEPS_PER_CYCLE * sqrt(omega_sq_here))

        return -DEFAULT_RELATIVE_STEP * z

    start_deriv = sol(start_z)[deriv_index]
    last_deriv = start_deriv

    found_zero = False

    last_z = start_z
    current_z = start_z + step_from(start_z)
    current_deriv = None
    while current_z > stop_z:
        current_deriv = sol(current_z)[deriv_index]

        # has there been a sign change since the last time we sampled the derivative?
        # we want the derivative to pass from negative to positive as z decreases, which is a
        # maximum of the solution
        if current_deriv * last_deriv < 0 and last_deriv < 0:
            found_zero = True
            break

        last_z = current_z

        current_z += step_from(current_z)
        last_deriv = current_deriv

    if not found_zero:
        raise RuntimeError(
            f"Did not find zero of derivative in the search window (start_z={start_z:.5g}, stop_z={stop_z:.5g}), current_z={current_z:.5g}, start Gprime={start_deriv:.5g}, last derivative={last_deriv:.5g}, current derivative={current_deriv:.5g}"
        )

    root = root_scalar(
        lambda z: sol(z)[deriv_index],
        bracket=(last_z, current_z),
        xtol=1e-6,
        rtol=1e-4,
    )

    if not root.converged:
        raise RuntimeError(
            f'root_scalar() did not converge to a solution: z_bracket=({last_z:.5g}, {current_z:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    root_z = root.root
    sol_root = sol(root_z)

    return {
        "z": root_z,
        "value": sol_root[value_index],
        "derivative": sol_root[deriv_index],
    }


# compatibility alias: the extremum is a maximum, but the historic name is used by callers and
# by reproduction scripts under docs/
find_phase_minimum = find_phase_extremum
