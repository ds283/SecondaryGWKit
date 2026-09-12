"""
The shared numeric driver for the two sectors' ODEs: ``GkNumericIntegration`` and
``TkNumericIntegration`` both integrate their second-order system through
:func:`numeric_with_phase_cut`, which runs SciPy's DOP853 over the requested redshift grid and,
in ``"stop"`` mode, cuts the result at a point of fixed phase.

**Why the integration is split at the cosmology's declared discontinuities.**

DOP853 chooses its step from an *embedded* error estimate: it forms two Runge-Kutta results of
different order over the same step and takes their difference as the local error. That difference
estimates the truncation error only while both Taylor expansions are valid over the step, which
requires the right-hand side to be smooth across it. Step over a point where the right-hand side
*jumps* and the estimate stops meaning anything: the method's effective order collapses from eight
to one, so its error falls like ``h`` rather than ``h^8`` and a decade of tolerance shrinks the
step by only ``10^(1/8) = 1.33``, buying about a quarter. Worse, *where* a step lands relative to
the jump changes discontinuously with the tolerance, so refinement is not even monotone --
tightening can make the answer worse. The remedy is not a tighter tolerance but a restart: end one
integration at the jump and begin the next from its final state, so that no step straddles it and
every step sees a smooth right-hand side. The restart is placed a hair on the *near* side of the
jump rather than exactly on it, for a reason that turns out to matter by three orders of magnitude:
see :data:`BREAK_POINT_STANDOFF`.

**How this module learns where those points are.** It asks the cosmology, through
``ComputeTargets.BackgroundModel._cosmology_break_points(..., kind=BREAK_POINT_DISCONTINUITY)``,
and it asks for *jumps only*. No equation-of-state knowledge lives here: a cosmology that declares
nothing -- every LambdaCDM model, ``RadiationModel``, every test stand-in -- is treated as smooth
and takes the single-``solve_ivp`` path this module has always taken, reproducing its numbers bit
for bit. The distinction between a jump and a kink is the cosmology's to make, and it matters here
in a way it does not in a quadrature: a fixed-order Gauss-Legendre panel has to be split at *every*
non-smooth point, kinks included, and on ``QCD_Cosmology``'s production range there are 404 of
those (the ``T(z)`` spline knots) against 3 jumps. An adaptive stepper absorbs a C2 point at the
cost of a few extra steps; restarting at all 407 would pay 408 startup transients to fix three.

**How the failure was detected, and why the test could not see it before.** A run's distance from
the converged answer is estimated by running the same integrator twice, a decade apart in
tolerance, and taking the difference: valid exactly while refinement is monotone, which is what a
jump destroys. On ``QCDModel`` that check failed at four of the fifty production wavenumbers, the
"reference" moving by 1.6e-6 to 6.2e-6 of the envelope against a criterion of 3.4e-8 -- so the
measurement was partly measuring its own reference, and no tolerance could have fixed it (SciPy
clamps ``rtol`` at 100 eps = 2.22e-14). ``docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`` §4 is that
measurement and §9 is the same measurement after the split.
"""

from math import expm1, fabs, log, pi, sqrt
from typing import Callable, List, Optional, Sequence

import numpy as np
import ray
from scipy.integrate import solve_ivp

from ComputeTargets import ModelProxy, BackgroundModel
from ComputeTargets.BackgroundModel import (
    BREAK_POINT_DISCONTINUITY,
    _cosmology_break_points,
)
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from LiouvilleGreen.integration_tools import find_phase_extremum
from Quadrature.integration_metadata import IntegrationData
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import check_units
from config.defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)

# State layout:
#    state[0] = value (T_k or G_k)
#    state[1] = derivative (T_k' or G_k')
VALUE_INDEX = 0
DERIV_INDEX = 1
EXPECTED_SOL_LENGTH = 2


def declared_discontinuities_in_z(
    model: BackgroundModel, z_lo: float, z_hi: float
) -> List[float]:
    """
    The redshifts strictly inside ``(z_lo, z_hi)`` at which the model's cosmology declares that a
    background quantity *jumps*, in descending order (the direction of integration).

    Duck-typed throughout: a model with no ``cosmology`` attribute, or a cosmology that does not
    implement ``integration_break_points``, or one whose equation of state declares no
    discontinuity temperatures, all give an empty list and hence the unsplit code path.

    The declaration is made in ``u = log(1+z)``, which is the campaign's integration variable, and
    is converted here with ``expm1``. That direction is the lossy one (``CLAUDE.md``), but the
    recovered ``z`` is used only as a limit of integration -- never in an equality-like comparison
    -- which is exactly the case the rule permits.
    """
    cosmology = getattr(model, "cosmology", None)
    if cosmology is None:
        return []

    u_points = _cosmology_break_points(
        cosmology, z_lo, z_hi, kind=BREAK_POINT_DISCONTINUITY
    )
    if len(u_points) == 0:
        return []

    z_points = {float(expm1(float(u))) for u in u_points}
    return sorted((z for z in z_points if z_lo < z < z_hi), reverse=True)


# How far on the near side of a declared discontinuity a segment boundary is placed, as a
# relative offset in (1+z) -- equivalently, an offset of this size in the campaign's integration
# variable u = log(1+z).
#
# This is not cosmetic. An explicit Runge-Kutta method evaluates a stage at the far end of every
# step, so a segment that *ends* exactly on the discontinuity evaluates its last stage exactly
# there -- and which branch of the equation of state answers at that point is decided by floating
# -point rounding of the cosmology's own internal lookup, which is a coin flip. When it lands on
# the far branch, the final step of the departing segment is a straddling step again, with the
# controller's full step size, and the split buys nothing.
#
# Measured on QCD_Cosmology's Tk run (prompts/GkTk-remedial, log 18): with the boundary placed
# exactly on the crossing, the reference-convergence drift at k = 4.972e7/Mpc stays at 2.99e-06
# of the envelope, against 4.6e-09 and 7.9e-09 at two neighbouring wavenumbers where the same
# coin came up the other way; with this standoff it falls to 6.5e-09. Displacing the boundary to
# the *far* side instead breaks those two neighbours symmetrically (4.6e-09 -> 5.9e-07,
# 7.9e-09 -> 8.5e-08), which is what identifies the mechanism.
#
# The value has to clear the cosmology's own evaluation noise -- a relative 1e-16 or so, from
# splines and root-solves -- by a wide margin, and has to be small enough that the sliver of the
# far side swept by the arriving segment contributes nothing: 1e-12 is four orders above the
# first and fourteen orders below the integration range in u. 1e-9 was measured to work equally
# well, so the choice is not delicate.
BREAK_POINT_STANDOFF = 1.0e-12


def _standoff_boundary(z_break: float) -> float:
    """
    The redshift at which to end the segment above ``z_break`` and begin the segment below it:
    :data:`BREAK_POINT_STANDOFF` on the near side, the near side being higher z because these
    integrations always run downwards.
    """
    return z_break + BREAK_POINT_STANDOFF * (1.0 + z_break)


class _SegmentedDenseOutput:
    """
    A dense-output callable assembled from one ``OdeSolution`` per segment, so that the stop-mode
    root-find (``find_phase_extremum``) can search a window that straddles a segment boundary.

    ``find_phase_extremum`` uses its ``sol`` argument only as ``sol(z) -> state vector``, so this
    is the whole protocol. Segments are held in descending order of redshift; the first whose
    lower bound lies at or below ``z`` is the one that contains it. The solution is continuous
    across a boundary even though the right-hand side is not, so which side a boundary redshift is
    evaluated on does not matter.
    """

    def __init__(self, segments):
        # segments: list of (z_start, z_end, dense_output), descending in z_start
        self._segments = segments

    def __call__(self, z: float):
        for _, z_end, dense in self._segments:
            if z >= z_end:
                return dense(z)

        # below the last segment's floor: the integration terminated on its event before reaching
        # it. Extrapolate from the final segment, which is what a single solve_ivp would do.
        return self._segments[-1][2](z)


class _SegmentedSolution:
    """
    The parts of a SciPy ``OdeResult`` that :func:`numeric_with_phase_cut` consumes, assembled
    from a sequence of per-segment solves: the sample grid and state (``t``, ``y``), the aggregate
    step count (``nfev``), the terminating ``status`` of the last segment executed, and the
    composite dense output (``sol``).

    ``success`` is always True: a failed segment raises inside :func:`_solve_segmented`, where the
    segment index and its redshift range are still known.
    """

    def __init__(self, t, y, nfev: int, status: int, sol, num_segments: int):
        self.t = t
        self.y = y
        self.nfev = nfev
        self.status = status
        self.success = True
        self.message = "The solver successfully reached the end of every segment."
        self.sol = sol
        self.num_segments = num_segments


def _solve_segmented(
    RHS,
    z_init: float,
    z_min: float,
    t_eval: Sequence[float],
    y0,
    break_z: Sequence[float],
    events,
    dense_output: bool,
    atol: float,
    rtol: float,
    args,
    task_label: str,
    k_inv_Mpc: float,
) -> _SegmentedSolution:
    """
    Integrate from ``z_init`` down to ``z_min`` in segments whose interior boundaries are the
    declared discontinuities ``break_z`` (descending, strictly inside the range), carrying the
    final state of each segment into the next as its initial condition.

    The requested output points ``t_eval`` are distributed across the segments without being
    moved: segment *j* takes the samples with ``z_end < s <= z_start``, and the lowest segment
    also takes a sample sitting exactly on ``z_min``. Each non-final segment additionally asks for
    its own lower boundary as an output point, purely so that the state there can be read off and
    handed to the next segment; that point is then dropped, so **no break point is ever rounded on
    to a sample** and the returned grid is exactly the grid requested.

    The interior boundaries are placed a standoff of :data:`BREAK_POINT_STANDOFF` on the *near*
    side of each declared discontinuity; :func:`_standoff_boundary` says why that is not a
    cosmetic detail.
    """
    z_top = float(z_init)
    z_bottom = float(z_min)

    # the standoff can in principle push a boundary out of the range, if a declared
    # discontinuity sits within it of an endpoint; such a boundary is simply dropped, which
    # leaves the endpoint itself as the restart and costs nothing
    interior = sorted(
        {
            boundary
            for boundary in (_standoff_boundary(float(z)) for z in break_z)
            if z_bottom < boundary < z_top
        },
        reverse=True,
    )
    boundaries = [z_top] + interior + [z_bottom]
    num_segments = len(boundaries) - 1

    samples = [float(z) for z in t_eval]
    num_samples = len(samples)

    # partition the requested output points by segment, preserving order
    chunks = []
    index = 0
    for j in range(num_segments):
        z_end = boundaries[j + 1]
        start = index
        if j == num_segments - 1:
            while index < num_samples and samples[index] >= z_end:
                index += 1
        else:
            while index < num_samples and samples[index] > z_end:
                index += 1
        chunks.append(samples[start:index])

    if index != num_samples:
        raise RuntimeError(
            f"{task_label}: {num_samples - index} requested sample points lie below the end of "
            f"the integration range (k={k_inv_Mpc}/Mpc, z_min={z_min}, first unassigned "
            f"z={samples[index]})"
        )

    t_pieces = []
    y_pieces = []
    dense_segments = []
    nfev = 0
    status = 0
    state = list(y0)

    for j in range(num_segments):
        z_start = boundaries[j]
        z_end = boundaries[j + 1]
        requested = chunks[j]
        is_last = j == num_segments - 1

        segment_t_eval = list(requested) if is_last else list(requested) + [z_end]

        sol = solve_ivp(
            RHS,
            method="DOP853",
            t_span=(z_start, z_end),
            y0=state,
            t_eval=segment_t_eval,
            events=events,
            dense_output=dense_output,
            atol=atol,
            rtol=rtol,
            args=args,
        )

        if not sol.success:
            reached = float(sol.t[-1]) if len(sol.t) > 0 else z_start
            raise RuntimeError(
                f"{task_label}: integration did not terminate successfully in segment "
                f"{j + 1} of {num_segments}, z in ({z_end}, {z_start}) "
                f'(k={k_inv_Mpc}/Mpc, z_source={z_init}, error at z={reached}, "{sol.message}")'
            )

        nfev += int(sol.nfev)
        if dense_output:
            dense_segments.append((z_start, z_end, sol.sol))

        num_requested = len(requested)

        # SciPy leaves sol.y as an empty *list* rather than an empty array when a solve with an
        # explicit t_eval produces no output points at all, which happens here whenever a
        # terminal event fires inside a segment before any of that segment's requested samples
        # (or its boundary point) is reached. Normalise before slicing.
        segment_t = np.asarray(sol.t, dtype=float).reshape(-1)
        segment_y = np.asarray(sol.y, dtype=float)
        if segment_y.size == 0:
            segment_y = np.empty((EXPECTED_SOL_LENGTH, 0), dtype=float)

        t_pieces.append(segment_t[:num_requested])
        y_pieces.append(segment_y[:, :num_requested])

        status = int(sol.status)
        if status == 1:
            # a terminal event fired inside this segment: the whole integration stops here, not
            # merely this segment
            break

        if not is_last:
            if len(segment_t) != num_requested + 1:
                raise RuntimeError(
                    f"{task_label}: segment {j + 1} of {num_segments} did not return the state "
                    f"at its lower boundary z={z_end} (k={k_inv_Mpc}/Mpc; expected "
                    f"{num_requested + 1} output points, got {len(segment_t)})"
                )
            state = list(segment_y[:, -1])

    if len(t_pieces) == 0:
        t = np.empty(0, dtype=float)
        y = np.empty((EXPECTED_SOL_LENGTH, 0), dtype=float)
    else:
        t = np.concatenate(t_pieces)
        y = np.concatenate(y_pieces, axis=1)

    return _SegmentedSolution(
        t=t,
        y=y,
        nfev=nfev,
        status=status,
        sol=_SegmentedDenseOutput(dense_segments) if dense_output else None,
        num_segments=num_segments,
    )


def scan_sample_grid_for_unresolved_osc(
    model: BackgroundModel,
    k: wavenumber,
    k_float: float,
    omega_sq: Callable[[BackgroundModel, float, float], float],
    sampled_z: Sequence[float],
    object_label: str,
    *,
    warn: bool = True,
) -> dict:
    """
    Test whether the *returned sample grid* is fine enough to resolve the oscillations the
    integrator stepped through, and print a warning if it is not.

    This is the oscillation-resolution diagnostic that used to be evaluated inside the ODE
    right-hand side, through ``NumericIntegrationSupervisor.report_wavelength``. Two things
    changed (prompt 11 of ``prompts/GkTk-remedial``):

    **What is tested.** The flag's documented meaning is "the sample grid *you supplied* is too
    coarse to resolve the oscillations I stepped through" (review §13.1) -- it is not a statement
    about the integrator's own accuracy, which review §10.1 measures independently at 2e-7 of the
    envelope. The test therefore now compares the local wavelength against the actual spacing of
    consecutive returned samples, rather than against a nominal spacing reconstructed from a
    ``delta_logz`` argument. That reconstruction carried two errors: ``delta_logz`` is supplied as
    a spacing in log10(1+z) and was consumed as a spacing in ln(1+z), understating the spacing by
    ln 10; and for ``GkNumericIntegration`` the value supplied describes the *source* grid while
    the samples are taken on the *response* grid, which is 12x sparser.

    **What is sampled.** On the right-hand side the test saw every internal step of the solver;
    evaluated on the sample grid it sees only the output points. A wavelength minimum that falls
    strictly between two output samples is therefore no longer caught. Since the subject of the
    test *is* the output grid, this is the more faithful form -- but the difference is real and is
    recorded here deliberately (review §13.1 asks that whoever implements this state which
    behaviour was chosen).

    The scan reports the **first** (largest-z) consecutive pair that fails, matching the old
    behaviour of latching on the first failure and reporting nothing afterwards.

    :param model: background model, supplying ``functions.Hubble``
    :param k: the wavenumber object, for the warning message
    :param k_float: dimensionful wavenumber, in the cosmology's units
    :param omega_sq: the sector's effective frequency, called as ``omega_sq(model, k_float, z)``
    :param sampled_z: the redshifts actually returned by the solver, in descending order
    :param object_label: label for the warning message
    :param warn: keyword-only; when False the two ``print`` calls are suppressed, and *only*
        those. Which pair fails, and the three returned values, do not depend on it. The
        production integrators pass ``warn=False`` because ``main.py`` now accumulates the flag
        over each work queue and prints one summary per wavenumber instead (prompt 16 of
        ``prompts/GkTk-remedial``, enacting README section 7 decision D2): with the corrected
        test of prompt 11 the flag fires on essentially every ``GkNumericIntegration`` object,
        so the per-object line would be ~1.3e5 lines per model. The default stays ``True`` so
        that any other caller -- a test, a script under ``docs/``, a future integrator -- keeps
        today's behaviour and the warning remains one argument away.
    :return: dict with ``has_unresolved_osc``, ``unresolved_z`` and ``unresolved_efolds_subh``
    """
    for i in range(len(sampled_z) - 1):
        z = float(sampled_z[i])

        omega_WKB_sq = omega_sq(model, k_float, z)
        if omega_WKB_sq <= 0.0:
            # not oscillatory here; there is nothing to resolve
            continue

        wavelength = 2.0 * pi / sqrt(omega_WKB_sq)
        grid_spacing = fabs(z - float(sampled_z[i + 1]))

        if wavelength < grid_spacing:
            efolds_subh = log((1.0 + z) * k_float / model.functions.Hubble(z))
            if warn:
                print(
                    f"!! WARNING: {object_label} integration for k = {k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}) may have developed unresolved oscillations"
                )
                print(
                    f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g} | approximate wavelength Delta z={wavelength:.5g}, approximate grid spacing at this z: {grid_spacing:.5g}"
                )
            return {
                "has_unresolved_osc": True,
                "unresolved_z": z,
                "unresolved_efolds_subh": efolds_subh,
            }

    return {
        "has_unresolved_osc": False,
        "unresolved_z": None,
        "unresolved_efolds_subh": None,
    }


@ray.remote
def numeric_with_phase_cut(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: redshift,
    z_sample: redshift_array,
    initial_value: float,
    initial_deriv: float,
    RHS,
    omega_sq: Optional[Callable[[BackgroundModel, float, float], float]] = None,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
    mode: str = None,
    stop_search_window_z_begin: Optional[float] = None,
    stop_search_window_z_end: Optional[float] = None,
    task_label: str = "numeric_with_phase_cut",
    object_label: str = "(object)",
    warn_unresolved_osc: bool = True,
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    # note the None test must come first: mode=None means "integrate the whole grid", and
    # calling .lower() on it raises AttributeError (review §10.2)
    if mode is not None:
        mode = mode.lower()

        if mode not in ["stop"]:
            raise ValueError(f'{task_label}: unknown compute mode "{mode}"')

    if mode in ["stop"]:
        if stop_search_window_z_begin is None:
            raise ValueError(
                "{label}: in 'stop' mode, stop_search_window_z_begin must be specified"
            )
        if stop_search_window_z_end is None:
            raise ValueError(
                "{label}: in 'stop' mode, stop_search_window_z_end must be specified"
            )

        if stop_search_window_z_begin < stop_search_window_z_end:
            stop_search_window_z_begin, stop_search_window_z_end = (
                stop_search_window_z_end,
                stop_search_window_z_begin,
            )
            print(
                f"## {task_label}: search window start/end arguments in the wrong order (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc). Now searching in interval: z in ({stop_search_window_z_begin}, {stop_search_window_z_end})"
            )

        max_z = z_init.z
        min_z = z_sample.min.z
        if stop_search_window_z_begin > max_z:
            raise ValueError(
                f"{task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window starting redshift z={stop_search_window_z_begin:.5g} exceeds source redshift z_source={max_z:.5g}"
            )
        if stop_search_window_z_end < min_z:
            print(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window ending redshift z={stop_search_window_z_end:.5g} is smaller than lowest z-response sample point z={min_z:.5g}. Search will terminate at z={min_z:.5g}."
            )
            stop_search_window_z_end = min_z

        if (
            fabs(stop_search_window_z_begin - stop_search_window_z_end)
            < DEFAULT_FLOAT_PRECISION
        ):
            raise ValueError(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified search window has effectively zero extent"
            )

        if (
            fabs(z_init.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_init.store_id == z_sample.min.store_id
        ):
            raise ValueError(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) in 'stop' mode, the source redshift and the lowest response redshift cannot be equal"
            )

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    k_float = k_wavenumber.k
    z_min = float(z_sample.min)

    # Ask the cosmology where its background quantities *jump* inside the integration range, and
    # integrate the pieces between those points in sequence rather than stepping across them: see
    # this module's docstring for why an adaptive Runge-Kutta method cannot be trusted across a
    # discontinuous right-hand side, and why kinks are deliberately not included. A cosmology
    # declaring none -- every LambdaCDM model, RadiationModel, every test stand-in -- gives an
    # empty list and the single-call path below, unchanged.
    break_z = declared_discontinuities_in_z(model, z_min, z_init.z)

    # delta_logz is still accepted and still handed to the supervisor, so that callers (main.py)
    # need not change; but the oscillation-resolution diagnostic no longer uses it. It now runs
    # after the solve, against the actual spacing of the returned samples -- see
    # scan_sample_grid_for_unresolved_osc and NumericIntegrationSupervisor.report_wavelength.
    with NumericIntegrationSupervisor(
        k_wavenumber, z_init, z_sample.min, object_label, delta_logz=delta_logz
    ) as supervisor:
        initial_state = [initial_value, initial_deriv]

        if mode == "stop":
            # set up an event to terminate the integration after the end of the search window
            def stop_event(z, state, model, k_float, supervisor):
                return z - stop_search_window_z_end + DEFAULT_FLOAT_PRECISION

            # mark stop_event as terminal
            stop_event.terminal = True

            events = [stop_event]

            # need dense output for the root-finding algorithm, used to cut at a point of fixed
            # phase (an extremum of the solution; see find_phase_extremum)
            dense_output = True
        else:
            events = None
            dense_output = False

        if len(break_z) == 0:
            sol = solve_ivp(
                RHS,
                method="DOP853",
                t_span=(z_init.z, z_min),
                y0=initial_state,
                t_eval=z_sample.as_float_list(),
                events=events,
                dense_output=dense_output,
                atol=atol,
                rtol=rtol,
                args=(
                    model,
                    k_float,
                    supervisor,
                ),
            )
        else:
            sol = _solve_segmented(
                RHS,
                z_init.z,
                z_min,
                z_sample.as_float_list(),
                initial_state,
                break_z,
                events,
                dense_output,
                atol,
                rtol,
                (
                    model,
                    k_float,
                    supervisor,
                ),
                task_label,
                k_wavenumber.k_inv_Mpc,
            )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'{task_label}: integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc}/Mpc, z_source={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    if mode == "stop" and sol.status != 1:
        # in "stop" mode, we expect the integration to finish at an event; if this doesn't happen, it implies
        # we somehow missed the termination criterion
        raise RuntimeError(
            f'{task_label}: mode is "{mode}", but integration did not finish at a termination event'
        )

    sampled_z = sol.t
    sampled_data = sol.y
    if len(sampled_z) > 0 and len(sampled_data) != EXPECTED_SOL_LENGTH:
        raise RuntimeError(
            f"{task_label}: solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(sampled_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(sampled_z)})"
        )
    if len(sampled_data) > 0:
        sampled_values = sampled_data[VALUE_INDEX]
        sampled_derivs = sampled_data[DERIV_INDEX]
    else:
        sampled_values = []
        sampled_derivs = []

    # if no data points returned, check if this is because the target z (ie., lowest z_response)
    # and the source z agree.
    # If so, then we know the correct value from the initial data.
    if (
        len(sampled_z) == 0
        and len(z_sample) == 0
        and (
            fabs(z_init.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_init.store_id == z_sample.min.store_id
        )
    ):
        sampled_z.append(z_init.z)
        sampled_values.append(0.0)
        sampled_derivs.append(1.0)

    returned_values = len(sampled_z)
    if mode != "stop":
        expected_values = len(z_sample)

        if returned_values != expected_values:
            raise RuntimeError(
                f"{task_label}: solve_ivp returned {returned_values} samples, but expected {expected_values}"
            )

    stop_deltaz_subh = None
    stop_value = None
    stop_deriv = None

    if mode == "stop":
        # find value of solution and derivative at a point of fixed phase. The extremum found is a
        # *maximum* of the solution -- the derivative passes from negative to positive as z
        # decreases -- and review §10.1 measures value/envelope = +1.000000 there in every run.
        # Cutting at a reproducible point of fixed phase is what matters; which extremum it is does
        # not, because the WKB objects' store() rotates arbitrary (value, derivative) initial data
        # into a pure sine. The older motivation, "cut at a minimum to avoid jitter in the phase",
        # is obsolete for that reason.
        #
        # When the sector's effective frequency is available the search steps in phase rather than
        # in relative z, so that its resolution does not depend on how wide the search window is
        # (review §10.2). The window is unchanged.
        payload = find_phase_extremum(
            sol.sol,
            start_z=stop_search_window_z_begin,
            stop_z=stop_search_window_z_end,
            value_index=VALUE_INDEX,
            deriv_index=DERIV_INDEX,
            omega_sq=(
                (lambda z: omega_sq(model, k_float, z))
                if omega_sq is not None
                else None
            ),
        )
        stop_deltaz_subh = k.z_exit - payload["z"]
        stop_value = payload["value"]
        stop_deriv = payload["derivative"]

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"{task_label}: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    # test whether the sample grid the caller supplied is fine enough to resolve the oscillations
    # we stepped through. This used to be done inside the ODE right-hand side, at a cost of 45 % of
    # the run (review §10.2); it is now done once, on the returned samples, which is the grid the
    # flag is actually about (review §13.1). See scan_sample_grid_for_unresolved_osc for the change
    # in what is sampled.
    #
    # warn_unresolved_osc gates the printed warning only -- never the test, and never the three
    # values returned. The two production integrators pass False and main.py summarises the flag
    # per wavenumber instead (README section 7 decision D2, taken by the user 2026-09-11).
    if omega_sq is not None:
        osc_diagnostic = scan_sample_grid_for_unresolved_osc(
            model,
            k_wavenumber,
            k_float,
            omega_sq,
            sampled_z,
            object_label,
            warn=warn_unresolved_osc,
        )
    else:
        # no effective frequency supplied: the diagnostic was not requested, and the flag fields
        # are reported as unknown, exactly as they were when delta_logz was omitted
        osc_diagnostic = {
            "has_unresolved_osc": None,
            "unresolved_z": None,
            "unresolved_efolds_subh": None,
        }

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value_sample": sampled_values,
        "deriv_sample": sampled_derivs,
        "solver_label": "solve_ivp+DOP853-stepping0",
        "has_unresolved_osc": osc_diagnostic["has_unresolved_osc"],
        "unresolved_z": osc_diagnostic["unresolved_z"],
        "unresolved_efolds_subh": osc_diagnostic["unresolved_efolds_subh"],
        "stop_deltaz_subh": stop_deltaz_subh,
        "stop_value": stop_value,
        "stop_deriv": stop_deriv,
    }
