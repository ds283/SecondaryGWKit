"""
The convergence facility for ``prompts/tolerance-convergence/`` (prompt 01, board item T1).

Every accuracy measurement this campaign makes is taken through here, so that the campaign's two
standing rules are the *easy* path rather than a thing each prompt has to remember:

* **A measurement is reported with its own error** (README §5 rule 5). There is no public way to
  obtain a drift figure without also obtaining whether it met its criterion: :func:`reference_drift`
  demands the smallest difference the caller intends to report, evaluates
  ``drift <= smallest_reported_difference / criterion_ratio`` itself, and returns a
  :class:`DriftVerdict` that carries both. ``GkTk-remedial`` prompt 17 reported candidate errors
  through a reference that had not converged on ``QCDModel`` at four wavenumbers; that is the
  failure this module exists to make awkward.
* **A measurement is reported with its grid generation** (README §5 rule 6). A geometry is built
  from a :class:`SourceGridSpec`, which names its generation and carries the sample count and the
  ``redshift_grid_digest`` of the grid it builds. There is no default.

**"One step tighter" is a property of the knob, not of the call site.** A tolerance pair steps by a
decade (:class:`TolerancePair`); a Gauss order steps by one order (:class:`GaussOrder`). Prompt 04
audits four integer orders and has no tolerance to move, so a facility that could only multiply a
float by 0.1 would serve half the campaign.

**The reference setting is the caller's.** Prompt 17's ``(1e-18, 1e-12)`` is an input, not a
default; prompts 03 and 04 choose different ones. Nothing here knows what is in
``config/defaults.py``, recommends a setting, or embeds a target: §6.1's target rule belongs to the
prompts that measure and to the user.

**The anchors.** Where the model is constant-``w`` every quantity this campaign audits has a closed
form, and README §3.1's table lists nine of them. They are exposed here as a registry
(:func:`radiation_anchors`) so that a self-convergence drift can be calibrated against truth at
every use rather than only in prompt 17 -- including the two the table adds that nobody had noticed:
``rho_G == 0`` identically in exact radiation, which makes the Green's-function residual a pure
quadrature-error measurement with no reference to build, and ``z_exit``, which is the elementary
inversion ``1 + z = k/(H0 e^N)``.

**What was folded in.** ``docs/gktk-remedial/tk_numeric_atol_sweep.py`` (``GkTk-remedial`` prompt
17) carried the sector machinery inline: the ``_Wavenumber`` / ``_KExit`` / ``_Proxy`` stand-ins,
the two sectors' geometries and solves, the envelope-relative sampling and the summary. Those are
here now, once, and that script imports them back. Its published figures must not move, and the
acceptance test of prompt 01 is that they do not.

No Ray and no datastore: remotes are called through their undecorated ``_function`` and the
stand-in models are ``ComputeTargets/tests/wkb_reference.py``'s (``CLAUDE.md``;
``GkTk-remedial`` README §5 rule 7).
"""

from dataclasses import dataclass, field
from decimal import Decimal
from math import exp, hypot, sqrt
from statistics import median
from typing import Callable, Optional, Sequence, Tuple

from ComputeTargets.BackgroundModel import BREAK_POINT_DISCONTINUITY
from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS
from ComputeTargets.TkNumericIntegration import RHS as Tk_RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.analytic_Gk import compute_analytic_G, compute_analytic_Gprime
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.tests.wkb_reference import (
    PRODUCTION_K_GRID_INV_MPC,
    PRODUCTION_RESPONSE_SPARSENESS,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_END,
    PRODUCTION_Z_INIT,
    SOURCE_GRID_GENERATIONS,
    SOURCE_GRID_V0,
    SOURCE_GRID_V2,
    difference_error,
    envelope_relative_error,
    horizon_exit_z,
    phase_error,
    production_source_grid,
    source_grid,
    to_redshift_array,
)
from CosmologyConcepts import redshift_array, redshift_grid_digest
from Quadrature.integrators.numeric_with_phase_cut import numeric_with_phase_cut
from Units import Mpc_units

UNITS = Mpc_units()

# main.py:630, :1199 -- part of the numeric call, no longer part of the diagnostic
PRODUCTION_DELTA_LOGZ = 1.0 / 100.0

# The criterion of GkTk-remedial prompt 17 §2.1, and of this campaign's README §0.2: a reference
# has converged when it moves, under one step of tightening, by at least an order of magnitude
# less than the smallest difference the measurement intends to report. It is a *default ratio*,
# not a default threshold -- the threshold is the caller's smallest reported difference divided by
# this, and the caller always supplies that.
CRITERION_RATIO = 10.0

# SciPy clamps rtol from below at 100 * eps, with a warning
# (scipy/integrate/_ivp/common.py:47-51 -- `rtol = np.maximum(rtol, 100 * EPS)`), so a tightening
# below this is ignored and a drift measured across it is measuring nothing. It is why
# GkTk-remedial prompt 17's reference-convergence check is one decade and not two.
# Reported by TolerancePair.rtol_step_is_effective and recorded in every DriftVerdict's notes.
SCIPY_RTOL_FLOOR = 2.220446049250313e-14


# ---------------------------------------------------------------------------------------------
# the knobs, and what "one step tighter" means for each
# ---------------------------------------------------------------------------------------------


class AccuracyKnob:
    """
    The setting whose convergence is being tested, and the rule for stepping it.

    Subclasses answer :meth:`tighter` and :meth:`looser` for themselves, which is the whole point:
    "one step" is a decade for an adaptive solver's tolerance pair and one order for a
    Gauss-Legendre rule, and a call site that had to know which would get it wrong in the sector it
    was not written for.
    """

    def tighter(self, steps: int = 1) -> "AccuracyKnob":
        raise NotImplementedError

    def looser(self, steps: int = 1) -> "AccuracyKnob":
        return self.tighter(-steps)

    @property
    def label(self) -> str:
        raise NotImplementedError

    def ladder(self, steps: int) -> Tuple["AccuracyKnob", ...]:
        """
        This setting and ``steps`` further ones, loose to tight.

        Loose to tight because README §6.1 rule 3 sweeps that way: the target is the *first*
        setting that clears the floor, and a ladder built the other way invites the reader to
        quote the tightest one.
        """
        if steps < 0:
            raise ValueError("AccuracyKnob.ladder: steps must be non-negative")
        return tuple(self.tighter(n) for n in range(steps + 1))

    def __str__(self) -> str:
        return self.label


def _shift_decades(value: float, steps: int) -> float:
    """
    ``value`` divided by ten ``steps`` times, **as a decimal shift**: 1e-18 tightened by a decade
    is 1e-19 and not 1.0000000000000001e-19.

    Binary floating point has no exact tenth, so the naive product walks off the round numbers a
    constant actually gets shipped as -- and an accuracy parameter is part of its object's
    datastore lookup key (README §2 (g)), so a value one ulp away from the one a later prompt
    writes into ``config/defaults.py`` is a different key.
    """
    if value == 0.0:
        return 0.0
    return float(Decimal(repr(float(value))).scaleb(-int(steps)))


@dataclass(frozen=True)
class TolerancePair(AccuracyKnob):
    """
    An adaptive solver's ``(atol, rtol)``. **One step is one decade.**

    ``axis`` says which of the two a step moves: ``"both"`` is what ``GkTk-remedial`` prompt 17's
    reference-convergence check does and what review §10.1's table does; ``"atol"`` and ``"rtol"``
    are the decoupled axes prompt 03 needs, because prompt 17 separated them in one sector and
    review §10.1 moved both at once in the other, so "the error is set by ``rtol``" has never been
    tested cleanly where it matters (README §2 (d)).
    """

    atol: float
    rtol: float
    axis: str = "both"

    DECADE = 10.0

    def __post_init__(self):
        if self.axis not in ("both", "atol", "rtol"):
            raise ValueError(
                f"TolerancePair: axis must be 'both', 'atol' or 'rtol', not {self.axis!r}"
            )

    def tighter(self, steps: int = 1) -> "TolerancePair":
        return TolerancePair(
            atol=(
                _shift_decades(self.atol, steps)
                if self.axis in ("both", "atol")
                else self.atol
            ),
            rtol=(
                _shift_decades(self.rtol, steps)
                if self.axis in ("both", "rtol")
                else self.rtol
            ),
            axis=self.axis,
        )

    @property
    def label(self) -> str:
        return f"(atol={self.atol:.3g}, rtol={self.rtol:.3g})"

    @property
    def rtol_step_is_effective(self) -> bool:
        """
        Whether ``solve_ivp`` would actually see this ``rtol``, or clamp it at
        :data:`SCIPY_RTOL_FLOOR` (``100 * eps``). A drift measured across a clamped step is zero
        by construction and says nothing about convergence.
        """
        return self.rtol >= SCIPY_RTOL_FLOOR


@dataclass(frozen=True)
class GaussOrder(AccuracyKnob):
    """
    The order of a fixed-order Gauss-Legendre rule -- ``TAU_GAUSS_ORDER``, ``CS_TAU_GAUSS_ORDER``,
    ``FRICTION_F_GAUSS_ORDER``, ``RHO_GAUSS_ORDER``, all 4 today. **One step is one order.**

    Four of the eight keyed object types have no tolerance to converge (README §2 (a)); this is
    the knob they do have, and it is an integer, so the step is ``+1`` and not a factor.
    """

    order: int
    name: str = "N"

    def __post_init__(self):
        if int(self.order) < 1:
            raise ValueError(f"GaussOrder: order must be >= 1, not {self.order!r}")

    def tighter(self, steps: int = 1) -> "GaussOrder":
        return GaussOrder(order=int(self.order) + int(steps), name=self.name)

    @property
    def label(self) -> str:
        return f"{self.name} = {int(self.order)}"


# ---------------------------------------------------------------------------------------------
# the error measures, and the summary of a sampled error
#
# The three definitions are wkb_reference.py's, re-exported rather than restated (prompt 01 §2.1
# item 5). They are GkTk-remedial README §6's and are not this campaign's to change.
# ---------------------------------------------------------------------------------------------

ERROR_MEASURES = {
    "envelope_relative": envelope_relative_error,
    "phase": phase_error,
    "difference": difference_error,
}


def summarise(errors) -> dict:
    """
    Maximum (and where it fell), second-largest, median, and the value at the **last** returned
    sample, of a list of ``(error, z, x)``.

    Folded unchanged from ``docs/gktk-remedial/tk_numeric_atol_sweep.py`` (prompt 17), whose
    docstring records why the terminal value is here and not in that prompt's list: the samples
    are in descending z, so the last entry is the deepest point reached, and it is the part of the
    run ``TkWKBIntegration`` reads as its initial condition -- which is what separates one bad
    sample from one bad step whose consequence is carried to the end.
    """
    ordered = sorted(errors, key=lambda e: e[0], reverse=True)
    worst = ordered[0]
    return {
        "max": worst[0],
        "max_z": worst[1],
        "max_x": worst[2],
        # a one-sample list has no second-largest; it reports the same value rather than raising,
        # which the folded version did. Nothing that fold has to reproduce ever sees one -- the
        # sectors sample hundreds -- but a Gauss order scored at a single node does
        "second": ordered[1][0] if len(ordered) > 1 else worst[0],
        "median": median(e[0] for e in errors),
        "terminal": errors[-1][0],
        "terminal_x": errors[-1][2],
        "samples": len(errors),
    }


# ---------------------------------------------------------------------------------------------
# the drift statistic, and the criterion evaluated
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftVerdict:
    """
    How far a reference moves under one step of tightening, **and whether that is small enough**.

    The two are one object on purpose. README §5 rule 5: a number quoted against a converged
    reference is quoted with that reference's drift beside it, and no conclusion is drawn from a
    signal that does not exceed it. A caller cannot get :attr:`max` out of this facility without
    also getting :attr:`passed` and :attr:`threshold`.
    """

    knob: AccuracyKnob
    tightened: AccuracyKnob
    drift: dict
    smallest_reported_difference: float
    criterion_ratio: float
    notes: Tuple[str, ...] = ()

    @property
    def threshold(self) -> float:
        """The largest drift this measurement may carry: the smallest reported difference / 10."""
        return self.smallest_reported_difference / self.criterion_ratio

    @property
    def passed(self) -> bool:
        return self.drift["max"] <= self.threshold

    @property
    def max(self) -> float:
        return self.drift["max"]

    @property
    def median(self) -> float:
        return self.drift["median"]

    @property
    def max_z(self) -> float:
        return self.drift["max_z"]

    @property
    def headroom(self) -> float:
        """How many times below the threshold the drift sits; < 1 means the criterion is missed."""
        return (
            self.threshold / self.drift["max"]
            if self.drift["max"] > 0.0
            else float("inf")
        )

    def __str__(self) -> str:
        verdict = "converged" if self.passed else "NOT CONVERGED"
        return (
            f"{verdict}: drift {self.drift['max']:.3g} at z={self.drift['max_z']:.5g} against a "
            f"threshold of {self.threshold:.3g} "
            f"({self.knob.label} -> {self.tightened.label})"
        )


def reference_drift(
    build: Callable[[AccuracyKnob], object],
    knob: AccuracyKnob,
    *,
    error_measure: Callable[[object, object], Sequence],
    smallest_reported_difference: float,
    criterion_ratio: float = CRITERION_RATIO,
    reference=None,
    tightened_knob: Optional[AccuracyKnob] = None,
) -> DriftVerdict:
    """
    Build the reference at ``knob``, build it again one step tighter, and score the difference.

    ``smallest_reported_difference`` is **required and has no default**: it is the smallest
    difference the caller intends to report, and without it there is no criterion, only a number.
    That is the whole of README §5 rule 5 expressed as a signature.

    :param build: ``build(knob)`` returns whatever the caller's error measure consumes -- a solver
        payload, a table, a phase -- for that setting
    :param knob: the reference setting. It is the caller's; nothing here has a preferred one
    :param error_measure: ``error_measure(candidate, reference)`` returns a list of
        ``(error, z, x)`` in the caller's own measure
    :param reference: the reference payload, if the caller has already built it at ``knob``
    :param tightened_knob: the one-step-tighter setting, if the caller wants a different step from
        the knob's own (prompt 03 moves one tolerance axis at a time)
    :return: a :class:`DriftVerdict`, which carries the numbers and the verdict together
    """
    if smallest_reported_difference <= 0.0:
        raise ValueError(
            "reference_drift: smallest_reported_difference must be positive -- it is the smallest "
            "difference this measurement intends to report, and the convergence criterion is a "
            "tenth of it (README §5 rule 5)"
        )

    tightened = knob.tighter() if tightened_knob is None else tightened_knob

    notes = []
    for label, candidate in (("reference", knob), ("tightened reference", tightened)):
        if (
            isinstance(candidate, TolerancePair)
            and not candidate.rtol_step_is_effective
        ):
            notes.append(
                f"the {label}'s rtol = {candidate.rtol:.3g} is below SciPy's clamp of "
                f"{SCIPY_RTOL_FLOOR:.3g} and is silently ignored"
            )

    if reference is None:
        reference = build(knob)
    tightened_payload = build(tightened)

    return DriftVerdict(
        knob=knob,
        tightened=tightened,
        drift=summarise(error_measure(tightened_payload, reference)),
        smallest_reported_difference=float(smallest_reported_difference),
        criterion_ratio=float(criterion_ratio),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class ConvergedReference:
    """
    A reference for one (target, model, k, geometry), **with the evidence that it is one**.

    :attr:`payload` is the reference; :attr:`drift` is the :class:`DriftVerdict` that says whether
    anything measured through it means anything. ``bool(reference)`` is the verdict.
    """

    knob: AccuracyKnob
    payload: object
    drift: DriftVerdict
    error_measure: Callable[[object, object], Sequence] = field(repr=False)

    @property
    def converged(self) -> bool:
        return self.drift.passed

    def __bool__(self) -> bool:
        return self.drift.passed

    def errors(self, candidate) -> Sequence:
        """The caller's error measure, of ``candidate`` against this reference."""
        return self.error_measure(candidate, self.payload)

    def score(self, candidate) -> dict:
        """:func:`summarise` of :meth:`errors`, with the reference's own drift beside it."""
        out = summarise(self.errors(candidate))
        out["reference_drift"] = self.drift.max
        out["reference_converged"] = self.drift.passed
        return out


def converged_reference(
    build: Callable[[AccuracyKnob], object],
    knob: AccuracyKnob,
    *,
    error_measure: Callable[[object, object], Sequence],
    smallest_reported_difference: float,
    criterion_ratio: float = CRITERION_RATIO,
    tightened_knob: Optional[AccuracyKnob] = None,
) -> ConvergedReference:
    """
    Build a reference at the caller's setting and measure its drift in one call.

    See :func:`reference_drift` for the arguments. The reference is built once and reused, so the
    cost is two solves rather than three.
    """
    payload = build(knob)
    drift = reference_drift(
        build,
        knob,
        error_measure=error_measure,
        smallest_reported_difference=smallest_reported_difference,
        criterion_ratio=criterion_ratio,
        reference=payload,
        tightened_knob=tightened_knob,
    )
    return ConvergedReference(
        knob=knob, payload=payload, drift=drift, error_measure=error_measure
    )


# ---------------------------------------------------------------------------------------------
# the anchors: the closed forms a constant-w model provides
#
# README §3.1's table, all nine rows. The facility exposes them; the prompt that measures decides
# which apply. A drift quoted for a quantity in this table without the oracle error beside it is
# uncalibrated (§5 rule 5), and this is the mechanism that makes obeying that the easy path.
#
# THE ONE VALIDITY BOUND: rho_T's primitive needs omega_T^2 > 0, i.e. 1 + z < k/(sqrt6 H0). A
# caller that walks it outside that bound has chosen its z_init wrongly; RadiationModel raises
# rather than returning a complex root, and that raise is not a finding about the representation.
# ---------------------------------------------------------------------------------------------


def exact_z_exit(H0: float, k_inv_Mpc: float, efolds_subh: float = 0.0) -> float:
    """
    The exact horizon-crossing redshift on an exact-radiation background:
    ``k(1+z)/H = k/(H0 s)`` when ``H = H0 s^2``, so ``1 + z = k/(H0 e^N)``.

    README §3.1 consequence 3: board item T6 records ``wavenumber_exit_time``'s root solve as
    "never measured", and the reason it was never measured is that nobody noticed the radiation
    case is a one-line inversion. It is not an approximation and there is no reference to build.
    """
    return k_inv_Mpc / (H0 * exp(efolds_subh)) - 1.0


def analytic_G(model, k_inv_Mpc: float, z_source: float, z: float) -> float:
    """``compute_analytic_G`` at the model's own constant ``w``, in the harness's variables."""
    w = model.functions.wBackground(z)
    return compute_analytic_G(
        k_inv_Mpc,
        w,
        model.tau(z_source),
        model.tau(z),
        model.functions.Hubble(z_source),
    )


def analytic_Gprime(model, k_inv_Mpc: float, z_source: float, z: float) -> float:
    """``compute_analytic_Gprime`` likewise."""
    w = model.functions.wBackground(z)
    return compute_analytic_Gprime(
        k_inv_Mpc,
        w,
        model.tau(z_source),
        model.tau(z),
        model.functions.Hubble(z_source),
        model.functions.Hubble(z),
    )


def analytic_T(model, k_inv_Mpc: float, z: float) -> float:
    """
    ``compute_analytic_T``: ``2^n Gamma(n+1) (k c_s tau)^-n J_n(k c_s tau)``, ``n = 3/2 + b``,
    reducing at ``w = 1/3`` to ``3(sin x - x cos x)/x^3``.
    """
    return compute_analytic_T(
        k_inv_Mpc, model.functions.wPerturbations(z), model.tau(z)
    )


def analytic_Tprime(model, k_inv_Mpc: float, z: float) -> float:
    """``compute_analytic_Tprime`` likewise."""
    return compute_analytic_Tprime(
        k_inv_Mpc,
        model.functions.wPerturbations(z),
        model.tau(z),
        model.functions.Hubble(z),
    )


def radiation_anchors(model) -> dict:
    """
    Every closed form README §3.1's table lists, bound to an exact-radiation control.

    The keys are the table's rows. ``value`` anchors take ``(k, z)`` or ``(k, z_source, z)``;
    ``interval`` anchors take ``(z_a, z_b)`` and are the ones that matter for the order-governed
    targets, because §6's difference error is relative to the interval and never to the absolute
    primitive -- ``tau_delta`` and ``cs_tau_delta`` carry ~1e-16 against ~5e-15 for the naive
    difference of two primitives (README §3.1 consequence 2).

    Two rows are the reason this registry exists rather than a pair of oracles:

    * ``rho_G`` is **identically zero** (``C == 0`` in exact radiation), so the Green's-function
      residual is a pure quadrature-error measurement with no reference to build at all;
    * ``z_exit`` is the elementary inversion :func:`exact_z_exit`, and board item T6 says that
      quantity has never been measured.

    :param model: a ``wkb_reference.RadiationModel``
    :return: ``{name: (kind, callable)}``
    """
    if not hasattr(model, "tau_delta"):
        raise TypeError(
            f"radiation_anchors: {type(model).__name__} is not an exact constant-w control; the "
            "closed forms of README §3.1 exist on RadiationModel and on nothing else in the "
            "harness"
        )

    return {
        "T": ("value", lambda k, z: analytic_T(model, k, z)),
        "Tprime": ("value", lambda k, z: analytic_Tprime(model, k, z)),
        "G": ("value", lambda k, z_source, z: analytic_G(model, k, z_source, z)),
        "Gprime": (
            "value",
            lambda k, z_source, z: analytic_Gprime(model, k, z_source, z),
        ),
        "tau": ("interval", model.tau_delta),
        "cs_tau": ("interval", model.cs_tau_delta),
        "friction_F": ("interval", model.friction_F_delta),
        "theta_G": ("phase", model.theta_G),
        "rho_G": ("phase", model.rho_G),
        "rho_T": ("phase", model.rho_T),
        "z_exit": ("value", lambda k, N=0.0: exact_z_exit(model.H0, k, N)),
    }


def anchor_error(
    kind: str, value: float, exact: float, envelope: float = None
) -> float:
    """
    The error of ``value`` against a closed form, in the measure that row of §3.1's table calls
    for: ``interval`` rows use :func:`difference_error`, ``phase`` rows :func:`phase_error`, and
    ``value`` rows :func:`envelope_relative_error` and therefore need an envelope.
    """
    if kind == "interval":
        return difference_error(value, exact)
    if kind == "phase":
        return phase_error(value, exact)
    if kind == "value":
        if envelope is None:
            raise ValueError(
                "anchor_error: a 'value' anchor is scored envelope-relative (GkTk-remedial "
                "README §6), so an envelope is required"
            )
        return envelope_relative_error(value, exact, envelope)
    raise ValueError(f"anchor_error: unknown anchor kind {kind!r}")


# ---------------------------------------------------------------------------------------------
# the two numeric sectors: stand-ins, geometry, solve, sampled error
#
# Folded from docs/gktk-remedial/tk_numeric_atol_sweep.py (GkTk-remedial prompt 17, extended by
# prompts 18-20), which is now a caller. The only thing that changed in the fold is that a
# geometry has to be told which source-grid generation it is built on.
# ---------------------------------------------------------------------------------------------


class _Wavenumber:
    def __init__(self, k: float, store_id: int, units):
        self.k = float(k)
        self.k_inv_Mpc = float(k)
        self.store_id = store_id
        self.units = units


class _KExit:
    """A ``wavenumber_exit_time`` stand-in: ``.k`` and ``.z_exit``."""

    def __init__(self, k: float, units, z_exit: float, store_id: int = 1):
        self.k = _Wavenumber(k, store_id, units)
        self.z_exit = z_exit


class _Proxy:
    """A ``ModelProxy`` stand-in: ``.get()`` and ``.units`` (for ``check_units``)."""

    def __init__(self, model, units):
        self._model = model
        self.units = units

    def get(self):
        return self._model


class _GridCosmologyView:
    """
    A cosmology view that can answer ``wPerturbations(z)``, for a stand-in that keeps it on
    ``.functions`` instead.

    ``pre_grid_background_proxy`` needs ``Hubble`` and ``wPerturbations`` *on the cosmology*, and
    ``wkb_reference.RadiationModel`` -- which is its own cosmology -- exposes ``wPerturbations``
    only through ``functions``. This forwards that one accessor and nothing else, so a model that
    declares break points still declares them. It is not a stand-in cosmology: it never supplies a
    value the wrapped object does not already have.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        inner = object.__getattribute__(self, "_inner")
        if name == "wPerturbations" and not hasattr(inner, "wPerturbations"):
            return inner.functions.wPerturbations
        return getattr(inner, name)


def grid_cosmology(cosmology):
    """``cosmology``, wrapped only if it cannot answer ``wPerturbations`` itself."""
    if hasattr(cosmology, "wPerturbations"):
        return cosmology
    return _GridCosmologyView(cosmology)


@dataclass(frozen=True)
class SourceGridSpec:
    """
    Which source grid a geometry is built on, **named**, and what it turned out to be.

    README §5 rule 6: a figure that does not say whether it was taken on version 0, 1 or 2 of the
    source grid cannot be compared with any other figure in the record. This carries the
    generation, and after :meth:`build` it carries the sample count and the
    ``redshift_grid_digest`` as well, so a table can print them beside the number.

    ``universal`` is the second axis, and it is not the same question as the generation:

    * ``universal=False`` is what ``GkTk-remedial`` prompt 17 did -- one lattice per wavenumber,
      run from that wavenumber's own ``z_source``. It is only available on version 0, which is
      the only construction that consults no cosmology, and it is what reproduces that prompt's
      published figures;
    * ``universal=True`` is what ``main.py`` does -- **one** grid per model, built from the
      earliest-exiting wavenumber's ``z_exit_suph_e5``, then truncated per work item
      (``main.py:1209-1211``). Versions 1 and 2 are built around what the cosmology declares and
      exist only in this form.
    """

    generation: str
    universal: bool
    z_init: float = PRODUCTION_Z_INIT
    z_end: float = PRODUCTION_Z_END
    samples_per_log10z: int = PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z
    k_inv_Mpc: Sequence[float] = tuple(float(k) for k in PRODUCTION_K_GRID_INV_MPC)

    def __post_init__(self):
        if self.generation not in SOURCE_GRID_GENERATIONS:
            raise ValueError(
                f"SourceGridSpec: unknown generation {self.generation!r}; the named generations "
                f"are {', '.join(repr(g) for g in SOURCE_GRID_GENERATIONS)}"
            )
        if self.generation != SOURCE_GRID_V0 and not self.universal:
            raise ValueError(
                f"SourceGridSpec: generation {self.generation!r} is built around what the "
                "cosmology declares over the whole grid, so it exists only as the universal grid "
                "main.py builds; universal=False is a version-0 construction only"
            )

    @property
    def label(self) -> str:
        return f"{self.generation} {'universal' if self.universal else 'per-k'}"

    def build(self, cosmology) -> "BuiltSourceGrid":
        """
        The universal grid for this cosmology, once. On a per-k spec there is no universal grid
        and :meth:`BuiltSourceGrid.for_wavenumber` builds each one as it is asked for.
        """
        return BuiltSourceGrid(self, cosmology)


class BuiltSourceGrid:
    """The grid a :class:`SourceGridSpec` describes, built for one cosmology."""

    def __init__(self, spec: SourceGridSpec, cosmology):
        self.spec = spec
        self.cosmology = cosmology

        if not spec.universal:
            self.z_values = None
            self.grid = None
            return

        grid = source_grid(
            spec.generation,
            spec.z_init,
            spec.z_end,
            spec.samples_per_log10z,
            cosmology=(
                None if spec.generation == SOURCE_GRID_V0 else grid_cosmology(cosmology)
            ),
            k_inv_Mpc=spec.k_inv_Mpc,
        )
        self.z_values = grid.z_values
        self.grid = to_redshift_array(grid.z_values)

    @property
    def generation(self) -> str:
        return self.spec.generation

    @property
    def samples(self) -> Optional[int]:
        return None if self.z_values is None else len(self.z_values)

    @property
    def digest(self) -> Optional[str]:
        return None if self.z_values is None else redshift_grid_digest(self.z_values)

    @property
    def label(self) -> str:
        if self.z_values is None:
            return f"{self.spec.label} (built per wavenumber)"
        return f"{self.spec.label}, {self.samples} samples, digest {self.digest}"

    def for_wavenumber(self, z_source: float) -> redshift_array:
        """
        The source grid one work item is sampled on, before the sector's own lower truncation.

        On a universal spec this is ``main.py:1209``'s ``truncate(z_exit_suph_e5, keep='lower')``.
        On the per-k version-0 spec it is prompt 17's own lattice from ``z_source`` down, which is
        what reproduces that prompt's figures.
        """
        if self.grid is None:
            return production_source_grid(
                z_source, self.spec.z_end, self.spec.samples_per_log10z
            )
        return self.grid.truncate(z_source, keep="lower")


#: prompt 17's geometry: one bare ``logspace`` per wavenumber. Every published tolerance figure in
#: this repository was scored on it, and it is what the construction check of prompt 01 §4.1
#: reproduces. It is **not** what production builds.
V0_PER_K_GRID = SourceGridSpec(generation=SOURCE_GRID_V0, universal=False)

#: the grid ``main.py`` builds: ``SOURCE_GRID_CONSTRUCTION_VERSION = 2``, one per model.
V2_PRODUCTION_GRID = SourceGridSpec(generation=SOURCE_GRID_V2, universal=True)


def _horizon_geometry(cosmology, k_inv_Mpc: float) -> dict:
    z_exit = horizon_exit_z(cosmology, k_inv_Mpc, 0.0)
    return {
        "z_exit": z_exit,
        "z_e3": horizon_exit_z(cosmology, k_inv_Mpc, 3.0),
        "z_e6": horizon_exit_z(cosmology, k_inv_Mpc, 6.0),
        "z_source": horizon_exit_z(
            cosmology, k_inv_Mpc, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
        ),
    }


def tk_geometry(cosmology, k_inv_Mpc: float, grid: BuiltSourceGrid) -> dict:
    """
    ``main.py``'s ``build_Tk_numeric_work`` geometry for one wavenumber: the source grid from five
    e-folds outside the horizon, truncated below at ``0.85 z_e6``, with the ``(z_e3, z_e6)`` stop
    window (``main.py:1197-1211``).

    ``cosmology`` is anything with ``Hubble(z)`` and ``H0`` -- the ``RadiationModel`` stand-in
    itself, or the real cosmology behind ``LambdaCDMModel`` / ``QCDModel``. ``grid`` says which
    generation the sample locations come from and **there is no default**.
    """
    geo = _horizon_geometry(cosmology, k_inv_Mpc)
    geo["grid"] = grid.for_wavenumber(geo["z_source"]).truncate(
        0.85 * geo["z_e6"], keep="higher-include"
    )
    geo["grid_spec"] = grid
    return geo


def gk_geometry(cosmology, k_inv_Mpc: float, grid: BuiltSourceGrid) -> dict:
    """
    ``main.py``'s ``build_Gk_numeric_work`` geometry for one object: the source redshift five
    e-folds outside the horizon, the **response** grid -- ``winnow(12)`` of the source grid, which
    is what ``GkNumericIntegration`` is sampled on -- cut to that source redshift above and to
    ``0.85 z_e6`` below, and the ``(z_e3, z_e6)`` stop window.

    ``GkNumericIntegration`` is one object per ``(k, z_source)``; one source redshift per k is
    taken here, the outermost, which is the longest and therefore the least favourable run.
    """
    geo = _horizon_geometry(cosmology, k_inv_Mpc)
    source = grid.for_wavenumber(geo["z_source"])
    geo["grid"] = (
        source.winnow(sparseness=PRODUCTION_RESPONSE_SPARSENESS)
        .truncate(source.max, keep="lower")
        .truncate(0.85 * geo["z_e6"], keep="higher-include")
    )
    geo["grid_spec"] = grid
    return geo


def gk_geometry_at_source(
    cosmology, k_inv_Mpc: float, grid: BuiltSourceGrid, z_source: float
) -> dict:
    """
    :func:`gk_geometry` at a source redshift the **caller** supplies, instead of the outermost one.

    ``GkNumericIntegration`` is one object per ``(k, z_source)``, and :func:`gk_geometry` takes one
    source redshift per wavenumber -- the outermost, five e-folds outside the horizon -- on the
    stated grounds that it is the longest and therefore the least favourable run. That is a
    *claim*, and prompt 03 of ``prompts/tolerance-convergence`` (§2.2) is required to test it
    rather than inherit it: if it holds, the whole ~65,000-object sector is bounded by fifty runs
    per model, which is a stronger result than any sub-sample of the second axis would give.

    Everything else is :func:`gk_geometry`'s, including the horizon geometry and the
    ``(z_e3, z_e6)`` stop window, which do not depend on ``z_source``; only the top of the response
    grid moves. ``z_source`` is expected to lie on the source grid, as production's does
    (``main.py:1770-1791`` iterates over the source-grid redshifts), but nothing here requires it.

    :param z_source: the source redshift, in the interval where ``main.py`` builds this target --
        above ``z_exit_subh_e4`` and at or below the outermost source redshift
    """
    geo = _horizon_geometry(cosmology, k_inv_Mpc)
    geo["z_source"] = float(z_source)
    source = grid.for_wavenumber(geo["z_source"])
    geo["grid"] = (
        source.winnow(sparseness=PRODUCTION_RESPONSE_SPARSENESS)
        .truncate(source.max, keep="lower")
        .truncate(0.85 * geo["z_e6"], keep="higher-include")
    )
    geo["grid_spec"] = grid
    return geo


def _numeric_run(
    model,
    k_inv_Mpc: float,
    geo: dict,
    atol: float,
    rtol: float,
    *,
    RHS,
    omega_sq,
    initial_value: float,
    initial_deriv: float,
    task_label: str,
    object_label: str,
    break_point_kind: str,
) -> dict:
    grid = geo["grid"]
    z_init = grid.max
    return numeric_with_phase_cut._function(
        _Proxy(model, UNITS),
        _KExit(k_inv_Mpc, UNITS, geo["z_exit"]),
        z_init,
        grid,
        initial_value=initial_value,
        initial_deriv=initial_deriv,
        RHS=RHS,
        omega_sq=omega_sq,
        atol=atol,
        rtol=rtol,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        mode="stop",
        stop_search_window_z_begin=min(geo["z_e3"], z_init.z),
        stop_search_window_z_end=geo["z_e6"],
        task_label=task_label,
        object_label=object_label,
        warn_unresolved_osc=False,
        break_point_kind=break_point_kind,
    )


def tk_run(
    model,
    k_inv_Mpc: float,
    geo: dict,
    atol: float,
    rtol: float,
    ic=None,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,
    task_label: str = "tk_numeric_atol_sweep",
) -> dict:
    """
    One ``TkNumericIntegration`` solve through the undecorated ``numeric_with_phase_cut``.

    ``ic`` is ``(T, dT/dz)`` at the top of the grid; ``None`` means the production
    ``T = 1, T' = 0``.

    ``break_point_kind`` defaults to ``BREAK_POINT_DISCONTINUITY`` -- what
    ``numeric_with_phase_cut`` asked for unconditionally when ``TK-NUMERIC-ATOL-SWEEP.md`` §9 was
    measured, so that §9's entry point reproduces §9. **The production ``TkNumericIntegration``
    call site passes ``BREAK_POINT_ALL``** since ``GkTk-remedial`` prompt 19, and a prompt
    measuring this sector on the tree as it stands has to say so.
    """
    value, deriv = (1.0, 0.0) if ic is None else ic
    return _numeric_run(
        model,
        k_inv_Mpc,
        geo,
        atol,
        rtol,
        RHS=Tk_RHS,
        omega_sq=Tk_omegaEff_sq,
        initial_value=value,
        initial_deriv=deriv,
        task_label=task_label,
        object_label="Tk(z)",
        break_point_kind=break_point_kind,
    )


def gk_run(
    model,
    k_inv_Mpc: float,
    geo: dict,
    atol: float,
    rtol: float,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,
    task_label: str = "gk_break_point_sweep",
) -> dict:
    """
    One ``GkNumericIntegration`` solve through the undecorated ``numeric_with_phase_cut``.

    ``break_point_kind`` is this sector's production policy *and* the module default; it is named
    here for the same reason the production call site names it (``GkTk-remedial`` prompt 19).
    """
    return _numeric_run(
        model,
        k_inv_Mpc,
        geo,
        atol,
        rtol,
        RHS=Gk_RHS,
        omega_sq=Gk_omegaEff_sq,
        initial_value=0.0,
        initial_deriv=1.0,
        task_label=task_label,
        object_label="Gr_k(z, z')",
        break_point_kind=break_point_kind,
    )


SECTORS = {
    "Tk": {"geometry": tk_geometry, "run": tk_run, "omega_sq": Tk_omegaEff_sq},
    "Gk": {"geometry": gk_geometry, "run": gk_run, "omega_sq": Gk_omegaEff_sq},
}


def x_local(model, k_inv_Mpc: float, z: float) -> float:
    """
    The transfer function's dimensionless phase variable, ``x = k c_s (1+z)/H``.

    In exact radiation this is ``k c_s tau`` identically; on a real background it is the same
    quantity evaluated locally. It is used only to *label* where an error falls.
    """
    functions = model.functions
    c_s = sqrt(functions.wPerturbations(z))
    return k_inv_Mpc * c_s * (1.0 + z) / functions.Hubble(z)


def sector_errors(sector: str, model, k_inv_Mpc: float, geo, candidate, reference):
    """
    Envelope-relative error of ``candidate`` against ``reference``, sample by sample, for either
    numeric sector.

    ``GkTk-remedial`` README §6: the denominator is the local Liouville-Green envelope
    ``hypot(T, T'/omega)`` of the **reference** run, with ``omega`` the sector's own effective
    frequency; samples where ``omega^2 <= 0`` are skipped, the mode not being oscillatory there.

    :return: list of ``(error, z, x)``, in the returned (descending z) order
    """
    omega_sq_fn = SECTORS[sector]["omega_sq"]
    out = []
    for z, value, ref_value, ref_deriv in zip(
        geo["grid"],
        candidate["value_sample"],
        reference["value_sample"],
        reference["deriv_sample"],
    ):
        omega_sq = omega_sq_fn(model, k_inv_Mpc, z.z)
        if omega_sq <= 0.0:
            continue
        envelope = hypot(ref_value, ref_deriv / sqrt(omega_sq))
        out.append(
            (
                envelope_relative_error(value, ref_value, envelope),
                z.z,
                x_local(model, k_inv_Mpc, z.z),
            )
        )
    return out


def sector_error_measure(sector: str, model, k_inv_Mpc: float, geo):
    """
    :func:`sector_errors` as the two-argument ``error_measure(candidate, reference)`` that
    :func:`reference_drift` and :func:`converged_reference` take.
    """
    return lambda candidate, reference: sector_errors(
        sector, model, k_inv_Mpc, geo, candidate, reference
    )


def sector_reference(
    sector: str,
    model,
    cosmology,
    k_inv_Mpc: float,
    grid: BuiltSourceGrid,
    knob: TolerancePair,
    *,
    smallest_reported_difference: float,
    criterion_ratio: float = CRITERION_RATIO,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,
    geo: Optional[dict] = None,
) -> Tuple[dict, ConvergedReference]:
    """
    The whole of §0.2's self-convergence test for one (sector, model, k, grid generation), in one
    call: build the geometry, build the reference at the caller's ``knob``, build it one decade
    tighter, and return the geometry beside a :class:`ConvergedReference`.

    :return: ``(geo, reference)``. ``geo['grid_spec'].label`` is the grid generation the figures
        must be reported with (README §5 rule 6)
    """
    if geo is None:
        geo = SECTORS[sector]["geometry"](cosmology, k_inv_Mpc, grid)
    run_fn = SECTORS[sector]["run"]

    def build(setting: TolerancePair):
        return run_fn(
            model,
            k_inv_Mpc,
            geo,
            setting.atol,
            setting.rtol,
            break_point_kind=break_point_kind,
        )

    return geo, converged_reference(
        build,
        knob,
        error_measure=sector_error_measure(sector, model, k_inv_Mpc, geo),
        smallest_reported_difference=smallest_reported_difference,
        criterion_ratio=criterion_ratio,
    )
