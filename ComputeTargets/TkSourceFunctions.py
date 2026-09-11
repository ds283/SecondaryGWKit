"""
A two-region representation of the transfer function T_k(z) for downstream consumers.

This is the transfer-function analogue of `GkSourceFunctions`
(`ComputeTargets/GkSourcePolicyData.py:19-32`, built in `_create_functions()` at
`GkSourcePolicyData.py:573-702`), with one simplification: the numeric -> WKB hand-over for
the transfer function is a *single, already determined* redshift, so there is no overlap
region and no policy object to choose a crossover.

`main.py:505-507` integrates `TkNumericIntegration` in `mode="stop"` from `z_exit_suph_e5` down
to a phase minimum inside the window `[z_exit_subh_e3, 0.85*z_exit_subh_e6]`, and
`main.py:686-700` starts `TkWKBIntegration` at exactly that stop point,
`z_init = k_exit.z_exit - Tk.stop_deltaz_subh`, with `T_init = Tk.stop_T`,
`Tprime_init = Tk.stop_Tprime`. The hand-over redshift is therefore recoverable from either
object, and this class cross-checks the two readings against each other.

Nothing here is a new computed result: every number is a re-reading of stored
`TkNumericIntegration`/`TkWKBIntegration` values, plus closed-form model functions. The object
is deliberately **not** a compute target and is not persisted; persisting it would only
duplicate rows.

Three regions, as z descends:

  * above the numeric region, T = 1 and dT/dz = 0 exactly (spec 01 section 2.8; this is the
    default `compute_quad_source` already substitutes at `QuadSource.py:116-121`);
  * the *numeric* region, where T and dT/dz are smooth and are supplied as splines in
    log(1+z) through the stored `TkNumericValue.T`, `.Tprime` samples;
  * the *WKB* (Liouville-Green) region, where T = M(z) sin(theta(z)) and the consumer is given
    the amplitude M, its logarithmic derivative d ln M/dz (closed form; see below), the phase
    theta as a `PrimitivePhase`, and d(theta)/dz = omega_eff (closed form).

Amplitude
---------

From `TkWKBIntegration.store()` (`TkWKBIntegration.py:491-503`), with `cos_coeff = 0`
(set at `TkWKBIntegration.py:443`),

    M(z) = sin_coeff * sqrt(H_init / H(z)) * omega_eff(z)^(-1/2) * exp(F(z))

with H_init = H(crossover_z) (the `H_ratio` stored per value is exactly H_init/H),
omega_eff^2 = `WKB_Tk.Tk_omegaEff_sq(model, k, z)` and F the Liouville-Green friction integral,
which the background model tabulates once per cosmology as `model.functions.friction_F`. F is
read from that table as

    F(z) = friction_F.delta(crossover_z, z),

exact to ~1e-14 absolute (prompts/GkTk-remedial log 04) and **not splined**: every factor of M
is now a closed form or a table lookup, so nothing in the amplitude or in its derivative
differentiates a spline. The stored `friction` samples are still read, but only to cross-check
the table at construction: they must agree to `FRICTION_CROSS_CHECK_RTOL` of the largest |F| on
the sampled range, which they do bit-for-bit for a `TkWKBIntegration` written by prompt 07 of
`prompts/GkTk-remedial`. A datastore written by the *retired* friction ODE disagrees by
2.261e-07 absolute in F (log 04) and is refused with a message saying so.

The friction integrand is dF/dz = (3/2)(1 + c_s^2)/(1+z) with c_s^2 = `wPerturbations(z)` --
exactly the integrand of spec 01 R23. (It used to be the right-hand side of a per-object ODE in
`TkWKBIntegration`, which prompt 07 retired; that right-hand side survives verbatim in
`ComputeTargets/tests/test_background_cs_tau_friction.py`, where prompt 04's
`TestFrictionODEComparison` measures what it cost.) With d ln H/dz = epsilon/(1+z),
R23/R24 give the closed form used by `dlnM_dz`:

    d ln M / dz = - epsilon(z) / (2(1+z))
                  - (1/2) d ln omega_eff / dz
                  + (3/2) (1 + c_s^2(z)) / (1+z)

the three terms being d ln sqrt(H_init/H)/dz, the derivative of omega_eff^(-1/2), and dF/dz
respectively.

Phase
-----

The phase convention is the code's: d(theta)/dz = +omega_eff integrated towards *smaller* z
from theta(z_init) = 0, so theta is negative and *decreasing* as z falls -- equivalently
theta is an increasing function of z (audit `TK-report.md` TK-8(a)).

The phase is a `ComputeTargets.primitive_phase.PrimitivePhase`, not a cubic spline of the
stored theta samples. A cubic spline of theta itself carries h^4 x_T/384 (review section 5, section
12.6): 3.5e-3 rad at k = 1e5/Mpc and ~10 rad at k = 3e8/Mpc on the production 100-per-decade
grid, because the phase it interpolates grows like x_T and reaches 1.4e10. Instead

    theta(z) = + k * cs_tau.delta(z, z_init) + phi(z)

with `cs_tau` the background model's sound-horizon table int c_s dz/H, evaluated through its
double-double interval accessor, and phi the small remainder -- the phase residual
rho_T ~ -0.09 rad (review section 12.2) plus the initial-data offset deltaTheta that
`TkWKBIntegration.store()` folded into the stored samples. phi is O(0.1) rad and smooth, so a
cubic spline of *it* is limited by h^4 |phi''''|/384 instead.

`sign = +1` here, against -1 for the Green's function: `PrimitivePhase` measures the leading
term from a fixed anchor, and the transfer function's anchor is `z_init` at the *top* of its
WKB region (the Green's function's is the response redshift at the bottom), so
theta = +k cs_tau.delta(z, z_init) = -k cs_tau.delta(z_init, z), which is what
`TkWKBIntegration` stores. `increasing` is no longer a concept: there is no spline of theta to
order, and the sign convention is explicit. It is checked at construction against the stored
samples -- with the wrong sign the residual would be *twice* the leading term rather than
negligible beside it.

`PrimitivePhase` computes the leading part of d(theta)/dz in closed form as
`sign * k * rate(z)`, where `rate(z)` is whatever makes d/dz[leading.delta(z, anchor)] =
rate(z). For the Green's function the leading primitive is tau and rate = 1/H, which is
`PrimitivePhase`'s own default; for the sound horizon d(cs_tau)/dz = -c_s/H, so
rate = c_s/H, and `_sound_horizon_rate` below builds exactly that callable, passed to
`PrimitivePhase` as its `rate` argument. `omega()` returns the closed-form
sqrt(Tk_omegaEff_sq), and now equals `phase.theta_deriv(z)` to the accuracy of phi's spline
derivative rather than to that of a spline of theta.

`sin_coeff` is exposed as well: `M` already includes it (so `M` may be negative, and no
absolute value is taken anywhere), but a consumer that wants T's sign convention needs it.

Duck-typed input protocol
-------------------------

The background model and the two integration objects are consumed through the following
attributes only, so that tests and future callers can pass synthetic stand-ins:

  `model` (a `BackgroundModel`, or anything exposing `.functions` with):
      `.Hubble`, `.epsilon`, `.wPerturbations`  -- callables of z, as before
      `.cs_tau`     -- the sound-horizon primitive, with `delta(z_a, z_b) = cs_tau(z_b) -
                       cs_tau(z_a)`; `BackgroundModel.TablePrimitive` in production
      `.friction_F` -- the Liouville-Green friction primitive, same `delta` convention
    Both are new requirements of this class (prompt 10 of `prompts/GkTk-remedial`); a stand-in
    that leaves them at their `None` default is refused by name at construction.

  `Tk_numeric` (a `TkNumericIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z` (float redshift),
                       `.T` and `.Tprime` (= dT/dz)
      `.stop_deltaz_subh`  -- float, the "stop" mode hand-over offset below z_exit
      `.z_exit`     -- optional float; used for the crossover cross-check if `k` does not
                       carry one

  `Tk_WKB` (a `TkWKBIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z`, `.theta_div_2pi`,
                       `.theta_mod_2pi`, `.friction`. `.friction` is no longer *used* to build
                       the amplitude -- it is cross-checked against `friction_F` and then
                       discarded -- but it is still required, because that cross-check is what
                       detects a datastore built by the retired friction ODE
      `.sin_coeff`, `.cos_coeff` -- floats; `cos_coeff` must vanish
      `.z_init`     -- float, the hand-over redshift

`.z_sample` is deliberately *not* used: in `mode="stop"` a `TkNumericIntegration` holds fewer
values than its `z_sample` (`TkNumericIntegration.py:424-426`), and the WKB `z_sample` may
begin below `z_init` (`main.py:695-697`), so the value lists are the authoritative record of
what was actually sampled.

Regions and the sampled ranges
------------------------------

`numeric_region` and `WKB_region` are the ranges over which the returned callables may
actually be evaluated, i.e. the *sampled* ranges clipped at `crossover_z`:

    numeric_region = (largest numeric sample z, smallest numeric sample z >= crossover_z)
    WKB_region     = (largest WKB sample z <= crossover_z, smallest WKB sample z)

so `WKB_region[0] <= crossover_z <= numeric_region[1]`, with equality in both places when the
sampling grid happens to contain the hand-over point. In production the two grids are the same
`z_source_sample` grid, so at most one grid step separates `WKB_region[0]` from
`numeric_region[1]`; nothing is extrapolated into that interval, because `PrimitivePhase` keeps
the same range discipline the stored-phase spline it replaced had, and refuses to be evaluated
outside the sampled range of its residual. Consumers that partition an integral at the
hand-over should clamp their nodes to these two ranges.
"""

from math import log, exp, sqrt, sin, fabs
from typing import Tuple

from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples
from ComputeTargets.spline_wrappers import ZSplineWrapper
from LiouvilleGreen.constants import TWO_PI
from config.defaults import DEFAULT_FLOAT_PRECISION

MIN_SPLINE_DATA_POINTS = 5

# PrimitivePhase measures the leading term from a fixed anchor. For the transfer function the
# anchor is z_init, at the *top* of the WKB region, so
#
#     theta(z) = +k * cs_tau.delta(z, z_init) = -k * cs_tau.delta(z_init, z)
#
# which is what TkWKBIntegration stores. (GkSourcePolicyData anchors at the response redshift,
# at the bottom of its range, and carries sign = -1.)
TK_PHASE_SIGN = +1

# The stored `friction` samples must agree with model.functions.friction_F to this fraction of
# the largest |F| on the sampled range. Prompt 07 makes the two bit-equal; the retired friction
# ODE it replaced disagrees by 2.261e-07 absolute in F (prompts/GkTk-remedial log 04), which is
# ~4e-9 of the production max |F| = 55.07. The table's own accuracy is 1.42e-14 absolute, i.e.
# ~3e-16 of the same scale, so any tolerance between ~1e-14 and ~1e-9 separates the two cases;
# 1e-12 sits in the middle of that band.
FRICTION_CROSS_CHECK_RTOL = 1.0e-12


def _one_plus_z(z: float, z_is_log: bool) -> Tuple[float, float]:
    """
    Return (z, log(1+z)) from either representation.
    """
    if z_is_log:
        return exp(z) - 1.0, z

    return z, log(1.0 + z)


def _sound_horizon_rate(functions):
    """
    Build the `rate(z)` callable `PrimitivePhase` needs for the sound-horizon leading term.

    `PrimitivePhase` evaluates the derivative of its leading term in closed form as
    `sign * k * rate(z)` (its `rate` constructor parameter). For the Green's function the
    leading primitive is tau with d(tau)/dz = -1/H, so `rate = 1/H` -- `PrimitivePhase`'s own
    default. The transfer function's leading primitive is cs_tau with d(cs_tau)/dz = -c_s/H,
    so what it needs here is

        d/dz [k cs_tau.delta(z, z_anchor)] = + k c_s(z)/H(z),

    i.e. `rate(z) = c_s(z)/H(z)`, with c_s^2 = `wPerturbations(z)` (the author's convention:
    `wPerturbations` is c_s^2 in the transfer-function sector, `wBackground` is w_0). Unlike the
    retired `_SoundHorizonRate` adapter this replaces, `functions` (the model's real
    `ModelFunctions`) is passed to `PrimitivePhase` unmodified -- `rate` carries the c_s/H
    correction on its own, so `model_functions.Hubble` on the resulting `PrimitivePhase` still
    returns the genuine Hubble rate.
    """

    def rate(z: float) -> float:
        cs_sq = functions.wPerturbations(z)
        if cs_sq <= 0.0:
            raise RuntimeError(
                f"TkSourceFunctions: the sound speed squared c_s^2 = wPerturbations(z) must be "
                f"positive to evaluate the transfer-function phase derivative (got {cs_sq:.8g} "
                f"at z={z:.8g})"
            )

        return sqrt(cs_sq) / functions.Hubble(z)

    return rate


class TkSourceFunctions:
    """
    Non-persisted description of T_k(z) over the whole source-redshift range, in a numeric
    region and a Liouville-Green region. See the module docstring for the input protocol and
    the closed forms used.
    """

    def __init__(
        self,
        model: BackgroundModel,
        k,
        Tk_numeric,
        Tk_WKB,
    ):
        """
        :param model: a BackgroundModel (only `model.functions` is used)
        :param k: the wavenumber; `float(k)` must give k/a0 in the same units as H(z). If it
            carries a `z_exit` attribute (a `wavenumber_exit_time` does), that value is used
            for the crossover cross-check.
        :param Tk_numeric: numeric-region data; see the module docstring for the protocol
        :param Tk_WKB: WKB-region data; see the module docstring for the protocol
        """
        self._model = model
        self._k = k
        self._k_float = float(k)

        # --- the two background tables this class now reads ------------------------------
        # Both are prompt 04's TablePrimitive objects in production. A ModelFunctions built
        # before prompt 04 leaves them at their None default, and a stand-in may simply not
        # supply them; say which one is missing rather than failing later on None.delta.
        for name in ("cs_tau", "friction_F"):
            accessor = getattr(model.functions, name, None)
            if accessor is None or not hasattr(accessor, "delta"):
                raise RuntimeError(
                    f"TkSourceFunctions: model.functions.{name} must be a primitive accessor "
                    f"with an interval method delta(z_a, z_b) (got {accessor!r}). The transfer "
                    f"function's phase and amplitude are read from the background model's "
                    f"sound-horizon and Liouville-Green friction tables; a BackgroundModel "
                    f"built before prompt 04 of prompts/GkTk-remedial does not carry them, and "
                    f"its datastore must be regenerated."
                )

        self._cs_tau = model.functions.cs_tau
        self._friction_F = model.functions.friction_F

        # --- hand-over redshift, and its cross-check -------------------------------------
        self._crossover_z = float(Tk_WKB.z_init)

        z_exit = getattr(k, "z_exit", None)
        if z_exit is None:
            z_exit = getattr(Tk_numeric, "z_exit", None)

        stop_deltaz_subh = getattr(Tk_numeric, "stop_deltaz_subh", None)
        if z_exit is not None and stop_deltaz_subh is not None:
            expected = float(z_exit) - float(stop_deltaz_subh)
            if fabs(expected - self._crossover_z) > DEFAULT_FLOAT_PRECISION * max(
                1.0, fabs(self._crossover_z)
            ):
                raise RuntimeError(
                    f"TkSourceFunctions: inconsistent hand-over redshift: TkWKBIntegration.z_init={self._crossover_z:.8g}, "
                    f"but z_exit - stop_deltaz_subh = {float(z_exit):.8g} - {float(stop_deltaz_subh):.8g} = {expected:.8g} "
                    f"(relative difference {fabs(expected - self._crossover_z)/max(1.0, fabs(self._crossover_z)):.5g})"
                )

        cos_coeff = float(Tk_WKB.cos_coeff)
        if fabs(cos_coeff) > DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"TkSourceFunctions: expected the Liouville-Green representation of T_k to be a pure sine "
                f"(TkWKBIntegration.store() sets cos_coeff = 0 at TkWKBIntegration.py:458), but cos_coeff={cos_coeff:.8g}"
            )
        self._sin_coeff = float(Tk_WKB.sin_coeff)

        self._build_numeric(Tk_numeric)
        self._build_WKB(Tk_WKB)

    # ------------------------------------------------------------------------------------
    # construction

    def _build_numeric(self, Tk_numeric) -> None:
        one_plus_crossover = 1.0 + self._crossover_z

        # keep only samples at or above the hand-over: below it the numeric solution is either
        # absent (mode="stop") or superseded by the Liouville-Green representation
        data = [
            v
            for v in Tk_numeric.values
            if (1.0 + v.z.z) >= one_plus_crossover * (1.0 - DEFAULT_FLOAT_PRECISION)
        ]
        if len(data) < MIN_SPLINE_DATA_POINTS:
            raise RuntimeError(
                f"TkSourceFunctions: too few numeric samples at or above the hand-over redshift "
                f"z={self._crossover_z:.5g} ({len(data)} found, {MIN_SPLINE_DATA_POINTS} required)"
            )

        data.sort(key=lambda v: v.z.z)

        self._numeric_min_z = data[0].z.z
        self._numeric_max_z = data[-1].z.z

        log_x = [log(1.0 + v.z.z) for v in data]

        self._T_spline = ZSplineWrapper(
            make_interp_spline(log_x, [v.T for v in data]),
            "numeric T_k",
            self._numeric_max_z,
            self._numeric_min_z,
            log_z=True,
        )

        # spline the *stored* dT/dz samples; do not differentiate the T spline
        self._dT_dz_spline = ZSplineWrapper(
            make_interp_spline(log_x, [v.Tprime for v in data]),
            "numeric dT_k/dz",
            self._numeric_max_z,
            self._numeric_min_z,
            log_z=True,
        )

    def _build_WKB(self, Tk_WKB) -> None:
        one_plus_crossover = 1.0 + self._crossover_z

        # keep only samples at or below the hand-over
        data = [
            v
            for v in Tk_WKB.values
            if (1.0 + v.z.z) <= one_plus_crossover * (1.0 + DEFAULT_FLOAT_PRECISION)
        ]
        if len(data) < MIN_SPLINE_DATA_POINTS:
            raise RuntimeError(
                f"TkSourceFunctions: too few WKB samples at or below the hand-over redshift "
                f"z={self._crossover_z:.5g} ({len(data)} found, {MIN_SPLINE_DATA_POINTS} required)"
            )

        data.sort(key=lambda v: v.z.z)

        self._WKB_min_z = data[0].z.z
        self._WKB_max_z = data[-1].z.z

        z_points = [v.z.z for v in data]

        # F is no longer splined: it comes from the background table as
        # friction_F.delta(crossover_z, z). The stored samples are read once, here, to confirm
        # that the table and the datastore describe the same friction integral.
        self._check_friction_samples(z_points, [v.friction for v in data])

        # theta = +k cs_tau.delta(z, z_init) + phi, with phi the small residual splined in
        # log(1+z). See the module docstring for why the growing part is not interpolated.
        theta_points = [v.theta_div_2pi * TWO_PI + v.theta_mod_2pi for v in data]
        phi_points = build_phi_samples(
            self._k_float,
            self._cs_tau,
            self._crossover_z,
            z_points,
            theta_points,
            sign=TK_PHASE_SIGN,
        )
        self._check_phase_sign(z_points, theta_points, phi_points)

        self._phase = PrimitivePhase(
            self._k_float,
            self._cs_tau,
            self._crossover_z,
            z_points,
            phi_points,
            sign=TK_PHASE_SIGN,
            model_functions=self._model.functions,
            rate=_sound_horizon_rate(self._model.functions),
            label="T_k WKB phase",
        )

        self._H_init = self._model.functions.Hubble(self._crossover_z)

    def _check_friction_samples(self, z_points, friction_points) -> None:
        """
        The stored `friction` samples must reproduce `friction_F.delta(crossover_z, z)`.

        They are bit-equal for a `TkWKBIntegration` written by prompt 07 of
        `prompts/GkTk-remedial`, which stores exactly that quantity; the retired friction ODE
        they replaced carried ~2e-7 absolute in F (prompt 04's `TestFrictionODEComparison`).
        The discrepancy is therefore a clean signal that the datastore and the background model
        disagree -- most likely a datastore predating prompt 07 read against a table-built
        model -- and refusing to build is better than silently using a table that does not
        describe the stored amplitude.

        The comparison is absolute, scaled by the largest |F| on the sampled range: F vanishes
        at the hand-over by construction, so a per-sample relative test would divide by zero
        there.
        """
        table = [self._friction_F.delta(self._crossover_z, z) for z in z_points]

        scale = max(fabs(value) for value in table)
        if scale <= 0.0:
            return

        worst = 0.0
        worst_z = z_points[0]
        for z, stored, expected in zip(z_points, friction_points, table):
            err = fabs(float(stored) - expected)
            if err > worst:
                worst = err
                worst_z = z

        if worst > FRICTION_CROSS_CHECK_RTOL * scale:
            raise RuntimeError(
                f"TkSourceFunctions: the stored Liouville-Green friction samples disagree with "
                f"model.functions.friction_F by {worst:.5g} ({worst/scale:.5g} of the largest "
                f"|F| = {scale:.5g} on the sampled range, tolerance "
                f"{FRICTION_CROSS_CHECK_RTOL:.5g}), worst at z={worst_z:.8g}. The amplitude is "
                f"built from the table, so the two must describe the same integral: this "
                f"normally means the TkWKBIntegration was written by the retired friction ODE "
                f"(~2e-7 absolute in F) and its datastore must be regenerated."
            )

    def _check_phase_sign(self, z_points, theta_points, phi_points) -> None:
        """
        Confirm `TK_PHASE_SIGN` against the stored samples.

        With the right sign the leading term is removed and phi is the small remainder; with
        the wrong one it is *added*, and phi would span twice the leading term instead. So it
        is enough to require that phi vary by less than the leading term does over the same
        samples -- a test that cannot fail on correctly signed data (phi is O(0.1) rad against
        a leading span of 1e4 to 1e10 rad) and cannot pass on data of the opposite sign.
        """
        if len(z_points) < 2:
            return

        phi_span = max(phi_points) - min(phi_points)
        leading_span = fabs(
            self._k_float * self._cs_tau.delta(max(z_points), min(z_points))
        )

        if phi_span >= leading_span:
            theta_span = max(theta_points) - min(theta_points)
            raise RuntimeError(
                f"TkSourceFunctions: the stored phase samples are not consistent with the "
                f"transfer-function sign convention theta(z) = +k cs_tau.delta(z, z_init) "
                f"(TK_PHASE_SIGN = {TK_PHASE_SIGN:+d}): after removing the leading term the "
                f"residual still spans {phi_span:.5g} rad against a leading span of "
                f"{leading_span:.5g} rad (the stored phase spans {theta_span:.5g} rad). Either "
                f"the samples carry the opposite sign convention, or they were not produced "
                f"from this background model's sound horizon."
            )

    # ------------------------------------------------------------------------------------
    # region bookkeeping

    @property
    def crossover_z(self) -> float:
        return self._crossover_z

    @property
    def numeric_region(self) -> Tuple[float, float]:
        return self._numeric_max_z, self._numeric_min_z

    @property
    def WKB_region(self) -> Tuple[float, float]:
        return self._WKB_max_z, self._WKB_min_z

    @property
    def sin_coeff(self) -> float:
        return self._sin_coeff

    @property
    def phase(self) -> PrimitivePhase:
        return self._phase

    def _check_numeric(self, z: float) -> None:
        if (1.0 + z) < (1.0 + self._numeric_min_z) * (1.0 - DEFAULT_FLOAT_PRECISION):
            raise RuntimeError(
                f"TkSourceFunctions: numeric region of T_k evaluated below its lower limit "
                f"(z={z:.5g}, numeric region = ({self._numeric_max_z:.5g}, {self._numeric_min_z:.5g}), "
                f"hand-over z={self._crossover_z:.5g})"
            )

    def _check_WKB(self, z: float, label: str) -> None:
        if (1.0 + z) > (1.0 + self._WKB_max_z) * (1.0 + DEFAULT_FLOAT_PRECISION):
            raise RuntimeError(
                f"TkSourceFunctions: {label} evaluated above the top of the WKB region "
                f"(z={z:.5g}, WKB region = ({self._WKB_max_z:.5g}, {self._WKB_min_z:.5g}), "
                f"hand-over z={self._crossover_z:.5g})"
            )
        if (1.0 + z) < (1.0 + self._WKB_min_z) * (1.0 - DEFAULT_FLOAT_PRECISION):
            raise RuntimeError(
                f"TkSourceFunctions: {label} evaluated below the bottom of the WKB region "
                f"(z={z:.5g}, WKB region = ({self._WKB_max_z:.5g}, {self._WKB_min_z:.5g}))"
            )

    # ------------------------------------------------------------------------------------
    # numeric region

    def T(self, z: float, z_is_log: bool = False) -> float:
        """
        T_k in the numeric region. Above the numeric region T = 1 exactly (spec 01 section 2.8).
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)

        if (1.0 + raw_z) > (1.0 + self._numeric_max_z) * (
            1.0 + DEFAULT_FLOAT_PRECISION
        ):
            return 1.0

        self._check_numeric(raw_z)
        return self._T_spline(log_z, z_is_log=True)

    def dT_dz(self, z: float, z_is_log: bool = False) -> float:
        """
        dT_k/dz in the numeric region, splined from the stored `Tprime` samples. Above the
        numeric region dT/dz = 0 exactly.
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)

        if (1.0 + raw_z) > (1.0 + self._numeric_max_z) * (
            1.0 + DEFAULT_FLOAT_PRECISION
        ):
            return 0.0

        self._check_numeric(raw_z)
        return self._dT_dz_spline(log_z, z_is_log=True)

    # ------------------------------------------------------------------------------------
    # WKB region

    def friction(self, z: float, z_is_log: bool = False) -> float:
        """
        The Liouville-Green friction integral F(z) = friction_F.delta(crossover_z, z), read
        from the background model's table. No spline: this is exact to the table's own ~1e-14
        absolute, and it is negative below the hand-over (`dF/dz > 0`).
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "the T_k friction function")
        return self._friction_F.delta(self._crossover_z, raw_z)

    def M(self, z: float, z_is_log: bool = False) -> float:
        """
        Liouville-Green amplitude, WKB region only:
        M = sin_coeff sqrt(H_init/H) omega_eff^(-1/2) exp(F). Carries the sign of sin_coeff.
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "the T_k LG amplitude")

        H = self._model.functions.Hubble(raw_z)
        omega = sqrt(Tk_omegaEff_sq(self._model, self._k_float, raw_z))

        return (
            self._sin_coeff
            * sqrt(self._H_init / H / omega)
            * exp(self._friction_F.delta(self._crossover_z, raw_z))
        )

    def dlnM_dz(self, z: float, z_is_log: bool = False) -> float:
        """
        d ln M/dz in closed form (spec 01 R23/R24; see the module docstring). No spline is
        differentiated.
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "the T_k LG amplitude derivative")

        one_plus_z = 1.0 + raw_z
        eps = self._model.functions.epsilon(raw_z)
        cs2 = self._model.functions.wPerturbations(raw_z)
        d_ln_omega_dz = Tk_d_ln_omegaEff_dz(self._model, self._k_float, raw_z)

        return (
            -eps / (2.0 * one_plus_z)
            - d_ln_omega_dz / 2.0
            + (3.0 / 2.0) * (1.0 + cs2) / one_plus_z
        )

    def omega(self, z: float, z_is_log: bool = False) -> float:
        """
        d(theta)/dz = omega_eff, in closed form (not the spline derivative of the phase).
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "the T_k LG frequency")

        return sqrt(Tk_omegaEff_sq(self._model, self._k_float, raw_z))

    def T_WKB(self, z: float, z_is_log: bool = False) -> float:
        """
        Convenience: M(z) sin(theta(z) mod 2pi), for tests and diagnostics.
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "T_k^WKB")

        return self.M(log_z, z_is_log=True) * sin(
            self._phase.theta_mod_2pi(log_z, x_is_log=True)
        )
