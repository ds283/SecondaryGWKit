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
    theta as a `phase_spline`, and d(theta)/dz = omega_eff (closed form).

Amplitude
---------

From `TkWKBIntegration.store()` (`TkWKBIntegration.py:494-506`), with `cos_coeff = 0`
(set at `TkWKBIntegration.py:458`),

    M(z) = sin_coeff * sqrt(H_init / H(z)) * omega_eff(z)^(-1/2) * exp(F(z))

with H_init = H(crossover_z) (the `H_ratio` stored per value is exactly H_init/H),
omega_eff^2 = `WKB_Tk.Tk_omegaEff_sq(model, k, z)` and F the Liouville-Green friction integral
stored per value as `friction`. Only F is splined here: the other factors are closed forms and
are evaluated exactly, so the product is never splined and the amplitude's derivative never
differentiates a spline.

The friction integrand is `TkWKBIntegration.friction_RHS` (`TkWKBIntegration.py:25-49`, the
returned value at `:49`): dF/dz = (3/2)(1 + c_s^2)/(1+z), with c_s^2 = `wPerturbations(z)`.
That is exactly the integrand of spec 01 R23. With d ln H/dz = epsilon/(1+z), R23/R24 give the
closed form used by `dlnM_dz`:

    d ln M / dz = - epsilon(z) / (2(1+z))
                  - (1/2) d ln omega_eff / dz
                  + (3/2) (1 + c_s^2(z)) / (1+z)

the three terms being d ln sqrt(H_init/H)/dz, the derivative of omega_eff^(-1/2), and dF/dz
respectively.

Phase
-----

The phase convention is the code's: d(theta)/dz = +omega_eff integrated towards *smaller* z
from theta(z_init) = 0, so theta is negative and *decreasing* as z falls -- equivalently
theta is an increasing function of z (audit `TK-report.md` TK-8(a)). The phase spline is
therefore built with `increasing=True`, unlike `GkSourcePolicyData._create_functions()`
(`:657-671`), where theta at fixed response redshift is a decreasing function of the *source*
redshift. `omega()` returns the closed-form sqrt(Tk_omegaEff_sq) rather than the spline
derivative, and equals `phase.theta_deriv(z)` up to spline error.

`sin_coeff` is exposed as well: `M` already includes it (so `M` may be negative, and no
absolute value is taken anywhere), but a consumer that wants T's sign convention needs it.

Duck-typed input protocol
-------------------------

The two integration objects are consumed through the following attributes only, so that tests
and future callers can pass synthetic stand-ins:

  `Tk_numeric` (a `TkNumericIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z` (float redshift),
                       `.T` and `.Tprime` (= dT/dz)
      `.stop_deltaz_subh`  -- float, the "stop" mode hand-over offset below z_exit
      `.z_exit`     -- optional float; used for the crossover cross-check if `k` does not
                       carry one

  `Tk_WKB` (a `TkWKBIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z`, `.theta_div_2pi`,
                       `.theta_mod_2pi`, `.friction`
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
`numeric_region[1]`; nothing is extrapolated into that interval, because `phase_spline` refuses
to be evaluated outside its own sampled range. Consumers that partition an integral at the
hand-over should clamp their nodes to these two ranges.
"""

from math import log, exp, sqrt, sin, fabs
from typing import Tuple

from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.spline_wrappers import ZSplineWrapper
from LiouvilleGreen.phase_spline import phase_spline
from config.defaults import DEFAULT_FLOAT_PRECISION

MIN_SPLINE_DATA_POINTS = 5

# chunking used for the phase spline; matches GkSourcePolicyData._create_functions (:664-671)
PHASE_SPLINE_CHUNK_LOGSTEP = 125


def _one_plus_z(z: float, z_is_log: bool) -> Tuple[float, float]:
    """
    Return (z, log(1+z)) from either representation.
    """
    if z_is_log:
        return exp(z) - 1.0, z

    return z, log(1.0 + z)


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

        log_x = [log(1.0 + v.z.z) for v in data]

        # F is smooth and monotone; it is the only part of the amplitude that needs a spline
        self._friction_spline = ZSplineWrapper(
            make_interp_spline(log_x, [v.friction for v in data]),
            "T_k WKB friction",
            self._WKB_max_z,
            self._WKB_min_z,
            log_z=True,
        )

        # theta is an increasing function of z under the code's convention (see the module
        # docstring), hence increasing=True
        self._phase = phase_spline(
            log_x,
            [v.theta_div_2pi for v in data],
            [v.theta_mod_2pi for v in data],
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=PHASE_SPLINE_CHUNK_LOGSTEP,
            increasing=True,
        )

        self._H_init = self._model.functions.Hubble(self._crossover_z)

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
    def phase(self) -> phase_spline:
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
        The Liouville-Green friction integral F(z), splined from the stored samples.
        """
        raw_z, log_z = _one_plus_z(z, z_is_log)
        self._check_WKB(raw_z, "the T_k friction function")
        return self._friction_spline(log_z, z_is_log=True)

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
            * exp(self._friction_spline(log_z, z_is_log=True))
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
