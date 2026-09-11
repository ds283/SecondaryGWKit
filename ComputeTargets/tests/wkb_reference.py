"""
Reference harness for the Gk/Tk WKB phase remedial campaign
(``prompts/GkTk-remedial/``, prompt 01).

This module holds the *measurement infrastructure* every later prompt in the campaign is scored
against: three duck-typed stand-in background models, helpers that reproduce the production
redshift grids, the campaign's three error definitions, and a loader for the cached reference
values in ``wkb_reference_data.json``.

**Independence.** A reference built from the object under test conceals common error, so this
module deliberately imports nothing from ``Quadrature/integrators/WKB_phase_function.py``,
``LiouvilleGreen/phase_spline.py``, or (once they exist) ``ComputeTargets/cumulative_table.py``,
``ComputeTargets/phase_residual.py``, ``ComputeTargets/primitive_phase.py``. It does import the
cosmology models, ``ModelFunctions``, ``redshift``/``redshift_array`` and
``ComputeTargets.spline_wrappers`` -- those are the things the references are *for*, or plumbing
with no bearing on the phase.

**Conventions used throughout, fixed by ``prompts/GkTk-remedial/README.md`` §2.**

* ``tau`` is the conformal-time primitive (the code's ``a0 tau``), with ``dtau/dz = -1/H``; the
  campaign's interval accessor is ``tau.delta(z_a, z_b) = tau(z_b) - tau(z_a)``
  ``= int_{z_b}^{z_a} dz/H``, positive when ``z_b < z_a`` (README §2 (c)).
* ``cs_tau`` is the sound-horizon primitive, ``d(cs_tau)/dz = -c_s/H`` with
  ``c_s^2 = wPerturbations``.
* ``friction_F`` is the primitive of ``TkWKBIntegration.friction_RHS``, i.e.
  ``dF/dz = +(3/2)(1 + c_s^2)/(1+z)``.  Note the sign: ``F`` *decreases* towards lower redshift,
  where ``tau`` and ``cs_tau`` increase.
* The Green's-function phase splits as ``theta(z; z_i) = -k tau.delta(z_i, z) - rho_G(z; z_i)``
  with ``rho_G(z; z_i) = int_z^{z_i} C/(omega + k/H) dz`` and
  ``omega^2 = (k/H)^2 + C`` (review §6). The transfer-function phase splits the same way with
  ``cs_tau``, ``omega_T^2 = c_s^2 (k/H)^2 + C_T`` and ``rho_T``.

Nothing in this module needs Ray, a datastore, or ``mpmath``.
"""

import json
from math import log, log1p, exp, expm1, sqrt, log10, asin, fabs
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

from ComputeTargets.BackgroundModel import ModelFunctions, compute_background
from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyConcepts import redshift, redshift_array
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

# ---------------------------------------------------------------------------------------------
# production geometry (main.py, on 9ff59d5)
# ---------------------------------------------------------------------------------------------

# main.py:69-71
PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z = 100
PRODUCTION_RESPONSE_SPARSENESS = 12
PRODUCTION_Z_END = 0.1

# main.py:2873-2891: source and response k arrays both run 1e5 .. 3e8 /Mpc
PRODUCTION_SMALLEST_K_INV_MPC = 1.0e5
PRODUCTION_LARGEST_K_INV_MPC = 3.0e8

# main.py:411 -- the universal source grid starts 5 e-folds *outside* the horizon for the
# earliest-exiting (largest) k
PRODUCTION_SUPERHORIZON_EFOLDS = 5

# the wavenumbers at which the JSON carries residual references
REFERENCE_K_VALUES = (1.0e5, 1.0e7, 3.0e8)

# the three model keys used in the JSON
MODEL_KEYS = ("RadiationModel", "LambdaCDMModel", "QCDModel")

REFERENCE_DATA_PATH = Path(__file__).parent / "wkb_reference_data.json"


# ---------------------------------------------------------------------------------------------
# production grids
# ---------------------------------------------------------------------------------------------


def horizon_exit_z(cosmology, k_inv_Mpc: float, efolds_subh: float = 0.0) -> float:
    """
    Solve k(1+z)/H(z) = exp(efolds_subh) for z, mirroring
    ``CosmologyConcepts.wavenumber._solve_horizon_exit``: positive ``efolds_subh`` is inside the
    horizon, negative is outside.

    :param cosmology: a cosmology exposing ``Hubble(z)`` and ``H0``
    :param k_inv_Mpc: comoving wavenumber, in the cosmology's units (Mpc_units: 1/Mpc)
    :param efolds_subh: e-folds inside the horizon (negative for outside)
    :return: the redshift of the requested crossing
    """

    def q(log_opz: float) -> float:
        z = expm1(log_opz)
        return log(k_inv_Mpc * (1.0 + z) / cosmology.Hubble(z)) - efolds_subh

    log_opz_guess = log(k_inv_Mpc / cosmology.H0) - efolds_subh

    lo = log_opz_guess
    hi = log_opz_guess
    step = log(1.5)
    for _ in range(200):
        if q(lo) * q(hi) < 0.0:
            break
        lo -= step
        hi += step
    else:
        raise RuntimeError(
            f"horizon_exit_z: failed to bracket the crossing for k={k_inv_Mpc:.5g}, "
            f"efolds_subh={efolds_subh}"
        )

    root = root_scalar(q, bracket=(lo, hi), xtol=1e-14, rtol=1e-14)
    if not root.converged:
        raise RuntimeError(
            f"horizon_exit_z: root_scalar did not converge for k={k_inv_Mpc:.5g}"
        )

    return expm1(root.root)


def production_source_z_values(
    z_init: float,
    z_end: float = PRODUCTION_Z_END,
    samples_per_log10z: int = PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
) -> np.ndarray:
    """
    Reproduce ``wavenumber_exit_time.populate_z_sample`` (CosmologyConcepts/wavenumber.py:250-292):
    a descending grid, log-spaced **in z** (not in 1+z), with ``samples_per_log10z`` points per
    decade of z between ``z_init`` and ``z_end``.

    :return: a descending ``numpy`` array of redshifts
    """
    num = int(round(samples_per_log10z * (log10(z_init) - log10(z_end)) + 0.5, 0))
    return np.logspace(log10(z_init), log10(z_end), num=num)


def to_redshift_array(z_values: Sequence[float]) -> redshift_array:
    """Wrap plain floats as a ``redshift_array`` (descending, as the class enforces)."""
    return redshift_array(
        [redshift(store_id=i, z=float(z)) for i, z in enumerate(z_values)]
    )


def production_source_grid(
    z_init: float,
    z_end: float = PRODUCTION_Z_END,
    samples_per_log10z: int = PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
) -> redshift_array:
    """The production source grid as a ``redshift_array`` (main.py:410-419)."""
    return to_redshift_array(
        production_source_z_values(z_init, z_end, samples_per_log10z)
    )


def production_response_grid(
    source_grid: redshift_array,
    sparseness: int = PRODUCTION_RESPONSE_SPARSENESS,
) -> redshift_array:
    """The production response grid: ``source_grid.winnow(sparseness)`` (main.py:424)."""
    return source_grid.winnow(sparseness=sparseness)


# ---------------------------------------------------------------------------------------------
# stand-in background models
# ---------------------------------------------------------------------------------------------


class RadiationModel:
    """
    Exact radiation control: ``H = H0 (1+z)^2``, ``epsilon = 2``, ``w = c_s^2 = 1/3``.

    Every quantity the campaign needs has a closed form (``s = 1+z``, ``a = k c_s / H0``):

    * ``tau(z)  = 1/(H0 s)``            (the additive constant fixed by ``tau -> 0`` as ``z -> inf``,
      which is the same convention as ``compute_background``'s ``tau_init`` asymptote)
    * ``cs_tau(z) = tau(z)/sqrt(3)``
    * ``friction_F(z) - friction_F(z_i) = 2 log((1+z)/(1+z_i))``
    * ``theta_G(z; z_i) = k(1/s_i - 1/s)`` and ``rho_G == 0`` identically (``C == 0``)
    * ``rho_T(z; z_i) = g(s) - g(s_i)`` with
      ``g(s) = -2s/(sqrt(a^2 - 2 s^2) + a) + sqrt(2) arcsin(sqrt(2) s / a)``,
      the exact primitive of ``C_T/(omega_T + k c_s/H)`` for ``C_T = -2/s^2``. Asymptotically
      ``g(s) -> 1/x + 2/(3 x^3)`` with ``x = k c_s tau``, so ``rho_T -> 1/x - 1/x_i``; note the
      sign, which is opposite to the ``1/x_i - 1/x`` quoted in review §12.4 (that quotation is in
      the convention ``rho = theta - (x_i - x)``, this module's is
      ``theta = -(x - x_i) - rho``, README §2 (a), (c)).
    """

    name = "RadiationModel"

    def __init__(self, H0: float = 1.0):
        self.H0 = float(H0)
        self.cosmology = None
        self.functions = ModelFunctions(
            Hubble=self.Hubble,
            epsilon=lambda z: 2.0,
            d_epsilon_dz=lambda z: 0.0,
            d2_epsilon_dz2=lambda z: 0.0,
            wBackground=lambda z: 1.0 / 3.0,
            wPerturbations=lambda z: 1.0 / 3.0,
            tau=self.tau,
            T_photon=lambda z: 0.0,
            d_lnH_dz=lambda z: 2.0 / (1.0 + z),
            d2_lnH_dz2=lambda z: -2.0 / (1.0 + z) ** 2,
            d3_lnH_dz3=lambda z: 4.0 / (1.0 + z) ** 3,
            d_wPerturbations_dz=lambda z: 0.0,
            d2_wPerturbations_dz2=lambda z: 0.0,
        )

    def Hubble(self, z: float) -> float:
        return self.H0 * (1.0 + z) * (1.0 + z)

    def tau(self, z: float) -> float:
        return 1.0 / (self.H0 * (1.0 + z))

    def tau_delta(self, z_a: float, z_b: float) -> float:
        """
        ``tau.delta(z_a, z_b) = tau(z_b) - tau(z_a) = (z_a - z_b)/(H0 (1+z_a)(1+z_b))``,
        positive when ``z_b < z_a`` (README §2 (c)).

        The factored form is used rather than the difference of two ``tau`` values because the
        latter loses a digit for every factor of ten by which the baseline is shorter than the
        primitive -- the same fact the campaign's double-double node table exists to defeat
        (review §13.3). Over one production grid interval the naive difference carries
        ~5e-15 relative; this form carries ~1e-16.
        """
        return (z_a - z_b) / (self.H0 * (1.0 + z_a) * (1.0 + z_b))

    def cs_tau(self, z: float) -> float:
        return self.tau(z) / sqrt(3.0)

    def cs_tau_delta(self, z_a: float, z_b: float) -> float:
        """``cs_tau.delta(z_a, z_b) = tau.delta(z_a, z_b)/sqrt(3)``."""
        return self.tau_delta(z_a, z_b) / sqrt(3.0)

    def friction_F_delta(self, z: float, z_ref: float) -> float:
        """``F(z) - F(z_ref) = 2 log((1+z)/(1+z_ref))``."""
        return 2.0 * (log1p(z) - log1p(z_ref))

    def theta_G(self, k: float, z: float, z_init: float) -> float:
        """``theta_G(z; z_init) = k(1/s_init - 1/s)``, negative for ``z < z_init``."""
        return k * (self.tau(z_init) - self.tau(z))

    def rho_G(self, k: float, z: float, z_init: float) -> float:
        """``C == 0`` in exact radiation, so the residual vanishes identically."""
        return 0.0

    def _rho_T_primitive(self, k: float, z: float) -> float:
        a = k / (sqrt(3.0) * self.H0)
        s = 1.0 + z
        disc = a * a - 2.0 * s * s
        if disc <= 0.0:
            raise ValueError(
                f"RadiationModel._rho_T_primitive: omega_T^2 < 0 at z={z:.6g} for k={k:.6g} "
                "(the mode is not sub-horizon here)"
            )
        return -2.0 * s / (sqrt(disc) + a) + sqrt(2.0) * asin(sqrt(2.0) * s / a)

    def rho_T(self, k: float, z: float, z_init: float) -> float:
        """
        ``rho_T(z; z_init) = int_z^{z_init} C_T/(omega_T + k c_s/H) dz``, negative for
        ``z < z_init``. Exact.
        """
        return self._rho_T_primitive(k, z) - self._rho_T_primitive(k, z_init)

    def x_T(self, k: float, z: float) -> float:
        """``x_T = k c_s tau``."""
        return k * self.cs_tau(z)


def _model_functions_from_background(
    cosmology, z_sample: redshift_array, payload: dict
) -> ModelFunctions:
    """
    Assemble a ``ModelFunctions`` from a ``compute_background`` payload, reproducing
    ``BackgroundModel._create_functions`` (ComputeTargets/BackgroundModel.py:387-458): analytic
    cosmology methods win where they exist, otherwise a ``make_interp_spline`` in ``log(1+z)``
    wrapped in a ``ZSplineWrapper``.

    ``tau`` is left as ``None``: nothing in the reference harness uses the production pointwise
    accessor, and prompt 03 replaces it.
    """
    z_values = [v.z for v in z_sample]
    min_z = min(z_values)
    max_z = max(z_values)

    def _build(attr: str, samples):
        if hasattr(cosmology, attr):
            return getattr(cosmology, attr)

        data = sorted(zip((log1p(z) for z in z_values), samples), key=lambda p: p[0])
        x_data, y_data = zip(*data)
        return ZSplineWrapper(
            make_interp_spline(x_data, y_data),
            label=attr,
            min_z=min_z,
            max_z=max_z,
            log_z=True,
        )

    d_lnH_dz = _build("d_lnH_dz", payload["d_lnH_dz_sample"])
    d2_lnH_dz2 = _build("d2_lnH_dz2", payload["d2_lnH_dz2_sample"])
    d3_lnH_dz3 = _build("d3_lnH_dz3", payload["d3_lnH_dz3_sample"])
    d_wPerturbations_dz = _build(
        "d_wPerturbations_dz", payload["d_wPerturbations_dz_sample"]
    )
    d2_wPerturbations_dz2 = _build(
        "d2_wPerturbations_dz2", payload["d2_wPerturbations_dz2_sample"]
    )
    T_photon = _build("T_photon", payload["T_photon_sample"])

    return ModelFunctions(
        Hubble=cosmology.Hubble,
        epsilon=lambda z: (1.0 + z) * d_lnH_dz(z),
        d_epsilon_dz=lambda z: d_lnH_dz(z) + (1.0 + z) * d2_lnH_dz2(z),
        d2_epsilon_dz2=lambda z: 2.0 * d2_lnH_dz2(z) + (1.0 + z) * d3_lnH_dz3(z),
        wBackground=cosmology.wBackground,
        wPerturbations=cosmology.wPerturbations,
        tau=None,
        T_photon=T_photon,
        d_lnH_dz=d_lnH_dz,
        d2_lnH_dz2=d2_lnH_dz2,
        d3_lnH_dz3=d3_lnH_dz3,
        d_wPerturbations_dz=d_wPerturbations_dz,
        d2_wPerturbations_dz2=d2_wPerturbations_dz2,
    )


class LambdaCDMModel:
    """
    ``LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())`` with the analytic
    ``d_lnH_dz``, ``d2_lnH_dz2``, ``d3_lnH_dz3``, ``d_wPerturbations_dz``,
    ``d2_wPerturbations_dz2`` wired into ``epsilon``, ``d_epsilon_dz``, ``d2_epsilon_dz2``
    exactly as ``BackgroundModel._create_functions`` does when a cosmology supplies them
    (this is the path ``docs/gk-wkb-review-fable-2026-09-09/realbg.py`` uses).

    ``functions.tau`` is ``None``: nothing here needs the production pointwise accessor.
    """

    name = "LambdaCDMModel"

    def __init__(self, cosmology=None):
        self.cosmology = (
            cosmology
            if cosmology is not None
            else LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        )
        c = self.cosmology
        self.functions = ModelFunctions(
            Hubble=c.Hubble,
            epsilon=lambda z: (1.0 + z) * c.d_lnH_dz(z),
            d_epsilon_dz=lambda z: c.d_lnH_dz(z) + (1.0 + z) * c.d2_lnH_dz2(z),
            d2_epsilon_dz2=lambda z: 2.0 * c.d2_lnH_dz2(z)
            + (1.0 + z) * c.d3_lnH_dz3(z),
            wBackground=c.wBackground,
            wPerturbations=c.wPerturbations,
            tau=None,
            T_photon=c.T_photon,
            d_lnH_dz=c.d_lnH_dz,
            d2_lnH_dz2=c.d2_lnH_dz2,
            d3_lnH_dz3=c.d3_lnH_dz3,
            d_wPerturbations_dz=c.d_wPerturbations_dz,
            d2_wPerturbations_dz2=c.d2_wPerturbations_dz2,
        )


class QCDModel:
    """
    ``QCD_Cosmology(store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20)`` -- the
    production ``max_z`` of ``config/model_list.py``.

    ``QCD_Cosmology`` supplies no analytic derivatives, so its ``ModelFunctions`` is built by
    calling the **undecorated** ``compute_background`` on the supplied grid (the pattern of
    ``ComputeTargets/tests/test_background_derivatives.py``) and splining the returned derivative
    samples exactly as ``BackgroundModel._create_functions`` does.

    Construction is expensive (see the campaign log for prompt 01): build one and reuse it.
    """

    name = "QCDModel"

    def __init__(
        self,
        z_sample: redshift_array,
        cosmology=None,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ):
        self.cosmology = (
            cosmology
            if cosmology is not None
            else QCD_Cosmology(
                store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
            )
        )
        self.z_sample = z_sample

        payload = compute_background._function(
            self.cosmology, z_sample, atol=atol, rtol=rtol
        )
        self.background_payload = payload
        self.functions = _model_functions_from_background(
            self.cosmology, z_sample, payload
        )


# ---------------------------------------------------------------------------------------------
# a closed-form stand-in for a ``TablePrimitive``
# ---------------------------------------------------------------------------------------------


class ClosedFormPrimitive:
    """
    A stand-in for ``BackgroundModel.TablePrimitive`` whose primitive is known in closed form:
    ``__call__(z) = f(z)`` and ``delta(z_a, z_b) = f(z_b) - f(z_a)``, the same sign convention as
    the real accessor (README §2 (c)).

    **Why a difference of two pointwise values is acceptable here, and only here.** The whole
    point of the production accessor is that it never forms ``delta`` that way: ``tau`` reaches
    ~1.4e4 Mpc, so half an ulp of the primitive is 9e-4 rad of phase at ``k = 3e8/Mpc`` *however
    short the interval*, and the campaign's node table is stored as (hi, lo) pairs precisely so
    that a short baseline is not scored against the size of the whole primitive (review §13.3,
    README §2 (c)). That failure is a *relative* one: the rounding of ``f(z_b) - f(z_a)`` is
    ~1 ulp of ``max|f|``, i.e. ~1 ulp of the accumulated phase once multiplied by ``k``.

    In a test fixture whose accumulated phase ``x = k c_s tau`` stays below ~1e4 rad, one ulp of
    the phase is ~2e-12 rad, which is below every threshold such a fixture asserts and below the
    spline error of the residual it is there to exhibit. (At ``x = 1e6``, where
    ``test_tk_source_functions`` exercises the consumer decomposition, it is ~1.2e-10 rad --
    still three orders below that test's 1e-7 rad bound, but no longer negligible against a
    tighter one.) Do not use this class for anything that runs on the production background:
    there ``x`` reaches 1.4e10 and the ulp is the dominant error.

    :param f: the primitive, a callable of ``z``
    :param label: a name, for error messages and diagnostics
    """

    def __init__(self, f, label: str = ""):
        self._f = f
        self._label = label

    def __call__(self, z: float) -> float:
        return float(self._f(float(z)))

    def delta(self, z_a: float, z_b: float) -> float:
        return float(self._f(float(z_b))) - float(self._f(float(z_a)))

    @property
    def label(self) -> str:
        return self._label


# ---------------------------------------------------------------------------------------------
# error definitions (README §6; used unchanged by every later prompt)
# ---------------------------------------------------------------------------------------------


def phase_error(theta: float, theta_ref: float) -> float:
    """
    **Phase error**: the absolute difference, in radians, of the *unwrapped* phase against the
    reference evaluated at the supplied double ``z``.

    Both arguments must already be unwrapped (``div*2pi + mod``, or the raw accumulated phase);
    nothing here reduces mod 2pi, because the campaign's errors are routinely many cycles.
    """
    return fabs(float(theta) - float(theta_ref))


def difference_error(delta: float, delta_ref: float) -> float:
    """
    **Difference error**: the relative error of an *interval* quantity against the reference
    interval, never against the absolute primitive.

    ``delta`` and ``delta_ref`` are both increments (for example ``tau.delta(z_a, z_b)``); the
    denominator is ``|delta_ref|``, so a short baseline is scored against its own size.
    """
    denom = fabs(float(delta_ref))
    if denom == 0.0:
        raise ValueError("difference_error: reference interval is zero")
    return fabs(float(delta) - float(delta_ref)) / denom


def envelope_relative_error(value: float, value_ref: float, envelope: float) -> float:
    """
    **Envelope-relative error**: the error of ``G`` or ``T`` divided by the *local
    Liouville-Green envelope*, not by the value, so that a zero crossing does not manufacture an
    infinite relative error.
    """
    denom = fabs(float(envelope))
    if denom == 0.0:
        raise ValueError("envelope_relative_error: envelope is zero")
    return fabs(float(value) - float(value_ref)) / denom


# ---------------------------------------------------------------------------------------------
# reference loader
# ---------------------------------------------------------------------------------------------


def load_references(path: Optional[Path] = None) -> dict:
    """
    Load ``wkb_reference_data.json`` as nested dicts, keyed
    model -> quantity -> checkpoint (see the file's own ``"schema"`` block, and the log of
    prompt 01, for the layout and the sign conventions).
    """
    with open(path if path is not None else REFERENCE_DATA_PATH, "r") as f:
        return json.load(f)
