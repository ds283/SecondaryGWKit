from collections import namedtuple
from math import sqrt, log
from typing import Optional, List, Union

import numpy as np
import ray
from ray import ObjectRef
from scipy.interpolate import make_interp_spline

from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyConcepts import redshift_array, redshift, wavenumber
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.supervisors.base import RHS_timer, IntegrationSupervisor
from Units.base import UnitsLike
from config.defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

# Gauss-Legendre order per production interval for the conformal-time table. Fixed by measurement
# in prompts/GkTk-remedial/logs/02-qcd-residual-convergence.md (N_tau = 4): order 4 is at the
# double-precision floor on LambdaCDM (review §7) and, once every interval is split at the
# cosmology's break points, on QCD_Cosmology as well. Raising it buys nothing.
TAU_GAUSS_ORDER = 4

# The IntegrationSolver label under which the table is registered in main.py; "stepping" carries
# the Gauss order, following the existing "<label>-stepping<n>" convention.
TAU_SOLVER_LABEL_BASE = "cumulative-GL"
TAU_SOLVER_LABEL = f"{TAU_SOLVER_LABEL_BASE}-stepping{TAU_GAUSS_ORDER}"

# Settings for the private grid on which _build_derivative fits its splines when a cosmology model
# supplies no analytic derivative. See the comment in compute_background().
# - number of extra points added beyond each end of the production grid
DERIVATIVE_FIT_PAD_POINTS = 12
# - number of sub-intervals each production interval is divided into
DERIVATIVE_FIT_REFINE = 3
# - the low-end padding is never allowed to take 1+z below this multiple of 1+z_min. With z_min >= 0
#   this keeps the padded grid at z >= -0.1, inside the range over which the GenericEOS models build
#   their T(z) spline (DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2)
DERIVATIVE_FIT_PAD_FLOOR = 0.9
# - the padding at either end never extends the fitted range in log(1+z) by more than this fraction
DERIVATIVE_FIT_PAD_FRACTION = 0.05
# - degree of the interpolating spline that is differentiated
DERIVATIVE_SPLINE_ORDER = 5


def _build_derivative_fit_grid(z_sample: redshift_array):
    """
    Build the private, padded and refined grid in x = log(1+z) on which compute_background fits the
    splines it differentiates.
    Returns (fit_x, fit_z, fit_select, sample_order) where fit_x[fit_select][sample_order] are the
    production redshifts in the order they appear in z_sample.
    """
    z_prod = np.array([z.z for z in z_sample], dtype=float)

    # z_sample may run in either direction; fit on an ascending grid and record how to get back
    ascending = np.argsort(z_prod)
    sample_order = np.empty_like(ascending)
    sample_order[ascending] = np.arange(len(z_prod))

    x_prod = np.log1p(z_prod[ascending])

    # refine: insert DERIVATIVE_FIT_REFINE-1 equally spaced points inside each production interval
    refine = max(int(DERIVATIVE_FIT_REFINE), 1)
    if refine > 1:
        subdivided = x_prod[:-1, None] + (x_prod[1:, None] - x_prod[:-1, None]) * (
            np.arange(refine)[None, :] / refine
        )
        x_core = np.concatenate([subdivided.reshape(-1), x_prod[-1:]])
    else:
        x_core = x_prod

    # pad: extend beyond each end at the local grid spacing, clamping the low end so that
    # 1+z cannot approach (or cross) zero on a coarse grid
    pad = max(int(DERIVATIVE_FIT_PAD_POINTS), 0)
    if pad > 0 and len(x_core) >= 2:
        # on any sensibly dense grid the padding is pad grid spacings; the two caps only bite on a
        # very coarse or very short grid, and mirror the 5% buffer LambdaCDM_GenericEOS uses
        max_extension = DERIVATIVE_FIT_PAD_FRACTION * (x_core[-1] - x_core[0])
        h_lo = min(
            x_core[1] - x_core[0],
            -log(DERIVATIVE_FIT_PAD_FLOOR) / pad,
            max_extension / pad,
        )
        h_hi = min(x_core[-1] - x_core[-2], max_extension / pad)
        lo = x_core[0] - h_lo * np.arange(pad, 0, -1)
        hi = x_core[-1] + h_hi * np.arange(1, pad + 1)
        fit_x = np.concatenate([lo, x_core, hi])
    else:
        pad = 0
        fit_x = x_core

    fit_select = pad + refine * np.arange(len(x_prod))

    fit_z = np.expm1(fit_x)

    # restore the production redshifts exactly at the points we will select, so that a cosmology
    # supplying analytic derivatives is evaluated at exactly the requested z (no expm1(log1p(z))
    # round trip) and its stored values are unchanged
    fit_x[fit_select] = x_prod
    fit_z[fit_select] = z_prod[ascending]

    return fit_x, fit_z, fit_select, sample_order


ModelFunctions = namedtuple(
    "ModelFunctions",
    [
        "Hubble",
        "epsilon",
        "d_epsilon_dz",
        "d2_epsilon_dz2",
        "wBackground",
        "wPerturbations",
        "tau",
        "T_photon",
        "d_lnH_dz",
        "d2_lnH_dz2",
        "d3_lnH_dz3",
        "d_wPerturbations_dz",
        "d2_wPerturbations_dz2",
    ],
)


def _cosmology_break_points(cosmology, z_lo: float, z_hi: float) -> np.ndarray:
    """
    The points in u = log(1+z), strictly inside (log(1+z_lo), log(1+z_hi)), at which the
    cosmology's background quantities lose smoothness, as an ascending array; empty if the
    cosmology declares none. Duck-typed like the analytic-derivative shortcuts below: a cosmology
    that does not implement ``integration_break_points`` (LambdaCDM, the test stand-ins) is
    treated as smooth. LambdaCDM_GenericEOS implements it (the T(z) spline knots and the
    equation-of-state branch temperatures); see prompts/GkTk-remedial/logs/02 for why every
    Gauss panel has to be split there.
    """
    method = getattr(cosmology, "integration_break_points", None)
    if method is None:
        return np.empty(0, dtype=float)
    return np.asarray(method(z_lo, z_hi), dtype=float)


@ray.remote
def compute_background(
    cosmology: BaseCosmology,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> dict:
    """
    Tabulate the background quantities on ``z_sample``.

    The conformal time tau = a_0 eta (with dtau/dz = -1/H) is *not* integrated as an ODE: it is
    accumulated as a Gauss-Legendre cumulative table on the sample grid itself
    (``ComputeTargets.cumulative_table.CumulativeTable``, order ``TAU_GAUSS_ORDER`` per interval,
    split at the cosmology's break points), held as double-double (hi, lo) pairs so that
    downstream phase differences ``k [tau(z_a) - tau(z_b)]`` over short baselines do not inherit
    the rounding of the absolute tau (review §7, §13.2, §13.3). The absolute normalisation is the
    author's radiation-era closed form ``tau_init = sqrt(3) M_P / sqrt(rho(z_init)) (1 + z_init)``
    at the top of the grid, added to the table in double-double.

    ``atol`` and ``rtol`` are accepted for signature compatibility with ``BackgroundModel.compute``
    and because they remain part of the datastore lookup key; the table has no tolerances.
    """
    z_nodes = np.array(z_sample.as_float_list(), dtype=float)
    z_init = float(z_nodes[0])
    z_stop = float(z_nodes[-1])

    break_points = _cosmology_break_points(cosmology, z_stop, z_init)

    with IntegrationSupervisor() as supervisor:

        def inverse_Hubble(z: float) -> float:
            with RHS_timer(supervisor):
                return 1.0 / cosmology.Hubble(z)

        table = CumulativeTable(
            z_nodes,
            inverse_Hubble,
            TAU_GAUSS_ORDER,
            break_points=break_points,
            label="tau",
        )

        # the author's radiation-era asymptote for tau at the top of the grid; a convention, kept
        rho_init = cosmology.rho(z_init)
        tau_init = (
            sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) * (1.0 + z_init)
        )
        table = table.shifted(tau_init)

    tau_hi_sample = [float(v) for v in table.hi]
    tau_lo_sample = [float(v) for v in table.lo]

    # each BaseCosmology instance provides methods to evaluate H(z), rho(z), and the value of the equation of state
    # for the background and perturbations
    H_sample = [cosmology.Hubble(z.z) for z in z_sample]
    rho_sample = [cosmology.rho(z.z) for z in z_sample]
    T_photon_sample = [cosmology.T_photon(z.z) for z in z_sample]
    wBackground_sample = [cosmology.wBackground(z.z) for z in z_sample]
    wPerturbations_sample = [cosmology.wPerturbations(z.z) for z in z_sample]

    # further, each BaseCosmology instance may provide methods to evaluate the derivatives of H(z) and w(z), but if it doesn't,
    # we estimate these derivatives using a spline.
    #
    # A spline fitted only on the production grid is badly biased at the two ends: the not-a-knot end
    # condition has no data to constrain it, and each differentiation of a stacked derivative amplifies
    # that error. To avoid this we fit on a *private* grid that is padded beyond both ends of the
    # production grid and refined between its points, evaluate the derivative there, and select the
    # production points only at the end. The padding has to be carried through the whole stack
    # (d2_lnH_dz2 is built from d_lnH_dz, d3_lnH_dz3 from d2_lnH_dz2, ...), so every level of the stack
    # is computed on the padded grid; only the returned samples are truncated.
    # LambdaCDM_GenericEOS._build_T_z_spline already buffers its own grid for exactly this reason.

    fit_x, fit_z, fit_select, sample_order = _build_derivative_fit_grid(z_sample)
    fit_opz = fit_z + 1.0

    # a not-a-knot quintic is used rather than the cubic of the original implementation: with the
    # stacked derivatives the cubic's discontinuous third derivative is the dominant residual error
    # once the end bias has been removed by padding
    fit_k = DERIVATIVE_SPLINE_ORDER if len(fit_x) >= DERIVATIVE_SPLINE_ORDER + 1 else 3

    def _build_derivative(attr: str, f_to_diff=None, fit_sample_to_diff=None):
        """
        Evaluate a derivative on the *padded fit grid*, either from the cosmology's own analytic
        method, or by differentiating a spline through the supplied samples/function.
        """
        if f_to_diff is None and fit_sample_to_diff is None:
            raise RuntimeError(
                "compute_background._build_derivative: f_to_diff and fit_sample_to_diff cannot both be None"
            )

        if hasattr(cosmology, attr):
            method = getattr(cosmology, attr)
            return np.array([method(z) for z in fit_z])

        if f_to_diff is not None:
            y_data = np.array([f_to_diff(z) for z in fit_z])
        else:
            y_data = np.asarray(fit_sample_to_diff)

        deriv = make_interp_spline(fit_x, y_data, k=fit_k).derivative()

        # the spline computes d/d(log(1+z)), so divide by 1+z to obtain the raw z-derivative
        return np.asarray(deriv(fit_x)) / fit_opz

    def _truncate(fit_values) -> List[float]:
        """Select the production grid points, restoring the ordering of z_sample."""
        return [float(v) for v in np.asarray(fit_values)[fit_select][sample_order]]

    d_lnH_dz_fit = _build_derivative(
        "d_lnH_dz", f_to_diff=lambda z: log(cosmology.Hubble(z))
    )
    d2_lnH_dz2_fit = _build_derivative("d2_lnH_dz2", fit_sample_to_diff=d_lnH_dz_fit)
    d3_lnH_dz3_fit = _build_derivative("d3_lnH_dz3", fit_sample_to_diff=d2_lnH_dz2_fit)
    d_wPerturbations_dz_fit = _build_derivative(
        "d_wPerturbations_dz", f_to_diff=cosmology.wPerturbations
    )
    d2_wPerturbations_dz2_fit = _build_derivative(
        "d2_wPerturbations_dz2", fit_sample_to_diff=d_wPerturbations_dz_fit
    )

    d_lnH_dz_sample = _truncate(d_lnH_dz_fit)
    d2_lnH_dz2_sample = _truncate(d2_lnH_dz2_fit)
    d3_lnH_dz3_sample = _truncate(d3_lnH_dz3_fit)
    d_wPerturbations_dz_sample = _truncate(d_wPerturbations_dz_fit)
    d2_wPerturbations_dz2_sample = _truncate(d2_wPerturbations_dz2_fit)

    return {
        # compute_steps is the node count and RHS_evaluations the number of Hubble evaluations
        # spent building the table (one per Gauss abscissa)
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=len(table),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "tau_hi_sample": tau_hi_sample,
        "tau_lo_sample": tau_lo_sample,
        "tau_order": TAU_GAUSS_ORDER,
        "H_sample": H_sample,
        "rho_sample": rho_sample,
        "T_photon_sample": T_photon_sample,
        "wBackground_sample": wBackground_sample,
        "wPerturbations_sample": wPerturbations_sample,
        "d_lnH_dz_sample": d_lnH_dz_sample,
        "d2_lnH_dz2_sample": d2_lnH_dz2_sample,
        "d3_lnH_dz3_sample": d3_lnH_dz3_sample,
        "d_wPerturbations_dz_sample": d_wPerturbations_dz_sample,
        "d2_wPerturbations_dz2_sample": d2_wPerturbations_dz2_sample,
        "solver_label": TAU_SOLVER_LABEL,
    }


class TablePrimitive:
    """
    A background primitive held as a ``CumulativeTable``: ``functions.tau`` (and, from prompt 04
    of ``prompts/GkTk-remedial``, ``cs_tau`` and ``friction_F``).

    Two accessors, both returning plain floats:

    * ``primitive(z)`` -- the absolute value pointwise, for the existing consumers
      (``compute_analytic_G/T``, the eta limits in ``QuadSourceIntegral``, main.py's Bessel
      ``x_max``). Carries half an ulp of the primitive itself.
    * ``primitive.delta(z_a, z_b) = primitive(z_b) - primitive(z_a)``, the interval accessor the
      WKB phase uses. Formed from the double-double node table and local Gauss partials, never
      as a difference of two pointwise values (README §2 (c); ``CumulativeTable.delta``).
      For tau this is ``int_{z_b}^{z_a} dz/H``, positive when ``z_b < z_a``.
    """

    def __init__(self, table: CumulativeTable, label: str):
        self._table = table
        self._label = label

    def __call__(self, z: float) -> float:
        return self._table.value(float(z))

    def delta(self, z_a: float, z_b: float) -> float:
        return self._table.delta(float(z_a), float(z_b))

    @property
    def table(self) -> CumulativeTable:
        return self._table

    @property
    def label(self) -> str:
        return self._label


class BackgroundModel(DatastoreObject):
    """
    Encapsulates the time history of a cosmological model.
    This bakes-in all the quantities we need such as the conformal time \tau (for the WKB phases,
    and for analytic approximations to the transfer functions and Green's functions). \tau is
    tabulated on the sample grid as a double-double Gauss-Legendre cumulative table rather than
    integrated as an ODE, and ``functions.tau`` exposes it both pointwise, ``tau(z)``, and as an
    interval, ``tau.delta(z_a, z_b)`` (see ``TablePrimitive``).
    It also means we have an explicit record in the database of the values of H(z), w(z), etc.,
    that yielded a particular set of results
    """

    # the solver label compute_background reports, and the registration main.py must make
    TAU_GAUSS_ORDER = TAU_GAUSS_ORDER
    TAU_SOLVER_LABEL_BASE = TAU_SOLVER_LABEL_BASE
    TAU_SOLVER_LABEL = TAU_SOLVER_LABEL

    def __init__(
        self,
        payload,
        solver_labels: dict,
        cosmology: BaseCosmology,
        atol: tolerance,
        rtol: tolerance,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._solver_labels = solver_labels
        self._z_sample = z_sample

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None
            self._solver = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._data: Optional[IntegrationData] = payload["data"]
            self._solver: Optional[IntegrationSolver] = payload["solver"]
            self._values: Optional[List[BackgroundModelValue]] = payload["values"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._cosmology = cosmology
        self._units = cosmology.units

        self._functions = None

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def cosmology(self):
        return self._cosmology

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_sample(self):
        return self._z_sample

    def efolds_subh(self, k: wavenumber, z: Union[redshift, float]) -> float:
        if isinstance(z, redshift):
            z_float = z.z
        else:
            z_float = float(z)

        H = self.functions.Hubble(z_float)
        return log((1.0 + z_float) * k.k / H)

    @property
    def data(self) -> IntegrationData:
        if self.values is None:
            raise RuntimeError("values have not yet been populated")

        return self._data

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    @property
    def functions(self) -> ModelFunctions:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        def _build_func(attr: str):
            if hasattr(self._cosmology, attr):
                return getattr(self._cosmology, attr)

            data = [(log(1.0 + v.z.z), getattr(v, attr)) for v in self.values]
            data.sort(key=lambda pair: pair[0])

            x_data, y_data = zip(*data)
            spline = make_interp_spline(x_data, y_data)
            return ZSplineWrapper(
                spline,
                label=attr,
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        # tau is reconstructed from the persisted (hi, lo) limbs with no quadrature; the integrand
        # is needed only for off-grid partials (the per-object anchor z_init of a numeric hand-over,
        # RECONCILIATION.md §2 item 5). No cosmology supplies an analytic tau, and a pointwise
        # analytic tau could not supply delta at the required accuracy, so there is no
        # hasattr(cosmology, "tau") shortcut here (RECONCILIATION.md §2 item 2). A cubic spline of
        # the nodes -- the previous accessor -- is 1.4e-9 relative off-grid, ~2 rad of phase at
        # k = 1e5/Mpc (review §7, §13.2).
        tau_func = self._build_tau_primitive()

        T_photon_func = _build_func("T_photon")
        d_lnH_dz_func = _build_func("d_lnH_dz")
        d2_lnH_dz2_func = _build_func("d2_lnH_dz2")
        d3_lnH_dz3_func = _build_func("d3_lnH_dz3")
        d_wPerturbations_dz_func = _build_func("d_wPerturbations_dz")
        d2_wPerturbations_dz2_func = _build_func("d2_wPerturbations_dz2")

        def epsilon(z: float) -> float:
            """
            Evaluate the conventional epsilon parameter eps = -dot(H)/H^2
            :param z: redshift of evaluation
            :return:
            """
            one_plus_z = 1.0 + z
            return one_plus_z * d_lnH_dz_func(z)

        def d_epsilon_dz(z: float) -> float:
            """
            Evaluate the z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return d_lnH_dz_func(z) + one_plus_z * d2_lnH_dz2_func(z)

        def d2_epsilon_dz2(z: float) -> float:
            """
            Evaluate the 2nd z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return 2.0 * d2_lnH_dz2_func(z) + one_plus_z * d3_lnH_dz3_func(z)

        # the underlying cosmology object is guaranteed to provide methods for Hubble, rho, wBackground, and wPerturbations.
        # it may or may not provide methods for other quantities.
        # The functions built above will pass through to the underlying cosmology object when it can perform the computation
        # (so we can make use of e.g. analytic expressions where they are available), but otherwise we use splines
        # constructed by differentiation (usually of another spline). Differentiating a spline gives us much less noisy
        # results than finite difference formulae.
        self._functions = ModelFunctions(
            Hubble=self._cosmology.Hubble,
            epsilon=epsilon,
            d_epsilon_dz=d_epsilon_dz,
            d2_epsilon_dz2=d2_epsilon_dz2,
            wBackground=self._cosmology.wBackground,
            wPerturbations=self._cosmology.wPerturbations,
            tau=tau_func,
            T_photon=T_photon_func,
            d_lnH_dz=d_lnH_dz_func,
            d2_lnH_dz2=d2_lnH_dz2_func,
            d3_lnH_dz3=d3_lnH_dz3_func,
            d_wPerturbations_dz=d_wPerturbations_dz_func,
            d2_wPerturbations_dz2=d2_wPerturbations_dz2_func,
        )

    def _build_tau_primitive(self) -> TablePrimitive:
        values = sorted(self.values, key=lambda v: v.z.z, reverse=True)
        z_nodes = [v.z.z for v in values]
        cosmology = self._cosmology

        def inverse_Hubble(z: float) -> float:
            return 1.0 / cosmology.Hubble(z)

        table = CumulativeTable(
            z_nodes,
            inverse_Hubble,
            TAU_GAUSS_ORDER,
            hi=[v.tau for v in values],
            lo=[v.tau_lo for v in values],
            break_points=_cosmology_break_points(cosmology, z_nodes[-1], z_nodes[0]),
            label="tau",
        )
        return TablePrimitive(table, label="tau")

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values has not yet been populated")

        if self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        self._compute_ref = compute_background.remote(
            self.cosmology,
            self._z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkWKBIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._data = data["data"]
        self._values = self.values_from_payload(self._z_sample, data)
        self._solver = self._solver_labels[data["solver_label"]]

        return True

    @staticmethod
    def values_from_payload(
        z_sample: redshift_array, data: dict
    ) -> List["BackgroundModelValue"]:
        """
        Build the per-redshift ``BackgroundModelValue`` list from a ``compute_background``
        payload, in ``z_sample`` order. Shared by ``store()`` and the offline tests.
        """
        H_sample = data["H_sample"]
        wB_sample = data["wBackground_sample"]
        wP_sample = data["wPerturbations_sample"]
        rho_sample = data["rho_sample"]
        T_photon_sample = data["T_photon_sample"]
        tau_hi_sample = data["tau_hi_sample"]
        tau_lo_sample = data["tau_lo_sample"]

        d_lnH_ds_sample = data["d_lnH_dz_sample"]
        d2_lnH_dz2_sample = data["d2_lnH_dz2_sample"]
        d3_lnH_dz3_sample = data["d3_lnH_dz3_sample"]

        d_wPerturbations_dz_sample = data["d_wPerturbations_dz_sample"]
        d2_wPerturbations_dz2_sample = data["d2_wPerturbations_dz2_sample"]

        values = []
        for i in range(len(H_sample)):
            values.append(
                BackgroundModelValue(
                    None,
                    z_sample[i],
                    Hubble=H_sample[i],
                    wBackground=wB_sample[i],
                    wPerturbations=wP_sample[i],
                    rho=rho_sample[i],
                    tau=tau_hi_sample[i],
                    T_photon=T_photon_sample[i],
                    d_lnH_dz=d_lnH_ds_sample[i],
                    d2_lnH_dz2=d2_lnH_dz2_sample[i],
                    d3_lnH_dz3=d3_lnH_dz3_sample[i],
                    d_wPerturbations_dz=d_wPerturbations_dz_sample[i],
                    d2_wPerturbations_dz2=d2_wPerturbations_dz2_sample[i],
                    tau_lo=tau_lo_sample[i],
                )
            )
        return values


class BackgroundModelValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        Hubble: float,
        wBackground: float,
        wPerturbations: float,
        rho: float,
        tau: float,
        T_photon: float,
        d_lnH_dz: float,
        d2_lnH_dz2: Optional[float] = None,
        d3_lnH_dz3: Optional[float] = None,
        d_wPerturbations_dz: Optional[float] = None,
        d2_wPerturbations_dz2: Optional[float] = None,
        tau_lo: float = 0.0,
    ):
        """
        ``tau`` is the high limb and ``tau_lo`` the low limb of the double-double conformal time
        at this redshift (``tau + tau_lo`` is the value to ~1e-32 relative). ``tau_lo`` is a
        keyword with a zero default so that stand-ins and the factory's ``build()`` path keep
        constructing.
        """
        DatastoreObject.__init__(self, store_id)

        self._z = z

        self._Hubble: float = Hubble
        self._wBackground: float = wBackground
        self._wPerturbations: float = wPerturbations

        self._rho: float = rho
        self._tau: float = tau
        self._tau_lo: float = tau_lo
        self._T_photon: float = T_photon

        self._d_lnH_dz: float = d_lnH_dz
        self._d2_lnH_dz2: float = d2_lnH_dz2
        self._d3_lnH_dz3: float = d3_lnH_dz3

        self._d_wPerturbations_dz: float = d_wPerturbations_dz
        self._d2_wPerturbations_dz2: float = d2_wPerturbations_dz2

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def Hubble(self) -> float:
        return self._Hubble

    @property
    def wBackground(self) -> float:
        return self._wBackground

    @property
    def wPerturbations(self) -> float:
        return self._wPerturbations

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def tau(self) -> float:
        """The high limb of the double-double conformal time at this redshift."""
        return self._tau

    @property
    def tau_lo(self) -> float:
        """The low limb of the double-double conformal time at this redshift."""
        return self._tau_lo

    @property
    def T_photon(self) -> float:
        return self._T_photon

    @property
    def d_lnH_dz(self) -> float:
        return self._d_lnH_dz

    @property
    def d2_lnH_dz2(self) -> Optional[float]:
        return self._d2_lnH_dz2

    @property
    def d3_lnH_dz3(self) -> Optional[float]:
        return self._d3_lnH_dz3

    @property
    def d_wPerturbations_dz(self) -> Optional[float]:
        return self._d_wPerturbations_dz

    @property
    def d2_wPerturbations_dz2(self) -> Optional[float]:
        return self._d2_wPerturbations_dz2


class ModelProxy:
    def __init__(self, model: BackgroundModel):
        self._ref: ObjectRef = ray.put(model)

        self._store_id: int = model.store_id if model.available else None

        self._units: UnitsLike = model.cosmology.units
        self._cosmology: BaseCosmology = model.cosmology

    @property
    def store_id(self) -> int:
        return self._store_id

    @property
    def available(self) -> bool:
        return self._store_id is not None

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def cosmology(self) -> BaseCosmology:
        return self._cosmology

    def get(self) -> BackgroundModel:
        """
        The return value should only be held locally and not persisted, otherwise the entire
        BackgroundModel instance may be serialized when it is passed around by Ray.
        That would defeat the purpose of the proxy.
        :return:
        """
        return ray.get(self._ref)
