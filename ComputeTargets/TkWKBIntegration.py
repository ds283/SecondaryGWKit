from math import sqrt, log
from typing import Optional, List

import ray

from ComputeTargets import ModelProxy, BackgroundModel
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEffPrime_dz
from CosmologyConcepts import wavenumber_exit_time, redshift_array, wavenumber
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from Units import check_units
from defaults import DEFAULT_FLOAT_PRECISION

THETA_INDEX = 0
Q_INDEX = 0
EXPECTED_SOL_LENGTH = 1

# how large do we allow the WKB phase theta to become, before we terminate the integration and
# reset to a small value?
# we need to resolve the phase on the scale of (0, 2pi), otherwise we will compute cos(theta),
# sin(theta) and hence the transfer function incorrectly
DEFAULT_PHASE_RUN_LENGTH = 1e4

# how large do we allow omega_WKB_sq to get before switching to a "stage #2" integration?
DEFAULT_OMEGA_WKB_SQ_MAX = 1e6


class TkWKBIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the Liouville-Green (WKB)
    phase function for the transfer function
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_init: Optional[float] = None,
        T_init: Optional[float] = None,
        Tprime_init: Optional[float] = None,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model)

        self._solver_labels = solver_labels
        self._z_sample = z_sample

        self._z_init = z_init
        self._T_init = T_init
        self._Tprime_init = Tprime_init

        if payload is None:
            DatastoreObject.__init__(self, None)

            self._stage_1 = IntegrationData(
                compute_time=None,
                compute_steps=None,
                RHS_evaluations=None,
                mean_RHS_time=None,
                max_RHS_time=None,
                min_RHS_time=None,
            )
            self._stage_2 = IntegrationData(
                compute_time=None,
                compute_steps=None,
                RHS_evaluations=None,
                mean_RHS_time=None,
                max_RHS_time=None,
                min_RHS_time=None,
            )

            self._has_WKB_violation = None
            self._WKB_violation_z = None
            self._WKB_violation_efolds_subh = None

            self._init_efolds_suph = None
            self._metadata = None

            self._solver = None

            self._sin_coeff = None
            self._cos_coeff = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._stage_1 = payload["stage_1_data"]
            self._stage_2 = payload["stage_2_data"]

            self._has_WKB_violation = payload["has_WKB_violation"]
            self._WKB_violation_z = payload["WKB_violation_z"]
            self._WKB_violation_efolds_subh = payload["WKB_violation_efolds_subh"]

            self._init_efolds_subh = payload["init_efolds_subh"]
            self._metadata = payload["metadata"]

            self._solver = payload["solver"]

            self._sin_coeff = payload["sin_coeff"]
            self._cos_coeff = payload["cos_coeff"]
            self._values = payload["values"]

        # we only expect the WKB method to give a good approximation on sufficiently subhorizon
        # scales, so we should validate that none of the sample points are too close to horizon re-entry
        if z_sample is not None:
            if z_init is None:
                raise ValueError(
                    f"If z_sample is not None, then z_init must also be set"
                )

            z_limit: float = k.z_exit_subh_e3

            for z in z_sample:
                # check that each sample redshift is not too close to the horizon, or outside it, for this k-mode
                z_float = float(z)

                if z_float > z_limit - DEFAULT_FLOAT_PRECISION:
                    raise ValueError(
                        f"Specified response redshift z={z_float:.5g} is closer than 3-folds to horizon re-entry for wavenumber k={k_wavenumber:.5g}/Mpc"
                    )

                # also, check that each response redshift is later than the specified initial redshift
                if z_float > z_init:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} exceeds initial redshift z={z_init:.5g}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._model_proxy = model

        self._k_exit = k

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def z_exit(self) -> float:
        return self._k_exit.z_exit

    @property
    def z_init(self) -> float:
        return self._z_init

    @property
    def T_init(self) -> Optional[float]:
        return self._T_init

    @property
    def Tprime_init(self) -> Optional[float]:
        return self._Tprime_init

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_sample(self):
        return self._z_sample

    @property
    def stage_1_data(self) -> IntegrationData:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._stage_1

    @property
    def stage_2_data(self) -> IntegrationData:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._stage_2

    @property
    def metadata(self) -> dict:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._metadata

    @property
    def has_WKB_violation(self) -> bool:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._has_WKB_violation

    @property
    def WKB_violation_z(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        if not self._has_WKB_violation:
            return None

        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        if not self._has_WKB_violation:
            return None

        return self._WKB_violation_efolds_subh

    @property
    def init_efolds_subh(self) -> float:
        if self._init_efolds_subh is None:
            raise RuntimeError("init_efolds_subh has not yet been populated")

        return self._init_efolds_subh

    @property
    def sin_coeff(self) -> float:
        if self._sin_coeff is None:
            raise RuntimeError("sin_coeff has not yet been populated")

        return self._sin_coeff

    @property
    def cos_coeff(self) -> float:
        if self._cos_coeff is None:
            raise RuntimeError("cos_coeff has not yet been populated")

        return self._cos_coeff

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "TkWKBIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "TkWKBIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_sample is None or self._z_init is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_init or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        model: BackgroundModel = self._model_proxy.get()

        # check that WKB approximation is likely to be valid at the specified initial time
        omega_WKB_sq_init = Tk_omegaEff_sq(model, self._k_exit.k.k, self._z_init)
        d_ln_omega_WKB_init = Tk_d_ln_omegaEffPrime_dz(
            model, self._k_exit.k.k, self.z_init
        )

        if omega_WKB_sq_init < 0.0:
            raise ValueError(
                f"omega_WKB^2 must be non-negative at the initial time (k={self._k_exit.k.k_inv_Mpc}/Mpc, z_init={self._z_init:.5g}, omega_WKB^2={omega_WKB_sq_init:.5g})"
            )

        WKB_criterion_init = d_ln_omega_WKB_init / sqrt(omega_WKB_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (TkWKBIntegration) k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc")
            print(
                f"|    WKB diagnostic |d(omega) / omega^2| exceeds unity at initial time (value={WKB_criterion_init:.5g}, z_init={self._z_init:.5g})"
            )
            print(f"     This may lead to meaningless results.")

        self._compute_ref = compute_Tk_WKB.remote(
            self._model_proxy,
            self._k_exit,
            self._z_init,
            self._z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "TkWKBIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate our internal fields from the payload
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._stage_1 = data["stage_1_data"]
        self._stage_2 = data["stage_2_data"]

        self._has_WKB_violation = data["has_WKB_violation"]
        self._WKB_violation_z = data["WKB_violation_z"]
        self._WKB_violation_efolds_subh = data["WKB_violation_efolds_subh"]

        self._metadata = data["metadata"]

        model: BackgroundModel = self._model_proxy.get()

        H_init = model.functions.Hubble(self._z_init)
        eps_init = model.functions.epsilon(self._z_init)

        one_plus_z_init = 1.0 + self._z_init
        k_over_aH = one_plus_z_init * self.k.k / H_init
        self._init_efolds_subh = log(k_over_aH)

        # can assume here that omega_WKB_sq_init > 0 and the WKB criterion < 1.
        # This has been checked in compute()
        omega_WKB_sq_init = Tk_omegaEff_sq(model, self._k_exit.k.k, self._z_init)
        d_ln_omega_WKB_init = Tk_d_ln_omegaEffPrime_dz(
            model, self._k_exit.k.k, self._z_init
        )

        # in the Green's function WKB calculation we adjust the phase so that the cos part of the
        # solution is absent. This is because when we compute Gr_k(z, z') for source z' inside the horizon,
        # the boundary conditions kill the cos part. Hence, when we fix the response time z and
        # assemble Gr_k for all possible values of z', we will pick up a discontinuity in the phase and
        # coefficient if we don't also kill the cos for z' outside the horizon.

        # there is no similar effect for the transfer function, but we still make the same shift
        # so that we only get a sin solution. This at least simplifies the representation a bit.

        omega_WKB_init = sqrt(omega_WKB_sq_init)
        sqrt_omega_WKB_init = sqrt(omega_WKB_init)

        num = sqrt_omega_WKB_init * self._G_init
        den = (
            self._Gprime_init
            + (self._G_init / 2.0) * (d_ln_omega_WKB_init + eps_init / one_plus_z_init)
        ) / sqrt_omega_WKB_init

        deltaTheta = atan2(num, den)
        alpha = sqrt(num * num + den * den)

        sin_deltaTheta = sin(deltaTheta)
        sgn_sin_deltaTheta = +1 if sin_deltaTheta >= 0.0 else -1
        sgn_G = +1 if self._G_init >= 0.0 else -1

        self._cos_coeff = 0.0
        self._sin_coeff = sgn_sin_deltaTheta * sgn_G * alpha

        # estimate tau at the source redshift
        H_source = model.functions.Hubble(self._z_source.z)
        tau_source = model.functions.tau(self._z_source.z)

        theta_div_2pi_sample = data["theta_div_2pi_sample"]
        theta_mod_2pi_sample = data["theta_mod_2pi_sample"]

        # shift theta by deltaTheta in order to put everything into the sin coefficient, rather than a linear
        # combination of sin and cos
        # We do this so that we get smooth functions of sin and cos when we reassemble the Green's functions as
        # functions of the source redshift (the ones with late source times are always expressed as pure sin with zero
        # cos mode)
        def wrap_theta(theta: float) -> Tuple[int, float]:
            if theta > 0.0:
                return +1, theta - TWO_PI
            if theta <= -TWO_PI:
                return -1, theta + TWO_PI

            return 0, theta

        theta_sample_shifts = [
            wrap_theta(theta_mod_2pi_sample + deltaTheta)
            for theta_mod_2pi_sample in theta_mod_2pi_sample
        ]
        theta_div_2pi_shift, theta_mod_2pi_sample = zip(*theta_sample_shifts)

        theta_div_2pi_shift_base = theta_div_2pi_shift[0]
        theta_div_2pi_sample = [
            d + shift - theta_div_2pi_shift_base
            for (d, shift) in zip(theta_div_2pi_sample, theta_div_2pi_shift)
        ]

        self._values = []
        for i in range(len(theta_mod_2pi_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = model.functions.Hubble(current_z_float)
            tau = model.functions.tau(current_z_float)
            wBackground = model.functions.wBackground(current_z_float)

            analytic_G_rad = compute_analytic_G(
                self.k.k, 1.0 / 3.0, tau_source, tau, H_source
            )
            analytic_Gprime_rad = compute_analytic_Gprime(
                self.k.k, 1.0 / 3.0, tau_source, tau, H_source, H
            )

            analytic_G_w = compute_analytic_G(
                self.k.k, wBackground, tau_source, tau, H_source
            )
            analytic_Gprime_w = compute_analytic_Gprime(
                self.k.k, wBackground, tau_source, tau, H_source, H
            )

            # should be safe to assume omega_WKB_sq > 0, since otherwise this would have been picked up during the integration
            omega_WKB_sq = Gk_omegaEff_sq(model, self._k_exit.k.k, current_z_float)
            omega_WKB = sqrt(omega_WKB_sq)

            d_ln_omega_WKB = Gk_d_ln_omegaEffPrime_dz(
                model, self._k_exit.k.k, current_z_float
            )
            WKB_criterion = fabs(d_ln_omega_WKB) / omega_WKB

            H_ratio = H_init / H
            norm_factor = sqrt(H_ratio / omega_WKB)

            # no need to include theta div 2pi in the calculation of G_WKB
            # (and indeed it may be more accurate if we don't)
            G_WKB = norm_factor * (
                self._cos_coeff * cos(theta_mod_2pi_sample[i])
                + self._sin_coeff * sin(theta_mod_2pi_sample[i])
            )

            # create new GkWKBValue object
            self._values.append(
                GkWKBValue(
                    None,
                    current_z,
                    H_ratio,
                    theta_mod_2pi_sample[i],
                    theta_div_2pi_sample[i],
                    omega_WKB_sq=omega_WKB_sq,
                    WKB_criterion=WKB_criterion,
                    G_WKB=G_WKB,
                    analytic_G_rad=analytic_G_rad,
                    analytic_Gprime_rad=analytic_Gprime_rad,
                    analytic_G_w=analytic_G_w,
                    analytic_Gprime_w=analytic_Gprime_w,
                    sin_coeff=self._sin_coeff,
                    cos_coeff=self._cos_coeff,
                    z_init=self._z_init,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True
