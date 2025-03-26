from math import log, sqrt, fabs, cos, sin, atan2
from typing import Optional, List

import ray

from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Gk import (
    compute_analytic_G,
    compute_analytic_Gprime,
)
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from Datastore import DatastoreObject
from LiouvilleGreen.WKBtools import shift_theta_sample
from LiouvilleGreen.constants import TWO_PI
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.integrators.WKB_phase_function import WKB_phase_function
from Units import check_units
from defaults import (
    DEFAULT_FLOAT_PRECISION,
)


class GkWKBIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the Liouville-Green (WKB)
    phase function for the tensor Green's function
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_init: Optional[float] = None,
        G_init: Optional[float] = 0.0,
        Gprime_init: Optional[float] = 1.0,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model)

        self._solver_labels = solver_labels
        self._z_sample = z_sample

        self._z_init = z_init
        self._G_init = G_init
        self._Gprime_init = Gprime_init

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
            z_limit: float = k.z_exit_subh_e3
            z_initial_float: float = z_init if z_init is not None else z_source.z

            for z in z_sample:
                # check that each sample redshift is not too close to the horizon, or outside it, for this k-mode
                z_float = float(z)

                if z_float > z_limit - DEFAULT_FLOAT_PRECISION:
                    raise ValueError(
                        f"Specified response redshift z={z_float:.5g} is closer than 3-folds to horizon re-entry for wavenumber k={k_wavenumber:.5g}/Mpc"
                    )

                # also, check that each response redshift is later than the specified source redshift
                if z_float > z_initial_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} exceeds initial redshift z={z_initial_float:.5g}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._model_proxy = model

        self._k_exit = k
        self._z_source = z_source

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
        if self._z_init is not None:
            return self._z_init

        return self._z_source.z

    @property
    def G_init(self) -> Optional[float]:
        return self._G_init

    @property
    def Gprime_init(self) -> Optional[float]:
        return self._Gprime_init

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_source(self):
        return self._z_source

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
                "GkWKBIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkWKBIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_source is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_source or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        model: BackgroundModel = self._model_proxy.get()

        initial_z = self._z_init if self._z_init is not None else self._z_source.z

        # check that WKB approximation is likely to be valid at the specified initial time
        omega_sq_init = Gk_omegaEff_sq(model, self._k_exit.k.k, initial_z)
        d_ln_omega_init = Gk_d_ln_omegaEff_dz(model, self._k_exit.k.k, initial_z)

        if omega_sq_init < 0.0:
            raise ValueError(
                f"omega_WKB^2 must be non-negative at the initial time (k={self._k_exit.k.k_inv_Mpc}/Mpc, z_init={initial_z:.5g}, omega_WKB^2={omega_sq_init:.5g})"
            )

        WKB_criterion_init = d_ln_omega_init / sqrt(omega_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (GkWKBIntegration) k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc")
            print(
                f"|    WKB diagnostic |d(omega) / omega^2| exceeds unity at initial time (value={WKB_criterion_init:.5g}, z_init={initial_z:.5g})"
            )
            print(f"     This may lead to meaningless results.")

        self._compute_ref = WKB_phase_function.remote(
            self._model_proxy,
            self._k_exit,
            initial_z,
            self._z_sample,
            omega_sq=Gk_omegaEff_sq,
            d_ln_omega_dz=Gk_d_ln_omegaEff_dz,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            task_label="compute_Gk_WKB_phase",
            object_label="Gr_k(z, z')",
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

        initial_z = self._z_init if self._z_init is not None else self._z_source.z

        H_init = model.functions.Hubble(initial_z)
        eps_init = model.functions.epsilon(initial_z)

        one_plus_z_init = 1.0 + initial_z
        k_over_aH = one_plus_z_init * self.k.k / H_init
        self._init_efolds_subh = log(k_over_aH)

        # can assume here that omega_WKB_sq_init > 0 and the WKB criterion < 1.
        # This has been checked in compute()
        omega_sq_init = Gk_omegaEff_sq(model, self._k_exit.k.k, initial_z)
        d_ln_omega_init = Gk_d_ln_omegaEff_dz(model, self._k_exit.k.k, initial_z)

        # we try to adjust the phase so that the cos part of the solution is absent.
        # This is because when we compute Gr_k(z, z') for a source z' inside the horizon,
        # the boundary conditions kill the cos part. Hence, when we fix the response time z and
        # assemble Gr_k for all possible values of z', we will pick up a discontinuity in the phase and
        # coefficient if we don't also kill the cos for z' outside the horizon.

        # So we have to keep the coefficient of the cos always zero.
        # This requires a corresponding adjustment to the phase, which we must calculate.

        # STEP 1. USE INITIAL DATA TO FIX THE COEFFICIENTS OF THE LIOUVILLE-GREEN SOLUTION
        #   alpha sin( theta ) + beta cos( theta )
        # (for now we are ignoring the effective friction term, except for its contribution
        # to fixing the values of alpha and beta
        omega_init = sqrt(omega_sq_init)
        sqrt_omega_init = sqrt(omega_init)

        raw_cos_coeff = sqrt_omega_init * self._G_init
        raw_sin_coeff = (
            self._Gprime_init
            + (self._G_init / 2.0) * (d_ln_omega_init + eps_init / one_plus_z_init)
        ) / sqrt_omega_init

        # STEP 2. WRITE THE SOLUTION IN THE FORM
        #   B sin ( theta + deltaTheta )
        deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
        B = sqrt(raw_cos_coeff * raw_cos_coeff + raw_sin_coeff * raw_sin_coeff)

        # fix the sign of B by comparison with the original G
        sin_deltaTheta = sin(deltaTheta)
        sgn_sin_deltaTheta = +1 if sin_deltaTheta >= 0.0 else -1
        sgn_G = +1 if self._G_init >= 0.0 else -1

        # evaluate the new coefficients of the sin and cos terms
        self._cos_coeff = 0.0
        self._sin_coeff = sgn_sin_deltaTheta * sgn_G * B

        # STEP 3. APPLY THE SHIFT TO THE PHASE FUNCTION
        # change theta to theta + deltaTheta, and then update the result mod 2pi
        theta_div_2pi_sample, theta_mod_2pi_sample = shift_theta_sample(
            div_2pi_sample=data["theta_div_2pi_sample"],
            mod_2pi_sample=data["theta_mod_2pi_sample"],
            shift=deltaTheta,
        )

        # STEP 4. EVALUATE THE FULL LIOUVILLE-GREEN SOLUTIONS

        # estimate tau at the source redshift, needed to obtain the analytical results
        # for comparison
        H_source = model.functions.Hubble(self._z_source.z)
        tau_source = model.functions.tau(self._z_source.z)

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
            omega_sq = Gk_omegaEff_sq(model, self._k_exit.k.k, current_z_float)
            omega = sqrt(omega_sq)

            d_ln_omega_WKB = Gk_d_ln_omegaEff_dz(
                model, self._k_exit.k.k, current_z_float
            )
            WKB_criterion = fabs(d_ln_omega_WKB) / omega

            H_ratio = H_init / H
            norm_factor = sqrt(H_ratio / omega)

            # no need to include theta div 2pi in the calculation of the sin/cos functions in G_WKB
            # (and indeed it is likely to be more accurate if we don't)
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
                    omega_WKB_sq=omega_sq,
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

        self._solver = self._solver_labels[data["phase_solver_label"]]

        return True


class GkWKBValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        H_ratio: float,
        theta_mod_2pi: float,
        theta_div_2pi: int,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
        G_WKB: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        z_init: Optional[float] = None,
        analytic_G_rad: Optional[float] = None,
        analytic_Gprime_rad: Optional[float] = None,
        analytic_G_w: Optional[float] = None,
        analytic_Gprime_w: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._H_ratio = H_ratio

        self._theta_mod_2pi = theta_mod_2pi
        self._theta_div_2pi = theta_div_2pi
        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion
        self._G_WKB = G_WKB

        self._analytic_G_rad = analytic_G_rad
        self._analytic_Gprime_rad = analytic_Gprime_rad

        self._analytic_G_w = analytic_G_w
        self._analytic_Gprime_w = analytic_Gprime_w

        self._sin_coeff = sin_coeff
        self._cos_coeff = cos_coeff

        self._z_init = z_init

    def __float__(self):
        """
        Cast to float. Returns value of G_k estimated using the WKB approximation
        :return:
        """
        return self._G_WKB

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def H_ratio(self) -> float:
        return self._H_ratio

    @property
    def theta_mod_2pi(self) -> float:
        return self._theta_mod_2pi

    @property
    def theta_div_2pi(self) -> int:
        return self._theta_div_2pi

    @property
    def theta(self) -> int:
        return self._theta_div_2pi * TWO_PI + self._theta_mod_2pi

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion

    @property
    def G_WKB(self) -> Optional[float]:
        return self._G_WKB

    @property
    def analytic_G_rad(self) -> Optional[float]:
        return self._analytic_G_rad

    @property
    def analytic_Gprime_rad(self) -> Optional[float]:
        return self._analytic_Gprime_rad

    @property
    def analytic_G_w(self) -> Optional[float]:
        return self._analytic_G_rad

    @property
    def analytic_Gprime_w(self) -> Optional[float]:
        return self._analytic_Gprime_rad

    @property
    def sin_coeff(self) -> Optional[float]:
        return self._sin_coeff

    @property
    def cos_coeff(self) -> Optional[float]:
        return self._cos_coeff

    @property
    def z_init(self) -> Optional[float]:
        return self._z_init
