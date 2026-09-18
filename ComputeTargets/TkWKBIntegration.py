from math import sqrt, log, atan2, sin, fabs, cos, exp
from typing import Optional, List

import ray

import ComputeTargets.phase_residual as phase_residual
from ComputeTargets import ModelProxy, BackgroundModel
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from CosmologyConcepts import wavenumber_exit_time, redshift_array, wavenumber, redshift
from Datastore import DatastoreObject
from LiouvilleGreen.WKBtools import apply_phase_offset
from LiouvilleGreen.constants import TWO_PI
from MetadataConcepts import store_tag
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from Quadrature.integrators.WKB_phase_function import WKB_phase_function
from Units import check_units
from config.defaults import DEFAULT_FLOAT_PRECISION


class TkWKBIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the Liouville-Green (WKB)
    phase function for the transfer function.

    Both of the redshift-dependent quantities this class stores come from the background
    model's cumulative Gauss-Legendre tables (``prompts/GkTk-remedial``, prompts 03, 04, 06,
    07), not from an ODE: the phase is
    ``theta_T(z; z_init) = -[k cs_tau.delta(z_init, z) + rho_T.delta(z_init, z)]`` with
    ``cs_tau`` the sound horizon ``int c_s dz/H``, and the Liouville-Green friction exponent is
    ``friction_F.delta(z_init, z)`` with ``dF/dz = +(3/2)(1 + c_s^2)/(1+z)``, negative for
    ``z < z_init``. The friction ODE that used to supply the latter -- a DOP853 solve of
    ``dF/dz`` at the production tolerances, whose right-hand side lived in this module -- carried
    2.3e-7 to 4.1e-7 of relative amplitude error set by ``rtol`` (review §12.3), where the table
    is at 1e-14 absolute in ``F``. It survives only inside
    ``ComputeTargets/tests/test_background_cs_tau_friction.py``, the test that measures what it
    cost.

    There is no tolerance anywhere in this object. ``atol`` and ``rtol`` lost their referent in
    the computation when the tables replaced the ODEs, and prompt 05 of
    ``prompts/tolerance-convergence`` removed them from the datastore lookup key as well: what
    keys this target now is ``rho_gauss_order``, the order of the phase residual table, which is
    the one accuracy parameter it has.
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
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
            self._friction_data = IntegrationData(
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

            self._init_efolds_subh = None
            self._metadata = None

            self._phase_solver = None
            self._friction_solver = None

            self._sin_coeff = None
            self._cos_coeff = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._stage_1 = payload["stage_1_data"]
            self._stage_2 = payload["stage_2_data"]
            self._friction_data = payload["friction_data"]

            self._has_WKB_violation = payload["has_WKB_violation"]
            self._WKB_violation_z = payload["WKB_violation_z"]
            self._WKB_violation_efolds_subh = payload["WKB_violation_efolds_subh"]

            self._init_efolds_subh = payload["init_efolds_subh"]
            self._metadata = payload["metadata"]

            self._phase_solver = payload["phase_solver"]
            self._friction_solver = payload["friction_solver"]

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

    # The Gauss-Legendre order of the phase residual table, and -- since prompt 05 of
    # prompts/tolerance-convergence -- this object's datastore lookup key. The accessor resolves
    # the single declaration in ComputeTargets/phase_residual.py *at call time*, which is also how
    # WKB_phase_function reaches it and how the factory reaches it for both store() and build().
    # There is no keyword, no payload key and no default on the path, so the order a row records,
    # the order a lookup asks for and the order the residual was tabulated at are the same object.
    @property
    def rho_gauss_order(self) -> int:
        return phase_residual.RHO_GAUSS_ORDER

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
    def friction_data(self) -> IntegrationData:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._friction_data

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
    def phase_solver(self) -> IntegrationSolver:
        if self._phase_solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._phase_solver

    @property
    def friction_solver(self) -> IntegrationSolver:
        if self._friction_solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._friction_solver

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
        d_ln_omega_WKB_init = Tk_d_ln_omegaEff_dz(model, self._k_exit.k.k, self.z_init)

        if omega_WKB_sq_init < 0.0:
            raise ValueError(
                f"omega_WKB^2 must be non-negative at the initial time (k={self._k_exit.k.k_inv_Mpc}/Mpc, z_init={self._z_init:.5g}, omega_WKB^2={omega_WKB_sq_init:.5g})"
            )

        WKB_criterion_init = fabs(d_ln_omega_WKB_init) / sqrt(omega_WKB_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (TkWKBIntegration) k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc")
            print(
                f"|    WKB diagnostic |d(omega) / omega^2| exceeds unity at initial time (value={WKB_criterion_init:.5g}, z_init={self._z_init:.5g})"
            )
            print(f"     This may lead to meaningless results.")

        # phase and friction both come from the background model's tables (sound horizon and
        # friction_F); no tolerances, and since prompt 05 of prompts/tolerance-convergence none in
        # the lookup key either -- what is keyed is phase_residual.RHO_GAUSS_ORDER, which
        # WKB_phase_function reads from the same declaration self.rho_gauss_order does.
        self._compute_ref = WKB_phase_function.remote(
            self._model_proxy,
            self._k_exit,
            self._z_init,
            self._z_sample,
            sector="Tk",
            omega_sq=Tk_omegaEff_sq,
            d_ln_omega_dz=Tk_d_ln_omegaEff_dz,
            friction=True,
            task_label="compute_Tk_WKB_phase",
            object_label="T_k(z)",
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
        self._friction_data = data["friction_data"]

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
        d_ln_omega_WKB_init = Tk_d_ln_omegaEff_dz(model, self._k_exit.k.k, self._z_init)
        cs2_init = model.functions.wPerturbations(self._z_init)

        # in the Green's function WKB calculation we adjust the phase so that the cos part of the
        # solution is absent. This is because when we compute Gr_k(z, z') for a source z' inside the horizon,
        # the boundary conditions kill the cos part. Hence, when we fix the response time z and
        # assemble Gr_k for all possible values of z', we will pick up a discontinuity in the phase and
        # coefficient if we don't also kill the cos for z' outside the horizon.

        # there is no similar effect for the transfer function, but we still make the same shift
        # so that we only get a sin solution. This at least simplifies the representation a bit.

        # STEP 1. USE INITIAL DATA TO FIX THE COEFFICIENTS OF THE LIOUVILLE-GREEN SOLUTION
        #   alpha sin( theta ) + beta cos( theta )
        # (for now we are ignoring the effective friction term, except for its contribution
        # to fixing the values of alpha and beta
        omega_WKB_init = sqrt(omega_WKB_sq_init)
        sqrt_omega_WKB_init = sqrt(omega_WKB_init)

        raw_cos_coeff = sqrt_omega_WKB_init * self._T_init
        raw_sin_coeff = (
            self._Tprime_init
            + (self._T_init / 2.0)
            * (
                d_ln_omega_WKB_init
                + (eps_init - 3.0 * (1.0 + cs2_init)) / one_plus_z_init
            )
        ) / sqrt_omega_WKB_init

        # STEP 2. WRITE THE SOLUTION IN THE FORM
        #   B sin ( theta + deltaTheta )
        # With deltaTheta = atan2(raw_cos, raw_sin) the sign of B sin(deltaTheta) is the sign of
        # raw_cos, i.e. of T_init, so B > 0 already reproduces the initial data: the sign
        # correction that used to follow here was provably +1 in every case (review §8.1, and
        # the identical code in GkWKBIntegration.store() that prompt 06 removed) and has been
        # removed.
        deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
        B = sqrt(raw_cos_coeff * raw_cos_coeff + raw_sin_coeff * raw_sin_coeff)

        # evaluate the new coefficients of the sin and cos terms
        self._cos_coeff = 0.0
        self._sin_coeff = B

        # STEP 3. APPLY THE SHIFT TO THE PHASE FUNCTION
        # change theta to theta + deltaTheta per sample, wrapping each sample's remainder back
        # into (-2pi, 0] and adding the resulting cycle shift to that sample's div 2pi. No
        # cross-sample rebase (README §2 (e); review §8.1). There is only one TkWKBIntegration
        # object per k, so the cross-object stitching the Green's function needs does not arise
        # here (review §12.7); the exactness is still worth having, because the stored
        # theta + deltaTheta is then the phase and nothing else.
        theta_div_2pi_sample, theta_mod_2pi_sample = apply_phase_offset(
            data["theta_div_2pi_sample"],
            data["theta_mod_2pi_sample"],
            deltaTheta,
        )

        # friction_sample[i] is friction_F.delta(z_init, z_i) = F(z_i) - F(z_init), evaluated
        # from the background model's friction_F table (prompt 04). It is negative for
        # z_i < z_init, which is the sign exp() below needs; the retired ODE accumulated the
        # same quantity from F(z_init) = 0.
        friction_sample = data["friction_sample"]

        # STEP 4. EVALUATE THE FULL LIOUVILLE-GREEN SOLUTIONS
        self._values = []
        for i in range(len(theta_mod_2pi_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = model.functions.Hubble(current_z_float)
            tau = model.functions.tau(current_z_float)
            wPerturbations = model.functions.wPerturbations(current_z_float)

            analytic_T_rad = compute_analytic_T(self.k.k, 1.0 / 3.0, tau)
            analytic_Tprime_rad = compute_analytic_Tprime(self.k.k, 1.0 / 3.0, tau, H)

            analytic_T_w = compute_analytic_T(self.k.k, wPerturbations, tau)
            analytic_Tprime_w = compute_analytic_Tprime(
                self.k.k, wPerturbations, tau, H
            )

            omega_sq = Tk_omegaEff_sq(model, self.k.k, current_z_float)
            omega = sqrt(omega_sq)

            WKB_criterion = fabs(
                Tk_d_ln_omegaEff_dz(model, self.k.k, current_z_float)
            ) / sqrt(fabs(omega_sq))

            H_ratio = H_init / H
            norm_factor = sqrt(H_ratio / omega)

            # no need to include theta div 2pi in the calculation of the Liouville-Green solution
            # (and indeed it may be more accurate if we don't)
            T_WKB = (
                norm_factor
                * exp(friction_sample[i])
                * (
                    self._cos_coeff * cos(theta_mod_2pi_sample[i])
                    + self._sin_coeff * sin(theta_mod_2pi_sample[i])
                )
            )

            # create new GkWKBValue object
            self._values.append(
                TkWKBValue(
                    None,
                    current_z,
                    H_ratio,
                    theta_mod_2pi_sample[i],
                    theta_div_2pi_sample[i],
                    friction_sample[i],
                    omega_WKB_sq=omega_sq,
                    WKB_criterion=WKB_criterion,
                    T_WKB=T_WKB,
                    analytic_T_rad=analytic_T_rad,
                    analytic_Tprime_rad=analytic_Tprime_rad,
                    analytic_T_w=analytic_T_w,
                    analytic_Tprime_w=analytic_Tprime_w,
                    sin_coeff=self._sin_coeff,
                    cos_coeff=self._cos_coeff,
                    z_init=self._z_init,
                )
            )

        self._phase_solver = self._solver_labels[data["phase_solver_label"]]
        self._friction_solver = self._solver_labels[data["friction_solver_label"]]

        return True


class TkWKBValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        H_ratio: float,
        theta_mod_2pi: float,
        theta_div_2pi: int,
        friction: float,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
        T_WKB: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        z_init: Optional[float] = None,
        analytic_T_rad: Optional[float] = None,
        analytic_Tprime_rad: Optional[float] = None,
        analytic_T_w: Optional[float] = None,
        analytic_Tprime_w: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._H_ratio = H_ratio

        self._theta_mod_2pi = theta_mod_2pi
        self._theta_div_2pi = theta_div_2pi
        self._friction = friction
        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion
        self._T_WKB = T_WKB

        self._analytic_T_rad = analytic_T_rad
        self._analytic_Tprime_rad = analytic_Tprime_rad

        self._analytic_T_w = analytic_T_w
        self._analytic_Tprime_w = analytic_Tprime_w

        self._sin_coeff = sin_coeff
        self._cos_coeff = cos_coeff

        self._z_init = z_init

    def __float__(self):
        """
        Cast to float. Returns value of G_k estimated using the WKB approximation
        :return:
        """
        return self._T_WKB

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
    def friction(self) -> float:
        return self._friction

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion

    @property
    def T_WKB(self) -> Optional[float]:
        return self._T_WKB

    @property
    def analytic_T_rad(self) -> Optional[float]:
        return self._analytic_T_rad

    @property
    def analytic_Tprime_rad(self) -> Optional[float]:
        return self._analytic_Tprime_rad

    @property
    def analytic_T_w(self) -> Optional[float]:
        return self._analytic_T_w

    @property
    def analytic_Tprime_w(self) -> Optional[float]:
        return self._analytic_Tprime_w

    @property
    def sin_coeff(self) -> Optional[float]:
        return self._sin_coeff

    @property
    def cos_coeff(self) -> Optional[float]:
        return self._cos_coeff

    @property
    def z_init(self) -> Optional[float]:
        return self._z_init
