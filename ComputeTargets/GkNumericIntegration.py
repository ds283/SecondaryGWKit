from math import fabs, log, sqrt
from typing import Optional, List

import ray

from ComputeTargets.BackgroundModel import (
    BREAK_POINT_DISCONTINUITY,
    BackgroundModel,
    ModelProxy,
)
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Gk import compute_analytic_G, compute_analytic_Gprime
from CosmologyConcepts import wavenumber, redshift, redshift_array, wavenumber_exit_time
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.integrators.numeric_with_phase_cut import (
    numeric_with_phase_cut,
    VALUE_INDEX,
    DERIV_INDEX,
)
from Quadrature.supervisors.base import RHS_timer
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import check_units
from config.defaults import (
    DEFAULT_FLOAT_PRECISION,
)


# RHS of ODE system
#
# State layout:
#   state[0] = G_k(z, z')
#   state[1] = Gprime_k(z, z')
def RHS(
    z: float,
    state: List[float],
    model: BackgroundModel,
    k_float: float,
    supervisor: NumericIntegrationSupervisor,
) -> List[float]:
    """
    k *must* be measured using the same units used for H(z) in the cosmology, otherwise we will not get
    correct dimensionless ratios
    """
    with RHS_timer(supervisor) as timer:
        G = state[VALUE_INDEX]
        Gprime = state[DERIV_INDEX]

        if supervisor.notify_available:
            supervisor.message(
                z,
                f"current state: Gr(k) = {G:.5g}, dGr(k)/dz = {Gprime:.5g}",
            )
            supervisor.reset_notify_time()

        H = model.functions.Hubble(z)
        eps = model.functions.epsilon(z)

        one_plus_z = 1.0 + z
        one_plus_z_2 = one_plus_z * one_plus_z

        dG_dz = Gprime

        k_over_H = k_float / H
        k_over_H_2 = k_over_H * k_over_H

        dGprime_dz = (
            -eps * Gprime / one_plus_z - (k_over_H_2 + (eps - 2.0) / one_plus_z_2) * G
        )

        # NOTE: the oscillation-resolution diagnostic used to live here, evaluating
        # Gk_omegaEff_sq (three extra background calls) at every step of the solver in order to
        # feed NumericIntegrationSupervisor.report_wavelength. That cost 45 % of the run
        # (review §10.2). The test is still performed, and still prints its warning, but it now
        # runs once after the solve, on the returned sample grid -- which is the grid the flag is
        # documented to be about (review §13.1). Gk_omegaEff_sq is handed to
        # numeric_with_phase_cut as its omega_sq argument in compute(), below.

    return [dG_dz, dGprime_dz]


class GkNumericIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    tensor Green's function, labelled by a wavenumber k, and two redshifts:
    the source redshift, and the response redshift.
    A single integration fixes the source redshift and determines the Green function
    as a function of the response redshift.
    However, once these have been computed and cached, we can obtain the result as
    a function of the source redshift if we wish
    """

    # The break-point policy this sector integrates under: which of the break points the cosmology
    # declares the ODE is split at. This is the *single* declaration of that policy -- it is passed
    # to numeric_with_phase_cut by compute(), written to the datastore by the factory's store(),
    # and filtered on by the factory's build() -- because since prompt 19 it is a genuine degree of
    # freedom that moves the stored answer, so three independent copies would be three chances to
    # look up a row computed under a policy other than the one asked for (prompt 20 of
    # prompts/GkTk-remedial, the user's decision of 2026-09-13).
    #
    # The value, and why it is this one rather than BREAK_POINT_ALL, is argued at the
    # numeric_with_phase_cut call site in compute() below.
    #
    # This follows GkWKBIntegration.PHASE_SOLVER_LABEL_BASE and BackgroundModel.TAU_SOLVER_LABEL:
    # a class constant, so that a caller which has only the class can read it.
    BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
        delta_logz: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model)

        self._solver_labels = solver_labels
        self._delta_logz = delta_logz
        self._mode = mode.lower() if mode is not None else None

        if self._mode is not None and self._mode not in ["stop"]:
            raise ValueError(
                f'GkNumericIntegration: unknown compute mode "{self._mode}"'
            )

        # search for a handover point (to the WKB calculation) from 3 to 6 e-folds inside the horizon
        self._stop_search_window_start_attr = "z_exit_subh_e3"
        self._stop_search_window_end_attr = "z_exit_subh_e6"

        self._z_sample = z_sample
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None
            self._compute_time = None
            self._compute_steps = None
            self._RHS_evaluations = None
            self._mean_RHS_time = None
            self._max_RHS_time = None
            self._min_RHS_time = None

            self._has_unresolved_osc = None
            self._unresolved_z = None
            self._unresolved_efolds_subh = None

            self._init_efolds_suph = None
            self._stop_deltaz_subh = None
            self._stop_G = None
            self._stop_Gprime = None

            self._solver = None

            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._data = payload["data"]

            self._has_unresolved_osc = payload["has_unresolved_osc"]
            self._unresolved_z = payload["unresolved_z"]
            self._unresolved_efolds_subh = payload["unresolved_efolds_subh"]

            self._init_efolds_suph = payload["init_efolds_suph"]
            self._stop_deltaz_subh = payload["stop_deltaz_subh"]
            self._stop_G = payload["stop_G"]
            self._stop_Gprime = payload["stop_Gprime"]

            self._solver = payload["solver"]

            self._values = payload["values"]

        # check that all sample points are *later* than the specified source redshift
        if z_source is not None and self._z_sample is not None:
            z_source_float = float(z_source)
            for z in self._z_sample:
                z_float = float(z)
                if z_float > z_source_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} is earlier than source redshift z={z_source_float:.5g}"
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
    def break_point_kind(self) -> str:
        """
        The break-point policy this object was (or will be) integrated under. It is a property of
        the sector, not of the instance, so it reads the class constant rather than any stored
        state: an instance deserialized from the datastore was selected by build() on exactly this
        value, so there is no second value it could carry.
        """
        return self.BREAK_POINT_KIND

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def z_exit(self) -> float:
        return self._k_exit.z_exit

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
    def data(self) -> IntegrationData:
        if self.values is None:
            raise RuntimeError("values have not yet been populated")

        return self._data

    @property
    def has_unresolved_osc(self) -> bool:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")
        return self._has_unresolved_osc

    @property
    def unresolved_z(self) -> float:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")

        return self._unresolved_z

    @property
    def unresolved_efolds_subh(self) -> float:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")

        return self._unresolved_efolds_subh

    @property
    def init_efolds_suph(self) -> float:
        if self._init_efolds_suph is None:
            raise RuntimeError("init_efolds_suph has not yet been populated")

        return self._init_efolds_suph

    @property
    def stop_deltaz_subh(self) -> float:
        if self._stop_deltaz_subh is None:
            raise RuntimeError("stop_deltaz_subh has not yet been populated")

        return self._stop_deltaz_subh

    @property
    def stop_G(self) -> float:
        if self._stop_G is None:
            raise RuntimeError("stop_G has not yet been populated")

        return self._stop_G

    @property
    def stop_Gprime(self) -> float:
        if self._stop_Gprime is None:
            raise RuntimeError("stop_Gprime has not yet been populated")

        return self._stop_Gprime

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkNumericIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkNumericIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_source is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_source or z_sample is missing). It can only represent a query."
            )

        # check that source redshift is not too far inside the horizon for this k-mode
        z_limit = self._k_exit.z_exit_subh_e4
        if self._z_source.z < z_limit - DEFAULT_FLOAT_PRECISION:
            raise ValueError(
                f"Specified source redshift z_source={self._z_source.z:.5g} is more than 4-efolds inside the horizon for k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc (horizon re-entry at z_entry={self._k_exit.z_exit:.5g})"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        # set up limits for the search window used to obtain an initial condition for a subsequent
        # WKB integral.
        # This is done by always cutting at a point of fixed phase where G' = 0. That point is a
        # *maximum* of G -- G' passes from negative to positive as z decreases, and review §10.1
        # measures G/envelope = +1.000000 there in every run -- not the minimum the comment here
        # used to claim. Which extremum it is does not matter: GkWKBIntegration.store() rotates
        # arbitrary (G, G') initial data into a pure sine, so the phase does not depend on where in
        # the cycle we cut. The search is performed within a fixed window.
        payload = {}
        if self._mode in ["stop"]:
            payload["mode"] = self._mode

            search_begin = getattr(self._k_exit, self._stop_search_window_start_attr)
            search_end = getattr(self._k_exit, self._stop_search_window_end_attr)

            if search_begin > self._z_source.z:
                search_begin = self._z_source.z

            if search_begin < search_end:
                raise RuntimeError(
                    f"Search window in incorrect order (search_begin={search_begin:.5g}, search_end={search_end:.5g})"
                )

            payload["stop_search_window_z_begin"] = search_begin
            payload["stop_search_window_z_end"] = search_end

        # The initial condition here is for the Green's function defined in David's analytic calculation,
        # which has source -delta(z-z'). This gives the condition dG/dz = +1 at z = z'.
        # This has the advantage that it gives us a simple, clean boundary condition.
        # The more familiar Green's function defined in conformal time tau with source
        # delta(tau-tau') is related to this via G_us = - H(z') G_them.
        self._compute_ref = numeric_with_phase_cut.remote(
            self._model_proxy,
            self._k_exit,
            self._z_source,
            self._z_sample,
            initial_value=0.0,
            initial_deriv=1.0,
            RHS=RHS,
            omega_sq=Gk_omegaEff_sq,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            delta_logz=self._delta_logz,
            task_label="compute_Gk",
            object_label="Gr_k(z, z')",
            # the per-object warning is suppressed here, not the test: has_unresolved_osc,
            # unresolved_z and unresolved_efolds_subh are computed and returned exactly as
            # before, and main.py accumulates them over the work queue and prints one summary
            # per wavenumber (README section 7 decision D2, taken by the user 2026-09-11;
            # prompt 16 of prompts/GkTk-remedial). A direct caller of numeric_with_phase_cut
            # still gets the warning, which defaults to on.
            warn_unresolved_osc=False,
            # this sector splits its integration only where the cosmology declares that a
            # background quantity *jumps*, not at its kinks. This is the module's default, but it
            # is passed explicitly because it is a decision taken on measurement rather than a
            # default inherited by omission (prompt 19 of prompts/GkTk-remedial, the user's
            # decision of 2026-09-13).
            #
            # Unnecessary, and prompt 19's reason still holds with better numbers. That prompt
            # measured this sector below the campaign's convergence criterion at all 50 production
            # wavenumbers on all three models with the jumps alone -- worst 8.41e-09 of the
            # envelope on QCD against 3.4e-08 (docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md §9.1,
            # §10.1). Re-measured on the background prompts 04-07 corrected, and this time under
            # *both* policies rather than only this one (prompt 08 of
            # prompts/qcd-background-audit, docs/qcd-background-audit/PER-SECTOR-POLICY.md §2):
            # worst 3.67e-09 on QCD under either policy, at the same wavenumber, none of the 50
            # above the criterion. The other policy buys nothing here, exactly as it did not
            # before.
            #
            # And it is no longer expensive, which is worth recording because the cost was half of
            # prompt 19's argument: asking for the kinks as well is +0.57 % of the production
            # right-hand-side evaluations (13,343 -> 13,419 per object), where it was +155 % when
            # the T(z) spline's knots were declared. At ~65,000 objects per model that was several
            # core-hours; it is now minutes. The decision stands on the accuracy measurement alone.
            #
            # Read from the class constant rather than written as a literal: since prompt 20 the
            # same value is also stored in, and filtered on by, the datastore lookup key, and the
            # three uses must not be able to drift apart.
            break_point_kind=self.BREAK_POINT_KIND,
            **payload,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkNumericIntegration: store() called, but no compute() is in progress"
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

        self._has_unresolved_osc = data["has_unresolved_osc"]
        self._unresolved_z = data["unresolved_z"]
        self._unresolved_efolds_subh = data["unresolved_efolds_subh"]

        self._stop_deltaz_subh = data.get("stop_deltaz_subh", None)
        self._stop_G = data.get("stop_value", None)
        self._stop_Gprime = data.get("stop_deriv", None)

        model: BackgroundModel = self._model_proxy.get()

        Hsource = model.functions.Hubble(self.z_source.z)
        k_over_aH = (1.0 + self.z_source.z) * self.k.k / Hsource
        self._init_efolds_suph = -log(k_over_aH)

        G_sample = data["value_sample"]
        Gprime_sample = data["deriv_sample"]

        tau_source = model.functions.tau(self._z_source.z)

        # need to be aware that G_sample may not be as long as self._z_sample, if we are working in "stop" mode
        self._values = []
        for i in range(len(G_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = model.functions.Hubble(current_z_float)
            tau = model.functions.tau(current_z_float)
            wBackground = model.functions.wBackground(current_z_float)

            analytic_G_rad = compute_analytic_G(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource
            )
            analytic_Gprime_rad = compute_analytic_Gprime(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource, H
            )
            analytic_G_w = compute_analytic_G(
                self.k.k, wBackground, tau_source, tau, Hsource
            )
            analytic_Gprime_w = compute_analytic_Gprime(
                self.k.k, wBackground, tau_source, tau, Hsource, H
            )

            omega_WKB_sq = Gk_omegaEff_sq(model, self.k.k, current_z_float)
            WKB_criterion = fabs(
                Gk_d_ln_omegaEff_dz(model, self.k.k, current_z_float)
            ) / sqrt(fabs(omega_WKB_sq))

            # create new GkNumericValue object
            self._values.append(
                GkNumericValue(
                    None,
                    current_z,
                    G_sample[i],
                    Gprime_sample[i],
                    analytic_G_rad=analytic_G_rad,
                    analytic_Gprime_rad=analytic_Gprime_rad,
                    analytic_G_w=analytic_G_w,
                    analytic_Gprime_w=analytic_Gprime_w,
                    omega_WKB_sq=omega_WKB_sq,
                    WKB_criterion=WKB_criterion,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class GkNumericValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor Green's transfer functions.
    Parameters such as wavenumber k, source redshift z_source, etc., are held by the
    owning GkNumericIntegration object
    """

    def __init__(
        self,
        store_id: int,
        z: redshift,
        G: float,
        Gprime: float,
        analytic_G_rad: Optional[float] = None,
        analytic_Gprime_rad: Optional[float] = None,
        analytic_G_w: Optional[float] = None,
        analytic_Gprime_w: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._G = G
        self._Gprime = Gprime

        self._analytic_G_rad = analytic_G_rad
        self._analytic_Gprime_rad = analytic_Gprime_rad

        self._analytic_G_w = analytic_G_w
        self._analytic_Gprime_w = analytic_Gprime_w

        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion

    def __float__(self):
        """
        Cast to float. Returns value of the Green's function
        :return:
        """
        return self.G

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def G(self) -> float:
        return self._G

    @property
    def Gprime(self) -> float:
        return self._Gprime

    @property
    def analytic_G_rad(self) -> Optional[float]:
        return self._analytic_G_rad

    @property
    def analytic_Gprime_rad(self) -> Optional[float]:
        return self._analytic_Gprime_rad

    @property
    def analytic_G_w(self) -> Optional[float]:
        return self._analytic_G_w

    @property
    def analytic_Gprime_w(self) -> Optional[float]:
        return self._analytic_Gprime_w

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion
