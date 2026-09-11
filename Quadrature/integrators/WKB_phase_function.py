"""
The Liouville-Green (WKB) phase of the tensor Green's function and of the transfer function,
evaluated from the background model's cumulative tables.

**What is computed.** With the frequency split into a leading part and a correction
(``ComputeTargets/phase_residual.py``; review §6),

    omega^2 = omega_0^2 + C,     omega_0 = k/H          (``sector="Gk"``)
    omega_T^2 = omega_{T,0}^2 + C_T,  omega_{T,0} = k c_s/H   (``sector="Tk"``)

the phase from the anchor ``z_init`` to a sample ``z`` is, exactly,

    theta(z; z_init) = -[ k * leading.delta(z_init, z) + rho.delta(z_init, z) ]

where ``leading`` is ``model.functions.tau`` (``Gk``) or ``model.functions.cs_tau`` (``Tk``) --
the double-double Gauss-Legendre tables ``compute_background`` builds once per model -- and
``rho`` is the residual table ``build_phase_residual`` builds once per call from the non-leading
terms of ``*_omegaEff_sq``, never as ``omega^2 - omega_0^2``. Every ``X.delta(z_a, z_b)`` is
``X(z_b) - X(z_a)``, positive for ``z_b < z_a`` (``prompts/GkTk-remedial/README.md`` §2 (c)),
so ``theta`` is negative and decreasing towards lower redshift, the author's convention.

Each sample's phase is then split into a cycle count and a remainder by ``WKB_mod_2pi`` in the
negative-remainder convention, ``theta_mod_2pi in (-2pi, 0]`` (README §2 (e)). The unreduced
``theta`` is formed once and reduced once, per sample; nothing is rebased across samples.

**What this replaced, and why** (review §2-§4, §7, §8.2). Until prompt 06 of
``prompts/GkTk-remedial`` this module integrated ``dtheta/dz = omega`` as an ODE in two stages: a
``DOP853`` run with a reset every 1e4 rad, then a change of variable to ``Q`` with
``theta = theta_i + omega_i (1+u) Q``. Its error was a fixed fraction of the *accumulated*
phase -- 13.9 rad at ``k = 1e5/Mpc`` and 7366 rad at ``k = 3e8/Mpc`` at ``z = 0.1`` on the
production background, so the stored ``theta_mod_2pi`` below ``z ~ 1e3`` was noise for the
largest ``k`` -- and its cost was proportional to the span: 2.5e6 right-hand-side evaluations and
63.7 s per object at ``k = 3e8``, ~13 CPU-hours per ``k``. The phase is ``k`` times a
``k``-independent function of redshift plus a residual of at most 1.5e-3 rad (review §6), so one
table per model and one small residual table per ``(model, k)`` deliver it to the
double-precision floor in milliseconds. Nothing of the ODE remains: no ODE solver, no stages, no
resets, no ``Q``, no supervisors.

**The anchor is off the grid.** For numeric-initialised objects ``z_init`` is a ``root_scalar``
root, not a node of the background grid (``RECONCILIATION.md`` §2 item 5). The leading table
reaches it through ``delta``'s off-grid partial -- one local Gauss panel per sample -- and the
residual table is built with ``z_init`` as its own top node, so ``rho.delta`` needs no partial.

**Diagnostics.** The WKB-validity criterion ``|d ln omega/dz| / omega`` is evaluated at
``z_init`` (an error if it exceeds one, as before) and at every sample; ``has_WKB_violation``,
``WKB_violation_z`` and ``WKB_violation_efolds_subh`` describe the first sample that exceeds it.
The semantics are those the ODE supervisors had, now evaluated on the sample grid rather than at
the solver's steps.
"""

import time
from math import fabs, log, sqrt
from typing import List, Optional, Sequence

import numpy as np
import ray

from ComputeTargets import BackgroundModel, ModelProxy
from ComputeTargets.phase_residual import RHO_GAUSS_ORDER, build_phase_residual
from CosmologyConcepts import redshift_array, wavenumber, wavenumber_exit_time
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from Quadrature.integration_metadata import IntegrationData
from Units import check_units

# The IntegrationSolver label under which the primitive-based phase is recorded. "stepping"
# carries the residual's Gauss order, in the pattern of BackgroundModel.TAU_SOLVER_LABEL
# (prompt 03); the leading tables' order is the background model's. The label is distinct from
# the background tables' "cumulative-GL" so that the phase solver and the background solver are
# separate IntegrationSolver rows ([03-integrationsolver-stepping-minimum-lookup]). main.py
# registers it through the GkWKBIntegration class attributes of the same names.
PHASE_SOLVER_LABEL_BASE = "wkb-primitive"
PHASE_SOLVER_STEPPING = RHO_GAUSS_ORDER
PHASE_SOLVER_LABEL = f"{PHASE_SOLVER_LABEL_BASE}-stepping{PHASE_SOLVER_STEPPING}"

# which background primitive carries the leading term k * delta in each sector
SECTOR_LEADING_PRIMITIVE = {"Gk": "tau", "Tk": "cs_tau"}


def _empty_integration_data() -> IntegrationData:
    """An ``IntegrationData`` with every field ``None`` -- the shape ``GkWKBIntegration`` and
    ``TkWKBIntegration`` initialise with, and what the datastore factories write as NULLs.
    """
    return IntegrationData(
        compute_time=None,
        compute_steps=None,
        RHS_evaluations=None,
        mean_RHS_time=None,
        max_RHS_time=None,
        min_RHS_time=None,
    )


def residual_nodes(
    grid: Sequence[float], z_sample: Sequence[float], z_init: float
) -> np.ndarray:
    """
    The nodes the residual table is built on: the background model's own grid restricted to
    ``[min(z_sample), z_init]``, together with every sample redshift and ``z_init`` itself,
    strictly descending.

    Building on the background grid (rather than on the sample grid, which for the Green's
    function is the 12x sparser response grid) keeps the residual's Gauss panels at the width
    prompt 02 measured order 4 at the floor on (``QCD_Cosmology`` in particular is split further
    at its break points inside ``build_phase_residual``). Making ``z_init`` the top node puts the
    anchor on-grid for ``rho``, so the only off-grid partials are the leading table's.
    """
    grid = np.asarray(grid, dtype=float)
    z_init = float(z_init)
    z_min = min(float(z) for z in z_sample)
    inside = grid[(grid >= z_min) & (grid <= z_init)]
    nodes = set(inside.tolist())
    nodes.update(float(z) for z in z_sample)
    nodes.add(z_init)
    return np.array(sorted(nodes, reverse=True), dtype=float)


def _leading_primitive(model: BackgroundModel, sector: str, task_label: str):
    if sector not in SECTOR_LEADING_PRIMITIVE:
        raise ValueError(
            f"{task_label}: unknown sector '{sector}' (expected one of "
            f"{tuple(SECTOR_LEADING_PRIMITIVE)})"
        )
    name = SECTOR_LEADING_PRIMITIVE[sector]
    leading = getattr(model.functions, name)
    if leading is None or not hasattr(leading, "delta"):
        raise RuntimeError(
            f"{task_label}: model.functions.{name} does not provide an interval accessor "
            f"'delta'; the background model predates prompts/GkTk-remedial prompt "
            f"{'03' if name == 'tau' else '04'} and must be regenerated"
        )
    return leading


@ray.remote
def WKB_phase_function(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: float,
    z_sample: redshift_array,
    *,
    sector: str,
    omega_sq,
    d_ln_omega_dz,
    friction: bool = False,
    task_label: str = "WKB_phase_function",
    object_label: str = "(object)",
) -> dict:
    """
    Evaluate the Liouville-Green phase ``theta(z; z_init)`` at every redshift of ``z_sample``
    from the background model's tables (module docstring), as a ``(theta_div_2pi,
    theta_mod_2pi)`` pair per sample.

    :param model_proxy: the ``ModelProxy`` of the background model; ``.get()`` supplies the
        ``BackgroundModel`` whose ``functions.tau``/``cs_tau``/``friction_F`` tables are used
    :param k: the ``wavenumber_exit_time`` of the mode
    :param z_init: the anchor redshift at which ``theta = 0``; the numeric hand-over point for
        numeric-initialised objects (off the grid), the source redshift otherwise
    :param z_sample: the response redshifts, all ``<= z_init``
    :param sector: ``"Gk"`` (leading primitive ``tau``) or ``"Tk"`` (``cs_tau``)
    :param omega_sq: ``Gk_omegaEff_sq`` or ``Tk_omegaEff_sq``, used only for the WKB-criterion
        diagnostic; the phase itself is built from the split functions inside
        ``build_phase_residual``
    :param d_ln_omega_dz: the matching ``*_d_ln_omegaEff_dz``, for the same diagnostic
    :param friction: if ``True`` (the ``Tk`` sector) also return
        ``friction_F.delta(z_init, z)`` for every sample, the exponent of the transfer function's
        friction factor
    :return: the payload described in the module docstring; keys ``"stage_1_data"``,
        ``"stage_2_data"``, ``"theta_div_2pi_sample"``, ``"theta_mod_2pi_sample"``,
        ``"phase_solver_label"``, ``"has_WKB_violation"``, ``"WKB_violation_z"``,
        ``"WKB_violation_efolds_subh"``, ``"metadata"``; and, when ``friction`` is set,
        ``"friction_sample"``, ``"friction_data"``, ``"friction_solver_label"``
    """
    start_time = time.perf_counter()

    k_wavenumber: wavenumber = k.k
    k_float = float(k_wavenumber.k)
    z_init = float(z_init)

    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    # the initial-time guards are unchanged from the ODE implementation
    omega_sq_init = omega_sq(model, k_float, z_init)
    d_ln_omega_dz_init = d_ln_omega_dz(model, k_float, z_init)

    if omega_sq_init < 0.0:
        raise RuntimeError(
            f"{task_label}: omega_WKB^2 is negative at the initial time for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, omega_WKB^2={omega_sq_init:.5g})"
        )

    WKB_criterion_init = fabs(d_ln_omega_dz_init) / sqrt(fabs(omega_sq_init))
    if WKB_criterion_init > 1.0:
        raise RuntimeError(
            f"{task_label}: WKB criterion |d(omega_WKB)/omega_WKB^2| > 1 at the initial time for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, WKB criterion={WKB_criterion_init:.5g})"
        )

    z_sample_list: List[float] = z_sample.as_float_list()
    if len(z_sample_list) == 0:
        raise RuntimeError(f"{task_label}: no sample redshifts were supplied")

    metadata = {
        "solver": PHASE_SOLVER_LABEL_BASE,
        "sector": sector,
        "N_rho": RHO_GAUSS_ORDER,
    }

    # the exact zero-length case: a single sample at the anchor itself, where theta = 0
    if len(z_sample_list) == 1 and z_sample_list[0] == z_init:
        metadata["initial_data_only"] = True

        payload = {
            "stage_1_data": _empty_integration_data(),
            "stage_2_data": _empty_integration_data(),
            "theta_div_2pi_sample": [0],
            "theta_mod_2pi_sample": [0.0],
            "phase_solver_label": PHASE_SOLVER_LABEL,
            "has_WKB_violation": False,
            "WKB_violation_z": None,
            "WKB_violation_efolds_subh": None,
        }
        if friction:
            payload["friction_sample"] = [0.0]
            payload["friction_data"] = _empty_integration_data()
            payload["friction_solver_label"] = PHASE_SOLVER_LABEL

        return payload | {"metadata": metadata}

    leading = _leading_primitive(model, sector, task_label)
    leading_table = leading.table
    metadata["N_lead"] = leading_table.order

    # the residual table, anchored at z_init as its top node
    nodes = residual_nodes(leading_table.z_nodes, z_sample_list, z_init)
    rho = build_phase_residual(model, k_float, nodes, sector)

    # the phase at every sample, formed once and reduced once per sample; the leading table
    # supplies the partial panel at an off-grid anchor itself, and never differences two
    # pointwise values (README §2 (c))
    leading_evaluations_before = leading_table.total_evaluations

    sampled_theta_div_2pi: List[int] = []
    sampled_theta_mod_2pi: List[float] = []
    for z in z_sample_list:
        theta = -(k_float * leading.delta(z_init, z) + rho.delta(z_init, z))
        theta_div_2pi, theta_mod_2pi = WKB_mod_2pi(theta)
        sampled_theta_div_2pi.append(theta_div_2pi)
        sampled_theta_mod_2pi.append(theta_mod_2pi)

    leading_partial_evaluations = (
        leading_table.total_evaluations - leading_evaluations_before
    )

    # the WKB-criterion diagnostic on the sample grid (the supervisors' semantics: the first
    # sample at which the criterion exceeds unity is recorded, and one warning is printed)
    has_WKB_violation = False
    WKB_violation_z: Optional[float] = None
    WKB_violation_efolds_subh: Optional[float] = None
    for z in z_sample_list:
        omega_sq_value = omega_sq(model, k_float, z)
        if omega_sq_value < 0.0:
            raise ValueError(
                f"{task_label}: omega_WKB^2 cannot be negative inside the WKB region (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z={z:.5g}, omega_WKB^2={omega_sq_value:.5g})"
            )
        WKB_criterion = fabs(d_ln_omega_dz(model, k_float, z)) / sqrt(omega_sq_value)
        if WKB_criterion > 1.0:
            H = model.functions.Hubble(z)
            efolds_subh = log((1.0 + z) * k_float / H)
            print(
                f"!! WARNING: {object_label} WKB theta_k(z) evaluation for k = {k_wavenumber.k_inv_Mpc:.5g}/Mpc (store_id={k_wavenumber.store_id}) may have violated the validity criterion for the WKB approximation"
            )
            print(f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g}")
            has_WKB_violation = True
            WKB_violation_z = z
            WKB_violation_efolds_subh = efolds_subh
            break

    payload = {}
    if friction:
        friction_F = model.functions.friction_F
        if friction_F is None or not hasattr(friction_F, "delta"):
            raise RuntimeError(
                f"{task_label}: model.functions.friction_F does not provide an interval accessor 'delta'; the background model predates prompts/GkTk-remedial prompt 04 and must be regenerated"
            )
        payload["friction_sample"] = [
            friction_F.delta(z_init, z) for z in z_sample_list
        ]
        # the friction integral is a table lookup with no solve of its own; the ODE it replaced
        # reported here, and the datastore column set is nullable (prompt 07 removes this key's
        # consumer)
        payload["friction_data"] = _empty_integration_data()
        payload["friction_solver_label"] = PHASE_SOLVER_LABEL

    elapsed = time.perf_counter() - start_time

    anchor_off_grid = leading_table.node_index(z_init) is None
    off_grid_samples = sum(
        1 for z in z_sample_list if leading_table.node_index(z) is None
    )

    # compact: the metadata is persisted as a JSON string in a String(DEFAULT_STRING_LENGTH)
    # column
    metadata.update(
        {
            "rho_nodes": int(len(nodes)),
            "rho_evals": int(rho.evaluations),
            "lead_partials": int(
                len(z_sample_list) * anchor_off_grid + off_grid_samples
            ),
            "lead_evals": int(leading_partial_evaluations),
            "offgrid_init": bool(anchor_off_grid),
            "rho_end": float(rho.delta(z_init, z_sample_list[-1])),
        }
    )

    # "stage_1_data" records the primitive's cost: the integrand evaluations spent building the
    # residual table and on the leading table's off-grid partials, and the wall time of this
    # call. There is no second stage; the per-evaluation timings the ODE supervisors measured
    # have no counterpart here and are left None (the columns are nullable).
    stage_1_data = IntegrationData(
        compute_time=elapsed,
        compute_steps=len(z_sample_list),
        RHS_evaluations=int(rho.evaluations + leading_partial_evaluations),
        mean_RHS_time=None,
        max_RHS_time=None,
        min_RHS_time=None,
    )

    payload.update(
        {
            "stage_1_data": stage_1_data,
            "stage_2_data": _empty_integration_data(),
            "theta_div_2pi_sample": sampled_theta_div_2pi,
            "theta_mod_2pi_sample": sampled_theta_mod_2pi,
            "phase_solver_label": PHASE_SOLVER_LABEL,
            "has_WKB_violation": has_WKB_violation,
            "WKB_violation_z": WKB_violation_z,
            "WKB_violation_efolds_subh": WKB_violation_efolds_subh,
        }
    )

    return payload | {"metadata": metadata}
