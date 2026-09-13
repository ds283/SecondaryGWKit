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
``rho`` is the residual table ``cached_phase_residual`` builds **once per**
``(model, k, sector)`` from the non-leading terms of ``*_omegaEff_sq``, never as
``omega^2 - omega_0^2``. Every ``X.delta(z_a, z_b)`` is
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
root, not a node of the background grid (``RECONCILIATION.md`` §2 item 5). Both tables reach it
through ``delta``'s off-grid partial. The leading table pays one local Gauss panel per sample;
the residual table pays *one per object*, because the anchor is split off at the table node
nearest to it,

    rho.delta(z_init, z) = rho.delta(z_init, n) + rho.delta(n, z),      n a node,

so ``rho.delta(n, z)`` is free for an on-grid sample. Prompt 06 instead built the residual table
with ``z_init`` as its own top node, which made the table per-object and cost 5,536 of the 6,000
integrand evaluations of an object at ``k = 3e8`` on LambdaCDM
(``[06-residual-table-per-object]``). Anchoring on the grid and splitting at ``n`` moves the
residual by the rounding of one extra double-double difference of a quantity below 0.1 rad:
measured against the per-object table over three models and three wavenumbers in both sectors,
``rho`` moves by at most 1.4e-17 rad and ``theta`` itself is **bit-identical** at every sample
(prompt 14).

**Diagnostics.** The WKB-validity criterion ``|d ln omega/dz| / omega`` is evaluated at
``z_init`` (an error if it exceeds one, as before) and at every sample; ``has_WKB_violation``,
``WKB_violation_z`` and ``WKB_violation_efolds_subh`` describe the first sample that exceeds it.
The semantics are those the ODE supervisors had, now evaluated on the sample grid rather than at
the solver's steps.
"""

import time
from math import fabs, log, log1p, sqrt
from typing import List, Optional

import numpy as np
import ray

from ComputeTargets import BackgroundModel, ModelProxy
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.phase_residual import RHO_GAUSS_ORDER, cached_phase_residual
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


def nearest_table_node(table: CumulativeTable, z: float) -> float:
    """
    The redshift of the table node nearest to ``z`` in ``u = log(1+z)``, or ``z`` itself when it
    is a node.

    This mirrors ``CumulativeTable``'s own nearest-node rule, so that splitting an interval at
    the returned node leaves a partial no wider than the one ``delta`` would have formed anyway.
    Lookup of an exact node is by exact ``z`` (README §5 rule 9); only the off-grid branch goes
    through ``log(1+z)``, where it is a quadrature endpoint and nothing more.
    """
    z = float(z)
    if table.node_index(z) is not None:
        return z

    u_asc = np.asarray(table.u_nodes)[::-1]
    n = u_asc.size
    u = log1p(z)
    i = int(np.searchsorted(u_asc, u))
    if i <= 0:
        j = 0
    elif i >= n:
        j = n - 1
    else:
        j = i if (u_asc[i] - u) < (u - u_asc[i - 1]) else i - 1
    return float(table.z_nodes[n - 1 - j])


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
        ``cached_phase_residual``
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

    # the residual table: one per (model, k, sector), built on the background grid cut at the
    # top of the WKB region and shared by every object of this wavenumber
    rho, rho_reused = cached_phase_residual(
        model,
        k_float,
        leading_table.z_nodes,
        sector,
        store_id=getattr(model_proxy, "store_id", None),
    )
    rho_evaluations_before = rho.total_evaluations if rho_reused else 0

    # the anchor and every sample must lie inside the table's own range, so that delta only ever
    # forms interior partials (CumulativeTable refuses an endpoint more than one grid interval
    # beyond either end). Asserted rather than assumed.
    rho_z_top = float(rho.z_nodes[0])
    rho_z_bottom = float(rho.z_nodes[-1])
    z_sample_max = max(z_sample_list)
    z_sample_min = min(z_sample_list)
    if (
        z_init > rho_z_top
        or z_init < rho_z_bottom
        or z_sample_max > rho_z_top
        or z_sample_min < rho_z_bottom
    ):
        raise RuntimeError(
            f"{task_label}: the anchor z_init={z_init:.8g} and the samples "
            f"[{z_sample_min:.8g}, {z_sample_max:.8g}] must lie inside the residual table's "
            f"range [{rho_z_bottom:.8g}, {rho_z_top:.8g}] for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc "
            f"(sector {sector})"
        )

    # the anchor is split off at the nearest node, so the residual's off-grid partial is paid
    # once per object rather than once per sample: rho.delta(z_init, z) = rho_anchor +
    # rho.delta(rho_anchor_node, z), and the second term is free for an on-grid sample. Exactly
    # zero, with no integrand call, when z_init is itself a node.
    rho_anchor_node = nearest_table_node(rho, z_init)
    rho_anchor = rho.delta(z_init, rho_anchor_node)

    # the phase at every sample, formed once and reduced once per sample; the leading table
    # supplies the partial panel at an off-grid anchor itself, and never differences two
    # pointwise values (README §2 (c))
    leading_evaluations_before = leading_table.total_evaluations

    sampled_theta_div_2pi: List[int] = []
    sampled_theta_mod_2pi: List[float] = []
    rho_delta = 0.0
    for z in z_sample_list:
        rho_delta = rho_anchor + rho.delta(rho_anchor_node, z)
        theta = -(k_float * leading.delta(z_init, z) + rho_delta)
        theta_div_2pi, theta_mod_2pi = WKB_mod_2pi(theta)
        sampled_theta_div_2pi.append(theta_div_2pi)
        sampled_theta_mod_2pi.append(theta_mod_2pi)

    leading_partial_evaluations = (
        leading_table.total_evaluations - leading_evaluations_before
    )
    rho_evaluations = rho.total_evaluations - rho_evaluations_before

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
            "rho_nodes": int(len(rho)),
            "rho_evals": int(rho_evaluations),
            "rho_reused": bool(rho_reused),
            "lead_partials": int(
                len(z_sample_list) * anchor_off_grid + off_grid_samples
            ),
            "lead_evals": int(leading_partial_evaluations),
            "offgrid_init": bool(anchor_off_grid),
            "rho_end": float(rho_delta),
        }
    )

    # "stage_1_data" records the primitive's cost *of this call*: the integrand evaluations spent
    # building the residual table -- none when it was reused -- and on both tables' off-grid
    # partials, plus the wall time. There is no second stage; the per-evaluation timings the ODE
    # supervisors measured have no counterpart here and are left None (the columns are nullable).
    stage_1_data = IntegrationData(
        compute_time=elapsed,
        compute_steps=len(z_sample_list),
        RHS_evaluations=int(rho_evaluations + leading_partial_evaluations),
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
