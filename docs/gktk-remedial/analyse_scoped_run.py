"""
Layer 2 of the ``prompts/GkTk-remedial`` close-out (prompt 13 §2): read back a scoped pipeline
run's datastore and check what it stored.

Four checks, all read-only (every shard is opened ``mode=ro``):

1. **The background tables round-trip.** ``BackgroundModelValue`` now persists ``tau_Mpc`` as the
   *high* limb of a double-double Gauss-Legendre table with ``tau_lo_Mpc`` beside it, and the same
   for ``cs_tau_Mpc``/``cs_tau_lo_Mpc``, with ``friction_F`` a single double (prompts 03, 04). The
   stored limbs are differenced against the top node and compared with prompt 01's references at
   the JSON checkpoints.

2. **The stored WKB phases are what the production producer gives offline, bit for bit.** For a
   sample of ``GkWKBIntegration`` and ``TkWKBIntegration`` rows the object's own initial data
   (``z_init``, its sample grid) is read out of the datastore, the background tables are
   *reconstructed from the persisted limbs* -- zero quadrature, exactly as
   ``BackgroundModel._build_tau_primitive`` does on load -- and ``WKB_phase_function`` is re-run
   offline. Every ``(theta_div_2pi, theta_mod_2pi)`` pair must match exactly: the same code ran.

3. **The consumers ran and recorded a single phase spline.** ``GkSourcePolicyData`` rows exist and
   are not ``fail``; every ``QuadSourceIntegral`` row's ``WKB_phase_spline_chunks`` is 1, which is
   what prompt 08's de-chunking means in stored data.

4. **The solver provenance is the primitive, not the ODE.** Every WKB row points at the
   ``wkb-primitive-stepping4`` ``IntegrationSolver``, and every stage-2 column is NULL.

Usage::

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/analyse_scoped_run.py \\
        --shards-glob '/path/to/store-shard*.sqlite' --model LambdaCDMModel
"""

import argparse
import glob
import json
import sqlite3
import sys
from math import fabs, fsum
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComputeTargets.BackgroundModel import (  # noqa: E402
    CS_TAU_GAUSS_ORDER,
    FRICTION_F_GAUSS_ORDER,
    TAU_GAUSS_ORDER,
    TablePrimitive,
    _cosmology_break_points,
    _cs_over_Hubble,
    _friction_integrand,
)
from ComputeTargets.WKB_Gk import Gk_d_ln_omegaEff_dz, Gk_omegaEff_sq  # noqa: E402
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq  # noqa: E402
from ComputeTargets.cumulative_table import CumulativeTable  # noqa: E402
from ComputeTargets.tests.test_gk_wkb_phase import (  # noqa: E402
    _run_Gk,
    _run_Tk,
    store_algebra,
)
from ComputeTargets.tests.test_tk_wkb_phase import tk_store_algebra  # noqa: E402
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    LambdaCDMModel,
    QCDModel,
    to_redshift_array,
)
from LiouvilleGreen.WKBtools import apply_phase_offset  # noqa: E402

PHASE_SOLVER_LABEL = "wkb-primitive-stepping4"
TAU_SOLVER_LABEL = "cumulative-GL-stepping4"


def open_shards(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"analyse_scoped_run: no shard matched {pattern!r}")
    return [(p, sqlite3.connect(f"file:{p}?mode=ro", uri=True)) for p in paths]


def read_background(conns):
    """``(z_nodes descending, hi/lo arrays)`` of the single background model in the store."""
    for _, c in conns:
        serials = [r[0] for r in c.execute("select serial from BackgroundModel")]
        if not serials:
            continue
        redshifts = {r[0]: r[1] for r in c.execute("select serial, z from redshift")}
        rows = list(
            c.execute(
                "select z_serial, tau_Mpc, tau_lo_Mpc, cs_tau_Mpc, cs_tau_lo_Mpc, "
                "friction_F from BackgroundModelValue where model_serial = ?",
                (serials[0],),
            )
        )
        if not rows:
            continue
        rows.sort(key=lambda r: redshifts[r[0]], reverse=True)
        return {
            "model_serial": serials[0],
            "z": np.array([redshifts[r[0]] for r in rows], dtype=float),
            "tau_hi": np.array([r[1] for r in rows], dtype=float),
            "tau_lo": np.array([r[2] for r in rows], dtype=float),
            "cs_tau_hi": np.array([r[3] for r in rows], dtype=float),
            "cs_tau_lo": np.array([r[4] for r in rows], dtype=float),
            "friction_F": np.array([r[5] for r in rows], dtype=float),
        }
    raise RuntimeError("analyse_scoped_run: no BackgroundModelValue rows found")


def build_model(model_key, bg):
    """
    The reference harness's stand-in for this cosmology, with its three primitives **replaced by
    the ones the datastore holds** -- ``CumulativeTable`` reconstructed from the persisted (hi, lo)
    limbs, so no quadrature runs and the table is bit-for-bit the one production loaded.
    """
    z = bg["z"]
    if model_key == "LambdaCDMModel":
        model = LambdaCDMModel()
    else:
        model = QCDModel(to_redshift_array([float(zz) for zz in z]))
    cosmology = model.cosmology

    breaks = _cosmology_break_points(cosmology, float(z[-1]), float(z[0]))
    tau = CumulativeTable(
        z,
        lambda zz: 1.0 / cosmology.Hubble(zz),
        TAU_GAUSS_ORDER,
        hi=bg["tau_hi"],
        lo=bg["tau_lo"],
        break_points=breaks,
        label="tau",
    )
    cs_tau = CumulativeTable(
        z,
        _cs_over_Hubble(cosmology),
        CS_TAU_GAUSS_ORDER,
        hi=bg["cs_tau_hi"],
        lo=bg["cs_tau_lo"],
        break_points=breaks,
        label="cs_tau",
    )
    friction_F = CumulativeTable(
        z,
        _friction_integrand(cosmology),
        FRICTION_F_GAUSS_ORDER,
        hi=bg["friction_F"],
        lo=np.zeros_like(bg["friction_F"]),
        break_points=breaks,
        label="friction_F",
    )
    model.functions = model.functions._replace(
        tau=TablePrimitive(tau, "tau"),
        cs_tau=TablePrimitive(cs_tau, "cs_tau"),
        friction_F=TablePrimitive(friction_F, "friction_F"),
    )
    return model


# ---------------------------------------------------------------------------------------------
# check 1 -- the background tables against prompt 01's references
# ---------------------------------------------------------------------------------------------


def check_background(model_key, bg, model, n_nodes):
    """
    The scoped run's own grid is not prompt 01's -- ``z_init`` follows the largest wavenumber of
    the run -- so the JSON checkpoints do not fall on its nodes. The stored limbs are therefore
    scored against an **independently constructed** reference on the run's own grid: mpmath at 40
    digits for `LambdaCDM` (whose Hubble rate is analytic), and converged adaptive Gauss-Legendre
    of the double-precision integrand for `QCD_Cosmology` (whose T(z) is a spline), both from
    ``docs/gktk-remedial/reference_lib.py``, which prompts 01 and 02 built.
    """
    print("\n-- 1. BackgroundModelValue limbs against an independent reference")
    sys.path.insert(0, str(Path(__file__).parent))
    import reference_lib as rl  # noqa: E402

    z = bg["z"]
    z_top = float(z[0])
    cosmology = model.cosmology
    tau = bg["tau_hi"] + bg["tau_lo"]
    cs_tau = bg["cs_tau_hi"] + bg["cs_tau_lo"]

    # nodes spread over the grid, plus the bottom one
    idx = sorted({len(z) // 10, len(z) // 2, 9 * len(z) // 10, len(z) - 1})[
        :n_nodes
    ] or [len(z) - 1]

    if model_key == "LambdaCDMModel":
        mp_bg = rl.MpLambdaCDM(model.cosmology)

        def reference(name, z_j):
            quantity = {"tau": "tau", "cs_tau": "cs_tau", "friction_F": "friction"}[
                name
            ]
            value = float(rl.mp_increment(mp_bg, quantity, float(z_j), z_top))
            return -value if name == "friction_F" else value

        method = "mpmath, 40 digits"
    else:
        integrands = {
            "tau": rl.integrand_tau_double(model.functions),
            "cs_tau": rl.integrand_cs_tau_double(model.functions),
            "friction_F": rl.integrand_friction_double(model.functions),
        }

        # `QCD_Cosmology` is not smooth: `QCD_EOS.G(T)` switches branch at three temperatures and
        # T(z) is a 500-point spline, so a break-*unaware* composite rule does not converge over
        # the whole range however fine it is -- prompt 02's `[01-qcd-eos-branch-boundaries]`
        # measured it stalling at N^-2. The reference is therefore prompt 02's `branch+knots`
        # scheme: split at every declared break point, and again at each decade of 1+z so no
        # panel spans more than one decade of dynamic range, then `gauss_bisect` each piece.
        all_breaks = _cosmology_break_points(cosmology, float(z[-1]), z_top)

        def reference(name, z_j):
            u_lo, u_hi = np.log1p(float(z_j)), np.log1p(z_top)
            edges = {u_lo, u_hi}
            edges.update(u for u in all_breaks if u_lo < u < u_hi)
            ln10 = np.log(10.0)
            n = int(np.floor(u_lo / ln10)) + 1
            while n * ln10 < u_hi:
                if u_lo < n * ln10:
                    edges.add(n * ln10)
                n += 1
            edges = sorted(edges)
            parts = []
            for a, b in zip(edges[:-1], edges[1:]):
                v, _, _ = rl.gauss_bisect(integrands[name], a, b, rtol=1e-15)
                parts.append(v)
            value = float(fsum(parts))
            return -value if name == "friction_F" else value

        method = (
            f"break-aware converged adaptive Gauss-Legendre of the double integrand "
            f"(prompt 02's `branch+knots` scheme; {len(all_breaks)} declared break points "
            f"in range)"
        )

    print(f"   reference: {method}; grid top node z = {z_top:.10g}, {len(z)} nodes")
    worst = {"tau": 0.0, "cs_tau": 0.0, "friction_F": 0.0}
    worst_z = {"tau": None, "cs_tau": None, "friction_F": None}
    for j in idx:
        z_j = float(z[j])
        print(
            f"   z = {z_j:14.8g}:  tau_Mpc = {bg['tau_hi'][j]:.17g}  "
            f"tau_lo_Mpc = {bg['tau_lo'][j]:.17g}"
        )
        for name, stored in (
            ("tau", tau),
            ("cs_tau", cs_tau),
            ("friction_F", bg["friction_F"]),
        ):
            ref = reference(name, z_j)
            got = float(stored[j] - stored[0])
            rel = fabs(got - ref) / max(1e-300, fabs(ref))
            print(
                f"       {name:>10} - {name}(top) = {got:.17g}  vs {ref:.17g}  "
                f"({rel:.3e} relative)"
            )
            if rel > worst[name]:
                worst[name], worst_z[name] = rel, z_j
    for name in ("tau", "cs_tau", "friction_F"):
        print(f"   worst relative {name}: {worst[name]:.4e} at z = {worst_z[name]}")
    return {"nodes": [float(z[j]) for j in idx], "worst": worst, "worst_z": worst_z}


# ---------------------------------------------------------------------------------------------
# check 2 -- stored phases against the offline producer, bit for bit
# ---------------------------------------------------------------------------------------------


def check_phases(conns, model, bg, n_objects):
    """Re-run the production producer offline on each object's own stored initial data, apply the
    production ``store()`` algebra and ``apply_phase_offset``, and require the resulting
    ``(theta_div_2pi, theta_mod_2pi)`` pair to equal the stored one **exactly**."""
    print(
        "\n-- 2. stored WKB phases against the offline producer (must be bit-identical)"
    )
    units = model.cosmology.units
    fn = model.functions
    results = {}

    for sector in ("Gk", "Tk"):
        checked_objects = 0
        checked_samples = 0
        mismatched = 0
        worst_case = None
        for _, c in conns:
            if checked_objects >= n_objects:
                break
            redshifts = {
                r[0]: r[1] for r in c.execute("select serial, z from redshift")
            }
            k_of_exit = {
                r[0]: r[1]
                for r in c.execute(
                    "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
                    "join wavenumber w on w.serial = e.wavenumber_serial"
                )
            }
            if sector == "Gk":
                query = (
                    "select serial, wavenumber_exit_serial, z_init, G_init, Gprime_init "
                    "from GkWKBIntegration where validated = 1 order by serial"
                )
                value_table = "GkWKBValue"
            else:
                query = (
                    "select serial, wavenumber_exit_serial, z_init, T_init, Tprime_init "
                    "from TkWKBIntegration where validated = 1 order by serial"
                )
                value_table = "TkWKBValue"

            for serial, exit_serial, z_init, v_init, vprime_init in c.execute(query):
                if checked_objects >= n_objects:
                    break
                k = float(k_of_exit[exit_serial])
                z_init = float(z_init)
                samples = list(
                    c.execute(
                        f"select z_serial, theta_div_2pi, theta_mod_2pi "
                        f"from {value_table} where wkb_serial = ?",
                        (serial,),
                    )
                )
                if len(samples) < 2:
                    continue
                samples.sort(key=lambda r: redshifts[r[0]], reverse=True)
                z_list = [redshifts[r[0]] for r in samples]

                if sector == "Gk":
                    payload = _run_Gk(
                        model, units, k, z_init, to_redshift_array(z_list)
                    )
                    omega_sq_init = Gk_omegaEff_sq(model, k, z_init)
                    d_ln_omega_init = Gk_d_ln_omegaEff_dz(model, k, z_init)
                    _, delta, _, _ = store_algebra(
                        omega_sq_init,
                        d_ln_omega_init,
                        fn.epsilon(z_init),
                        z_init,
                        float(v_init),
                        float(vprime_init),
                    )
                else:
                    payload = _run_Tk(
                        model, units, k, z_init, to_redshift_array(z_list)
                    )
                    omega_sq_init = Tk_omegaEff_sq(model, k, z_init)
                    d_ln_omega_init = Tk_d_ln_omegaEff_dz(model, k, z_init)
                    _, delta, _, _ = tk_store_algebra(
                        omega_sq_init,
                        d_ln_omega_init,
                        fn.epsilon(z_init),
                        fn.wPerturbations(z_init),
                        z_init,
                        float(v_init),
                        float(vprime_init),
                    )

                div, mod = apply_phase_offset(
                    payload["theta_div_2pi_sample"],
                    payload["theta_mod_2pi_sample"],
                    delta,
                )
                for (_, stored_div, stored_mod), d, m in zip(samples, div, mod):
                    checked_samples += 1
                    if stored_div != d or stored_mod != m:
                        mismatched += 1
                        residual = fabs(
                            (stored_div - d) * 2.0 * np.pi + (stored_mod - m)
                        )
                        if worst_case is None or residual > worst_case[0]:
                            worst_case = (residual, k, z_init, serial)
                checked_objects += 1

        results[sector] = {
            "objects": checked_objects,
            "samples": checked_samples,
            "mismatched": mismatched,
            "worst": worst_case,
        }
        detail = (
            ""
            if worst_case is None
            else f" (worst residual {worst_case[0]:.3e} rad at k = {worst_case[1]:.4g}/Mpc, "
            f"z_init = {worst_case[2]:.6g}, serial {worst_case[3]})"
        )
        print(
            f"   {sector}: {checked_objects} objects, {checked_samples} samples; "
            f"{mismatched} not bit-identical to the offline producer{detail}"
        )
    return results


# ---------------------------------------------------------------------------------------------
# check 3 -- the consumers, and WKB_phase_spline_chunks
# ---------------------------------------------------------------------------------------------


def check_consumers(conns):
    print("\n-- 3. the consumers: GkSourcePolicyData and QuadSourceIntegral")
    names = {0: "numeric", 1: "WKB", 2: "mixed", 99: "fail"}
    types = {}
    chunks = {}
    n_qsi = 0
    n_policy = 0
    for _, c in conns:
        for (t,) in c.execute("select type from GkSourcePolicyData"):
            label = names.get(t, f"<{t}>")
            types[label] = types.get(label, 0) + 1
            n_policy += 1
        for (n,) in c.execute("select WKB_phase_spline_chunks from QuadSourceIntegral"):
            chunks[n] = chunks.get(n, 0) + 1
            n_qsi += 1
    print(f"   GkSourcePolicyData rows: {n_policy}, by type {types}")
    print(
        f"   QuadSourceIntegral rows: {n_qsi}, WKB_phase_spline_chunks histogram {chunks}"
    )
    return {"policy": n_policy, "types": types, "qsi": n_qsi, "chunks": chunks}


# ---------------------------------------------------------------------------------------------
# check 4 -- solver provenance and the retired stage 2
# ---------------------------------------------------------------------------------------------


def check_solvers(conns):
    print("\n-- 4. solver provenance, and the retired stage-2 columns")
    out = {}
    for _, c in conns:
        labels = {
            r[0]: r[1] for r in c.execute("select serial, label from IntegrationSolver")
        }
        for table, col in (
            ("GkWKBIntegration", "solver_serial"),
            ("TkWKBIntegration", "phase_solver_serial"),
            ("TkWKBIntegration", "friction_solver_serial"),
            ("BackgroundModel", "solver_serial"),
        ):
            for (s,) in c.execute(f"select distinct {col} from {table}"):
                key = f"{table}.{col}"
                out.setdefault(key, set()).add(labels.get(s, f"<serial {s}>"))
        for table in ("GkWKBIntegration", "TkWKBIntegration"):
            n_total = c.execute(f"select count(*) from {table}").fetchone()[0]
            n_null = c.execute(
                f"select count(*) from {table} where stage_2_compute_time is null "
                f"and stage_2_RHS_evaluations is null"
            ).fetchone()[0]
            key = f"{table}.stage_2 NULL"
            prev = out.get(key, (0, 0))
            out[key] = (prev[0] + n_null, prev[1] + n_total)
    for key, value in sorted(out.items()):
        if isinstance(value, set):
            print(f"   {key}: {sorted(value)}")
        else:
            print(f"   {key}: {value[0]} of {value[1]}")
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-glob", required=True)
    parser.add_argument(
        "--model", required=True, choices=("LambdaCDMModel", "QCDModel")
    )
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--objects", type=int, default=6)
    args = parser.parse_args()

    conns = open_shards(args.shards_glob)
    print(f"** {len(conns)} shards: {[p for p, _ in conns]}")

    bg = read_background(conns)
    print(
        f"** background grid: {len(bg['z'])} nodes, z in [{bg['z'][-1]:.6g}, {bg['z'][0]:.6g}]"
    )

    model = build_model(args.model, bg)
    check_background(args.model, bg, model, args.nodes)
    check_phases(conns, model, bg, args.objects)
    check_consumers(conns)
    check_solvers(conns)


if __name__ == "__main__":
    main()
