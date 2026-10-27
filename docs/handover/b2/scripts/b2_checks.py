"""
handover prompt 13 (B2), Part B: the checks made before the launch (prompt 13 section 2.5).

    python -B b2_checks.py construction    # 1: every cell's objects, built, not run
    python -B b2_checks.py completeness    # 2: E'+all five = R, R-all five = E', bit for bit
    python -B b2_checks.py determinism     # 3: E, E', R twice in one process, together 980
    python -B b2_checks.py transparency    # 4: R with and without the wrapper, together 980
    python -B b2_checks.py controls        # 5: E against A2's Table 2; N_E' - N_E; R against A4
    python -B b2_checks.py stationary      # 6: the phase groups' theta' over R's partition

Each writes `state/check_<name>.json` and `state/check_<name>.txt` beside this directory, prints a
summary and exits 0 when the check holds, 1 when it does not (a check that only records, such as
R against A4's arm P, never fails).

* **completeness** builds, at `together` 980, (i) the hybrid with *no* term delegated -- E' with
  all five terms made realistic -- and compares it with R itself, unwrapped; and (ii) the hybrid
  with *every* term delegated -- R with all five made exact, which is E' -- and compares it with
  an independent construction of the exact flavour on R's geometry,
  `test_quadsource_integral.exact_with_the_geometry_of(R)` (`ExactTkFunctions` breaking at R's
  switch nodes, `ExactGk`, `FakeQuadSource(exact=True)`). Both must be bit-identical in `total`
  and `total_abserr`. Disjointness is shown from the mapping (`b2_cells.ACCESSORS`), by listing.
* **stationary**: for every rung, every Levin sub-interval of R's partition and every phase group
  in it, the group's d theta / d log(1+z') from the exact closed forms (`_ExactPhase.theta_deriv`
  of `ExactGk` and `ExactTk`, combined with the group's signs), on 4001 points uniform in
  log(1+z') per sub-interval, ends included: min and max |theta'|, and any sign change.

No Ray, no datastore, no lock. After landing, this file is a record of its own SHA-256.
"""

import json
import re
import sys
import time
from math import log, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b2_cells as C  # noqa: E402

STATE = HERE.parent / "state"
A4_P_TOGETHER_980 = +3.623e-03  # GK-PHASE-COST-GROWTH.md Table H, arm P, at 28ebaf0 (pre-06)
SCAN_POINTS = 4001


def _hex(x):
    return float.hex(float(x))


def _N(record):
    return C.KT.KT_NORM * record["total"] / record["ref"]["predicted"]


def _err_N(record):
    return abs(_N(record)) * record["declared"]


# ------------------------------------------------------------------------------------------------


def construction(out):
    ok = True
    rows = []
    for shape_name, rungs in C.LADDER.items():
        for x_resp in rungs:
            for obj in C.OBJECTS:
                t0 = time.perf_counter()
                try:
                    case, delegation = C.build_case(shape_name, x_resp, obj)
                    objects = C.phase_objects(case)
                    problems = C.construction_problems(obj, objects, delegation)
                    below = None
                    if obj == "R":
                        # G's phase at log(1 + z_resp), as phase_group_Levin_integral forms it,
                        # and 1-8 ulp below it
                        phase = case.GkPolicy.functions.phase
                        a = log(1.0 + case.z_resp)
                        failures = []
                        u = a
                        for k in range(9):
                            try:
                                phase.theta_mod_2pi(u, x_is_log=True)
                                phase.raw_theta(u, x_is_log=True)
                                phase.theta_deriv(u, x_is_log=True, log_derivative=True)
                            except Exception as exc:  # noqa: BLE001
                                failures.append(f"a-{k}ulp: {type(exc).__name__}: {exc}")
                            u = float(np.nextafter(u, -np.inf))
                        below = failures
                        if failures:
                            problems.append(f"R: G's phase fails at or below z_resp: {failures}")
                    error = None
                except Exception as exc:  # noqa: BLE001
                    objects, problems, below = None, [], None
                    error = f"{type(exc).__name__}: {exc}"
                good = error is None and not problems
                ok = ok and good
                rows.append(
                    {
                        "shape": shape_name,
                        "x_resp": x_resp,
                        "object": obj,
                        "ok": good,
                        "error": error,
                        "problems": problems,
                        "objects": objects,
                        "below_anchor_failures": below,
                        "build_time": time.perf_counter() - t0,
                    }
                )
                if obj == "R" or not good:
                    G = (objects or {}).get("G", {})
                    Tq = (objects or {}).get("Tq", {})
                    Tr = (objects or {}).get("Tr", {})
                    out(
                        f"{shape_name:9s} {x_resp:8.0e} {obj:8s} {'ok ' if good else 'BAD'} "
                        f"G={G.get('class')}/{G.get('exact_remainder')}/{G.get('leading_class')} "
                        f"Tq={Tq.get('class')}/{Tq.get('exact_remainder')}/{Tq.get('leading_class')} "
                        f"WKB_top<z_init: {Tq.get('WKB_region_top', 0) < Tq.get('z_init', 0)} "
                        f"Tr={Tr.get('class')}/{Tr.get('exact_remainder')}/{Tr.get('leading_class')} "
                        f"WKB_top<z_init: {Tr.get('WKB_region_top', 0) < Tr.get('z_init', 0)}"
                        + ("" if below is None else f" G below z_resp: {len(below)} failures")
                        + (f" ERROR {error}" if error else "")
                        + (f" PROBLEMS {problems}" if problems else "")
                    )
    n_bad = sum(not r["ok"] for r in rows)
    out(f"{len(rows)} objects built ({len(rows) // len(C.OBJECTS)} rungs x {len(C.OBJECTS)}); {n_bad} with problems")
    return ok, {"cells": rows}


def _run_case(case):
    out = case.run(atol=C.REF_ATOL, rtol=C.REF_RTOL)
    return {k: out[k] for k in ("total", "total_abserr", "numeric_quad", "WKB_Levin")}


def completeness(out):
    shape_name, x_resp = "together", 980.0
    ok = True
    payload = {"cell": [shape_name, x_resp]}

    # (i) E' + all five = R
    R_case, _ = C.build_case(shape_name, x_resp, "R")
    R = _run_case(R_case)
    none_case, _ = C.build_case(shape_name, x_resp, "R")
    record_none = C.apply_hybrid(none_case, set())
    none = _run_case(none_case)
    same_i = {k: _hex(R[k]) == _hex(none[k]) for k in R}
    # (ii) R - all five = E', against the exact flavour on R's geometry
    all_case, _ = C.build_case(shape_name, x_resp, "Ep")
    Ep = _run_case(all_case)
    exact_geo = C.exact_with_the_geometry_of(R_case)
    G = _run_case(exact_geo)
    same_ii = {k: _hex(Ep[k]) == _hex(G[k]) for k in Ep}
    ok = same_i["total"] and same_i["total_abserr"] and same_ii["total"] and same_ii["total_abserr"]

    for label, a_name, a, b_name, b, same in (
        ("(i) E'+all five vs R", "hybrid, D = {}", none, "R unwrapped", R, same_i),
        ("(ii) R-all five vs E'", "hybrid, D = all", Ep, "exact_with_the_geometry_of(R)", G, same_ii),
    ):
        out(f"{label}:")
        for key in ("total", "total_abserr", "numeric_quad", "WKB_Levin"):
            out(
                f"   {key:13s} {a_name:16s} {a[key]!r:25s} {_hex(a[key]):24s} | {b_name} "
                f"{b[key]!r:25s} {_hex(b[key]):24s} {'bit-identical' if same[key] else 'DIFFER'}"
            )

    # disjointness, from the mapping
    seen = {}
    clashes = []
    for term, accessors in C.ACCESSORS.items():
        for acc in accessors:
            if acc in seen:
                clashes.append((acc, seen[acc], term))
            seen[acc] = term
    out("disjointness (b2_cells.ACCESSORS), accessor -> term:")
    for acc, term in seen.items():
        out(f"   {acc:30s} {term}")
    out(f"   {len(seen)} accessors, {len(clashes)} in two terms {clashes}")
    ok = ok and not clashes
    payload.update(
        {
            "i": {"hybrid_none": none, "R": R, "same": same_i, "delegation": record_none},
            "ii": {"hybrid_all": Ep, "exact_with_the_geometry_of": G, "same": same_ii},
            "accessor_to_term": seen,
            "clashes": clashes,
        }
    )
    return ok, payload


def _fingerprint(record):
    return {
        "total": _hex(record["total"]),
        "total_abserr": _hex(record["total_abserr"]),
        "groups": [
            [
                g["subinterval"],
                g["label"],
                g["num_regions"],
                g["max_depth"],
                g["max_region_depth"],
                g["num_phase_limited_regions"],
                _hex(g["raw_value"]),
                _hex(g["raw_abserr"]),
            ]
            for g in record["groups"]
        ],
    }


def determinism(out):
    ok = True
    objs = {}
    for obj in ("E", "Ep", "R"):
        first = C.run_cell("together", 980.0, obj)
        second = C.run_cell("together", 980.0, obj)
        a, b = _fingerprint(first), _fingerprint(second)
        same = a == b
        ok = ok and same
        objs[obj] = {"same": same, "first": a, "second": b}
        out(
            f"{obj:3s}: total {a['total']} / {b['total']}, total_abserr {a['total_abserr']} / "
            f"{b['total_abserr']}, {len(a['groups'])} groups, regions {first['Levin_regions']} / "
            f"{second['Levin_regions']}, max depth {first['max_depth']} / {second['max_depth']}: "
            f"{'bit-identical' if same else 'DIFFER'}"
        )
    return ok, {"cell": ["together", 980.0], "objects": objs}


def transparency(out):
    wrapped = C.run_cell("together", 980.0, "R", instrument=True)
    plain = C.run_cell("together", 980.0, "R", instrument=False)
    same = {
        key: _hex(wrapped[key]) == _hex(plain[key])
        for key in ("total", "total_abserr", "numeric_quad", "WKB_Levin")
    }
    same["Levin_regions_from_metadata"] = (
        wrapped["Levin_regions_from_metadata"] == plain["Levin_regions_from_metadata"]
    )
    ok = all(same.values())
    for key in ("total", "total_abserr", "numeric_quad", "WKB_Levin"):
        out(
            f"{key:14s} wrapped {wrapped[key]!r:26s} {_hex(wrapped[key])}  unwrapped "
            f"{plain[key]!r:26s} {_hex(plain[key])}  {'bit-identical' if same[key] else 'DIFFER'}"
        )
    out(
        f"Levin regions (metadata): wrapped {wrapped['Levin_regions_from_metadata']}, unwrapped "
        f"{plain['Levin_regions_from_metadata']}; the wrapper's own count {wrapped['Levin_regions']}"
    )
    return ok, {"same": same, "wrapped": {k: v for k, v in wrapped.items() if k != "groups"}, "unwrapped": plain}


def _a2_exact_closed():
    """A2's Table 2 exact, closed rows: {(shape, x_resp): (N + 9/8, declared)}."""
    text = (C.REPO / "docs" / "handover" / "REALISTIC-LARGE-X.md").read_text()
    rows = {}
    for line in text.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) >= 7 and cols[3] == "exact" and cols[4] == "closed":
            rows[(cols[0], float(cols[1]))] = (float(cols[5]), float(cols[6]), cols[5])
    return rows


def _agreeing_digits(a: float, b: float) -> int:
    """Significant digits of the 4-figure printed b that a reproduces (0-4)."""
    sa, sb = f"{a:+.3e}", f"{b:+.3e}"
    if sa[-4:] != sb[-4:] or sa[0] != sb[0]:
        return 0
    n = 0
    for ca, cb in zip(sa[1:6].replace(".", ""), sb[1:6].replace(".", "")):
        if ca != cb:
            break
        n += 1
    return n


def controls(out):
    ok = True
    a2 = _a2_exact_closed()
    rows = []
    out("E against A2's exact closed column (REALISTIC-LARGE-X.md Table 2, pre-B1 tree):")
    for shape_name, rungs in C.LADDER.items():
        for x_resp in rungs:
            E = C.run_cell(shape_name, x_resp, "E")
            Np = E["N_plus_9_8"]
            eN = _err_N(E)
            a2_Np, a2_decl, a2_text = a2[(shape_name, x_resp)]
            a2_eN = abs(a2_Np + C.KT.KT_NORM) * a2_decl  # |N| * declared, as A2's `term`
            diff = Np - a2_Np
            # A2's printed value carries a rounding of half a unit in its fourth figure
            rounding = 0.5e-3 * 10 ** np.floor(np.log10(abs(a2_Np))) if a2_Np != 0 else 0.0
            within = abs(diff) <= eN + a2_eN + rounding
            ok = ok and within
            digits = _agreeing_digits(Np, a2_Np)
            rows.append(
                {
                    "shape": shape_name,
                    "x_resp": x_resp,
                    "N_plus_9_8": Np,
                    "err_N": eN,
                    "A2_N_plus_9_8": a2_Np,
                    "A2_printed": a2_text,
                    "A2_err_N": a2_eN,
                    "A2_print_rounding": rounding,
                    "diff": diff,
                    "within": bool(within),
                    "agreeing_digits": digits,
                    "integral_time": E["integral_time"],
                }
            )
            out(
                f"   {shape_name:9s} {x_resp:8.0e} E {Np:+.6e} (err {eN:.1e}) A2 {a2_text} "
                f"(err {a2_eN:.1e}) diff {diff:+.2e} vs bound {eN + a2_eN + rounding:.1e}: "
                f"{'within' if within else 'OUTSIDE'}; {digits} of A2's 4 printed figures agree"
            )
    # geometry control
    E = C.run_cell("together", 980.0, "E")
    Ep = C.run_cell("together", 980.0, "Ep")
    R = C.run_cell("together", 980.0, "R")
    geo = _N(Ep) - _N(E)
    geo_err = _err_N(Ep) + _err_N(E)
    geo_ok = abs(geo) <= geo_err
    ok = ok and geo_ok
    out(
        f"geometry control, together 980: N_E' - N_E = {geo:+.3e}, sum of declared errors "
        f"{geo_err:.3e}: {'within' if geo_ok else 'OUTSIDE'} (N_E'+9/8 {Ep['N_plus_9_8']:+.6e}, "
        f"N_E+9/8 {E['N_plus_9_8']:+.6e})"
    )
    out(
        f"R at together 980: N+9/8 = {R['N_plus_9_8']:+.6e} (declared, in N, {_err_N(R):.1e}); "
        f"A4 arm P {A4_P_TOGETHER_980:+.3e} at 28ebaf0, before levin-phase-floor 06; difference "
        f"{R['N_plus_9_8'] - A4_P_TOGETHER_980:+.2e}, recorded, not judged"
    )
    return ok, {
        "E_vs_A2": rows,
        "geometry": {
            "N_Ep_minus_N_E": geo,
            "err": geo_err,
            "within": bool(geo_ok),
            "E": {k: E[k] for k in ("total", "total_abserr", "declared", "N_plus_9_8")},
            "Ep": {k: Ep[k] for k in ("total", "total_abserr", "declared", "N_plus_9_8")},
            "E_partition": E["partition_subintervals"],
            "Ep_partition": Ep["partition_subintervals"],
        },
        "R_vs_A4_P": {
            "R_N_plus_9_8": R["N_plus_9_8"],
            "R_err_N": _err_N(R),
            "A4_P": A4_P_TOGETHER_980,
        },
    }


def stationary(out):
    from ComputeTargets.phase_groups import group_signs, signs_label
    from ComputeTargets.tests.test_phase_groups import ExactGk, ExactTk

    ok = True
    rows = []
    per_shape = {}
    for shape_name, rungs in C.LADDER.items():
        for x_resp in rungs:
            case, _ = C.build_case(shape_name, x_resp, "R")
            shape, w, model = case.shape, case.w, case.model
            gk = ExactGk(shape.k, w, case.z_resp, model)
            Tq = ExactTk(shape.q, w, model)
            Tr = ExactTk(shape.r, w, model)
            partition = case.partition()
            for index, item in enumerate(partition["subintervals"]):
                if item["method"] != "Levin":
                    continue
                regime = tuple(item["regime"])
                u = np.linspace(log(1.0 + item["z_min"]), log(1.0 + item["z_max"]), SCAN_POINTS)
                d = {
                    "G": np.array([gk.phase.theta_deriv(x, x_is_log=True, log_derivative=True) for x in u]) if regime[0] else None,
                    "q": np.array([Tq.phase.theta_deriv(x, x_is_log=True, log_derivative=True) for x in u]) if regime[1] else None,
                    "r": np.array([Tr.phase.theta_deriv(x, x_is_log=True, log_derivative=True) for x in u]) if regime[2] else None,
                }
                for signs in group_signs(regime):
                    total = np.zeros_like(u)
                    scale = np.zeros_like(u)
                    for s, name in zip(signs, ("G", "q", "r")):
                        if s != 0 and d[name] is not None:
                            total += s * d[name]
                            scale = np.maximum(scale, np.abs(d[name]))
                    a = np.abs(total)
                    sign_change = bool(np.any(total == 0.0) or np.any(np.sign(total[:-1]) != np.sign(total[1:])))
                    i = int(np.argmin(a))
                    row = {
                        "shape": shape_name,
                        "x_resp": x_resp,
                        "subinterval": index,
                        "regime": "".join(n for n, f in zip("Gqr", regime) if f),
                        "z_max": item["z_max"],
                        "z_min": item["z_min"],
                        "group": signs_label(signs),
                        "min_abs_theta_prime": float(a[i]),
                        "max_abs_theta_prime": float(a.max()),
                        "min_relative": float(a[i] / scale[i]) if scale[i] > 0 else None,
                        "z_at_min": float(np.expm1(u[i])),
                        "sign_change": sign_change,
                    }
                    rows.append(row)
                    ok = ok and not sign_change
                    best = per_shape.get(shape_name)
                    if best is None or row["min_relative"] < best["min_relative"]:
                        per_shape[shape_name] = row
    out(f"{len(rows)} (rung, Levin sub-interval, group) rows, {SCAN_POINTS} points each; "
        f"{sum(r['sign_change'] for r in rows)} with a sign change")
    for shape_name, row in per_shape.items():
        out(
            f"   {shape_name:9s} nearest approach: x_resp {row['x_resp']:.0e}, sub-interval "
            f"{row['subinterval']} ({row['regime']}), group {row['group']}: min |theta'| = "
            f"{row['min_abs_theta_prime']:.4e} per log(1+z) = {row['min_relative']:.4f} of the "
            f"largest constituent's |theta'| there, at z = {row['z_at_min']:.6g}"
        )
    for shape_name in C.LADDER:
        mins = [r for r in rows if r["shape"] == shape_name]
        m = min(mins, key=lambda r: r["min_abs_theta_prime"])
        out(
            f"   {shape_name:9s} smallest absolute min |theta'|: {m['min_abs_theta_prime']:.4e} "
            f"per log(1+z) (x_resp {m['x_resp']:.0e}, {m['regime']}, {m['group']}; "
            f"{m['min_relative']:.4f} relative)"
        )
    return ok, {"rows": rows, "nearest_per_shape": per_shape}


def main():
    name = sys.argv[1]
    fn = {
        "construction": construction,
        "completeness": completeness,
        "determinism": determinism,
        "transparency": transparency,
        "controls": controls,
        "stationary": stationary,
    }[name]
    STATE.mkdir(parents=True, exist_ok=True)
    lines = []

    def out(text):
        print(text, flush=True)
        lines.append(text)

    started = time.time()
    import warnings

    warnings.simplefilter("ignore")
    ok, payload = fn(out)
    payload.update(
        {
            "check": name,
            "ok": ok,
            "b2_cells_sha256": C.SCRIPT_SHA256,
            "b2_checks_sha256": C.sha256_of(__file__),
            "started_unix": started,
            "elapsed": time.time() - started,
        }
    )
    with open(STATE / f"check_{name}.json", "w") as handle:
        json.dump(payload, handle, indent=1, default=repr)
    out(f"== check {name}: {'HOLDS' if ok else 'FAILS'} ({time.time() - started:.1f} s)")
    (STATE / f"check_{name}.txt").write_text("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
