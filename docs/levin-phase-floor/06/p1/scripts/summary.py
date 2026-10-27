"""levin-phase-floor 06, finding P1: the summary tables, from the JSON every other p1 script wrote.
Needs no tree. Every number here is read from those files; nothing is recomputed but sums,
differences and the (1 + z_response) scaling (one multiplication, as phase_group_Levin_integral
applies it).

The reference for a row's G+q+r under a tree is reference.py's Phi = 1e4 value (fine GL on the
window, Levin m = 9 at the stored rtol outside). Its error is quoted two ways:
  * "tight": the largest of |GL fine - GL coarse| (Phi = 1e4), the spread of the reference over
    every Phi computed, and |Levin outer m9 rtol 1e-7 - m9 rtol 1e-10| (Phi = 1e4);
  * "conservative": the tight error plus |Levin outer m9 - m6| (Phi = 1e4).
The corrected total of a store is its stored total minus its stored G+q+r value plus
(1 + z_response) times the reference under its own tree. "corrected_total_all_low_groups" also
replaces the other three low-sub-interval groups by their forced 2^12-piece Levin sums.

Usage: python -B summary.py   Writes out/summary.json.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import OUT, header, sha256  # noqa: E402

TREES = ("r13", "r14")
ROWS = ("row1", "row2")


def h(d):
    return float.fromhex(d["hex"])


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    V = {t: load(f"validate_{t}.json") for t in TREES}
    A = {t: load(f"analyse_{t}.json") for t in TREES}
    R = {t: load(f"reference_{t}.json") for t in TREES}
    D = {t: load(f"rootdiag_{t}.json") for t in TREES}
    inputs = [OUT / f"{k}_{t}.json" for k in ("validate", "analyse", "reference", "rootdiag") for t in TREES]
    out = {"header": header(Path(__file__), "p1 summary", inputs=inputs), "rows": {}}
    for row in ROWS:
        r = {}
        scale = {t: h(A[t]["rows"][row]["stored_scale_1_plus_z_response"]) for t in TREES}
        assert scale["r13"] == scale["r14"]
        s = scale["r13"]
        r["scale_1_plus_z_response"] = s
        stored = {}
        for t in TREES:
            items = {x["name"]: x for x in V[t]["rows"][row]["validation"]["items"]}
            stored[t] = {
                "total": h(items["total"]["stored"]),
                "total_abserr": h(items["total_abserr"]["stored"]),
                "Gqr": h(items["sub[1].G+q+r.value"]["stored"]),
                "Gqr_abserr": h(items["sub[1].G+q+r.abserr"]["stored"]),
                "all_identical": V[t]["rows"][row]["validation"]["all_identical"],
                "n_items": len(items),
            }
        r["stored"] = stored
        for t in TREES:
            g = A[t]["rows"][row]["groups"]["G+q+r"]
            m = {}
            for rt, v in g["rtol_sweep"].items():
                m[f"(b) Levin as QSI calls it, rtol {rt}"] = {
                    "value": h(v["value"]), "abserr": h(v["abserr"]), "regions": v["num_regions"],
                    "evaluations": v["evaluations"], "converged": v["converged"],
                }
            for mm, v in g["forced"].items():
                m[f"(c) forced 2^{mm} pieces, rtol 1e-07"] = {
                    "value": h(v["sum"]), "abserr": h(v["sum_abserr"]), "regions": v["num_regions_total"],
                    "converged": v["all_converged"],
                }
            ref = R[t]["rows"][row]["windows"]
            for phi, w in ref.items():
                m[f"(d) reference Phi {phi}: GL(window) + Levin m9 (outer)"] = {
                    "value": h(w["reference"]), "gl_err": h(w["gl_error_estimate"]),
                    "levin_outer_err_m6_vs_m9_or_rtol": h(w["levin_outer_error_estimate"]),
                    "window_z": [h(w["z_window"][0]), h(w["z_window"][1])],
                    "gl_pieces_fine": w["gl"]["dphi0.5_n20"]["pieces"],
                    "gl_max_piece_span_rad_fine": w["gl"]["dphi0.5_n20"]["max_piece_phase_span"],
                    "gl_window_fine": h(w["gl"]["dphi0.5_n20"]["value"]),
                    "levin_m9_over_window": h(w["levin_over_window_m9"]["value"]),
                }
            w4 = ref["10000"]
            refval = h(w4["reference"])
            spread = max(abs(h(w["reference"]) - refval) for w in ref.values())
            lo = w4["levin_outer"]
            m9 = h(lo["m9_rtol1e-07"]["sum"])
            m9t = h(lo["m9_rtol1e-10"]["sum"])
            m6 = h(lo["m6_rtol1e-07"]["sum"])
            tight = max(h(w4["gl_error_estimate"]), spread, abs(m9 - m9t))
            cons = tight + abs(m9 - m6)
            r.setdefault("methods", {})[t] = m
            r.setdefault("reference", {})[t] = {
                "value": refval, "err_tight": tight, "err_conservative": cons,
                "scaled": s * refval, "scaled_err_tight": s * tight, "scaled_err_conservative": s * cons,
                "Phi_spread": spread,
            }
        # errors of each store's G+q+r and total against the reference under its own tree
        comp = {}
        for t in TREES:
            ref_s = r["reference"][t]["scaled"]
            st = stored[t]
            corrected = st["total"] - st["Gqr"] + ref_s
            comp[t] = {
                "Gqr_stored_minus_reference": st["Gqr"] - ref_s,
                "Gqr_error_over_declared_abserr": abs(st["Gqr"] - ref_s) / st["Gqr_abserr"],
                "corrected_total": corrected,
                "total_stored_minus_corrected": st["total"] - corrected,
                "total_error_over_declared_total_abserr": abs(st["total"] - corrected) / st["total_abserr"],
            }
        # also: every group of the low sub-interval replaced (G+q+r by the reference, the other
        # three by their forced 2^12-piece Levin sums), the high sub-interval and quad parts as stored
        for t in TREES:
            items = {x["name"]: x for x in V[t]["rows"][row]["validation"]["items"]}
            g = A[t]["rows"][row]["groups"]
            st = stored[t]
            low_stored = sum(h(items[f"sub[1].{lab}.value"]["stored"]) for lab in ("G+q+r", "G+q-r", "G-q+r", "G-q-r"))
            low_new = r["reference"][t]["scaled"] + sum(
                s * h(g[lab]["forced"]["12"]["sum"]) for lab in ("G+q-r", "G-q+r", "G-q-r")
            )
            comp[t]["corrected_total_all_low_groups"] = st["total"] - low_stored + low_new
        r["comparison"] = comp
        # the two trees' references and corrected totals against each other
        r["reference_r14_minus_r13"] = r["reference"]["r14"]["value"] - r["reference"]["r13"]["value"]
        r["corrected_total_r14_minus_r13"] = comp["r14"]["corrected_total"] - comp["r13"]["corrected_total"]
        # the root decision (rootdiag)
        r["root"] = {
            t: {k: D[t]["rows"][row]["probes"][0][k] for k in
                ("estimate", "step4_residual", "relerr", "rtol", "roundoff_floor", "resolved", "phase_limited")}
            for t in TREES
        }
        out["rows"][row] = r
        print(f"== {row}: scale {s!r}")
        for t in TREES:
            st, c, rf = stored[t], comp[t], r["reference"][t]
            print(f"   {t}: stored G+q+r {st['Gqr']:.10e} (abserr {st['Gqr_abserr']:.2e}); reference x scale "
                  f"{rf['scaled']:.10e} +- {rf['scaled_err_tight']:.1e} (conservative {rf['scaled_err_conservative']:.1e}); "
                  f"stored - ref {c['Gqr_stored_minus_reference']:.4e} = {c['Gqr_error_over_declared_abserr']:.3g} x abserr")
            print(f"       total stored {st['total']:.10e}, corrected {c['corrected_total']:.10e}, stored - corrected "
                  f"{c['total_stored_minus_corrected']:.4e} = {c['total_error_over_declared_total_abserr']:.3g} x total_abserr")
        print(f"   all-low-groups corrected totals: r13 {comp['r13']['corrected_total_all_low_groups']:.10e}, "
              f"r14 {comp['r14']['corrected_total_all_low_groups']:.10e}")
        print(f"   reference r14 - r13 {r['reference_r14_minus_r13']:.3e} (unscaled); corrected totals r14 - r13 "
              f"{r['corrected_total_r14_minus_r13']:.3e}")
    path = OUT / "summary.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
