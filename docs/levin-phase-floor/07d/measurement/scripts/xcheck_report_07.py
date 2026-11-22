"""levin-phase-floor prompt 07, beyond section 6: the cross-check's tables as text, from out/xcheck/xcheck.json and
out/xcheck/projection.json (read only). New in 07.
Usage: python -B xcheck_report_07.py XCHECK_JSON PROJECTION_JSON OUT_MD"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402


def f(x):
    return "None" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))


def main():
    xp, pp, out = (Path(a) for a in sys.argv[1:4])
    X, P = json.loads(xp.read_text()), json.loads(pp.read_text())
    A, B = X["A"], X["B"]
    L = [l for l in header(Path(__file__).resolve(), "levin-phase-floor 07: the cross-check's tables", [xp, pp])]
    L += ["", f"rule: {X['rule']} sha256 {X['rule_sha256']}; checkpoints unchanged before/after: {X['checkpoints_unchanged']}", ""]
    L += ["## A. The cross-rtol check", "",
          f"Population: {A['population_calls']} calls (both rtols); {A['pairs']} pairs; unpaired {len(A['unpaired'])}.", "",
          "| | pairs | r_g <= 1 | r_06m <= 1 |", "|---|---|---|---|",
          f"| all | {A['overall']['pairs']} | {A['overall']['r_g<=1']} | {A['overall']['r_m<=1']} |"]
    for name, tbl in (("kind", A["by_kind"]), ("band", A["by_band"]), ("k", A["by_k"])):
        for k, v in tbl.items():
            L.append(f"| {name} {k} | {v['pairs']} | {v['r_g<=1']} | {v['r_m<=1']} |")
    L += ["", "Distributions (quantiles; bins):", ""]
    for k, v in A["dist"].items():
        L.append(f"- {k}: q0 {f(v['q0'])}, q10 {f(v['q10'])}, q50 {f(v['q50'])}, q90 {f(v['q90'])}, q99 {f(v['q99'])}, "
                 f"max {f(v['q100'])}; bins {v['bins']}")
    L += ["", "Every pair with r_g > 1:", "",
          "| row | kind | index | group | band | k | r_g | r_06m | d8_g | d8_06m | class | values (g7, g8, m7, m8) | abserrs (g7, g8, m7, m8) | depth | regions | converged |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in A["r_g_gt_1"]:
        L.append(f"| {p['row']} | {p['kind']} | {p['index']} | {p['group']} | {p['band']} | {p['k']} | {f(p['r_g'])} | {f(p['r_m'])} | "
                 f"{f(p['d8_g'])} | {f(p['d8_m'])} | {p['class']} | {list(p['values'].values())} | {list(p['abserrs'].values())} | "
                 f"{p['depth']} | {p['regions']} | {p['converged']} |")
    L += ["", "## B. The disagreement list", "", f"Listed calls: {B['n_listed_calls']}; by class {B['by_class']}; "
          f"by class and rtol {B['by_class_rtol']}; by class and group {B['by_class_kind']}.", "",
          "By class, band and k:", "", "| class | band | k | calls |", "|---|---|---|---|"]
    for key, n in B["by_class_band_k"].items():
        c, b, k = key.split("|")
        L.append(f"| {c} | {b} | {k} | {n} |")
    L += ["", f"In V4's set: {B['in_v4']}", "",
          f"Row totals of the {B['row_totals_summary']['rows']} units these calls move: |dtotal|/total_abserr_06m "
          f"{B['row_totals_summary']['dtotal_over_abserr_06m']}; /total_abserr_g {B['row_totals_summary']['dtotal_over_abserr_g']}", "",
          "Classes (a) and (e), in full:", "",
          "| unit | index | class | dev/abserr_g | dev/abserr_06m | r_g | r_06m | d8_g | d8_06m | depth g/06m | regions g/06m | V4 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in B["calls"]:
        if c["class"] in ("a", "b", "d", "e"):
            L.append(f"| {c['unit']} | {c['index']} | {c['class']} | {f(c['dev_over_abserr_g'])} | {f(c['dev_over_abserr_06m'])} | "
                     f"{f(c['r_g'])} | {f(c['r_m'])} | {f(c['d8_g'])} | {f(c['d8_m'])} | {c['depth_g']}/{c['depth_06m']} | "
                     f"{c['regions_g']}/{c['regions_06m']} | {c['v4']} |")
    L += ["", "## Class (c) and (d): the reference cost (refs_projection_07.py's model, unchanged)", "",
          f"{P['n_projected']} of {P['n_calls']} rows projected (one capture per row; the integrand is the same at both rtols). "
          f"All four readings: {P['total_all_four_s']:.0f} s; FCC r1 and r2 only: {P['total_fcc_s']:.0f} s; FCC plus QAWO where "
          f"each QAWO reading projects at 30 minutes or less: {P['total_fcc_plus_qawo_le_1800_s']:.0f} s.", "",
          "| sid | unit | index | class | QAWO r1 s | QAWO r2 s | FCC r1 s | FCC r2 s | total s | n_seg | span (rad) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(P["calls"], key=lambda r: r.get("total_s", float("inf"))):
        if "readings_s" not in r:
            L.append(f"| {r['sid']} | {r['unit']} | {r['index']} | {r['class']} | not projected: {r['not_projected']} | | | | | | |")
            continue
        g = r["readings_s"]
        L.append(f"| {r['sid']} | {r['unit']} | {r['index']} | {r['class']} | {g['qawo_r1']:.0f} | {g['qawo_r2']:.0f} | {g['fcc_r1']:.0f} | "
                 f"{g['fcc_r2']:.0f} | {r['total_s']:.0f} | {r['features']['n_seg']} | {r['features']['span_q']:.3g} |")
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
