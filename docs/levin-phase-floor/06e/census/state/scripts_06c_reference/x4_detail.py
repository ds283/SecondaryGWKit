"""levin-phase-floor prompt 06c section 5, X4: a descriptive breakdown of the rows that violate X4's
rule, after X4 was scored (not part of the rule). New in 06c.

For each row outside 06a's 720 stationary-point rows whose |total_g - total_06b| exceeds
total_abserr_06b + total_abserr_g (exactly, as analyse06c.py scores it): the row's Levin calls
(group and analytic) matched to 06b's, each call's |value_g - value_06b| against
abserr_g + abserr_06b, and the call that moved most. Tabulates, over the violating rows, the
kind of that call, whether the call itself violates the same test, and its 06b and g flags. Writes
state/x4_worst_calls.json: the most-violating call (by |dvalue| / (abserr_g + abserr_06b)) of each
of the N worst rows per rtol, in x1_capture.py's selection form, for a descriptive reference.

Usage: python -B x4_detail.py RUN_DIR [N]
"""

import collections
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06B, WIDER, header, sha256  # noqa: E402
from analyse06c import all_calls, fl, hx, match_calls, read_rtol, rows_by_key, scan06a  # noqa: E402


def main():
    run_dir = Path(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: X4's violating rows, call by call (descriptive)",
                    [RUN06B / "checkpoint.jsonl", run_dir / "checkpoint.jsonl"]):
        print(l)
    by_key = rows_by_key()
    _scans, sp = scan06a()
    picks = []
    for rtol in ("1e-7", "1e-8"):
        G = read_rtol(run_dir, rtol)
        B = read_rtol(RUN06B, rtol)
        tab = collections.Counter()
        worst_rows = []
        for rk, dg in G.items():
            if rk in sp:
                continue
            db = B[rk]
            rg, rb = dg["result"], db["result"]
            tg, tb, eg, eb = fl(rg["total"]), fl(rb["total"]), fl(rg["total_abserr"]), fl(rb["total_abserr"])
            if not (math.isfinite(eg) and math.isfinite(eb)):
                continue
            lhs = abs(Fraction(tg) - Fraction(tb))
            rhs = Fraction(eb) + Fraction(eg)
            if lhs <= rhs:
                continue
            matched, _p = match_calls(dg, db)
            best = None
            n_call_viol = 0
            for kind, cg, cb in matched:
                d = abs(fl(cg["value"]) - fl(cb["value"]))
                s = fl(cg["abserr"]) + fl(cb["abserr"])
                ratio = d / s if s > 0 else float("inf")
                n_call_viol += ratio > 1
                if best is None or d > best[0]:
                    best = (d, ratio, kind, cg, cb)
            d, ratio, kind, cg, cb = best
            # the call that violates the call-level test most (for the descriptive reference)
            wv = max(((abs(fl(c1["value"]) - fl(c2["value"])) / (fl(c1["abserr"]) + fl(c2["abserr"])) if fl(c1["abserr"]) + fl(c2["abserr"]) > 0 else float("inf"), k1, c1, c2)
                      for k1, c1, c2 in matched), key=lambda t: t[0])
            tab[("most-violating call kind", wv[1])] += 1
            tab[("most-violating call: 06b converged", wv[3]["converged"])] += 1
            tab[("most-violating call: g converged", wv[2]["converged"])] += 1
            tab[("rows", "")] += 1
            tab[("largest-moving call kind", kind)] += 1
            tab[("largest-moving call violates the call-level test", ratio > 1)] += 1
            tab[("rows with any call violating the call-level test", n_call_viol > 0)] += 1
            tab[("largest-moving call: 06b converged", cb["converged"])] += 1
            tab[("largest-moving call: g converged", cg["converged"])] += 1
            tab[("largest-moving call: 06b max_depth", cb["max_depth"] >= 20)] += 1
            tab[("largest-moving call: dvalue share of dtotal >= 0.5", d >= 0.5 * float(lhs))] += 1
            tab[("band", by_key[rk]["band"])] += 1
            worst_rows.append((float(lhs / rhs), rk, dg["unit"], kind, cg["index"], cg["notify_label"], ratio, hx(cg["value"]), hx(cg["abserr"]),
                               hx(cb["value"]), hx(cb["abserr"]), cg["converged"], cb["converged"], cg["max_depth"], cb["max_depth"], wv[2], wv[3], wv[0]))
        print(f"\n== {rtol}: violating rows {tab[('rows', '')]}")
        for k_, v in sorted(tab.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            if k_[0] != "rows":
                print(f"   {k_[0]}: {k_[1]} -> {v}")
        worst_rows.sort(key=lambda t: -t[0])
        print("   the ten worst rows: (row ratio, row, kind, call, label, call ratio, g value, g abserr, 06b value, 06b abserr, conv g, conv 06b, depth g, depth 06b)")
        for t in worst_rows[:10]:
            print(f"      {t[0]:.3g} {t[1]} {t[3]} {t[4]} {(t[5] or '')[-45:]} {t[6]:.3g} {t[7]} {t[8]} {t[9]} {t[10]} {t[11]} {t[12]} {t[13]} {t[14]}")
        print("   the most-violating call of each of the ten worst rows: (row, call, label, call ratio)")
        for t in worst_rows[:10]:
            print(f"      {t[1]} {t[15]['index']} {(t[15]['notify_label'] or '')[-45:]} {t[17]:.3g}")
        for t in worst_rows[:N]:
            cg, cb = t[15], t[16]
            R = by_key[t[1]]
            picks.append({"unit": t[2], "index": cg["index"], "notify_label": cg["notify_label"], "row": R["name"], "band": R["band"], "k_key": R["k_key"],
                          "k": R["k"], "category": f"X4 worst row ({rtol})", "census_value": hx(cg["value"]), "census_abserr": hx(cg["abserr"]),
                          "census_num_regions": cg["num_regions"], "census_max_depth": cg["max_depth"], "census_converged": cg["converged"],
                          "value_06b": hx(cb["value"]), "abserr_06b": hx(cb["abserr"]), "converged_06b": cb["converged"]})
        del G, B
    out = WIDER / "state" / "x4_worst_calls.json"
    out.write_text(json.dumps(picks, indent=1))
    print(f"\nwrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
