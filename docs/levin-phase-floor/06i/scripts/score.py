"""levin-phase-floor prompt 06i Sec 2-5: per-sample tables and the verdicts R1-R4. New in 06i.

Reads only: ../out/readings/sNN-{fcc,qawo}-r{1,2}[-bound].json (this prompt's readings),
06e's out/z3_scores.json (the census values and abserr under 06c and 06e, and Gauss-Legendre's
three resolutions), 06e's references/out/ref-sNN-P*.txt and 06c's orchestrator_check
02-/04-ref-sNN*.txt (Gauss-Legendre wall times, read only), ../out/closed_form_a.json and
../out/r0_jitter.json.

Per sample, exactly as the prompt writes it:
- four readings: FCC res 1 and 2, QAWO res 1 and 2;
- scatter = max - min of the four, against m = min(abserr_06c, abserr_06e); the reference counts iff
  scatter <= m/100 (Z3's rule);
- the reference value is FCC res 2 (mpmath, the finer of the two FCC readings);
- Sec 8's check: |FCC_r2 - QAWO_r2| against the larger of the two implementations' own scatters,
  |FCC_r1 - FCC_r2| and |QAWO_r1 - QAWO_r2|;
- the Gauss-Legendre difference |ref - S128(P64)| against Filon scatter + GL scatter + m/100 (R1);
- |v_06e - ref|/abserr_06e and |v_06c - ref|/abserr_06c (R3);
- wall time per reading (R4 for Route A: each under 600 s).
Descriptive: Route A against the closed form (samples 8, 10, 22, 23, 25); Route B's jitter.
All comparisons are made in mpmath at 50 digits on the 30-digit values (and on the hex of the
census values).
Usage: python -B score.py   (writes ../out/scores.json and ../out/results.txt)
"""

import json
import re
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prov  # noqa: E402

mp.mp.dps = 50
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
RD = OUT / "readings"
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
Z3 = MAIN / "docs/levin-phase-floor/06e/census/references/out/z3_scores.json"
GLOUT = MAIN / "docs/levin-phase-floor/06e/census/references/out"
ORCH = MAIN / "var/levin-phase-floor-06c/depth20-tiling/orchestrator_check"
ROUTE_A = (8, 9, 10, 22, 23, 25, 27)
ROUTE_B = (1, 2, 3, 13, 14, 15)
GL_COUNTED = {2, 3, 8, 9, 14, 15}
DROPPED = {"A": (10, 22, 23, 25, 27), "B": (1, 13)}


def hx(s):
    return mp.mpf(float.fromhex(s))


def reading(n, impl, res):
    for tag in (f"s{n:02d}-{impl}-r{res}", f"s{n:02d}-{impl}-r{res}-bound"):
        p = RD / f"{tag}.json"
        if p.exists():
            return json.loads(p.read_text()), p
    return None, None


def gl_walls(n):
    files = (
        [ORCH / f"02-ref-s{n:02d}.txt", ORCH / f"04-ref-s{n:02d}-P64.txt"]
        if n in (1, 9, 25)
        else [GLOUT / f"ref-s{n:02d}-P128.txt", GLOUT / f"ref-s{n:02d}-P64.txt"]
    )
    w = []
    for f in files:
        m = re.search(r"^wall ([0-9.]+) s", f.read_text(), re.M)
        w.append(float(m.group(1)) if m else None)
    return w, [str(f) for f in files]


def main():
    z3 = {r["n"]: r for r in json.loads(Z3.read_text())["rows"]}
    cf = {
        r["sample"]: r
        for r in json.loads((OUT / "closed_form_a.json").read_text())["samples"]
    }
    jit = {
        r["sample"]: r
        for r in json.loads((OUT / "r0_jitter.json").read_text())["samples"]
    }
    rows = []
    inputs = [Z3, OUT / "closed_form_a.json", OUT / "r0_jitter.json"]
    for n in ROUTE_A + ROUTE_B:
        route = "A" if n in ROUTE_A else "B"
        z = z3[n]
        R = {}
        for impl in ("fcc", "qawo"):
            for res in (1, 2):
                d, p = reading(n, impl, res)
                if d is not None:
                    R[f"{impl}_r{res}"] = d
                    inputs.append(p)
        row = dict(
            sample=n, route=route, block=z["block"], unit=z["unit"], index=z["index"]
        )
        row["readings"] = {
            k: dict(
                value_30=v["value_30"], value_hex=v["value_hex"], wall_s=v["wall_s"]
            )
            for k, v in R.items()
        }
        e_g, e_c = hx(z["abserr_g"]), hx(z["abserr_06c"])
        v_g, v_c = hx(z["value_g"]), hx(z["value_06c"])
        m = min(e_g, e_c)
        row["min_abserr"] = float(m)
        w, wf = gl_walls(n)
        row["gl_wall_s"] = w
        row["gl_files"] = wf
        if len(R) < 4:
            row["complete"] = False
            rows.append(row)
            continue
        row["complete"] = True
        val = {k: mp.mpf(v["value_30"]) for k, v in R.items()}
        vals = list(val.values())
        scatter = max(vals) - min(vals)
        ref = val["fcc_r2"]
        s_f = abs(val["fcc_r1"] - val["fcc_r2"])
        s_q = abs(val["qawo_r1"] - val["qawo_r2"])
        D = abs(val["fcc_r2"] - val["qawo_r2"])
        gl_ref = hx(z["S128_P64"])
        gl_scatter = mp.mpf(z["scatter_over_min"]) * m
        row.update(
            ref_30=mp.nstr(ref, 30),
            ref_hex=float(ref).hex(),
            scatter=float(scatter),
            scatter_over_min=float(scatter / m),
            counts=bool(scatter <= m / 100),
            impl_scatter_fcc=float(s_f),
            impl_scatter_qawo=float(s_q),
            impl_disagreement=float(D),
            impl_disagreement_over_larger_scatter=(
                float(D / max(s_f, s_q)) if max(s_f, s_q) > 0 else float("inf")
            ),
            sec8_implementations_disagree=bool(D > max(s_f, s_q)),
            gl_counted=n in GL_COUNTED,
            gl_ref_hex=z["S128_P64"],
            gl_scatter=float(gl_scatter),
            gl_diff=float(abs(ref - gl_ref)),
            gl_diff_allowance=float(scatter + gl_scatter + m / 100),
            gl_diff_over_allowance=float(
                abs(ref - gl_ref) / (scatter + gl_scatter + m / 100)
            ),
            gl_diff_over_sum_scatters=float(abs(ref - gl_ref) / (scatter + gl_scatter)),
            ratio_06e=float(abs(v_g - ref) / e_g),
            ratio_06c=float(abs(v_c - ref) / e_c),
            r3_violation=bool(abs(v_g - ref) > e_g),
            max_wall_s=max(v["wall_s"] for v in R.values()),
        )
        if route == "A" and n in cf:
            c = mp.mpf(cf[n]["value_30"])
            row["closed_form_30"] = cf[n]["value_30"]
            row["closed_form_diff"] = {k: float(v - c) for k, v in val.items()}
            row["closed_form_max_diff_over_scatter"] = (
                float(max(abs(v - c) for v in vals) / scatter)
                if scatter > 0
                else float("inf")
            )
            row["closed_form_max_diff_over_min_abserr"] = float(
                max(abs(v - c) for v in vals) / m
            )
        if route == "B":
            row["jitter"] = {
                k: R[k]["jitter"] for k in ("fcc_r1", "fcc_r2") if "jitter" in R[k]
            }
            row["jitter_floor_GL_density"] = jit[n]["jitter_floor_GL_P128_n96"]
            row["tmap"] = {k: R[k].get("tmap") for k in R}
            row["abscissa_correction"] = {
                k: R[k].get("abscissa_correction") for k in ("fcc_r1", "fcc_r2")
            }
            row["ratio_06e_with_jitter"] = {
                k: float(
                    abs(v_g - (ref + mp.mpf(R[k]["jitter"]["jittered_minus_smooth"])))
                    / e_g
                )
                for k in ("fcc_r1", "fcc_r2")
                if "jitter" in R[k]
            }
        if "stats" in R["qawo_r1"]:
            row["qawo_stats"] = {k: R[k]["stats"] for k in ("qawo_r1", "qawo_r2")}
        rows.append(row)

    by = {r["sample"]: r for r in rows}
    complete = all(r["complete"] for r in rows)
    verdicts = {}

    def r1_ok(n):
        r = by[n]
        return r["complete"] and r["counts"] and r["gl_diff"] <= r["gl_diff_allowance"]

    if complete:
        rA = all(r1_ok(n) for n in (8, 9))
        rB = all(r1_ok(n) for n in (2, 3, 14, 15))
        verdicts["R1"] = {
            "route_A": "holds" if rA else "fails",
            "route_B": "holds" if rB else "fails",
            "per_sample": {n: r1_ok(n) for n in (8, 9, 2, 3, 14, 15)},
        }
        r2 = {}
        for route, ok in (("A", rA), ("B", rB)):
            if not ok:
                r2[route] = "not scored (route fails R1)"
            else:
                r2[route] = (
                    "holds" if all(by[n]["counts"] for n in DROPPED[route]) else "fails"
                )
        verdicts["R2"] = r2
        held = [rt for rt, ok in (("A", rA), ("B", rB)) if ok]
        counted = [r for r in rows if r["route"] in held and r["counts"]]
        newly = [r for r in counted if not r["gl_counted"]]
        viol = [r["sample"] for r in counted if r["r3_violation"]]
        verdicts["R3"] = {
            "counted": [r["sample"] for r in counted],
            "newly_counted": [r["sample"] for r in newly],
            "violations": viol,
            "verdict": (
                "fails" if viol else ("undecided" if len(newly) < 5 else "holds")
            ),
        }
        aw = {
            r["sample"]: max(v["wall_s"] for v in r["readings"].values())
            for r in rows
            if r["route"] == "A"
        }
        verdicts["R4"] = {
            "max_route_A_reading_wall_s": aw,
            "verdict": "holds" if all(w < 600 for w in aw.values()) else "fails",
        }
    out = {
        "header": prov.header(inputs),
        "complete": complete,
        "verdicts": verdicts,
        "rows": rows,
    }
    (OUT / "scores.json").write_text(json.dumps(out, indent=1))

    L = []
    L.append(
        "sample route block | ref (FCC r2) hex | scatter/m | counts | GL diff/allowance | |v06e-ref|/e06e | |v06c-ref|/e06c | impl D/larger scatter | max wall s"
    )
    for r in rows:
        if not r["complete"]:
            L.append(
                f"s{r['sample']:02d} {r['route']} {r['block']} | INCOMPLETE ({len(r['readings'])} readings)"
            )
            continue
        L.append(
            f"s{r['sample']:02d} {r['route']} {r['block']} | {r['ref_hex']} | {r['scatter_over_min']:.3g} | "
            f"{'COUNTS' if r['counts'] else 'dropped'} | {r['gl_diff_over_allowance']:.3g}{' (GL counted)' if r['gl_counted'] else ''} | "
            f"{r['ratio_06e']:.4g}{' VIOLATES' if r['r3_violation'] else ''} | {r['ratio_06c']:.4g} | "
            f"{r['impl_disagreement_over_larger_scatter']:.3g}{' SEC8' if r['sec8_implementations_disagree'] else ''} | {r['max_wall_s']:.0f}"
        )
    L.append("")
    for r in rows:
        if r["complete"]:
            L.append(
                f"s{r['sample']:02d} readings: "
                + "; ".join(
                    f"{k} {v['value_30']} ({v['wall_s']:.0f} s)"
                    for k, v in r["readings"].items()
                )
                + f"; GL walls {r['gl_wall_s']}"
            )
    L.append("")
    for r in rows:
        if r.get("closed_form_30"):
            L.append(
                f"s{r['sample']:02d} closed form {r['closed_form_30']}: reading - CF "
                + ", ".join(f"{k} {v:+.3e}" for k, v in r["closed_form_diff"].items())
                + f"; max |diff|/scatter {r['closed_form_max_diff_over_scatter']:.3g}, /min abserr {r['closed_form_max_diff_over_min_abserr']:.3g}"
            )
    L.append("")
    for r in rows:
        if r["complete"] and r["route"] == "B":
            L.append(
                f"s{r['sample']:02d} jitter: "
                + "; ".join(
                    f"{k}: jittered-smooth {v['jittered_minus_smooth']:+.3e}, floor {v['floor_1sigma_at_these_nodes']:.3e}"
                    for k, v in r["jitter"].items()
                )
                + f"; GL-density floor {r['jitter_floor_GL_density']:.3e}; m/100 {r['min_abserr']/100:.3e}; |v06e-(ref+jit)|/e06e {r['ratio_06e_with_jitter']}"
            )
    L.append("")
    L.append(json.dumps(verdicts, indent=1))
    (OUT / "results.txt").write_text(
        json.dumps(out["header"], indent=1) + "\n\n" + "\n".join(L) + "\n"
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
