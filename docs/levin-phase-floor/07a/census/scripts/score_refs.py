"""levin-phase-floor prompt 06g section 5, the references: the scores, **scored after stage 2 of the
rerun batch** (the user's ruling, 2026-11-02: score now, while stage 3 runs). New in 06g.

Rules, and nothing else: the predictions file (state/predictions_20261101T175120Z.md, "as
written"), the ruling file (state/ruling_addendum3_20261101T201347Z.md, "as ruled"), the V1 and V8
rule files, addenda 4 and 5 (the usable rule for Route B readings). All arithmetic exact: census
and capture doubles from hex, reading values from their 30-digit strings, in mpmath at 50 digits.

**Readings used per sample:**
- 06i's 13: 06i's four readings (docs/levin-phase-floor/06i/out/readings/), the references kept;
- z00 and z12, and V1's w00..w11: only the rerun batch's readings (out/readings/<tag>-rerun.json, the
  tool at ff3e8394); their old readings are unscored ("made by the tool at 1c5336e, which inverted
  through a non-monotone segment; unsound by the tool's own rule");
- every other sample: its part 2 readings (out/readings/<tag>.json); and any -rerun reading that
  exists for it (stage 3 of the batch is still running: r24, r26, r53 are FCC-only, r47 has three
  readings, at the time of scoring).
**Usable (addendum 5):** a Route B reading is usable iff (i) ``tmap.all_monotone`` is True, or (ii)
its tool's SHA-256 is ff3e8394's ``tools/filon_reference.py`` (5ed56fc2...) and
``tmap.direct_segments`` equals its count of non-monotone segments. Route A readings have no such
condition. Only usable readings are used; the clause each meets is recorded.

**As written:** a reference counts iff its four readings are present, all usable under clause (i)
(the predictions' "every reading reports all_monotone true"), and their spread (largest minus
smallest) is at most tol = min(abserr_06h, abserr_g)/100. V3 fails if a counted reference has
|v_g - ref| > abserr_g, else is undecided if fewer than 45 of the 90 count, else holds. V4: holds if
z02, z08, z09, z14 all count and each |v_g - ref|/abserr_g <= 0.01; fails if a counted one exceeds;
else undecided.
**As ruled:** every sample with at least FCC's two usable readings is scored; ref is the FCC res-2
reading; its readings, their spread and spread/tol are stated; class "two implementations" (a QAWO
and an FCC reading) or "FCC only". V3 as ruled fails if any scored reference has |v_g - ref| >
abserr_g, else holds; the same restricted to two-implementation references beside it; FCC-only
violations, and FCC r1 vs r2 differences over tol, are findings.
**V1's table:** per call, the g capture's readings; tol = min(e23, e24)/100 with e23, e24 the
calls' own abserr (the store holds (1 + z_response) times them; the ratios are the same); which of
R23 and R24 is nearer the reference and whether that code's abserr covers its distance.
**V8's table:** per call, its readings, 06h's and g's value, abserr and converged, both ratios.
**The correction:** on every scored reference whose g record has a Clenshaw-Curtis leaf
(``leaves.cc`` > 0; ``has_cc_leaf`` beside it), |v_g - sumE - ref| against |v_g - ref| and abserr_g.
**z00 across hosts:** its four rerun readings' value_hex against the laptop's (addendum 5 item 4).

Writes out/scores_after_stage2/ (scores.json, report.md). Reads only. Usage: python -B score_refs.py
"""

import json
import sys
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

mp.mp.dps = 50
NEW_TOOL_SHA = "5ed56fc24e46eb03e49b33408bc73d73fbfae24c38ec26cf5fd6537b43c4d058"
KEYS = [(i, r) for i in RC.IMPLS for r in RC.RESS]
RERUN_ONLY = {f"w{i:02d}" for i in range(12)} | {"z00", "z12"}
LAPTOP_Z00 = {
    ("fcc", 1): "0x1.0c5f18f4cb423p-86",
    ("fcc", 2): "0x1.0c5f18f86f178p-86",
    ("qawo", 1): "0x1.0c5f18f86f93bp-86",
    ("qawo", 2): "0x1.0c5f18f86f3e1p-86",
}
OUT = RC.ROOT / "out" / "scores_after_stage2"
INPUTS = []


def rd(p):
    p = Path(p)
    if p.exists():
        INPUTS.append(p)
        return json.loads(p.read_text())
    return None


def hx(x):
    return float.fromhex(x["hex"] if isinstance(x, dict) else x)


def M(x):
    return mp.mpf(x)


def fh(x):
    return float(x).hex()


def usable(d):
    """(usable, clause) for one reading."""
    if d["route"] != "B":
        return True, "Route A"
    tm = d.get("tmap") or {}
    if tm.get("all_monotone") is True:
        return True, "(i) all_monotone"
    nonmono = sum(1 for s in d.get("segments") or [] if not s.get("monotone"))
    if d.get("tool_sha256") == NEW_TOOL_SHA and tm.get("direct_segments") == nonmono:
        return True, f"(ii) ff3e8394, direct_segments {nonmono}"
    return False, "not usable"


def readings_of(e):
    sid = e["sid"]
    out = {}
    if e.get("in_06i"):
        for impl, res in KEYS:
            for tag in (
                f"s{e['sample_no']:02d}-{impl}-r{res}",
                f"s{e['sample_no']:02d}-{impl}-r{res}-bound",
            ):
                d = rd(RC.I06_READINGS / f"{tag}.json")
                if d:
                    d.setdefault("route", e["i06"]["route"])
                    out[(impl, res)] = d
                    break
        return out, "06i's readings"
    for impl, res in KEYS:
        new = rd(RC.ROOT / "out" / "readings" / f"{sid}-{impl}-r{res}-rerun.json")
        old = (
            None
            if sid in RERUN_ONLY
            else rd(RC.ROOT / "out" / "readings" / f"{sid}-{impl}-r{res}.json")
        )
        d = new or old
        if d:
            out[(impl, res)] = d
    return out, (
        "rerun batch only" if sid in RERUN_ONLY else "part 2 (+ rerun where present)"
    )


def assess(reads, tol):
    use = {}
    clauses = {}
    for k, d in reads.items():
        ok, cl = usable(d)
        clauses[f"{k[0]}_r{k[1]}"] = cl
        if ok:
            use[k] = d
    vals = {k: M(d["value_30"]) for k, d in use.items()}
    r = {
        "readings": {
            f"{k[0]}_r{k[1]}": {
                "value_30": d["value_30"],
                "value_hex": d["value_hex"],
                "wall_s": d.get("wall_s"),
                "clause": clauses[f"{k[0]}_r{k[1]}"],
                "rerun": bool(d.get("rerun")),
            }
            for k, d in use.items()
        },
        "unusable": [k for k, c in clauses.items() if c == "not usable"],
        "tol": float(tol),
    }
    have_fcc = ("fcc", 1) in vals and ("fcc", 2) in vals
    r["scored"] = have_fcc
    r["n_readings"] = len(vals)
    r["methods"] = sorted(f"{k[0]} r{k[1]}" for k in vals)
    r["class"] = None
    if have_fcc:
        r["class"] = (
            "two implementations" if any(k[0] == "qawo" for k in vals) else "FCC only"
        )
        spread = max(vals.values()) - min(vals.values())
        r["spread"] = float(spread)
        r["spread_over_tol"] = float(spread / tol)
        r["fcc_r1_r2_over_tol"] = float(abs(vals[("fcc", 1)] - vals[("fcc", 2)]) / tol)
        r["ref_30"] = use[("fcc", 2)]["value_30"]
        r["ref_hex"] = use[("fcc", 2)]["value_hex"]
        r["ref"] = vals[("fcc", 2)]
    all_i = len(use) == 4 and all(
        clauses[f"{k[0]}_r{k[1]}"] in ("(i) all_monotone", "Route A") for k in use
    )
    r["counts_as_written"] = bool(all_i and have_fcc and r["spread"] <= tol)
    return r


def ratio(v, ref, e):
    return float(abs(v - ref) / e) if e else float("inf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    INPUTS.append(RC.ROOT / "out" / "sample" / "selection.json")
    sp720 = RC.scan06a_sp()
    rows = []
    for e in sel:
        g, h = e["g_record"], e["06h_record"]
        vg, eg, vh, eh = (
            M(hx(g["value"])),
            M(hx(g["abserr"])),
            M(hx(h["value"])),
            M(hx(h["abserr"])),
        )
        tol = min(eg, eh) / 100
        reads, source = readings_of(e)
        a = assess(reads, tol)
        row = {
            "sid": e["sid"],
            "source": e["source"],
            "kind": e["kind"],
            "unit": e["unit"],
            "index": e["index"],
            "rtol": e["rtol"],
            "label": e["notify_label"],
            "reading_source": source,
            **a,
            "g": {
                "value": g["value"]["hex"],
                "abserr": g["abserr"]["hex"],
                "converged": g["converged"],
            },
            "06h": {
                "value": h["value"]["hex"],
                "abserr": h["abserr"]["hex"],
                "converged": h["converged"],
            },
            "has_cc_leaf": g.get("has_cc_leaf"),
            "leaves_cc": (g.get("leaves") or {}).get("cc"),
            "sumE": (g.get("node_error_signed_sum") or {}).get("hex"),
            "stationary_point_row_720": RC.rowkey_of(e["unit"]) in sp720,
        }
        if e.get("in_06i"):
            row["i06_counted_06i_rule"] = e["i06"]["counts_06i"]
        if a["scored"]:
            ref = a.pop("ref")
            row.pop("ref", None)
            row["r_g"] = ratio(vg, ref, eg)
            row["r_06h"] = ratio(vh, ref, eh)
            row["violates"] = bool(abs(vg - ref) > eg)
            row["violates_06h"] = bool(abs(vh - ref) > eh)
            if row["leaves_cc"] and row["sumE"]:
                sE = M(float.fromhex(row["sumE"]))
                row["correction"] = {
                    "d_g": float(abs(vg - ref)),
                    "d_g_minus_sumE": float(abs(vg - sE - ref)),
                    "ratio": (
                        float(abs(vg - sE - ref) / abs(vg - ref)) if vg != ref else None
                    ),
                    "d_g_over_eg": float(abs(vg - ref) / eg),
                    "d_corr_over_eg": float(abs(vg - sE - ref) / eg),
                }
        row.pop("ref", None)
        rows.append(row)
    v3 = [r for r in rows if r["source"] in ("z3", "draw")]
    v8 = [r for r in rows if r["source"] == "v8"]
    assert len(v3) == 90 and len(v8) == 12
    # V3 as written
    cw = [r for r in v3 if r["counts_as_written"]]
    viol_w = [r for r in cw if r["violates"]]
    v3_written = "fails" if viol_w else ("undecided" if len(cw) < 45 else "holds")
    # V3 as ruled
    sc = [r for r in v3 if r["scored"]]
    viol_r = [r for r in sc if r["violates"]]
    two = [r for r in sc if r["class"] == "two implementations"]
    viol_two = [r for r in two if r["violates"]]
    v3_ruled = "fails" if viol_r else "holds"
    v3_ruled_two = "fails" if viol_two else "holds"
    findings = [r for r in sc if r["class"] == "FCC only" and r["violates"]]
    fcc_dis = [r for r in sc if r["fcc_r1_r2_over_tol"] > 1]
    out_tol = [r for r in sc if r["spread_over_tol"] > 1]
    # V4
    v4 = []
    for sid in ("z02", "z08", "z09", "z14"):
        r = next(x for x in rows if x["sid"] == sid)
        v4.append(
            {
                "sid": sid,
                "counts_as_written": r["counts_as_written"],
                "scored": r["scored"],
                "r_g": r.get("r_g"),
                "r_06h": r.get("r_06h"),
                "06e_breakdown": {
                    "z02": 0.0012,
                    "z14": 0.0012,
                    "z08": 0.00046,
                    "z09": 0.0015,
                }[sid],
            }
        )
    v4_written = (
        "fails"
        if any(x["counts_as_written"] and x["r_g"] > 0.01 for x in v4)
        else ("holds" if all(x["counts_as_written"] for x in v4) else "undecided")
    )
    v4_ruled = (
        "fails"
        if any(x["scored"] and x["r_g"] > 0.01 for x in v4)
        else ("holds" if all(x["scored"] for x in v4) else "undecided")
    )
    # V1
    v1cap = json.loads((RC.ROOT / "v1" / "out" / "v1_capture.json").read_text())
    INPUTS.append(RC.ROOT / "v1" / "out" / "v1_capture.json")
    v1 = []
    for c in v1cap:
        e23, e24 = M(hx(c["p"]["call_abserr"])), M(hx(c["g"]["call_abserr"]))
        v23, v24 = M(hx(c["p"]["call_value"])), M(hx(c["g"]["call_value"]))
        tol = min(e23, e24) / 100
        reads, _ = readings_of({"sid": c["wid"]})
        a = assess(reads, tol)
        row = {
            "wid": c["wid"],
            "k": c["k"],
            "q": c["q"],
            "r": c["r"],
            "z_response": c["z_response"],
            **a,
            "R23": {
                "value": c["p"]["call_value"],
                "abserr": c["p"]["call_abserr"],
                "store_value": c["p"]["store_value"],
                "store_abserr": c["p"]["store_abserr"],
                "converged": c["p"]["converged"],
            },
            "R24": {
                "value": c["g"]["call_value"],
                "abserr": c["g"]["call_abserr"],
                "store_value": c["g"]["store_value"],
                "store_abserr": c["g"]["store_abserr"],
                "converged": c["g"]["converged"],
            },
        }
        if a["scored"]:
            ref = a["ref"]
            d23, d24 = abs(v23 - ref), abs(v24 - ref)
            near = "R23" if d23 < d24 else ("R24" if d24 < d23 else "tie")
            per = {k: mp.mpf(v["value_30"]) for k, v in a["readings"].items()}
            row["v24_vs_each_over_e24"] = {
                k: float(abs(v24 - x) / e24) for k, x in per.items()
            }
            row["v23_vs_each_over_e23"] = {
                k: float(abs(v23 - x) / e23) for k, x in per.items()
            }
            row["nearer_vs_every_reading"] = (
                "R24"
                if all(abs(v24 - x) < abs(v23 - x) for x in per.values())
                else (
                    "R23"
                    if all(abs(v23 - x) < abs(v24 - x) for x in per.values())
                    else "mixed"
                )
            )
            row.update(
                d23=float(d23),
                d24=float(d24),
                r23=float(d23 / e23),
                r24=float(d24 / e24),
                nearer=near,
                nearer_covered=bool((d23 <= e23) if near == "R23" else (d24 <= e24)),
                r23_covered=bool(d23 <= e23),
                r24_covered=bool(d24 <= e24),
            )
        row.pop("ref", None)
        v1.append(row)
    # z00 across hosts
    z00 = []
    for k in KEYS:
        d = json.loads(
            (
                RC.ROOT / "out" / "readings" / f"z00-{k[0]}-r{k[1]}-rerun.json"
            ).read_text()
        )
        z00.append(
            {
                "reading": f"{k[0]} r{k[1]}",
                "mac_studio": d["value_hex"],
                "laptop": LAPTOP_Z00[k],
                "identical": d["value_hex"] == LAPTOP_Z00[k],
                "mac_wall_s": d["wall_s"],
                "difference": float(
                    abs(M(d["value_30"]) - M(float.fromhex(LAPTOP_Z00[k])))
                ),
            }
        )
    hdr = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the references, scored after stage 2 of the rerun batch",
        sorted(set(INPUTS)),
    )
    doc = {
        "header": hdr,
        "rule": __doc__,
        "V3": {
            "as_written": {
                "verdict": v3_written,
                "n_counted": len(cw),
                "violations": [r["sid"] for r in viol_w],
            },
            "as_ruled": {
                "verdict": v3_ruled,
                "n_scored": len(sc),
                "violations": [r["sid"] for r in viol_r],
                "two_implementations": {
                    "verdict": v3_ruled_two,
                    "n": len(two),
                    "violations": [r["sid"] for r in viol_two],
                },
                "n_fcc_only": len(sc) - len(two),
                "fcc_only_violations": [r["sid"] for r in findings],
                "fcc_r1_r2_over_tol": [r["sid"] for r in fcc_dis],
                "spread_over_tol": [r["sid"] for r in out_tol],
            },
            "not_scored": [r["sid"] for r in v3 if not r["scored"]],
        },
        "V4": {"as_written": v4_written, "as_ruled": v4_ruled, "rows": v4},
        "z00_across_hosts": z00,
        "rows": rows,
        "V1": v1,
    }
    (OUT / "scores.json").write_text(json.dumps(doc, indent=1, default=str))
    # the report
    L = []
    w = L.append
    w("# The references, scored after stage 2 of the rerun batch\n")
    w("\n".join(hdr) + "\n")
    w("## V3 — honest where referenced\n")
    w(
        f"- **As written: {v3_written}.** Counted {len(cw)} of 90 (four usable clause-(i) readings within tol); violations {len(viol_w)}: {[r['sid'] for r in viol_w]}."
    )
    w(
        f"- **As the ruling scores it: {v3_ruled}.** Scored {len(sc)} of 90 ({len(two)} two implementations, {len(sc) - len(two)} FCC only); violations {len(viol_r)}: {[r['sid'] for r in viol_r]}. Two-implementation references only: **{v3_ruled_two}** ({len(viol_two)} violations)."
    )
    w(
        f"- Not scored: {[r['sid'] for r in v3 if not r['scored']]}. Findings: FCC-only violations {[r['sid'] for r in findings]}; FCC r1 vs r2 over tol {[r['sid'] for r in fcc_dis]}; spread over tol {[r['sid'] for r in out_tol]}.\n"
    )
    w(
        "| sid | kind | class | readings | spread/tol | counts as written | \\|v_g−ref\\|/e_g | \\|v_06h−ref\\|/e_06h | clauses |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for r in v3:
        if not r["scored"]:
            w(
                f"| {r['sid']} | {r['kind']} | not scored | {r['methods']} | | no | | | |"
            )
            continue
        cl = sorted({x["clause"] for x in r["readings"].values()})
        w(
            f"| {r['sid']}{' (720)' if r['stationary_point_row_720'] else ''} | {r['kind']} | {r['class']} | {', '.join(r['methods'])} | {r['spread_over_tol']:.3g} | {'yes' if r['counts_as_written'] else 'no'} | "
            f"{r['r_g']:.4g}{' **VIOLATES**' if r['violates'] else ''} | {r['r_06h']:.4g} | {'; '.join(cl)} |"
        )
    w("\n### Every V3 violation (as ruled), in float.hex\n")
    for r in viol_r:
        w(
            f"- **{r['sid']}** ({r['class']}, {r['unit']} #{r['index']}): v_g {r['g']['value']}, ref {r['ref_hex']} ({r['ref_30']}), abserr_g {r['g']['abserr']}; "
            f"|v_g−ref|/e_g {r['r_g']:.6g}; 06h: v {r['06h']['value']}, abserr {r['06h']['abserr']}, ratio {r['r_06h']:.6g}; counts as written: {r['counts_as_written']}"
        )
    worst = max(sc, key=lambda r: r["r_g"])
    w(
        f"\nWorst |v_g−ref|/e_g over the scored: {worst['sid']} {worst['r_g']:.6g} (06h's ratio beside it {worst['r_06h']:.6g}).\n"
    )
    w("## V4 — the trusted leaf's value\n")
    w(f"- **As written: {v4_written}. As ruled: {v4_ruled}.**\n")
    w(
        "| sample | counts as written | \\|v_g−ref\\|/e_g | 06h ratio | 06e breakdown (with the quarters) |\n|---|---|---|---|---|"
    )
    for x in v4:
        w(
            f"| {x['sid']} | {x['counts_as_written']} | {x['r_g']:.6g} | {x['r_06h']:.6g} | {x['06e_breakdown']} |"
        )
    w(
        "\n## V1's 12 group calls (addendum 1; the rerun batch's readings at g against R24)\n"
    )
    w(
        "| wid | k, z_r | class | readings | spread/tol | R23 v / e | R24 v / e | ref (FCC r2) | \\|v23−ref\\|/e23 | \\|v24−ref\\|/e24 | nearer | its e covers? |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in v1:
        if not r["scored"]:
            w(f"| {r['wid']} | | not scored | {r['methods']} | | | | | | | | |")
            continue
        w(
            f"| {r['wid']} | {r['k']:.3g}, {r['z_response']:.6g} | {r['class']} | {', '.join(r['methods'])} | {r['spread_over_tol']:.3g} | "
            f"{r['R23']['value']} / {r['R23']['abserr']} | {r['R24']['value']} / {r['R24']['abserr']} | {r['ref_hex']} | {r['r23']:.4g} | {r['r24']:.4g} | "
            f"{r['nearer']} | {'yes' if r['nearer_covered'] else 'NO'} |"
        )
    w(
        "\n**Each reading against each code** (|v - reading|/e, each code's own e; FCC r2 is the reference by rule):\n"
    )
    w(
        "| wid | R24 vs fcc r1 / fcc r2 / qawo r1 / qawo r2 | R23 vs fcc r1 / fcc r2 / qawo r1 / qawo r2 | nearer against every reading |"
    )
    w("|---|---|---|---|")
    ks = ["fcc_r1", "fcc_r2", "qawo_r1", "qawo_r2"]
    for r in v1:
        if r["scored"]:
            w(
                f"| {r['wid']} | {' / '.join(format(r['v24_vs_each_over_e24'].get(k, float('nan')), '.3g') for k in ks)} | "
                f"{' / '.join(format(r['v23_vs_each_over_e23'].get(k, float('nan')), '.3g') for k in ks)} | {r['nearer_vs_every_reading']} |"
            )
    w(
        "\nClauses: every V1 reading meets clause (ii) (ff3e8394, direct_segments = its non-monotone count): "
        + str(sorted({x["clause"] for r in v1 for x in r["readings"].values()}))
        + ".\n"
    )
    w("## The V8 sample (addendum 2)\n")
    w(
        "| vid | unit #i | class | readings | spread/tol | g v / e / conv | 06h v / e / conv | ref | \\|v_g−ref\\|/e_g | \\|v_06h−ref\\|/e_06h |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in v8:
        if not r["scored"]:
            w(
                f"| {r['sid']} | {r['unit']} #{r['index']} | not scored | {r['methods']} | | | | | | |"
            )
            continue
        w(
            f"| {r['sid']} | {r['unit']} #{r['index']} | {r['class']} | {', '.join(r['methods'])} | {r['spread_over_tol']:.3g} | "
            f"{r['g']['value']} / {r['g']['abserr']} / {r['g']['converged']} | {r['06h']['value']} / {r['06h']['abserr']} / {r['06h']['converged']} | "
            f"{r['ref_hex']} | {r['r_g']:.4g}{' **>1**' if r['violates'] else ''} | {r['r_06h']:.4g}{' **>1**' if r['violates_06h'] else ''} |"
        )
    w(
        "\n## The correction −ΣE (descriptive; scored V3 references with a Clenshaw–Curtis leaf)\n"
    )
    w(
        "| sid | class | ΣE (hex) | \\|v_g−ref\\| | \\|v_g−ΣE−ref\\| | corrected / uncorrected | \\|v_g−ref\\|/e_g | \\|v_g−ΣE−ref\\|/e_g | has_cc_leaf |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    corr = [r for r in v3 if r.get("correction")]
    for r in corr:
        c = r["correction"]
        w(
            f"| {r['sid']} | {r['class']} | {r['sumE']} | {c['d_g']:.4g} | {c['d_g_minus_sumE']:.4g} | {c['ratio'] if c['ratio'] is None else format(c['ratio'], '.4g')} | "
            f"{c['d_g_over_eg']:.4g} | {c['d_corr_over_eg']:.4g} | {r['has_cc_leaf']} |"
        )
    better = sum(
        1
        for r in corr
        if r["correction"]["ratio"] is not None and r["correction"]["ratio"] < 1
    )
    w(
        f"\n{len(corr)} references with a CC leaf; the correction brings v nearer the reference on {better}.\n"
    )
    w("## z00 across two hosts (addendum 5 item 4)\n")
    w(
        "| reading | Mac Studio | laptop | identical | |difference| | Mac wall s |\n|---|---|---|---|---|---|"
    )
    for x in z00:
        w(
            f"| {x['reading']} | {x['mac_studio']} | {x['laptop']} | {x['identical']} | {x['difference']:.3g} | {x['mac_wall_s']:.0f} |"
        )
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
