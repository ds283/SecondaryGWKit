"""Orchestrator's own re-score of V4 (levin-phase-floor 07 Part B, step 9 check). Read only.

Usage:  python -B v4_rescore.py OUT_JSON

Independent of the census agent's score_v4_07.py. It reads:
- the 24 calls' identities from census/state/v4_selection.json (fixed before the census);
- 07's value and abserr from the census checkpoint, by unit and call index;
- 06m's value and abserr from 06m's checkpoint, at the offset census/state/index_06m.json gives, the
  line accepted only if its SHA-256 matches and it names the same unit;
- each reading from census/out/v4/readings/<sid>-<impl>-r<res>.json, accepted only if: its tool SHA-256 is
  d37e633b...; its input pickle on disk has the SHA-256 the reading records; route B; and every
  non-monotone segment is `direct`. float(value_30) is checked to round to value_hex.
- z00, z12 and Z3's seven from 06g's scores_stage3/scores.json `rows` (committed, read in tree g), each
  reading's value_30, 06g's `unusable` left out, the reference 06g's FCC r2.

Rule (prompt section 6 V4, as ruled at stop 4): tol = min(abserr_06m, abserr_g)/100; readings agree iff
max - min <= tol; reference FCC r2; honest iff |v_g - ref| <= abserr_g. Exact Fraction arithmetic throughout:
census doubles from hex, readings from their 30-digit decimal strings.
"""

import hashlib
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
G = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7")
C = MAIN / "var/levin-phase-floor-07/census"
RUN07 = MAIN / "var/runs/levin-phase-floor-07-census-20261108T171137"
RUN06M = MAIN / "var/runs/levin-phase-floor-06m-census-20261106T102153"
INDEX = C / "state/index_06m.json"
SEL = C / "state/v4_selection.json"
READ = C / "out/v4/readings"
S06G = G / "docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json"
TOOL = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
CK07 = "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"
ZS = ["z00", "z12", "z07", "z16", "z18", "z21", "z24", "z28", "z29"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return Fraction(float.fromhex(h))


def fd(s):
    return Fraction(Decimal(s))


def call_of(data, index):
    m = [c for c in data["calls"] + data["other_calls"] if c["index"] == index]
    assert len(m) == 1, (data["unit"], index, len(m))
    return m[0]


def vh(c, key):
    v = c[key]
    return v["hex"] if isinstance(v, dict) else float(v).hex()


def score(readings, vg, eg, v6, e6):
    tol = min(eg, e6) / 100
    vals = list(readings.values())
    spread = max(vals) - min(vals)
    agree = spread <= tol
    ref = readings["fcc_r2"]
    rg = abs(vg - ref) / eg
    r6 = abs(v6 - ref) / e6
    return {
        "tol": float(tol),
        "spread_over_tol": float(spread / tol),
        "agree": agree,
        "ratio_g": float(rg),
        "ratio_06m": float(r6),
        "verdict": ("honest" if rg <= 1 else "NOT honest") if agree else "reported, not scored",
        "honest_06m": r6 <= 1,
    }


def main():
    out_json = sys.argv[1]
    assert sha(RUN07 / "checkpoint.jsonl") == CK07
    sel = json.loads(SEL.read_text())["calls"]
    s06g = {r["sid"]: r for r in json.loads(S06G.read_text())["rows"]}
    want = {c["unit"] for c in sel} | {s06g[z]["unit"] for z in ZS}
    d07 = {}
    for line in open(RUN07 / "checkpoint.jsonl"):
        d = json.loads(line)["data"]
        if d["unit"] in want:
            d07[d["unit"]] = d
    idx = json.loads(INDEX.read_text())["units"]
    d06 = {}
    with open(RUN06M / "checkpoint.jsonl", "rb") as fh:
        for u in want:
            e = idx[u]
            fh.seek(e["offset"])
            raw = fh.read(e["length"])
            assert hashlib.sha256(raw).hexdigest() == e["line_sha256"], u
            dd = json.loads(raw)["data"]
            assert dd["unit"] == u
            d06[u] = dd
    rows, problems = [], []
    for c in sel:
        sid, u, i = c["sid"], c["unit"], c["index"]
        a, b = call_of(d07[u], i), call_of(d06[u], i)
        assert a["notify_label"] == b["notify_label"] and a["group"] == c["group"], sid
        assert vh(b, "value") == c["value_06m"] and vh(b, "abserr") == c["abserr_06m"], sid
        rd, chk = {}, {}
        for impl in ("qawo", "fcc"):
            for res in (1, 2):
                p = READ / f"{sid}-{impl}-r{res}.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                ok = {
                    "tool": r["tool_sha256"] == TOOL,
                    "input": sha(r["pickle"]) == r["pickle_sha256"],
                    "unit": r["unit"] == u and r["index"] == i,
                    "route": r["route"] == "B",
                    "nonmono_direct": all(s["direct"] for s in r["segments"] if not s["monotone"]),
                    "hex": float(r["value_30"]).hex() == r["value_hex"],
                }
                chk[f"{impl}_r{res}"] = ok
                if all(ok.values()):
                    rd[f"{impl}_r{res}"] = fd(r["value_30"])
                else:
                    problems.append((sid, impl, res, ok))
        basis = "four readings" if len(rd) == 4 else ("FCC r1 and r2" if set(rd) == {"fcc_r1", "fcc_r2"} else sorted(rd))
        s = score(rd, fx(vh(a, "value")), fx(vh(a, "abserr")), fx(vh(b, "value")), fx(vh(b, "abserr")))
        rows.append({"sid": sid, "unit": u, "index": i, "basis": basis, "converged_g": a["converged"],
                     "converged_06m": b["converged"], "checks": chk, **s})
    for z in ZS:
        r6 = s06g[z]
        u, i = r6["unit"], r6["index"]
        a, b = call_of(d07[u], i), call_of(d06[u], i)
        rd = {k: fd(v["value_30"]) for k, v in r6["readings"].items() if k not in r6["unusable"]}
        assert fd(r6["ref_30"]) == rd["fcc_r2"], z
        s = score(rd, fx(vh(a, "value")), fx(vh(a, "abserr")), fx(vh(b, "value")), fx(vh(b, "abserr")))
        rows.append({"sid": z, "unit": u, "index": i, "basis": sorted(rd), "unusable_06g": r6["unusable"],
                     "converged_g": a["converged"], "converged_06m": b["converged"],
                     "bitequal_07_06m": vh(a, "value") == vh(b, "value") and vh(a, "abserr") == vh(b, "abserr"), **s})
    # compare with the agent's
    ag = json.loads((C / "out/v4/v4_scores.json").read_text())
    agr = {x["sid"]: x for x in ag["calls"] + ag["z"]}
    cmp = []
    for r in rows:
        x = agr[r["sid"]]
        cmp.append({"sid": r["sid"], "verdict_same": x["verdict"] == r["verdict"], "basis_same": x["basis"] == r["basis"] if r["sid"].startswith("v") else None,
                    "ratio_g_reldiff": abs(x["ratio_g"] - r["ratio_g"]) / r["ratio_g"] if r["ratio_g"] else abs(x["ratio_g"]),
                    "ratio_06m_reldiff": abs(x["ratio_06m"] - r["ratio_06m"]) / r["ratio_06m"] if r["ratio_06m"] else abs(x["ratio_06m"]),
                    "honest_06m_same": x["honest_06m"] == r["honest_06m"]})
    out = {"script_sha256": sha(__file__), "checkpoint07": CK07, "rows": rows, "reading_problems": problems,
           "vs_agent": cmp,
           "n_readings_checked": sum(len(r.get("checks", {})) for r in rows),
           "not_honest": [r["sid"] for r in rows if r["verdict"] == "NOT honest"],
           "not_scored": [r["sid"] for r in rows if r["verdict"] == "reported, not scored"],
           "honest_06m": [r["sid"] for r in rows if r["honest_06m"]]}
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print("# v4_rescore.py sha256", out["script_sha256"])
    print(f"{'sid':4} {'basis':16} {'conv g/06m':10} {'spread/tol':>10} {'ratio_g':>10} {'ratio_06m':>10}  verdict")
    for r in rows:
        print(f"{r['sid']:4} {str(r['basis'])[:16]:16} {str(r['converged_g'])[0]}/{str(r['converged_06m'])[0]:8} "
              f"{r['spread_over_tol']:10.3g} {r['ratio_g']:10.3g} {r['ratio_06m']:10.3g}  {r['verdict']}")
    print("readings checked:", out["n_readings_checked"], "problems:", problems)
    print("not honest:", out["not_honest"], " not scored:", out["not_scored"])
    print("06m honest on:", len(out["honest_06m"]), out["honest_06m"])
    print("vs agent: verdicts differ:", [c["sid"] for c in cmp if not c["verdict_same"]],
          " basis differ:", [c["sid"] for c in cmp if c["basis_same"] is False],
          " honest_06m differ:", [c["sid"] for c in cmp if not c["honest_06m_same"]],
          " max ratio reldiff g/06m: %.3g / %.3g" % (max(c["ratio_g_reldiff"] for c in cmp), max(c["ratio_06m_reldiff"] for c in cmp)))


if __name__ == "__main__":
    main()
