"""levin-phase-floor prompt 06g section 5, the references: the tool's readings on 06i's 13 against 06i's
readings (prompt 06g section 7: "a reading disagrees with 06i's on one of its 13" is a stop; the code
log's deviation 12). New in 06g.

The rule, fixed in the predictions file before any reading (from 06i's own rule between a reference
and another reading, its R1: "differs ... by at most the sum of the two scatters plus 1/100 of the
minimum abserr"), applied reading by reading: for each sample n of the 13 and each of the four
readings k (QAWO and FCC, each at resolutions 1 and 2), with values taken from the 30-digit strings
and compared in mpmath at 50 digits,

    D_k = |tool_k - 06i_k|  <=  s_tool + s_06i + m/100,

where s_tool and s_06i are the two sets' four-reading scatters (largest minus smallest) and m is
06i's minimum abserr for the sample, min(abserr_06c, abserr_06e) (06i's scores.json ``min_abserr``,
the m of 06i's counting rule). The tool's readings are on the captures at g (state/inputs/zNN.pkl),
whose groups and integrand compare13.py found equal to 06i's inputs. **A sample with any k outside
the rule is a disagreement, and a stop.**

Sample 8, as 06i read it under the user's ruling (*"I accept sample 8."*, 06i amendment 2): 06i's
section 8 check, |FCC_r2 - QAWO_r2| against the larger of the two implementations' own
res-1/res-2 scatters, exceeded 1 (5.1x; 3.5e-25, 1.3e-8 of m), and the reference was accepted because
its four readings agree within the counting rule; section 8's statistic is not a counting or an
agreement criterion. So here it is computed for every sample, for the tool's readings and for 06i's,
and reported, not used: sample 8 is treated like the other twelve.

Also reported, not used: whether each reading is bit-identical to 06i's (value_30 and value_hex), D_k
over the allowance and over m, and the readings' structure against 06i's (panels or segments,
tables, K_exact, C_exact, u_breaks).

**The command-line check:** on two samples (CLI_SAMPLES, one per route) the tool's own command line,
``python -m tools.filon_reference PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6 --out
JSON``, was run by run_refs.py; each of its four readings must equal one_reading.py's reading of the
same capture bit for bit (value_30, imag_30 and value_hex). A difference is a stop too: the per-reading
driver would not then be the tool's command line.

Writes out/tool_vs_06i.json; exits 1 on any disagreement. Reads only.

Usage: python -B compare06i.py READINGS_DIR CLI_DIR
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
KEYS = [(impl, res) for impl in RC.IMPLS for res in RC.RESS]
CLI_SAMPLES = (25, 13)  # Route A (06i's fastest, 5 s per reading) and Route B (37 s)
STRUCT_KEYS = ("n_panels", "n_segments", "n_tables", "K_exact", "C_exact", "u_breaks")


def i06_reading(n, impl, res):
    for tag in (f"s{n:02d}-{impl}-r{res}", f"s{n:02d}-{impl}-r{res}-bound"):
        p = RC.I06_READINGS / f"{tag}.json"
        if p.exists():
            return json.loads(p.read_text()), p
    raise SystemExit(f"06i's reading s{n:02d} {impl} r{res} not found")


def sec8(v):
    s_f = abs(v[("fcc", 1)] - v[("fcc", 2)])
    s_q = abs(v[("qawo", 1)] - v[("qawo", 2)])
    D = abs(v[("fcc", 2)] - v[("qawo", 2)])
    return float(D / max(s_f, s_q)) if max(s_f, s_q) > 0 else float("inf")


def main():
    rd, cli_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scores = {r["sample"]: r for r in json.loads(RC.I06_SCORES.read_text())["rows"]}
    ins = [RC.I06_SCORES]
    rows = []
    n_bad = 0
    for n in RC.I06_SAMPLES:
        m = mp.mpf(scores[n]["min_abserr"])
        tool, ref = {}, {}
        tdoc, idoc = {}, {}
        for impl, res in KEYS:
            p = rd / f"z{n:02d}-{impl}-r{res}.json"
            tdoc[(impl, res)] = json.loads(p.read_text())
            idoc[(impl, res)], ip = i06_reading(n, impl, res)
            ins += [p, ip]
            tool[(impl, res)] = mp.mpf(tdoc[(impl, res)]["value_30"])
            ref[(impl, res)] = mp.mpf(idoc[(impl, res)]["value_30"])
        s_t = max(tool.values()) - min(tool.values())
        s_i = max(ref.values()) - min(ref.values())
        allow = s_t + s_i + m / 100
        per = {}
        ok = True
        for k in KEYS:
            D = abs(tool[k] - ref[k])
            good = D <= allow
            ok &= good
            t, i = tdoc[k], idoc[k]
            per[f"{k[0]}_r{k[1]}"] = {
                "tool_value_30": t["value_30"],
                "06i_value_30": i["value_30"],
                "D": float(D),
                "D_over_allowance": float(D / allow),
                "D_over_m": float(D / m),
                "agrees": bool(good),
                "bit_identical_value_30": t["value_30"] == i["value_30"],
                "bit_identical_value_hex": t["value_hex"] == i["value_hex"],
                "tool_wall_s": t["wall_s"],
                "06i_wall_s": i["wall_s"],
                "structure": {
                    s: [t.get(s), i.get(s)] for s in STRUCT_KEYS if s in t or s in i
                },
                "tool_ray_initialized": t.get("ray_initialized"),
            }
        rows.append(
            {
                "sample": n,
                "route": scores[n]["route"],
                "m_06i": float(m),
                "scatter_tool": float(s_t),
                "scatter_06i": float(s_i),
                "allowance": float(allow),
                "agrees": bool(ok),
                "readings": per,
                "sec8_tool": sec8(tool),
                "sec8_06i": sec8(ref),
                "ref_06i_fcc_r2": idoc[("fcc", 2)]["value_30"],
                "ref_tool_fcc_r2": tdoc[("fcc", 2)]["value_30"],
            }
        )
        n_bad += not ok
    cli = []
    for n in CLI_SAMPLES:
        p = cli_dir / f"z{n:02d}.json"
        ins.append(p)
        doc = json.loads(p.read_text())
        same = []
        for r in doc["readings"]:
            t = json.loads((rd / f"z{n:02d}-{r['impl']}-r{r['res']}.json").read_text())
            same.append(
                {
                    "impl": r["impl"],
                    "res": r["res"],
                    "identical": all(
                        r[x] == t[x] for x in ("value_30", "imag_30", "value_hex")
                    ),
                    "cli_value_30": r["value_30"],
                    "one_reading_value_30": t["value_30"],
                }
            )
        good = len(same) == 4 and all(x["identical"] for x in same)
        n_bad += not good
        cli.append(
            {
                "sample": n,
                "identical": good,
                "readings": same,
                "cli_scatter": doc["scatter"],
                "cli_counts": doc["counts"],
            }
        )
    hdr = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: the Filon tool on 06i's 13, against 06i's readings",
        ins,
    )
    out = RC.ROOT / "out" / "tool_vs_06i.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    out.write_text(json.dumps({"header": hdr, "rows": rows, "cli": cli}, indent=1))
    for l in hdr:
        print(l)
    print(
        "sample route | worst D/allowance | worst D/m | bit-identical (of 4) | scatter tool / 06i (over m) | sec8 tool / 06i | agrees"
    )
    for r in rows:
        per = r["readings"].values()
        print(
            f"s{r['sample']:02d} {r['route']} | {max(x['D_over_allowance'] for x in per):.3g} | {max(x['D_over_m'] for x in per):.3g} | "
            f"{sum(x['bit_identical_value_30'] for x in per)} | {r['scatter_tool'] / r['m_06i']:.3g} / {r['scatter_06i'] / r['m_06i']:.3g} | "
            f"{r['sec8_tool']:.3g} / {r['sec8_06i']:.3g} | {'AGREES' if r['agrees'] else 'DISAGREES'}"
        )
    for c in cli:
        print(
            f"command line on z{c['sample']:02d}: {'IDENTICAL to one_reading.py' if c['identical'] else 'DIFFERS'} ({[x['identical'] for x in c['readings']]})"
        )
    print(
        f"{13 - sum(not r['agrees'] for r in rows)} of 13 agree with 06i; wrote {out} sha256 {RC.sha256(out)}"
    )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
