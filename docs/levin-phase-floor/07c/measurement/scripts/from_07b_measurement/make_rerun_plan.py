"""levin-phase-floor prompt 06g section 5, the references: the rerun batch's plan, per addenda 4 and 5
(the user's rulings of 2026-11-02). New in 06g.

**Every reading uses the tool at ff3e8394** (worktree TOOL_TREE), one at a time with six workers,
with the harness each sample's captures need on PYTHONPATH after the tool's tree: the g census
harness (references/scripts/) for z, r and v samples, V1's rehearsal-store harness (v1/scripts/) for
w samples. Stages, in this order:
1. ``reread``: six clean readings of parts 1 and 2 re-read under the new tool and the out-of-order
   progress wrapper, each of which must reproduce its old ``value_hex``, ``value_30`` and
   ``imag_30`` bit for bit (REREADS, with why each is representative); one that does not is a stop;
2. ``stationary``: all four readings of each of the 14 stationary-point samples, w00..w11, z00, z12
   (addendum 5, item 2), replacing their old readings;
3. ``stopped``: the stopped QAWO readings that run (addenda 4 section 2, 5 item 1), shortest
   estimate first. Stage 3 runs after stage 2 so that its long readings hold nothing else up.
Each candidate's estimate is from its phase span, the sum of |t_span| over its FCC res-1 reading's
segments, at 0.06-0.10 us/rad for QAWO r1 and 4x that (0.24-0.40) for r2, both ends given; a
candidate runs unless the addenda skip it, its estimate is a day or more, its existing readings are
not all monotone, or its FCC and QAWO r1 already agree as tightly as r52's (spread 4.4e-5 of the
tolerance).

Writes out/rerun_plan.json and out/rerun_plan.md (refusing to overwrite). Reads only.
Usage: python -B make_rerun_plan.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402

import mpmath as mp  # noqa: E402

mp.mp.dps = 50
TOOL_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-ff3e839")
TOOL_SHA = "ff3e8394fa3cf81a3d27087708f6fc8acbc3fb63"
V1 = RC.ROOT / "v1"
R1_LO, R1_HI = 0.06e-6, 0.10e-6
R2_FACTOR = 4.0
DAY = 86400.0
REREADS = [  # (sid, impl, res, old reading, why)
    (
        "z13",
        "qawo",
        1,
        "readings_06i13",
        "Route B QAWO r1, part 1 (06i's sample 13, a G+q-r census group, 750 segments)",
    ),
    (
        "r55",
        "qawo",
        2,
        "readings",
        "Route B QAWO r2, part 2 (a drawn G-r group, 298 segments): the r2 QAWO path",
    ),
    (
        "z13",
        "fcc",
        1,
        "readings_06i13",
        "Route B FCC r1, part 1 (the same call as the QAWO re-read)",
    ),
    (
        "r56",
        "fcc",
        1,
        "readings",
        "Route B FCC r1, part 2 (a drawn G-q+r group, z_response 1.75e10, 204 segments)",
    ),
    (
        "z25",
        "qawo",
        2,
        "readings_06i13",
        "Route A QAWO r2, part 1 (06i's sample 25, a Y4 analytic block)",
    ),
    ("r09", "fcc", 1, "readings", "Route A FCC r1, part 2 (a drawn Y2 analytic block)"),
]
STATIONARY = [f"w{i:02d}" for i in range(12)] + ["z00", "z12"]
# the laptop's z00 under the new tool (addendum 5 item 4), and the extra over the old tool's FCC walls
LAPTOP_Z00 = {
    ("fcc", 1): 160.0,
    ("fcc", 2): 735.0,
    ("qawo", 1): 309.0,
    ("qawo", 2): 1023.0,
}
STOPPED = [  # (sid, impl, res, decision rule from the addenda)
    ("r47", "qawo", 2, "candidate"),
    ("r26", "qawo", 1, "candidate"),
    ("r26", "qawo", 2, "candidate"),
    ("r24", "qawo", 1, "candidate if under a day"),
    ("r53", "qawo", 1, "candidate (stopped by part 2 at 10:16Z)"),
    ("r53", "qawo", 2, "candidate (part 2 stopped it)"),
    (
        "r52",
        "qawo",
        2,
        "skip: addenda 4 and 5 (three readings agree to 4.4e-5 of the tolerance)",
    ),
    (
        "r24",
        "qawo",
        2,
        "skip: addenda 4 and 5 ('about 2 days' from part 2's ordered progress)",
    ),
]


def rd(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def span(sid):
    d = rd(RC.ROOT / "out" / "readings" / f"{sid}-fcc-r1.json")
    return sum(abs(s["t_span"]) for s in d["segments"]), d


def main():
    sel = {
        e["sid"]: e
        for e in json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    }
    v1cap = {
        e["wid"]: e for e in json.loads((V1 / "out" / "v1_capture.json").read_text())
    }
    for p in (RC.ROOT / "out" / "rerun_plan.json", RC.ROOT / "out" / "rerun_plan.md"):
        if p.exists():
            raise SystemExit(f"{p} exists")
    H = RC.H

    def entry(stage, sid, impl, res, lo, hi, **kw):
        if sid.startswith("w"):
            g = v1cap[sid]["g"]
            pkl, harness, unit, index, rtol = (
                V1 / "state" / "inputs" / f"{sid}-g.pkl",
                V1 / "scripts",
                f"R24 {g['notify_label'][:40]}",
                g["index"],
                "1e-7",
            )
        else:
            e = sel[sid]
            pkl, harness, unit, index, rtol = (
                RC.ROOT / "state" / "inputs" / f"{sid}.pkl",
                HERE,
                e["unit"],
                e["index"],
                e["rtol"],
            )
        return {
            "stage": stage,
            "sid": sid,
            "impl": impl,
            "res": res,
            "pickle": str(pkl),
            "harness": str(harness),
            "pythonpath": f"{TOOL_TREE}:{harness}",
            "unit": unit,
            "index": index,
            "rtol": rtol,
            "estimate_lo_s": lo,
            "estimate_hi_s": hi,
            **kw,
        }

    plan, md = [], []
    md.append("# The rerun batch's plan (addenda 4 and 5)\n")
    md.append(
        f"Tool: `{TOOL_TREE}/tools/filon_reference.py` at `{TOOL_SHA[:8]}`. One reading at a time, six workers; "
        "progress counted out of order (`one_reading.py --progress-order unordered`). Written by "
        "`scripts/make_rerun_plan.py` (its docstring is the rule), before any of these readings.\n"
    )
    # 1. re-reads
    md.append("## 1. Bit-for-bit re-reads (each a stop if it does not reproduce)\n")
    md.append(
        "| Reading | Old reading | Old wall s | Why it is representative |\n|---|---|---|---|"
    )
    for sid, impl, res, where, why in REREADS:
        old = RC.ROOT / "out" / where / f"{sid}-{impl}-r{res}.json"
        d = rd(old)
        if d.get("tmap") is not None and not d["tmap"]["all_monotone"]:
            raise SystemExit(f"{sid}: re-read candidate not monotone")
        plan.append(
            entry(
                "reread",
                sid,
                impl,
                res,
                d["wall_s"],
                2 * d["wall_s"] + 10,
                compare_with=str(old),
            )
        )
        md.append(
            f"| {sid} {impl} r{res} | `{where}/{sid}-{impl}-r{res}.json` | {d['wall_s']:.1f} | {why} |"
        )
    # 2. the 14 stationary-point samples
    md.append(
        "\n## 2. The 14 stationary-point samples, all four readings (replacing the old ones)\n"
    )
    md.append(
        "Estimates: low end the old tool's wall for that reading (its siblings' mean where it ended in error); "
        "high end that plus the laptop's z00 excess under the new tool (FCC r1 +129 s, r2 +615 s; QAWO r1 309 s, "
        "r2 1,023 s); z00 and z12 at the laptop's walls (low) and 1.5x (high).\n"
    )
    md.append(
        "| Sample | Phase span (rad) | Old readings' all_monotone | Readings | Estimate s (low-high) |\n|---|---|---|---|---|"
    )
    wmean = {}
    for impl in RC.IMPLS:
        for res in RC.RESS:
            ws = [
                rd(RC.ROOT / "out" / "readings" / f"w{i:02d}-{impl}-r{res}.json")
                for i in range(12)
            ]
            ws = [w["wall_s"] for w in ws if w]
            wmean[(impl, res)] = sum(ws) / len(ws)
    for sid in STATIONARY:
        sp, d = span(sid)
        lo_s = hi_s = 0.0
        for impl in RC.IMPLS:
            for res in RC.RESS:
                old = rd(RC.ROOT / "out" / "readings" / f"{sid}-{impl}-r{res}.json")
                if sid.startswith("z"):
                    lo, hi = LAPTOP_Z00[(impl, res)], 1.5 * LAPTOP_Z00[(impl, res)]
                else:
                    lo = old["wall_s"] if old else wmean[(impl, res)]
                    extra = LAPTOP_Z00[(impl, res)] - (
                        {("fcc", 1): 31.0, ("fcc", 2): 120.0}.get((impl, res), 0.0)
                    )
                    hi = lo + extra
                lo_s += lo
                hi_s += hi
                plan.append(
                    entry(
                        "stationary",
                        sid,
                        impl,
                        res,
                        lo,
                        hi,
                        replaces=(
                            f"out/readings/{sid}-{impl}-r{res}.json" if old else None
                        ),
                    )
                )
        md.append(
            f"| {sid} | {sp:.3g} | {d['tmap']['all_monotone']} | 4 | {lo_s:.0f}-{hi_s:.0f} |"
        )
    # 3. the stopped QAWO readings
    md.append("\n## 3. The stopped QAWO readings\n")
    md.append(
        "Estimate: phase span x 0.06-0.10 us/rad (r1), x 0.24-0.40 us/rad (r2).\n"
    )
    md.append(
        "| Reading | Phase span (rad) | Estimate s (low-high) | all_monotone (existing readings) | Existing spread / tolerance | Runs? |\n|---|---|---|---|---|---|"
    )
    runs = []
    for sid, impl, res, rule in STOPPED:
        sp, _ = span(sid)
        f = 1.0 if res == 1 else R2_FACTOR
        lo, hi = sp * R1_LO * f, sp * R1_HI * f
        mono = []
        vals = {}
        for k in ("qawo-r1", "qawo-r2", "fcc-r1", "fcc-r2"):
            d = rd(RC.ROOT / "out" / "readings" / f"{sid}-{k}.json")
            if d:
                mono.append(d["tmap"]["all_monotone"])
                vals[k] = mp.mpf(d["value_30"])
        e = sel[sid]
        tol = min(H(e["g_record"]["abserr"]), H(e["06h_record"]["abserr"])) / 100
        spread = float((max(vals.values()) - min(vals.values())) / tol)
        allmono = all(mono)
        if rule.startswith("skip"):
            run, why = False, rule
        elif lo >= DAY:
            run, why = False, "skip: estimated at a day or more"
        elif not allmono:
            run, why = False, "skip: an existing reading is not all monotone"
        elif "qawo-r1" in vals and spread <= 4.4e-5:
            run, why = False, "skip: FCC and QAWO r1 already agree as tightly as r52's"
        else:
            run, why = True, "runs" + (
                f" (its high end, {hi:.0f} s, is under a day by {DAY - hi:.0f} s)"
                if hi > 0.9 * DAY
                else ""
            )
        md.append(
            f"| {sid} {impl} r{res} | {sp:.4g} | {lo:.0f}-{hi:.0f} ({lo / 3600:.1f}-{hi / 3600:.1f} h) | {allmono} | "
            f"{spread:.3g} ({', '.join(sorted(vals))}) | {'**runs**' if run else 'no'}: {why} |"
        )
        if run:
            runs.append(entry("stopped", sid, impl, res, lo, hi, phase_span=sp))
    runs.sort(key=lambda x: x["estimate_lo_s"])
    plan += runs
    for i, x in enumerate(plan):
        x["n"] = i
        x["tag"] = f"{x['sid']}-{x['impl']}-r{x['res']}"
    tot = {
        st: [
            sum(x["estimate_lo_s"] for x in plan if x["stage"] == st),
            sum(x["estimate_hi_s"] for x in plan if x["stage"] == st),
        ]
        for st in ("reread", "stationary", "stopped")
    }
    T = [sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values())]
    md.append(
        f"\nOrder of stage 3 (shortest low estimate first): "
        + ", ".join(x["tag"] for x in runs)
        + ".\n"
    )
    md.append("## The batch's projected total\n")
    md.append("| Stage | Readings | Estimate (low-high) |\n|---|---|---|")
    for st, (lo, hi) in tot.items():
        md.append(
            f"| {st} | {sum(1 for x in plan if x['stage'] == st)} | {lo:.0f}-{hi:.0f} s ({lo / 3600:.1f}-{hi / 3600:.1f} h) |"
        )
    md.append(
        f"| **total** | {len(plan)} | **{T[0]:.0f}-{T[1]:.0f} s ({T[0] / 3600:.1f}-{T[1] / 3600:.1f} h)** |"
    )
    md.append("\n## The backstop (addenda 4 and 5)\n")
    md.append(
        "No reading is stopped for passing 30 minutes or for a multiple of its estimate. From 30 minutes, every 5 "
        "minutes, its progress (chunks finished, counted out of order) and projected finish (start + elapsed x total / "
        "done; with no chunk back yet, its estimate's high end) go to the heartbeat and `out/rerun_monitor.log`. It is "
        "stopped only when its projected total is a day (86,400 s) or more, or when it has run 24 h."
    )
    doc = {
        "tool_tree": str(TOOL_TREE),
        "tool_sha": TOOL_SHA,
        "rule": __doc__,
        "n_readings": len(plan),
        "estimate_total_s": T,
        "by_stage": tot,
        "plan": plan,
    }
    (RC.ROOT / "out" / "rerun_plan.json").write_text(json.dumps(doc, indent=1))
    (RC.ROOT / "out" / "rerun_plan.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
