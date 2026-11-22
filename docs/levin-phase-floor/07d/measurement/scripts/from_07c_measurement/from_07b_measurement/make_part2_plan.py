"""levin-phase-floor prompt 06g section 5, the references: part 2's reading plan, per addendum 3 (the
user's ruling, 2026-11-01). New in 06g.

**What runs, in this order** (addendum 3, section 1):
1. V1's 12 (addendum 1; w00..w11, the captures at g against R24, ``v1/state/inputs/wNN-g.pkl``):
   QAWO and FCC at resolutions 1 and 2;
2. V8's 12 (addendum 2; v00..v11): the same four;
3. the rest of V3's sample, the 77 new samples in selection order (Z3's 17 not in 06i, then
   r00..r59): the same four, except on the 17 large-span Route A calls (``LARGE``), which get FCC at
   both resolutions here and no QAWO res 2;
4. last, QAWO res 1 on the ten of those 17 whose projection is under about 2 h (``LONG_R1``, in the
   addendum's order); none on the other seven.
Within a sample the tool's order (QAWO r1, r2, FCC r1, r2) is kept.

**Each reading's projection** (the model of out/cost_projection_part2.txt (4), from part 1's 52
readings on 06i's 13): Route A QAWO res 1 and res 2 at a fixed time per radian of phase span
|K|(eta_b - eta_a) (the 7 Route A samples' time over their total span), FCC res 1 and res 2 at the
7 samples' mean; Route B each reading at a fixed time per segment (the 6 Route B samples' time
over their segments); plus 1.6 s of process start-up per reading (part 1's measured mean). The set
of 17 is checked against the model: exactly the part-2 Route A calls whose QAWO res-2 projection
exceeds 30 minutes.

**The check and the backstop** (addendum 3, section 2; the driver applies them, run_refs.py):
- from max(30 min, its projection) on, the driver logs the reading's progress (chunks done out of
  total, from one_reading.py's progress file) and its projected finish (start + elapsed x total /
  done), every 5 minutes, to the run's heartbeat and out/reading_monitor.log;
- **the backstop:** a reading is stopped (its process group sent SIGTERM, then SIGKILL after 30 s),
  recorded and reported, only if (a) after 30 minutes its progress projects a total of 24 h or more
  (the user's "a day or more"), or (b) its elapsed time reaches its ceiling, max(6 x projection,
  projection + 1 h) (the user's example: "a hard ceiling per reading at several times its
  projection"), or (c) 24 h. Otherwise it finishes.

Writes out/part2_plan.json (refusing to overwrite) and prints the plan and its projected total.

Usage: python -B make_part2_plan.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

LARGE = (
    "z04",
    "z05",
    "z06",
    "z07",
    "z11",
    "z16",
    "z17",
    "z18",
    "z19",
    "z20",
    "z21",
    "z24",
    "z26",
    "z28",
    "z29",
    "r07",
    "r30",
)
LONG_R1 = ("z20", "r30", "z19", "z26", "z05", "z17", "z11", "z16", "z28", "r07")
OVERHEAD_S = 1.6
DAY_S = 86400.0


def ceiling(proj):
    return min(DAY_S, max(6.0 * proj, proj + 3600.0))


def main():
    routes = json.loads((RC.ROOT / "out" / "routes.json").read_text())
    tv = json.loads((RC.ROOT / "out" / "tool_vs_06i.json").read_text())
    sel = json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())
    v1cap = json.loads((RC.ROOT / "v1" / "out" / "v1_capture.json").read_text())
    v1grp = {
        e["wid"]: e
        for e in json.loads(
            (RC.ROOT / "v1" / "out" / "v1_compare_groups.json").read_text()
        )
    }
    out_p = RC.ROOT / "out" / "part2_plan.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    ins = [
        RC.ROOT / "out" / n
        for n in ("routes.json", "tool_vs_06i.json", "sample/selection.json")
    ]
    ins += [
        RC.ROOT / "v1" / "out" / "v1_capture.json",
        RC.ROOT / "v1" / "out" / "v1_compare_groups.json",
    ]
    hdr = header(
        Path(__file__).resolve(),
        "levin-phase-floor 06g: part 2's reading plan (addendum 3)",
        ins,
    )
    for l in hdr:
        print(l)

    def span(sid):
        r = routes[sid]
        a, b = (float.fromhex(x) for x in r["x_span"])
        return abs(r["K_float"]) * abs(math.exp(b) - math.exp(a))

    keys = [f"{i}_r{r}" for i in RC.IMPLS for r in RC.RESS]
    tA = {k: 0.0 for k in keys}
    tB = {k: 0.0 for k in keys}
    sA = segB = 0.0
    nA = 0
    for r in tv["rows"]:
        sid = f"z{r['sample']:02d}"
        rt = routes[sid]["route"]
        for k in keys:
            (tA if rt == "A" else tB)[k] += r["readings"][k]["tool_wall_s"]
        if rt == "A":
            sA += span(sid)
            nA += 1
        else:
            segB += routes[sid]["n_segments"]
    model = {
        "A_qawo_r1_s_per_rad": tA["qawo_r1"] / sA,
        "A_qawo_r2_s_per_rad": tA["qawo_r2"] / sA,
        "A_fcc_r1_s": tA["fcc_r1"] / nA,
        "A_fcc_r2_s": tA["fcc_r2"] / nA,
        **{f"B_{k}_s_per_segment": tB[k] / segB for k in keys},
        "overhead_s": OVERHEAD_S,
    }

    def proj(route, impl, res, sid=None, nseg=None):
        if route == "A":
            if impl == "qawo":
                return model[f"A_qawo_r{res}_s_per_rad"] * span(sid)
            return model[f"A_fcc_r{res}_s"]
        return model[f"B_{impl}_r{res}_s_per_segment"] * nseg

    # the 17, checked against the model
    capped = sorted(
        e["sid"]
        for e in sel
        if not e["in_06i"]
        and routes[e["sid"]]["route"] == "A"
        and proj("A", "qawo", 2, e["sid"]) > RC.READING_LIMIT_S
    )
    if capped != sorted(LARGE):
        raise SystemExit(f"the model's large-span set {capped} is not addendum 3's 17")
    plan = []

    def add(stage, sid, pkl, unit, index, rtol, route, impl, res, nseg=None):
        p = proj(route, impl, res, sid, nseg)
        plan.append(
            {
                "n": len(plan),
                "stage": stage,
                "sid": sid,
                "pickle": str(pkl),
                "unit": unit,
                "index": index,
                "rtol": rtol,
                "route": route,
                "impl": impl,
                "res": res,
                "projection_s": p,
                "ceiling_s": ceiling(p),
            }
        )

    for e in v1cap:  # 1. V1's 12
        g = e["g"]
        nseg = v1grp[e["wid"]]["n_edges"] - 1
        for impl in RC.IMPLS:
            for res in RC.RESS:
                add(
                    "v1",
                    e["wid"],
                    RC.ROOT / "v1" / "state" / "inputs" / f"{e['wid']}-g.pkl",
                    f"R24 {g['notify_label'][:40]}",
                    g["index"],
                    "1e-7",
                    "B",
                    impl,
                    res,
                    nseg,
                )
    by = {e["sid"]: e for e in sel}
    for e in sel:  # 2. V8's 12
        if e["source"] == "v8":
            for impl in RC.IMPLS:
                for res in RC.RESS:
                    add(
                        "v8",
                        e["sid"],
                        RC.ROOT / "state" / "inputs" / f"{e['sid']}.pkl",
                        e["unit"],
                        e["index"],
                        e["rtol"],
                        routes[e["sid"]]["route"],
                        impl,
                        res,
                        routes[e["sid"]].get("n_segments"),
                    )
    for e in sel:  # 3. the rest of V3's sample
        if e["in_06i"] or e["source"] == "v8":
            continue
        rt = routes[e["sid"]]["route"]
        for impl in RC.IMPLS:
            for res in RC.RESS:
                if e["sid"] in LARGE and impl == "qawo":
                    continue
                add(
                    "ref",
                    e["sid"],
                    RC.ROOT / "state" / "inputs" / f"{e['sid']}.pkl",
                    e["unit"],
                    e["index"],
                    e["rtol"],
                    rt,
                    impl,
                    res,
                    routes[e["sid"]].get("n_segments"),
                )
    for sid in LONG_R1:  # 4. the long QAWO res-1 readings, last
        e = by[sid]
        add(
            "ref_long",
            sid,
            RC.ROOT / "state" / "inputs" / f"{sid}.pkl",
            e["unit"],
            e["index"],
            e["rtol"],
            "A",
            "qawo",
            1,
        )
    tot = sum(x["projection_s"] + OVERHEAD_S for x in plan)
    by_stage = {}
    for x in plan:
        by_stage.setdefault(x["stage"], [0, 0.0])
        by_stage[x["stage"]][0] += 1
        by_stage[x["stage"]][1] += x["projection_s"] + OVERHEAD_S
    doc = {
        "header": hdr,
        "rule": __doc__,
        "model": model,
        "large_span": list(LARGE),
        "long_qawo_r1": list(LONG_R1),
        "n_readings": len(plan),
        "projected_total_s": tot,
        "by_stage": by_stage,
        "plan": plan,
    }
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps(model, indent=1))
    for st, (n, s) in by_stage.items():
        print(f"stage {st}: {n} readings, projected {s:.0f} s ({s / 3600:.2f} h)")
    for x in plan:
        if x["stage"] == "ref_long" or x["projection_s"] > 600:
            print(
                f"   long: {x['stage']} {x['sid']} {x['impl']} r{x['res']}: projection {x['projection_s']:.0f} s, ceiling {x['ceiling_s']:.0f} s"
            )
    print(
        f"total: {len(plan)} readings, projected {tot:.0f} s = {tot / 3600:.2f} h; wrote {out_p} sha256 {RC.sha256(out_p)}"
    )


if __name__ == "__main__":
    main()
