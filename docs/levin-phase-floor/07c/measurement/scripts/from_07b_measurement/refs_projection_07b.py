"""levin-phase-floor prompt 07b Part A, section 2.4: project each reference reading's wall time before any is taken.
New in 07b, after 07a's refs_projection_07a.py (a record), whose fitted model it reuses unchanged.

- **Route B (group calls):** 07a's refitted model (07a census out/refs/refs_projection.json, ``fits``: per reading kind,
  wall = alpha n_seg + beta span + gamma D by non-negative least squares on 07's 74 V4 readings on this host, with the
  fit's largest actual/fitted ratio as a margin), applied to each capture's features by 07's ``features`` (imported
  unchanged from refs_projection_07.py).
- **Route A (analytic calls):** 07a's model has no Route A reading. Each capture's panel counts at each reading kind
  (the tool's ``RouteA(call).panels(impl, res)``) are recorded; the wall is projected from a per-panel time measured on
  this host by a pilot reading (``--route-a-pilot`` READING_JSON ...: one_reading.py outputs of one Route A call), with the
  pilot's own panel counts; until a pilot is given, Route A calls are reported unprojected.
- 07's ruling at its stop 4 decides the readings: FCC r1 and r2 on every call, QAWO r1 and r2 where each QAWO reading
  projects at 30 minutes or less with the margin.

Reads out/refs/inputs.json. Writes OUT_JSON. Usage: PYTHONPATH=<tree>:<scripts> python -B refs_projection_07b.py OUT_JSON
[--route-a-pilot JSON [JSON ...]]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07A, WIDER, header, sha256  # noqa: E402
from refs_projection_07 import features, load_capture  # noqa: E402  (07's, unchanged)

FITS = REC07A / "out" / "refs" / "refs_projection.json"
INPUTS = WIDER / "out" / "refs" / "inputs.json"
KINDS = [("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--route-a-pilot", nargs="*", default=[])
    a = ap.parse_args()
    import tools.filon_reference as F
    from mpmath import exp as mp_exp

    fits = json.loads(FITS.read_text())["fits"]
    ins = json.loads(INPUTS.read_text())["inputs"]
    per_panel = {}
    pilot_info = []
    for p in a.route_a_pilot:
        r = json.loads(Path(p).read_text())
        n = r["n_panels"]  # prompt 07b: one_reading.py's own panel count
        per_panel[f"{r['impl']}_r{r['res']}"] = r["wall_s"] / n
        pilot_info.append({"reading": p, "sha256": sha256(p), "sid": r["sample"], "impl": r["impl"], "res": r["res"],
                           "wall_s": r["wall_s"], "panels": n})
    calls = {}
    for sid, e in sorted(ins.items()):
        if sha256(e["path"]) != e["sha256"]:
            raise SystemExit(f"{sid}: input has another SHA-256")
        call = load_capture(e["path"])
        route = F.route_of(call)
        proj = {}
        if route == "B":
            f = features(call)
            for impl, res in KINDS:
                k = f"{impl}_r{res}"
                al, be, ga = fits[k]["alpha_beta_gamma"]
                w = al * f["n_seg"] + be * f["span_q" if impl == "qawo" else "span_f"] + ga * f["D_q" if impl == "qawo" else "D_f"]
                proj[k] = {"fit_s": w, "with_margin_s": w * max(1.0, fits[k]["max_actual_over_fit"])}
            # prompt 07b: FCC read with steep segments split (leafref_07b.py's rule, validated on v08): the pilot's
            # measured job CPU per segment (r1 0.097 s, r2 0.388 s; out/leafrefs/pilot_v08_split18.json) over six
            # workers, times 1.1 for the added edges and 2 for packing and scatter
            for r, cps in ((1, 0.097), (2, 0.388)):
                w = f["n_seg"] * 1.1 * cps / 6.0
                proj[f"fccsplit_r{r}"] = {"fit_s": w, "with_margin_s": 2.0 * w}
        else:
            A = F.RouteA(call)
            # prompt 07b: QAWO's dyadic tiles are counted, not built (a large K makes the list itself too big to build):
            # each u-segment's eta span over the tiles' largest width 2 h_max, as panels() would tile it
            import math as _m
            pc = {}
            for i, r in KINDS:
                if i == "fcc":
                    pc[f"{i}_r{r}"] = len(A.panels(i, r))
                    continue
                cfg = F.ROUTE_A_QAWO[r]
                ee = [float(mp_exp(u)) for u in A.u_segments(cfg["du"])]
                Kabs = abs(float(A.K_exact))
                n = 0
                for l_, r_ in zip(ee[:-1], ee[1:]):
                    h = min(cfg["Pmax"] / (2 * Kabs), (r_ - l_) / 2)
                    n += _m.ceil((r_ - l_) / (2 * h))
                pc[f"{i}_r{r}"] = n
            f = {"panels": pc, "K": float(A.K_exact), "u_breaks": len(A.u_breaks())}
            for impl, res in KINDS:
                k = f"{impl}_r{res}"
                if k in per_panel and f["panels"][k] < 10**7:
                    w = per_panel[k] * f["panels"][k]
                    proj[k] = {"fit_s": w, "with_margin_s": 2.0 * w}
        fcc = [("fccsplit", 1), ("fccsplit", 2)] if route == "B" else [("fcc", 1), ("fcc", 2)]
        if all(f"{i}_r{r}" in proj for i, r in fcc) and all(f"qawo_r{r}" in proj for r in (1, 2)):
            qawo_ok = all(proj[f"qawo_r{r}"]["with_margin_s"] <= 1800.0 for r in (1, 2))
            readings = ([("qawo", 1), ("qawo", 2)] if qawo_ok else []) + fcc
        elif route == "A" and all(f"{i}_r{r}" in proj for i, r in fcc):
            qawo_ok, readings = False, fcc
        else:
            qawo_ok, readings = None, None
        calls[sid] = {"route": route, "kind": e["kind"], "set": e["set"], "features": f, "projection": proj,
                      "qawo": qawo_ok, "readings": readings}
    done = [s for s in calls if calls[s]["readings"] is not None]
    total = sum(calls[s]["projection"][f"{i}_r{r}"]["with_margin_s"] for s in done for i, r in calls[s]["readings"])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: section 2.4's references, projected",
                 [FITS, INPUTS] + [Path(p) for p in a.route_a_pilot])
    doc = {"header": hdr, "fits": fits, "route_a_per_panel_s": per_panel, "route_a_pilot": pilot_info,
           "route_a_margin": 2.0, "calls": calls, "projected": len(done), "unprojected": sorted(set(calls) - set(done)),
           "total_with_margin_s": total}
    Path(a.out_json).write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    by_set = {}
    for s in done:
        c = calls[s]
        w = sum(c["projection"][f"{i}_r{r}"]["with_margin_s"] for i, r in c["readings"])
        by_set[c["set"]] = by_set.get(c["set"], 0.0) + w
        print(f"{s:28s} {c['route']} {c['set']:8s} " + " ".join(f"{k} {v['with_margin_s']:.0f}" for k, v in c["projection"].items())
              + f" | {'QAWO+FCC' if c['qawo'] else 'FCC only'} total {w:.0f} s")
    for s in sorted(set(calls) - set(done)):
        print(f"{s:28s} {calls[s]['route']} unprojected: {calls[s]['features']}")
    print("by set (s, with margin):", {k: round(v) for k, v in by_set.items()})
    print(f"projected {len(done)} of {len(calls)} calls; total with margin {total:.0f} s = {total / 3600:.2f} h")
    print(f"wrote {a.out_json} sha256 {sha256(a.out_json)}")


if __name__ == "__main__":
    main()
