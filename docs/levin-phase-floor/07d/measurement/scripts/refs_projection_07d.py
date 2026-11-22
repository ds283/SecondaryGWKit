"""levin-phase-floor prompt 07d Part A, section 2.6: project (Ra)'s and (Rt)'s reference readings before any is taken.
New in 07d, after 07c Part A's refs_projection_07c.py (its arithmetic kept).

**Route B** (group calls): 07c's model, read from 07c Part A's out/refs/refs_projection.json (read only, SHA-256 in the
header), not refitted: split FCC per resolution a_r n_split + g_r D_nonmono (the gamma D term for every segment holding a
stationary point), times the fit's largest actual/fitted ratio; QAWO per resolution alpha n_seg + beta span_q + gamma D_q
(07a's fit), times its largest ratio.
**Route A** (analytic blocks; 07c read none, so it has no model): panels (the tool's own count, out/refs/inputs_*.json)
times the largest wall per panel measured by refs_costA_07d.py (out/refs/costA.json, one process, the tool's chunk
functions, no value kept), times MARGIN_A = 1.5.
**QAWO** is planned for a reading where it projects at 30 minutes or less with its margin (state/refs_rule_calls_07d.txt,
each resolution on its own); otherwise it is dropped, and the projection says what it would have cost.
**The run** (07c's plan): three readings at a time, two workers each; elapsed = max(summed wall / 3, longest) x 1.20
(07b's census margin, as 07c). Beside it, 07c's measured margin (its issues.md item 9: the jobs' summed wall was 0.65 of the
model's) applied to Route B's split-FCC part.
Writes out/refs/refs_projection_07d.json. Usage: PYTHONPATH=<tree>:<scripts> python -B refs_projection_07d.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import MEAS07C, header, sha256  # noqa: E402

P07C = MEAS07C / "out" / "refs" / "refs_projection.json"
MARGIN_A = 1.5
QAWO_CAP_S = 1800.0
PARALLEL = 3
CENSUS_MARGIN = 1.20
MEASURED_07C = 0.65


def main():
    out = ROOT / "out" / "refs" / "refs_projection_07d.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    p7 = json.loads(P07C.read_text())
    fit, qfits = {int(k): v for k, v in p7["split_fit"].items()}, p7["qawo_fits"]
    cost_p = ROOT / "out" / "refs" / "costA.json"
    cost = json.loads(cost_p.read_text())["rows"]
    perA = {k: max(r["per_panel_s"][k] for r in cost.values()) for k in ("fcc_r1", "fcc_r2", "qawo_r1", "qawo_r2")}
    srcs = [P07C, cost_p]
    calls = {}
    for s in ("ra", "rt"):
        p = ROOT / "out" / "refs" / f"inputs_{s}.json"
        srcs.append(p)
        for key, e in json.loads(p.read_text())["inputs"].items():
            f = e["features"]
            proj = {}
            if e["route"] == "A":
                for r in (1, 2):
                    w = f[f"panels_fcc_r{r}"] * perA[f"fcc_r{r}"]
                    proj[f"fcc_r{r}"] = {"fit_s": w, "with_margin_s": w * MARGIN_A}
                    w = f[f"panels_qawo_r{r}"] * perA[f"qawo_r{r}"]
                    proj[f"qawo_r{r}"] = {"fit_s": w, "with_margin_s": w * MARGIN_A}
                base = ["fcc_r1", "fcc_r2"]
            else:
                for r in (1, 2):
                    a_, g_ = fit[r]["a_g"]
                    w = a_ * f["n_split"] + g_ * f["D_nonmono"]
                    proj[f"fccsplit_r{r}"] = {"fit_s": w, "with_margin_s": w * max(1.0, fit[r]["max_actual_over_fit"])}
                    al, be, ga = qfits[f"qawo_r{r}"]["alpha_beta_gamma"]
                    w = al * f["n_seg"] + be * f["span_q"] + ga * f["D_q"]
                    proj[f"qawo_r{r}"] = {"fit_s": w, "with_margin_s": w * max(1.0, qfits[f"qawo_r{r}"]["max_actual_over_fit"])}
                base = ["fccsplit_r1", "fccsplit_r2"]
            qawo = [f"qawo_r{r}" for r in (1, 2) if proj[f"qawo_r{r}"]["with_margin_s"] <= QAWO_CAP_S]
            dropped = [f"qawo_r{r}" for r in (1, 2) if f"qawo_r{r}" not in qawo]
            readings = base + qawo
            calls[key] = {"set": s, "sid": e["sid"], "route": e["route"], "features": f, "projection": proj, "readings": readings,
                          "dropped": dropped, "dropped_s": sum(proj[k]["with_margin_s"] for k in dropped),
                          "wall_s": sum(proj[k]["with_margin_s"] for k in readings),
                          "longest_reading_s": max(proj[k]["with_margin_s"] for k in readings)}
    job = sum(c["wall_s"] for c in calls.values())
    longest = max(c["longest_reading_s"] for c in calls.values())
    elapsed = max(job / PARALLEL, longest) * CENSUS_MARGIN
    split = sum(c["projection"][k]["with_margin_s"] for c in calls.values() for k in c["readings"] if k.startswith("fccsplit"))
    job_meas = job - split + MEASURED_07C * split
    elapsed_meas = max(job_meas / PARALLEL, longest) * CENSUS_MARGIN
    by = {}
    for c in calls.values():
        b = by.setdefault(f"{c['set']}|{c['route']}", {"n": 0, "readings": 0, "wall_s": 0.0, "qawo_dropped": 0})
        b["n"] += 1
        b["readings"] += len(c["readings"])
        b["wall_s"] += c["wall_s"]
        b["qawo_dropped"] += len(c["dropped"])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: (Ra) and (Rt), projected", srcs)
    doc = {"header": hdr, "per_panel_A_s": perA, "margin_A": MARGIN_A, "split_fit_07c": fit, "qawo_fits_07a": qfits,
           "calls": calls, "by_set_route": by, "job_wall_s": job, "longest_reading_s": longest, "parallel": PARALLEL,
           "census_margin": CENSUS_MARGIN, "elapsed_projection_s": elapsed, "measured_margin_07c": MEASURED_07C,
           "elapsed_with_07c_measured_margin_s": elapsed_meas, "n_readings": sum(len(c["readings"]) for c in calls.values())}
    out.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print("Route A wall per panel (largest measured):", perA)
    for k, c in sorted(calls.items(), key=lambda kv: -kv[1]["wall_s"]):
        print(f"{k:28s} {c['route']} " + " ".join(f"{r} {c['projection'][r]['with_margin_s']:.0f}" for r in c["readings"])
              + f" | {c['wall_s']:.0f} s; dropped {c['dropped']} ({c['dropped_s']:.3g} s)")
    print("by set and route:", json.dumps(by))
    print(f"{doc['n_readings']} readings; job wall {job:.0f} s ({job / 3600:.2f} h); longest reading {longest:.0f} s")
    print(f"elapsed, {PARALLEL} at once, x{CENSUS_MARGIN}: {elapsed:.0f} s = {elapsed / 3600:.2f} h; "
          f"with 07c's measured 0.65 on split FCC: {elapsed_meas / 3600:.2f} h")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
