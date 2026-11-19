"""levin-phase-floor prompt 07c Part A, section 2.5: the readings plan for R4, R5 and R31's record, fixed before any reading.
New in 07c, after 07b Part A's plan (state/plan_refs.json there).

Per integrand of out/refs/inputs.json: split FCC (leafref_07c.py on the whole span, ``--split-ratio 1.8``, resolutions 1 and
2 in one command, two workers), and QAWO at resolutions 1 and 2 (one_reading.py, two workers) where the projection
(out/refs/refs_projection.json) puts each QAWO reading at 30 minutes or less with its margin. Readings run two at a time
(four worker processes), longest projected first, leaving two process slots for the rest of Part A. The run's backstop is
36 h (section 8's limit for Part A), with 07's monitor after 30 minutes.

The rule for the reference and its stated error (fixed here, before any reading, and copied to state/refs_rule.txt):
07b Part A's ``score_07b.refs()``: the reference is split FCC r2 (``fccsplit-r2``, the sum of its pieces at resolution 2);
its stated error is the spread of every reading taken (largest minus smallest of fccsplit r1, fccsplit r2 and, where taken,
QAWO r1 and r2). A call is scored if that spread is at most min(abserr)/100 over the codes and variants scored on it (07a's
census record too, for R5's calls); otherwise it is reported, not scored. |v - ref| / abserr above 1 is dishonest.

Writes state/refs/specs/<key>.json, state/plan_refs.json, state/refs_rule.txt. Usage: python -B plan_refs_07c.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    plan_p = ROOT / "state" / "plan_refs.json"
    rule_p = ROOT / "state" / "refs_rule.txt"
    for p in (plan_p, rule_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    rule_p.write_text(__doc__)
    ins = json.loads((ROOT / "out" / "refs" / "inputs.json").read_text())["inputs"]
    proj = json.loads((ROOT / "out" / "refs" / "refs_projection.json").read_text())["calls"]
    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
    man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    import ray.cloudpickle as cp

    specd = ROOT / "state" / "refs" / "specs"
    specd.mkdir(parents=True, exist_ok=True)
    readd = ROOT / "out" / "refs" / "readings"
    readd.mkdir(parents=True, exist_ok=True)
    jobs = []
    for key, e in ins.items():
        rec = cp.loads(Path(e["path"]).read_bytes())
        a, b = rec["x_span"]
        spec = {"sid": key, "capture": {"path": e["path"], "offset": 0, "length": e["bytes"], "sha256": e["sha256"], "override": None},
                "spans": [[float(a).hex(), float(b).hex()]], "points": [float(a).hex(), float(b).hex()], "named": [], "all_leaves": None}
        sp = specd / f"{key}.json"
        if sp.exists():
            raise SystemExit(f"{sp} exists")
        sp.write_text(json.dumps(spec, indent=1))
        pr = proj[key]
        jobs.append({"name": f"{key}-fccsplit", "pythonpath": f"{TREE}:{HERE}", "requires": str(sp),
                     "projected_s": pr["projection"]["fccsplit_r1"]["with_margin_s"] + pr["projection"]["fccsplit_r2"]["with_margin_s"],
                     "argv": [PY, "-B", str(HERE / "leafref_07c.py"), str(sp), str(readd / f"{key}-fccsplit.json"),
                              "--workers", "2", "--split-ratio", "1.8"]})
        if pr["qawo"]:
            for r in (1, 2):
                jobs.append({"name": f"{key}-qawo-r{r}", "pythonpath": f"{TREE}:{HERE}", "requires": e["path"],
                             "projected_s": pr["projection"][f"qawo_r{r}"]["with_margin_s"],
                             "argv": [PY, "-B", str(HERE / "one_reading.py"), e["path"], "qawo", str(r),
                                      str(readd / f"{key}-qawo-r{r}.json"), "--sid", key, "--unit", rec["unit"],
                                      "--index", str(rec["index"]), "--rtol", str(rec["kw"]["rtol"]), "--workers", "2"]})
    jobs.sort(key=lambda j: -j["projected_s"])
    plan = {"label": "refs", "backstop_s": 129600.0, "check_after_s": 1800.0,
            "stages": [{"name": "readings", "max_parallel": 2, "jobs": jobs}]}
    plan_p.write_text(json.dumps(plan, indent=1))
    tot = sum(j["projected_s"] for j in jobs)
    print(f"{len(jobs)} readings over {len(ins)} integrands; projected job wall {tot:.0f} s; two at a time x1.2: {tot / 2 * 1.2 / 3600:.2f} h")
    print(f"wrote {plan_p}")


if __name__ == "__main__":
    main()
