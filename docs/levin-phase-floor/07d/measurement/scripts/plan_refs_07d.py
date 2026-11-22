"""levin-phase-floor prompt 07d Part A, section 2.6: the readings plan for (Ra) and (Rt), fixed before any reading. New in
07d, after 07c Part A's plan_refs_07c.py.

Per integrand of out/refs/inputs_ra.json and inputs_rt.json, the readings out/refs/refs_projection_07d.json plans
(state/refs_rule_calls_07d.txt, fixed at 10:37:01Z):
- Route A: one_reading_07d.py (07c's one_reading.py, copied) FCC at resolutions 1 and 2, and QAWO at each resolution the
  projection keeps, each a command of its own, two workers, on the integrand's pickle (state/refs/inputs/<key>.pkl);
- Route B: leafref_07d.py (07c's leafref_07c.py, copied) on the call's whole span as one span (points: its two ends),
  ``--split-ratio 1.8``, resolutions 1 and 2 in one command, two workers, on the replay manifest's capture of the call
  (no rtol override: the integrand does not depend on it); and QAWO (one_reading_07d.py) where kept.
Readings run three at a time (six worker processes, nothing else running), longest projected first. The backstop is
Part A's start (state/started_utc.txt) + 36 h - 1.5 h, less now, with ``backstop_divisor`` 3 (run_07d.py), 07c's
plan_refs2's form. Writes state/refs/specs/<key>.json and state/plan_refs.json (refusing to overwrite).
Usage: PYTHONPATH=<tree>:<scripts> python -B plan_refs_07d.py
"""

import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    plan_p = ROOT / "state" / "plan_refs.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists")
    from replay_07d import manifest

    man = manifest()
    proj_p = ROOT / "out" / "refs" / "refs_projection_07d.json"
    proj = json.loads(proj_p.read_text())["calls"]
    ins = {}
    for s in ("ra", "rt"):
        ins |= json.loads((ROOT / "out" / "refs" / f"inputs_{s}.json").read_text())["inputs"]
    import ray.cloudpickle as cp

    specd = ROOT / "state" / "refs" / "specs"
    specd.mkdir(parents=True, exist_ok=True)
    readd = ROOT / "out" / "refs" / "readings"
    readd.mkdir(parents=True, exist_ok=True)
    pp = f"{TREE}:{HERE}"
    jobs = []
    for key, e in sorted(ins.items()):
        pr = proj[key]
        pk = e["pickle"]
        rec = cp.loads(Path(pk).read_bytes())

        def one(impl, r):
            return {"name": f"{key}-{impl}-r{r}", "pythonpath": pp, "requires": pk,
                    "projected_s": pr["projection"][f"{impl}_r{r}"]["with_margin_s"],
                    "argv": [PY, "-B", str(HERE / "one_reading_07d.py"), pk, impl, str(r), str(readd / f"{key}-{impl}-r{r}.json"),
                             "--sid", key, "--unit", e["unit"], "--index", str(e["index"]), "--rtol", str(e["rtol"]),
                             "--workers", "2"]}

        if e["route"] == "A":
            jobs += [one("fcc", 1), one("fcc", 2)]
        else:
            c = man[f"d:{e['sid']}"]
            a, b = rec["x_span"]
            spec = {"sid": key, "capture": {"path": c["path"], "offset": c["offset"], "length": c["length"], "sha256": c["sha256"],
                                            "override": None},
                    "spans": [[float(a).hex(), float(b).hex()]], "points": [float(a).hex(), float(b).hex()]}
            sp = specd / f"{key}.json"
            if sp.exists():
                raise SystemExit(f"{sp} exists")
            sp.write_text(json.dumps(spec, indent=1))
            jobs.append({"name": f"{key}-fccsplit", "pythonpath": pp, "requires": str(sp),
                         "projected_s": pr["projection"]["fccsplit_r1"]["with_margin_s"] + pr["projection"]["fccsplit_r2"]["with_margin_s"],
                         "argv": [PY, "-B", str(HERE / "leafref_07d.py"), str(sp), str(readd / f"{key}-fccsplit.json"),
                                  "--workers", "2", "--split-ratio", "1.8"]})
        for q in pr["readings"]:
            if q.startswith("qawo"):
                jobs.append(one("qawo", int(q[-1])))
    jobs.sort(key=lambda j: -j["projected_s"])
    start = datetime.datetime.strptime((ROOT / "state" / "started_utc.txt").read_text().split()[0], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    backstop = (start + datetime.timedelta(hours=34.5) - now).total_seconds()
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: (Ra) and (Rt) readings plan", [proj_p])
    plan = {"label": "refs", "header": hdr, "backstop_s": backstop, "backstop_divisor": 3, "check_after_s": 1800.0,
            "stages": [{"name": "readings", "max_parallel": 3, "jobs": jobs}]}
    plan_p.write_text(json.dumps(plan, indent=1))
    tot = sum(j["projected_s"] for j in jobs)
    print(f"{len(jobs)} commands over {len(ins)} integrands; projected job wall {tot:.0f} s; three at a time x1.2: "
          f"{tot / 3 * 1.2 / 3600:.2f} h; backstop {backstop:.0f} s")
    print(f"wrote {plan_p} {sha256(plan_p)}")


if __name__ == "__main__":
    main()
