"""levin-phase-floor prompt 07c Part A, section 2.5: the plan for the references' second launch, after the first took no
reading (deviation 23). New in 07c. On the laptop orchestrator's ruling (2026-11-18, option (b)): the same 123 readings,
rule, seeds and projection model as state/plan_refs.json, three at a time (six worker processes, nothing else running),
and a backstop at what remains of Part A's 36 h less 1.5 h for scoring and records.

Reads state/plan_refs.json (its SHA-256 recorded) and state/started_utc.txt (Part A's start). Every job is kept as it
is, in the same order (longest projected first); only these change:
- ``label`` "refs2"; the stage's ``max_parallel`` 3;
- ``backstop_s``: Part A's start + 36 h − 1.5 h, less now, in seconds;
- ``backstop_divisor`` 3 (run_06j.py v2, marked "prompt 07c"): P divides the remaining projections by the commands run at
  once, so that P is a wall-clock projection.
Writes state/plan_refs2.json (refusing to overwrite). Usage: python -B plan_refs2_07c.py
"""

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = ROOT / "state" / "plan_refs.json"
    out = ROOT / "state" / "plan_refs2.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    plan = json.loads(src.read_text())
    start = dt.datetime.fromisoformat(
        (ROOT / "state" / "started_utc.txt").read_text().split()[0].replace("Z", "+00:00"))
    deadline = start + dt.timedelta(hours=36) - dt.timedelta(hours=1.5)
    now = dt.datetime.now(dt.timezone.utc)
    jobs = [j for st in plan["stages"] for j in st["jobs"]]
    assert len(jobs) == 123 and len(plan["stages"]) == 1
    plan2 = {"label": "refs2", "derived_from": {"path": str(src), "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
             "part_a_started": start.isoformat(), "deadline_less_scoring": deadline.isoformat(),
             "backstop_s": round((deadline - now).total_seconds()), "backstop_divisor": 3.0,
             "check_after_s": plan["check_after_s"],
             "stages": [{"name": "readings", "max_parallel": 3, "jobs": jobs}]}
    out.write_text(json.dumps(plan2, indent=1))
    tot = sum(j["projected_s"] for j in jobs)
    print(f"Part A started {start.isoformat()}; deadline less 1.5 h {deadline.isoformat()}; now {now.isoformat()}")
    print(f"{len(jobs)} readings; summed projection {tot:.0f} s; three at a time {tot / 3:.0f} s, x1.2 {tot / 3 * 1.2 / 3600:.2f} h; "
          f"backstop {plan2['backstop_s']} s ({plan2['backstop_s'] / 3600:.2f} h)")
    print(f"wrote {out} {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
