"""levin-phase-floor prompt 07c Part A: write a replay plan for run_06j.py (launched by launch_07c.py). New in 07c.

Each stage runs run_replays_07c.py once (it keeps at most six replay processes), so each stage is one command, run one
after another. Usage: python -B plan_replays_07c.py LABEL VARIANT[:instrument] [VARIANT ...] -- SET [SET ...]
Writes state/plan_<LABEL>.json with stages ``<variant>`` each running every call of the sets named (manifest ``set``
values; ``parta`` selects every ``parta:*`` set) under that variant, label <LABEL>.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    label = sys.argv[1]
    i = sys.argv.index("--")
    variants, sets = sys.argv[2:i], sys.argv[i + 1:]
    out = ROOT / "state" / f"plan_{label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
    if (ROOT / "state" / "calls_07c_r5.json").exists():
        man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    have = sorted({c["set"] for c in man.values()})
    want = sorted(s for s in have if s in sets or (s.startswith("parta:") and "parta" in sets))
    stages = []
    for v in variants:
        name, inst = (v.split(":")[0], v.endswith(":instrument"))
        stages.append({"name": name, "max_parallel": 1, "jobs": [{
            "name": f"replays_{name}_{label}", "pythonpath": TREE,
            "argv": [PY, "-B", str(HERE / "run_replays_07c.py"), name, label, "--sets", *want, "--workers", "6"]
            + (["--instrument"] if inst else [])}]})
    out.write_text(json.dumps({"label": label, "sets": want, "n_calls": sum(1 for c in man.values() if c["set"] in want),
                               "stages": stages}, indent=1))
    print(f"wrote {out}: {len(stages)} stages over {sum(1 for c in man.values() if c['set'] in want)} calls of sets {want}")


if __name__ == "__main__":
    main()
