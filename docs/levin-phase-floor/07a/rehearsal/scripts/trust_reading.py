"""The orchestrator's reading of issues.md item 1, settled from the replay's route_07a (levin-phase-floor 07a,
rehearsal agent, R30). New. Reads out/replay_calls.checks.json (replay07a_r30.py check). For each of the 49 group
records whose abserr moved against R29 with value, regions and location unchanged: the solves whose C2 route or LU
choice differs between V and 07's estimate, by caller and by whether their span is a returned region, with V and
the estimate; and over every replayed record, whether any region's own solve ("Levin subregion") differs.

    ./venv/bin/python -I -B trust_reading.py <checks.json>
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = sys.argv[1]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    print(f"# input {p} sha256 {sha256(p)}")
    d = json.loads(Path(p).read_text())
    for l in d["header"]:
        print(f"# (checks header) {l}")
    checks = d["checks"]
    ao = [c for c in checks if c["abserr_only_vs_r29"]]
    print(f"abserr-only records: {len(ao)}; reproduced {sum(c['reproduces'] for c in ao)}")
    per = Counter()
    nd = Counter()
    for c in ao:
        rt = c["route_07a"] or {}
        diffs = [(k, n) for k, n in (rt.get("counts") or {}).items() if "| " in k and not k.split(" || ")[1].startswith("same")]
        nd[sum(n for _, n in diffs)] += 1
        for k, n in diffs:
            per[k] += n
        own = [(k, n) for k, n in (rt.get("counts") or {}).items() if k.startswith("Levin subregion ||") and not k.split(" || ")[1].startswith("same")]
        ch = rt.get("changed") or []
        print(f"{c['key']} {c['path']} {c['label']}: differing solves {diffs}; own-solve differs {own}; "
              + "; ".join(f"[{x['desc']}: V {x['V_repr']} est {x['est_repr']} is_direct {x['is_direct']} returned {x['returned']}]" for x in ch))
    print(f"number of differing solves per record: {dict(nd)}")
    print("differing solves over the 49 records, by caller and class:")
    for k, n in sorted(per.items()):
        print(f"  {k}: {n}")
    allown = Counter()
    for c in checks:
        rt = c["route_07a"] or {}
        for k, n in (rt.get("counts") or {}).items():
            if k.startswith("Levin subregion ||"):
                allown[k.split(" || ")[1]] += n
    print(f"over all {len(checks)} checked records' calls: the regions' own solves (\"Levin subregion\") by class: {dict(allown)}")


if __name__ == "__main__":
    main()
