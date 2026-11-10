"""Orchestrator's check of the agents' KEPT_ON_MAC_STUDIO.txt lists (07 Part B, section B.3 check 9). Read only.

Usage:  python -I -B kept_check.py

For each list: every listed file exists with the listed SHA-256 and size; and completeness: every pickle and every
file over 2 MB under the agent's directory is listed (the list itself and anything under another agent's listed
directory excepted), and, for each run directory a list names, every file in it is listed.
Paths in the census list are relative to census/ or to var/runs/; in the rehearsal lists, to the main checkout.
"""

import hashlib
import os
import re
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
B = MAIN / "var/levin-phase-floor-07"
LISTS = {
    "census": (B / "census/KEPT_ON_MAC_STUDIO.txt", B / "census", [B / "census"]),
    "rehearsal": (B / "rehearsal/KEPT_ON_MAC_STUDIO.txt", MAIN, [B / "rehearsal"]),
    "r29": (B / "rehearsal/r29/KEPT_ON_MAC_STUDIO.txt", MAIN, [B / "rehearsal/r29"]),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def resolve(base, rel):
    for cand in (base / rel, MAIN / "var/runs" / rel, MAIN / rel):
        if cand.exists():
            return cand
    return base / rel


def main():
    print("# kept_check.py sha256", sha(__file__))
    allbad = 0
    for name, (lst, base, roots) in LISTS.items():
        entries, run_dirs = {}, set()
        for line in open(lst):
            m = re.match(r"^([0-9a-f]{64})\s+(\d+)\s+(\S+)", line)
            if m:
                p = resolve(base, m.group(3)).resolve()
                entries[p] = (m.group(1), int(m.group(2)))
            for rd in re.findall(r"var/runs/([A-Za-z0-9_.-]+)/?", line):
                if line.startswith("#") and (MAIN / "var/runs" / rd).is_dir():
                    run_dirs.add(MAIN / "var/runs" / rd)
        bad = []
        for p, (h, n) in entries.items():
            if not p.exists():
                bad.append(("missing", str(p)))
            elif p.stat().st_size != n:
                bad.append(("size", str(p), p.stat().st_size, n))
            elif sha(p) != h:
                bad.append(("sha", str(p)))
        unlisted = []
        for r in roots:
            for dp, dn, fn in os.walk(r):
                if name == "rehearsal" and Path(dp).resolve().is_relative_to((B / "rehearsal/r29").resolve()):
                    continue
                for f in fn:
                    p = (Path(dp) / f).resolve()
                    if p == lst.resolve():
                        continue
                    if (f.endswith(".pkl") or p.stat().st_size > 2 * 1024 * 1024) and p not in entries:
                        unlisted.append(str(p))
        for rd in sorted(run_dirs):
            for f in sorted(rd.iterdir()):
                if f.is_file() and f.resolve() not in entries:
                    unlisted.append(str(f))
        allbad += len(bad) + len(unlisted)
        print(f"== {name}: {lst}")
        print(f"   entries {len(entries)}; run directories named {[d.name for d in sorted(run_dirs)]}")
        print(f"   bad {len(bad)} {bad[:10]}")
        print(f"   unlisted {len(unlisted)} {unlisted[:10]}")
    print("ALL OK" if not allbad else f"PROBLEMS {allbad}")


if __name__ == "__main__":
    main()
