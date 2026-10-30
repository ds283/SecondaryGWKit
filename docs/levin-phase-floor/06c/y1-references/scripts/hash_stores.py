"""Hash the ten original store files (R17, R18) and, if present, their copies.

Usage: python -B hash_stores.py [--copies] [--compare OUT_FILE_OF_AN_EARLIER_RUN]
Prints ``sha256  size  path`` per file. With --compare, checks every path's hash against the
earlier output and prints whether all are unchanged.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yprov import COPY_DIR, ORIG_DIR, header, sha256, store_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", action="store_true")
    ap.add_argument("--compare", action="append", default=[])
    a = ap.parse_args()
    for l in header(Path(__file__).resolve(), "Y1 references: store hashes", [Path(p) for p in a.compare]):
        print(l)
    now = {}
    dirs = [("original", ORIG_DIR)] + ([("copy", COPY_DIR)] if a.copies else [])
    for kind, d in dirs:
        for store in ("R17", "R18"):
            for p in store_files(store, d):
                h = sha256(p)
                now[str(p)] = h
                print(f"{h}  {p.stat().st_size:>10}  {kind} {store} {p}")
    # sibling files (journals, -wal, -shm) next to either set
    for kind, d in dirs:
        extra = sorted(x.name for x in Path(d).iterdir() if "06c-r1" in x.name and not x.name.endswith(".sqlite"))
        print(f"# non-.sqlite files beside the {kind}s: {extra if extra else 'none'}")
    if a.copies:
        pairs = [(o, c) for s in ("R17", "R18") for o, c in zip(store_files(s, ORIG_DIR), store_files(s, COPY_DIR))]
        same = all(now[str(o)] == now[str(c)] for o, c in pairs)
        print(f"# every copy equals its original: {same}")
    ok = True
    for path in a.compare:
        before = {}
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            before[parts[-1]] = parts[0]
        for p, h in before.items():
            if p in now:
                same = now[p] == h
                ok &= same
                if not same:
                    print(f"CHANGED {p}: {h} -> {now[p]}")
        print(f"# compared with {path}: {len([p for p in before if p in now])} files, all unchanged: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
