"""levin-phase-floor prompt 07d Part A: KEPT_ON_MAC_STUDIO.txt, written last. New in 07d, after 07c Part A's.

Every pickle and every file over 2 MB (2,097,152 bytes) under var/levin-phase-floor-07d/measurement/ and the registered run
directories var/runs/levin-phase-floor-07d-*/, as SHA-256, size in bytes and path relative to the main checkout, sorted by
path. Run outside rec.sh, so that out/commands.log (itself listed) does not change after it is hashed.
Usage: python -B kept_07d.py
"""

import datetime
import hashlib
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
ROOT = MAIN / "var" / "levin-phase-floor-07d" / "measurement"
LIMIT = 2 * 1024 * 1024


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = ROOT / "KEPT_ON_MAC_STUDIO.txt"
    if out.exists():
        raise SystemExit(f"{out} exists")
    dirs = [ROOT] + sorted((MAIN / "var" / "runs").glob("levin-phase-floor-07d-*"))
    files = sorted({p for d in dirs for p in d.rglob("*") if p.is_file() and (p.suffix == ".pkl" or p.stat().st_size > LIMIT)})
    rows = [(sha(p), p.stat().st_size, p.relative_to(MAIN)) for p in files]
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    L = ["# levin-phase-floor 07d Part A: files kept on the Mac Studio (mps026298), not copied back.",
         "# Every pickle and every file over 2 MB (2,097,152 bytes) under var/levin-phase-floor-07d/measurement/ and the",
         "# registered run directories var/runs/levin-phase-floor-07d-*/, as SHA-256, size in bytes and path relative to the",
         f"# main checkout. Written last, {now}, after every other record, by scripts/kept_07d.py.",
         f"# {len(rows)} files, {sum(r[1] for r in rows)} bytes.", ""]
    L += [f"{h}  {n}  {p}" for h, n, p in rows]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
