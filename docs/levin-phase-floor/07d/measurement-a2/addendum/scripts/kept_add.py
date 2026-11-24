"""levin-phase-floor prompt 07d Part A': KEPT_ON_MAC_STUDIO.txt, written last. Every pickle and every file over 2 MB
(2,097,152 bytes) under var/levin-phase-floor-07d/measurement-a2/ and this Part's registered run directories
var/runs/levin-phase-floor-07d-add-*/, as SHA-256, size and path relative to the main checkout. New in Part A', after Part A's
kept_07d.py.
Usage: python -B kept_a2.py
"""
import datetime
import hashlib
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
ROOT = MAIN / "var" / "levin-phase-floor-07d" / "measurement-a2" / "addendum"
LIMIT = 2 * 1024 * 1024


def main():
    out = ROOT / "KEPT_ON_MAC_STUDIO.txt"
    if out.exists():
        raise SystemExit(f"{out} exists")
    roots = [ROOT] + sorted((MAIN / "var" / "runs").glob("levin-phase-floor-07d-add-*"))
    rows, total = [], 0
    for r in roots:
        for p in sorted(r.rglob("*")):
            if p.is_file() and (p.suffix == ".pkl" or p.stat().st_size > LIMIT):
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                n = p.stat().st_size
                total += n
                rows.append(f"{h}  {n}  {p.relative_to(MAIN)}")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hdr = ["# levin-phase-floor 07d Part A' addendum: files kept on the Mac Studio (mps026298), not copied back.",
           "# Every pickle and every file over 2 MB (2,097,152 bytes) under var/levin-phase-floor-07d/measurement-a2/addendum/ and the",
           "# registered run directories var/runs/levin-phase-floor-07d-add-*/, as SHA-256, size in bytes and path relative to the",
           f"# main checkout. Written last, {now}, after every other record, by addendum/scripts/kept_add.py.",
           f"# {len(rows)} files, {total} bytes.", ""]
    out.write_text("\n".join(hdr + rows) + "\n")
    print(len(rows), total)


if __name__ == "__main__":
    main()
