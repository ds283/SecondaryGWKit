"""levin-phase-floor prompt 07a Part A: KEPT_ON_MAC_STUDIO.txt, every pickle and every file over 2 MB under
var/levin-phase-floor-07a/measurement/ (and the registered run directories named on the command line), with SHA-256
and size; these are not copied back. New in 07a, after 07's kept_list_07.py. Usage: python -B kept_list_07a.py [RUN_DIR ...]
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIMIT = 2 * 1024 * 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def listing(base, rel_to):
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and (p.suffix == ".pkl" or p.stat().st_size > LIMIT):
            rows.append(f"{sha256(p)}  {p.stat().st_size}  {p.relative_to(rel_to)}")
    return rows


def main():
    out = ROOT / "KEPT_ON_MAC_STUDIO.txt"
    L = ["# levin-phase-floor 07a Part A measuring agent: files kept on the Mac Studio only (mps026298), not copied to the laptop.",
         f"# sha256  size_bytes  path. Written by scripts/kept_list_07a.py (sha256 {sha256(__file__)}).", "",
         "## pickles, and files over 2 MB, under var/levin-phase-floor-07a/measurement/"]
    L += [r for r in listing(ROOT, ROOT) if not r.endswith("KEPT_ON_MAC_STUDIO.txt")]
    for rd in sys.argv[1:]:
        rd = Path(rd)
        L += ["", f"## pickles, and files over 2 MB, under {rd}"]
        L += listing(rd, rd.parent)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}: {sum(1 for l in L if l and not l.startswith('#'))} files")


if __name__ == "__main__":
    main()
