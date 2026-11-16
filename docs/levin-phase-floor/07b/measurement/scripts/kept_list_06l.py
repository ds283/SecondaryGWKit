"""levin-phase-floor 06l: KEPT_ON_MAC_STUDIO.txt: every pickle and every file over 2 MB under
var/levin-phase-floor-06l/ (outside review/, the orchestrator's), and every file of 06l's run directories,
each with SHA-256 and size. New in 06l (06k's kept_list_06k.py with its names changed). Read only, except its output.
Usage: python -B kept_list_06l.py"""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs")
RUN_DIRS = sorted(p.name for p in RUNS.glob("levin-phase-floor-06l-*"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
L = ["# levin-phase-floor 06l: files kept on the Mac Studio only (mps026298), not copied to the laptop.",
     "# sha256  size_bytes  path (relative to var/levin-phase-floor-06l/, or to var/runs/). Written by scripts/kept_list_06l.py.", ""]
L.append("## pickles, and files over 2 MB, under var/levin-phase-floor-06l/ (review/ excluded)")
n = tot = 0
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.is_symlink() or p.relative_to(ROOT).parts[0] == "review" or p.name == "KEPT_ON_MAC_STUDIO.txt":
        continue
    if p.suffix == ".pkl" or p.stat().st_size > 2 * 1024 * 1024:
        s = p.stat().st_size
        L.append(f"{sha(p)}  {s}  {p.relative_to(ROOT)}")
        n += 1
        tot += s
L.append(f"# {n} files, {tot} bytes")
for rd in RUN_DIRS:
    L += ["", f"## run directory var/runs/{rd}/ (every file)"]
    m = t = 0
    for p in sorted((RUNS / rd).rglob("*")):
        if p.is_file():
            s = p.stat().st_size
            L.append(f"{sha(p)}  {s}  {rd}/{p.relative_to(RUNS / rd)}")
            m += 1
            t += s
    L.append(f"# {m} files, {t} bytes")
(ROOT / "KEPT_ON_MAC_STUDIO.txt").write_text("\n".join(L) + "\n")
print("\n".join(l for l in L if l.startswith("#") or l.startswith("##")))
