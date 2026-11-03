"""levin-phase-floor 06g references: KEPT_ON_MAC_STUDIO.txt: every pickle and every file over 2 MB
under var/levin-phase-floor-06g/references/ (symbolic links listed with their targets, not
followed), and every file of the agent's three run directories, each with SHA-256 and size."""
import hashlib, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs")
RUN_DIRS = ["levin-phase-floor-06g-references-20261101T175231", "levin-phase-floor-06g-references-20261101T201415",
            "levin-phase-floor-06g-references-rerun-20261102T185324"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
L = ["# levin-phase-floor 06g Part B, the references agent: files kept on the Mac Studio only",
     "# (mps026298), not copied to the laptop. sha256  size_bytes  path (relative to var/levin-phase-floor-06g/references/",
     "# or to var/runs/). Written by scripts/kept_list.py.", ""]
L.append("## pickles, and files over 2 MB, under var/levin-phase-floor-06g/references/")
n = tot = 0
links = []
for p in sorted(ROOT.rglob("*")):
    if p.is_symlink():
        links.append(f"symlink  {p.relative_to(ROOT)} -> {os.readlink(p)}")
        continue
    if not p.is_file():
        continue
    if p.suffix == ".pkl" or p.stat().st_size > 2 * 1024 * 1024:
        s = p.stat().st_size
        L.append(f"{sha(p)}  {s}  {p.relative_to(ROOT)}")
        n += 1; tot += s
L.append(f"# {n} files, {tot} bytes")
L.append("")
L.append("## symbolic links (v1/stores/, to the rehearsal stores R23 and R24 in var/datastores/; read only)")
L += links
for rd in RUN_DIRS:
    L.append("")
    L.append(f"## run directory var/runs/{rd}/ (every file)")
    m = t = 0
    for p in sorted((RUNS / rd).rglob("*")):
        if p.is_file():
            s = p.stat().st_size
            L.append(f"{sha(p)}  {s}  {rd}/{p.relative_to(RUNS / rd)}")
            m += 1; t += s
    L.append(f"# {m} files, {t} bytes")
(ROOT / "KEPT_ON_MAC_STUDIO.txt").write_text("\n".join(L) + "\n")
print("\n".join(l for l in L if l.startswith("#")))
