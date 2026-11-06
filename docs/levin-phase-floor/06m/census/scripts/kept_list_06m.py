"""levin-phase-floor prompt 06m section 5: the files the census agent made that stay on the Mac Studio (not copied to
the laptop): every file of its two run directories, every pickle under var/levin-phase-floor-06m/census/, and every
other file there over 2 MB. New in 06m (after 06l's kept_list_06l.py). SHA-256, size in bytes, path (relative to
var/levin-phase-floor-06m/census/, or to var/runs/). Usage: python -B kept_list_06m.py OUT_TXT"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT.parents[2] / "var" / "runs" if (ROOT.parents[2] / "var" / "runs").exists() else ROOT.parents[1] / "runs"
RUN_DIRS = ["levin-phase-floor-06m-census-20261106T102153", "levin-phase-floor-06m-k0-20261106T101118"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


out = Path(sys.argv[1])
lines = ["# levin-phase-floor 06m census agent: files kept on the Mac Studio only (mps026298), not copied to the laptop.",
         f"# sha256  size_bytes  path. Written by scripts/kept_list_06m.py (sha256 {sha(Path(__file__))}).", ""]
lines.append("## pickles, and files over 2 MB, under var/levin-phase-floor-06m/census/")
n = tot = 0
for root, _d, files in sorted(os.walk(ROOT)):
    for name in sorted(files):
        p = Path(root) / name
        if p == out.resolve():
            continue
        if name.endswith(".pkl") or p.stat().st_size > 2 * 1024 * 1024:
            lines.append(f"{sha(p)}  {p.stat().st_size}  {p.relative_to(ROOT)}")
            n += 1
            tot += p.stat().st_size
lines.append(f"# {n} files, {tot} bytes")
for rd in RUN_DIRS:
    run = RUNS / rd
    lines += ["", f"## run directory var/runs/{rd}/ (every file)"]
    n = tot = 0
    for root, _d, files in sorted(os.walk(run)):
        for name in sorted(files):
            p = Path(root) / name
            lines.append(f"{sha(p)}  {p.stat().st_size}  {p.relative_to(RUNS)}")
            n += 1
            tot += p.stat().st_size
    lines.append(f"# {n} files, {tot} bytes")
out.write_text("\n".join(lines) + "\n")
print("\n".join(l for l in lines if l.startswith("#")))
