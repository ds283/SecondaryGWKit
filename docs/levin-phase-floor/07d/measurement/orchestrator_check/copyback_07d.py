"""Orchestrator's scratch (not a record): 07d Part A's copy-back list, built on the Mac Studio, read-only.

Run with cwd the Mac Studio's main checkout. Prints three sections to stdout:
  COPY  sha256 size path   (files to copy, path relative to the main checkout)
  LEFT  sha256 size path   (the variant replay folders, left out by the orchestrator)
  INFO  summary lines and anomalies
"""
import hashlib, os, re
from pathlib import Path

ROOT = Path.cwd()
BASE = Path("var/levin-phase-floor-07d/measurement")
assert (ROOT / BASE).is_dir(), ROOT
MIB2 = 2 * 1024 * 1024
HEX = re.compile(r"^([0-9a-f]{64})\s+(\d+)\s+(\S.*?)\s*$")
REPLAY_DIRS = ("tree", "Ea", "Eg", "ET12", "ET20", "Ep", "T20")


def sha(p):
    h = hashlib.sha256()
    with open(ROOT / p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files_under(d):
    out = []
    for dp, dn, fn in os.walk(ROOT / d):
        for n in fn:
            p = Path(dp, n)
            if p.is_symlink():
                print(f"INFO anomaly: symlink skipped {p.relative_to(ROOT)}")
                continue
            out.append(p.relative_to(ROOT))
    return sorted(out)


anom = []
kept = {}
for line in (ROOT / BASE / "KEPT_ON_MAC_STUDIO.txt").read_text().splitlines():
    m = HEX.match(line)
    if m:
        kept[Path(m.group(3))] = (m.group(1), int(m.group(2)))
for p, (h, s) in kept.items():
    if not (ROOT / p).is_file():
        anom.append(f"kept list names a missing file: {p}")
    elif (ROOT / p).stat().st_size != s or sha(p) != h:
        anom.append(f"kept list hash/size differs: {p}")

runs = sorted(d.relative_to(ROOT) for d in (ROOT / "var/runs").iterdir()
              if d.is_dir() and d.name.startswith("levin-phase-floor-07d-"))
cands = files_under(BASE) + [p for r in runs for p in files_under(r)]
copy, left = [], []
for p in cands:
    size = (ROOT / p).stat().st_size
    rel = p.relative_to(BASE) if p.is_relative_to(BASE) else None
    if rel is not None and rel.parts[:2] and rel.parts[0] == "out" and len(rel.parts) > 2 \
            and rel.parts[1] == "replays" and rel.parts[2] in REPLAY_DIRS:
        if p not in kept:
            left.append((sha(p), size, p))
        continue
    if p in kept:
        continue
    if p.suffix in (".pkl", ".pickle") or size > MIB2:
        anom.append(f"pickle or over 2 MiB but not on the kept list: {p} {size}")
        continue
    copy.append((sha(p), size, p))

for h, s, p in copy:
    print(f"COPY {h} {s} {p}")
for h, s, p in left:
    print(f"LEFT {h} {s} {p}")
print(f"INFO runs: {[str(r) for r in runs]}")
print(f"INFO copy {len(copy)} files {sum(s for _, s, _ in copy)} bytes")
print(f"INFO left (replays) {len(left)} files {sum(s for _, s, _ in left)} bytes")
print(f"INFO kept list {len(kept)} files {sum(s for _, s in kept.values())} bytes, hashes checked")
print("INFO anomalies: " + ("none" if not anom else str(len(anom))))
for a in anom:
    print(f"INFO anomaly: {a}")
