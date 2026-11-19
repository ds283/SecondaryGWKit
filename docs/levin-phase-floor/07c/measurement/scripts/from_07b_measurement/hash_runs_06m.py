"""levin-phase-floor prompt 06m section 5: hash, read only, every file of route B's census run directory (06l's
"before") and of 06g's census run directory, and check them against the lists their prompts kept. New in 06m.

- Route B's run directory, levin-phase-floor-06l-census-routeB-20261105T100126: every file is checked by SHA-256 and
  size against docs/levin-phase-floor/06l/KEPT_ON_MAC_STUDIO.txt (its section for this run directory), and the file
  sets must be equal.
- 06g's run directory, levin-phase-floor-06g-census-20261101T114315: its checkpoint and its 16 region pickles are
  checked against docs/levin-phase-floor/06g/census/KEPT_ON_MAC_STUDIO.txt (that list names only those of the run's
  files); every other file is hashed for the before/after comparison.

Every file is opened 'rb'. Writes OUT_TXT: a provenance header, then one line per file "sha256 size run/relpath",
then the checks' verdicts. Exits 1 if a check fails.

Usage: PYTHONPATH=<tree> python -B hash_runs_06m.py LABEL OUT_TXT
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import (  # noqa: E402
    CHECKPOINT06G_SHA256,
    CHECKPOINT06L_B_SHA256,
    G_TREE,
    RUN06G,
    RUN06L_B,
    header,
    sha256,
)

KEPT06L = G_TREE / "docs" / "levin-phase-floor" / "06l" / "KEPT_ON_MAC_STUDIO.txt"
KEPT06G = G_TREE / "docs" / "levin-phase-floor" / "06g" / "census" / "KEPT_ON_MAC_STUDIO.txt"


def hash_dir(run):
    out = {}
    for root, _dirs, files in os.walk(run):
        for name in files:
            p = Path(root) / name
            out[str(p.relative_to(run.parent))] = (sha256(p), p.stat().st_size)
    return dict(sorted(out.items()))


def kept06l():
    want = {}
    sec = False
    for line in KEPT06L.read_text().splitlines():
        if line.startswith("## "):
            sec = RUN06L_B.name in line
            continue
        if sec and line and not line.startswith("#"):
            sha, size, rel = line.split(None, 2)
            want[rel] = (sha, int(size))
    return want


def kept06g():
    want = {}
    for line in KEPT06G.read_text().splitlines():
        if line and not line.startswith("#"):
            sha, size, path = line.split(None, 2)
            p = Path(path)
            if p.parent == RUN06G:
                want[str(p.relative_to(RUN06G.parent))] = (sha, int(size))
    return want


def main():
    label, out_p = sys.argv[1], Path(sys.argv[2])
    lines = header(Path(__file__).resolve(), f"levin-phase-floor 06m: route B's and 06g's census run directories, {label}",
                   [KEPT06L, KEPT06G])
    ok = True
    verdicts = []
    for run, kept, ck_sha in ((RUN06L_B, kept06l(), CHECKPOINT06L_B_SHA256), (RUN06G, kept06g(), CHECKPOINT06G_SHA256)):
        got = hash_dir(run)
        lines.append(f"# {run}: {len(got)} files, {sum(s for _, s in got.values())} bytes")
        for rel, (sha, size) in got.items():
            lines.append(f"{sha}  {size}  {rel}")
        ck = got.get(f"{run.name}/checkpoint.jsonl", (None, None))[0]
        verdicts.append(f"{run.name}: checkpoint sha256 {ck} {'EQUAL' if ck == ck_sha else 'DIFFERS FROM'} {ck_sha}")
        ok &= ck == ck_sha
        n_pkl = sum(1 for r in got if Path(r).name.startswith("regions_") and r.endswith(".pkl"))
        verdicts.append(f"{run.name}: {n_pkl} region pickles")
        ok &= n_pkl == 16
        bad = [r for r, v in kept.items() if got.get(r) != v]
        if run == RUN06L_B:
            extra = sorted(set(got) - set(kept))
            verdicts.append(f"{run.name}: against 06l's KEPT_ON_MAC_STUDIO.txt: {len(kept)} listed, {len(got)} present, "
                            f"{len(bad)} differing or missing {bad[:5]}, {len(extra)} not listed {extra[:5]}")
            ok &= not bad and not extra and len(kept) == 150
        else:
            verdicts.append(f"{run.name}: against 06g's census KEPT_ON_MAC_STUDIO.txt (checkpoint and region pickles): "
                            f"{len(kept)} listed, {len(bad)} differing or missing {bad[:5]}; {len(got)} files hashed")
            ok &= not bad and len(kept) == 17
    lines += ["# " + v for v in verdicts] + [f"# verdict: {'ALL HOLD' if ok else 'A CHECK FAILS'}"]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
