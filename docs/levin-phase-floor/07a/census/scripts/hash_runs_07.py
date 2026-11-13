"""levin-phase-floor prompt 07 section 6: hash, read only, every file of 06m's census run directory (07's "before"),
and check them against the list 06m kept. New in 07, after 06m's hash_runs_06m.py (a record, not edited).

- 06m's run directory, levin-phase-floor-06m-census-20261106T102153: every file is checked by SHA-256 and size against
  docs/levin-phase-floor/06m/census/KEPT_ON_MAC_STUDIO.txt (its section for this run directory, read in the tree g),
  the file sets must be equal (150 files), the checkpoint must be ea5bc3ab..., and there must be 16 region pickles.

Every file is opened 'rb'. Writes OUT_TXT: a provenance header, then one line per file "sha256 size run/relpath", then
the checks' verdicts. Exits 1 if a check fails.

Usage: python -B hash_runs_07.py LABEL OUT_TXT
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06M_SHA256, G_TREE, RUN06M, header, sha256  # noqa: E402

KEPT06M = G_TREE / "docs" / "levin-phase-floor" / "06m" / "census" / "KEPT_ON_MAC_STUDIO.txt"


def hash_dir(run):
    out = {}
    for root, _dirs, files in os.walk(run):
        for name in files:
            p = Path(root) / name
            out[str(p.relative_to(run.parent))] = (sha256(p), p.stat().st_size)
    return dict(sorted(out.items()))


def kept06m():
    want = {}
    sec = False
    for line in KEPT06M.read_text().splitlines():
        if line.startswith("## "):
            sec = RUN06M.name in line
            continue
        if sec and line and not line.startswith("#"):
            sha, size, rel = line.split(None, 2)
            want[rel] = (sha, int(size))
    return want


def main():
    label, out_p = sys.argv[1], Path(sys.argv[2])
    lines = header(Path(__file__).resolve(), f"levin-phase-floor 07: 06m's census run directory, {label}", [KEPT06M])
    kept = kept06m()
    got = hash_dir(RUN06M)
    lines.append(f"# {RUN06M}: {len(got)} files, {sum(s for _, s in got.values())} bytes")
    for rel, (sha, size) in got.items():
        lines.append(f"{sha}  {size}  {rel}")
    ok = True
    verdicts = []
    ck = got.get(f"{RUN06M.name}/checkpoint.jsonl", (None, None))[0]
    verdicts.append(f"{RUN06M.name}: checkpoint sha256 {ck} {'EQUAL' if ck == CHECKPOINT06M_SHA256 else 'DIFFERS FROM'} "
                    f"{CHECKPOINT06M_SHA256}")
    ok &= ck == CHECKPOINT06M_SHA256
    n_pkl = sum(1 for r in got if Path(r).name.startswith("regions_") and r.endswith(".pkl"))
    verdicts.append(f"{RUN06M.name}: {n_pkl} region pickles")
    ok &= n_pkl == 16
    bad = [r for r, v in kept.items() if got.get(r) != v]
    extra = sorted(set(got) - set(kept))
    verdicts.append(f"{RUN06M.name}: against 06m's KEPT_ON_MAC_STUDIO.txt: {len(kept)} listed, {len(got)} present, "
                    f"{len(bad)} differing or missing {bad[:5]}, {len(extra)} not listed {extra[:5]}")
    ok &= not bad and not extra and len(kept) == 150
    lines += ["# " + v for v in verdicts] + [f"# verdict: {'ALL HOLD' if ok else 'A CHECK FAILS'}"]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
