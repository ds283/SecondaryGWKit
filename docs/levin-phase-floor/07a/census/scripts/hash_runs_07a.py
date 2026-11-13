"""levin-phase-floor prompt 07a section 6: hash, read only, every file of 07's census run directory (07a's "before") and
every file of 07's census working records, and check them against the list 07 kept. New in 07a, after 07's
hash_runs_07.py (a record, not edited).

- 07's run directory, levin-phase-floor-07-census-20261108T171137: every file is checked by SHA-256 and size against
  docs/levin-phase-floor/07/census/KEPT_ON_MAC_STUDIO.txt (its section for this run directory, read in the tree h); the
  file sets must be equal (150 files), the checkpoint must be 03dfe786..., and there must be 16 region pickles.
- 07's working records, var/levin-phase-floor-07/census/: every file is hashed (for the before/after comparison), and
  the 152 files the same KEPT list names under its first section ("pickles, and files over 2 MB") are checked by
  SHA-256 and size.

Every file is opened 'rb'. Writes OUT_TXT: a provenance header, then one line per file "sha256 size path" (run files
relative to var/runs/, records files as "REC07/<relpath>"), then the checks' verdicts. Exits 1 if a check fails.

Usage: python -B hash_runs_07a.py LABEL OUT_TXT
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07_SHA256, H_TREE, REC07, RUN07, header, sha256  # noqa: E402

KEPT07 = H_TREE / "docs" / "levin-phase-floor" / "07" / "census" / "KEPT_ON_MAC_STUDIO.txt"


def hash_dir(root, rel_to):
    out = {}
    for r, _dirs, files in os.walk(root):
        for name in files:
            p = Path(r) / name
            out[str(p.relative_to(rel_to))] = (sha256(p), p.stat().st_size)
    return dict(sorted(out.items()))


def kept07(section_marker):
    want = {}
    sec = False
    for line in KEPT07.read_text().splitlines():
        if line.startswith("## "):
            sec = section_marker in line
            continue
        if sec and line and not line.startswith("#"):
            sha, size, rel = line.split(None, 2)
            want[rel] = (sha, int(size))
    return want


def main():
    label, out_p = sys.argv[1], Path(sys.argv[2])
    lines = header(Path(__file__).resolve(), f"levin-phase-floor 07a: 07's census run directory and working records, {label}",
                   [KEPT07])
    ok = True
    verdicts = []
    # the run directory
    kept = kept07(f"run directory var/runs/{RUN07.name}/")
    got = hash_dir(RUN07, RUN07.parent)
    lines.append(f"# {RUN07}: {len(got)} files, {sum(s for _, s in got.values())} bytes")
    for rel, (sha, size) in got.items():
        lines.append(f"{sha}  {size}  {rel}")
    ck = got.get(f"{RUN07.name}/checkpoint.jsonl", (None, None))[0]
    verdicts.append(f"{RUN07.name}: checkpoint sha256 {ck} {'EQUAL' if ck == CHECKPOINT07_SHA256 else 'DIFFERS FROM'} "
                    f"{CHECKPOINT07_SHA256}")
    ok &= ck == CHECKPOINT07_SHA256
    n_pkl = sum(1 for r in got if Path(r).name.startswith("regions_") and r.endswith(".pkl"))
    verdicts.append(f"{RUN07.name}: {n_pkl} region pickles")
    ok &= n_pkl == 16
    bad = [r for r, v in kept.items() if got.get(r) != v]
    extra = sorted(set(got) - set(kept))
    verdicts.append(f"{RUN07.name}: against 07's KEPT_ON_MAC_STUDIO.txt: {len(kept)} listed, {len(got)} present, "
                    f"{len(bad)} differing or missing {bad[:5]}, {len(extra)} not listed {extra[:5]}")
    ok &= not bad and not extra and len(kept) == 150
    # the working records
    kept_r = kept07("under var/levin-phase-floor-07/census/")
    got_r = hash_dir(REC07, REC07)
    lines.append(f"# {REC07}: {len(got_r)} files, {sum(s for _, s in got_r.values())} bytes")
    for rel, (sha, size) in got_r.items():
        lines.append(f"{sha}  {size}  REC07/{rel}")
    bad_r = [r for r, v in kept_r.items() if got_r.get(r) != v]
    verdicts.append(f"{REC07}: against 07's KEPT_ON_MAC_STUDIO.txt (pickles and files over 2 MB): {len(kept_r)} listed, "
                    f"{len(bad_r)} differing or missing {bad_r[:5]}; {len(got_r)} files hashed in all")
    ok &= not bad_r and len(kept_r) == 152
    lines += ["# " + v for v in verdicts] + [f"# verdict: {'ALL HOLD' if ok else 'A CHECK FAILS'}"]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
