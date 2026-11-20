"""levin-phase-floor prompt 07c section 6 (the census agent): 07b's census run directory, 07b's K0 run directory, 07b's
census working records and Part A's records (var/levin-phase-floor-07c/measurement/, whose replay records Y8 reads),
hashed, read only. New in 07c, after 07b's hash_runs_07b.py (a record, not edited), whose form it keeps.

- **07b's census run directory** (prov.RUN07B): every file, against 07b's KEPT_ON_MAC_STUDIO.txt (in the tree h,
  docs/levin-phase-floor/07b/census/), section "run directory var/runs/levin-phase-floor-07b-census-20261117T010200/":
  150 files listed, every one present with its SHA-256 and size, none unlisted; the checkpoint's SHA-256 396784bc...;
  16 region pickles.
- **07b's K0 run directory** (prov.RUN07B_K0): against the same file's section for it (5 files).
- **07b's census working records** (prov.REC07B): every file hashed; the pickles and files over 2 MB listed in 07b's
  KEPT_ON_MAC_STUDIO.txt (its first section, 54 files) present with their SHA-256 and size.
- **Part A's records** (var/levin-phase-floor-07c/measurement/): every file hashed (no list to check against; before
  and after are compared).

Writes OUT_TXT (header, one line per file: sha256, size, path) and prints the verdicts. Exits 1 unless every check holds.
Usage: python -B hash_runs_07c.py LABEL OUT_TXT
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07B_SHA256, H_TREE, MAIN, REC07B, RUN07B, RUN07B_K0, header, sha256  # noqa: E402

KEPT07B = H_TREE / "docs" / "levin-phase-floor" / "07b" / "census" / "KEPT_ON_MAC_STUDIO.txt"
PART_A = MAIN / "var" / "levin-phase-floor-07c" / "measurement"


def hash_dir(root, rel_to):
    out = {}
    for r, _dirs, files in os.walk(root):
        for name in files:
            p = Path(r) / name
            out[str(p.relative_to(rel_to))] = (sha256(p), p.stat().st_size)
    return dict(sorted(out.items()))


def kept07b(section_marker):
    want = {}
    sec = False
    for line in KEPT07B.read_text().splitlines():
        if line.startswith("## "):
            sec = section_marker in line
            continue
        if sec and line and not line.startswith("#"):
            sha, size, rel = line.split(None, 2)
            want[rel] = (sha, int(size))
    return want


def main():
    label, out_p = sys.argv[1], Path(sys.argv[2])
    lines = header(Path(__file__).resolve(),
                   f"levin-phase-floor 07c: 07b's census and K0 run directories and working records, and Part A's records, "
                   f"{label}", [KEPT07B])
    ok = True
    verdicts = []
    for run, n_want, n_pkl_want in ((RUN07B, 150, 16), (RUN07B_K0, 5, 0)):
        kept = kept07b(f"run directory var/runs/{run.name}/")
        got = hash_dir(run, run.parent)
        lines.append(f"# {run}: {len(got)} files, {sum(s for _, s in got.values())} bytes")
        for rel, (sha, size) in got.items():
            lines.append(f"{sha}  {size}  {rel}")
        if run == RUN07B:
            ck = got.get(f"{run.name}/checkpoint.jsonl", (None, None))[0]
            verdicts.append(f"{run.name}: checkpoint sha256 {ck} {'EQUAL' if ck == CHECKPOINT07B_SHA256 else 'DIFFERS FROM'} "
                            f"{CHECKPOINT07B_SHA256}")
            ok &= ck == CHECKPOINT07B_SHA256
        n_pkl = sum(1 for r in got if Path(r).name.startswith("regions_") and r.endswith(".pkl"))
        verdicts.append(f"{run.name}: {n_pkl} region pickles")
        ok &= n_pkl == n_pkl_want
        bad = [r for r, v in kept.items() if got.get(r) != v]
        extra = sorted(set(got) - set(kept))
        verdicts.append(f"{run.name}: against 07b's KEPT_ON_MAC_STUDIO.txt: {len(kept)} listed, {len(got)} present, "
                        f"{len(bad)} differing or missing {bad[:5]}, {len(extra)} not listed {extra[:5]}")
        ok &= not bad and not extra and len(kept) == n_want
    kept_r = kept07b("under var/levin-phase-floor-07b/census/")
    got_r = hash_dir(REC07B, REC07B)
    lines.append(f"# {REC07B}: {len(got_r)} files, {sum(s for _, s in got_r.values())} bytes")
    for rel, (sha, size) in got_r.items():
        lines.append(f"{sha}  {size}  REC07B/{rel}")
    bad_r = [r for r, v in kept_r.items() if got_r.get(r) != v]
    verdicts.append(f"{REC07B}: against 07b's KEPT_ON_MAC_STUDIO.txt (pickles and files over 2 MB): {len(kept_r)} listed, "
                    f"{len(bad_r)} differing or missing {bad_r[:5]}; {len(got_r)} files hashed in all")
    ok &= not bad_r and len(kept_r) == 54
    got_a = hash_dir(PART_A, PART_A)
    lines.append(f"# {PART_A}: {len(got_a)} files, {sum(s for _, s in got_a.values())} bytes")
    for rel, (sha, size) in got_a.items():
        lines.append(f"{sha}  {size}  PARTA/{rel}")
    verdicts.append(f"{PART_A}: {len(got_a)} files hashed (compared before against after)")
    lines += ["# " + v for v in verdicts] + [f"# verdict: {'ALL HOLD' if ok else 'A CHECK FAILS'}"]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
