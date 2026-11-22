"""levin-phase-floor prompt 07b section 6 (the census agent): 07a's census run directory, 07a's census working records and
Part A's records (var/levin-phase-floor-07b/measurement/, whose replay records U8 reads), hashed, read only. New in 07b,
after 07a's hash_runs_07a.py (a record, not edited), whose form it keeps.

- **07a's census run directory** (prov.RUN07A): every file, against 07a's KEPT_ON_MAC_STUDIO.txt (in the tree h,
  docs/levin-phase-floor/07a/census/), section "run directory var/runs/levin-phase-floor-07a-census-20261112T132744/":
  149 files listed, every one present with its SHA-256 and size, none unlisted; the checkpoint's SHA-256 461df2ec...;
  16 region pickles.
- **07a's census working records** (prov.REC07A): every file hashed; the pickles and files over 2 MB listed in 07a's
  KEPT_ON_MAC_STUDIO.txt (its first section) present with their SHA-256 and size.
- **Part A's records** (var/levin-phase-floor-07b/measurement/): every file hashed (no list to check against; before
  and after are compared).

Writes OUT_TXT (header, one line per file: sha256, size, path) and prints the verdicts. Exits 1 unless every check holds.
Usage: python -B hash_runs_07b.py LABEL OUT_TXT
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07A_SHA256, H_TREE, MAIN, REC07A, RUN07A, header, sha256  # noqa: E402

KEPT07A = H_TREE / "docs" / "levin-phase-floor" / "07a" / "census" / "KEPT_ON_MAC_STUDIO.txt"
PART_A = MAIN / "var" / "levin-phase-floor-07b" / "measurement"


def hash_dir(root, rel_to):
    out = {}
    for r, _dirs, files in os.walk(root):
        for name in files:
            p = Path(r) / name
            out[str(p.relative_to(rel_to))] = (sha256(p), p.stat().st_size)
    return dict(sorted(out.items()))


def kept07a(section_marker):
    want = {}
    sec = False
    for line in KEPT07A.read_text().splitlines():
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
                   f"levin-phase-floor 07b: 07a's census run directory and working records, and Part A's records, {label}",
                   [KEPT07A])
    ok = True
    verdicts = []
    kept = kept07a(f"run directory var/runs/{RUN07A.name}/")
    got = hash_dir(RUN07A, RUN07A.parent)
    lines.append(f"# {RUN07A}: {len(got)} files, {sum(s for _, s in got.values())} bytes")
    for rel, (sha, size) in got.items():
        lines.append(f"{sha}  {size}  {rel}")
    ck = got.get(f"{RUN07A.name}/checkpoint.jsonl", (None, None))[0]
    verdicts.append(f"{RUN07A.name}: checkpoint sha256 {ck} {'EQUAL' if ck == CHECKPOINT07A_SHA256 else 'DIFFERS FROM'} "
                    f"{CHECKPOINT07A_SHA256}")
    ok &= ck == CHECKPOINT07A_SHA256
    n_pkl = sum(1 for r in got if Path(r).name.startswith("regions_") and r.endswith(".pkl"))
    verdicts.append(f"{RUN07A.name}: {n_pkl} region pickles")
    ok &= n_pkl == 16
    bad = [r for r, v in kept.items() if got.get(r) != v]
    extra = sorted(set(got) - set(kept))
    verdicts.append(f"{RUN07A.name}: against 07a's KEPT_ON_MAC_STUDIO.txt: {len(kept)} listed, {len(got)} present, "
                    f"{len(bad)} differing or missing {bad[:5]}, {len(extra)} not listed {extra[:5]}")
    ok &= not bad and not extra and len(kept) == 149
    kept_r = kept07a("under var/levin-phase-floor-07a/census/")
    got_r = hash_dir(REC07A, REC07A)
    lines.append(f"# {REC07A}: {len(got_r)} files, {sum(s for _, s in got_r.values())} bytes")
    for rel, (sha, size) in got_r.items():
        lines.append(f"{sha}  {size}  REC07A/{rel}")
    bad_r = [r for r, v in kept_r.items() if got_r.get(r) != v]
    verdicts.append(f"{REC07A}: against 07a's KEPT_ON_MAC_STUDIO.txt (pickles and files over 2 MB): {len(kept_r)} listed, "
                    f"{len(bad_r)} differing or missing {bad_r[:5]}; {len(got_r)} files hashed in all")
    ok &= not bad_r and len(kept_r) == 91
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
