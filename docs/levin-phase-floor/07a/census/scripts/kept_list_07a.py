"""levin-phase-floor prompt 07a section 6: the list of files kept on the Mac Studio only (not copied to the laptop):
every file of the census, references and k0 run directories, and every pickle and every file over 2 MB under
var/levin-phase-floor-07a/census/, each with its SHA-256 and size. Copied from 07's kept_list_07.py (a record, not edited), changed only in its run directories, its paths and its title. Read only.
Usage: python -B kept_list_07a.py OUT_TXT"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, WIDER, sha256  # noqa: E402

RUNS = ["levin-phase-floor-07a-census-20261112T132744", "levin-phase-floor-07a-references-20261112T203555",  # prompt 07a
        "levin-phase-floor-07a-k0-20261112T130135"]  # prompt 07a


def main():
    out_p = Path(sys.argv[1])
    L = ["# levin-phase-floor 07a census agent: files kept on the Mac Studio only (mps026298), not copied to the laptop.",
         f"# sha256  size_bytes  path. Written by scripts/kept_list_07a.py (sha256 {sha256(Path(__file__))})."]
    L += ["", "## pickles, and files over 2 MB, under var/levin-phase-floor-07a/census/"]
    n = 0
    for root, _d, files in sorted(os.walk(WIDER)):
        for f in sorted(files):
            p = Path(root) / f
            if p == out_p.resolve():
                continue
            s = p.stat().st_size
            if f.endswith(".pkl") or s > 2 * 1024 * 1024:
                L.append(f"{sha256(p)}  {s}  {p.relative_to(WIDER)}")
                n += 1
    L.append(f"# {n} files")
    for run in RUNS:
        d = MAIN / "var" / "runs" / run
        L += ["", f"## run directory var/runs/{run}/ (every file)"]
        tot = cnt = 0
        for root, _d, files in sorted(os.walk(d)):
            for f in sorted(files):
                p = Path(root) / f
                s = p.stat().st_size
                L.append(f"{sha256(p)}  {s}  {p.relative_to(d.parent)}")
                tot += s
                cnt += 1
        L.append(f"# {cnt} files, {tot} bytes")
    out_p.write_text("\n".join(L) + "\n")
    print(f"wrote {out_p} ({len(L)} lines) sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
