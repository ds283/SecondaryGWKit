"""levin-phase-floor prompt 07 section 6: the list of files kept on the Mac Studio only (not copied to the laptop):
every file of the census, references and k0 run directories, and every pickle and every file over 2 MB under
var/levin-phase-floor-07/census/, each with its SHA-256 and size. After 06m's kept_list_06m.py. Read only.
Usage: python -B kept_list_07.py OUT_TXT"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, WIDER, sha256  # noqa: E402

RUNS = ["levin-phase-floor-07-census-20261108T171137", "levin-phase-floor-07-references-20261109T005328",
        "levin-phase-floor-07-k0-20261108T135240"]


def main():
    out_p = Path(sys.argv[1])
    L = ["# levin-phase-floor 07 census agent: files kept on the Mac Studio only (mps026298), not copied to the laptop.",
         f"# sha256  size_bytes  path. Written by scripts/kept_list_07.py (sha256 {sha256(Path(__file__))})."]
    L += ["", "## pickles, and files over 2 MB, under var/levin-phase-floor-07/census/"]
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
