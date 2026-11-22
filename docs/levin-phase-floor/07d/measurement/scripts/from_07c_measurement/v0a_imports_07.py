"""levin-phase-floor prompt 07 section 6, Gate V0's first clause: the files of the parent tree p that the K0 run's
workers loaded (each worker's ``bye``: ``modules_own_tree``), against ``git diff --name-only 48cf87c0 91472dbe -- .
':!docs' ':!prompts'``. New in 07 (after 06g's gate_v_clause1.py). Read only.
Usage: python -B v0a_imports_07.py RUN_K0_DIR OUT_TXT
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import P_TREE, header, sha256  # noqa: E402


def main():
    run_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    files = set()
    n_workers = 0
    for wdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for line in open(wdir / "emit.jsonl"):
            m = json.loads(line)
            if m["kind"] == "bye":
                n_workers += 1
                for _name, f in m["modules_own_tree"]:
                    files.add(str(Path(f).resolve().relative_to(P_TREE)))
    diff = subprocess.run(["git", "--no-optional-locks", "-C", str(P_TREE), "diff", "--name-only", "48cf87c0",
                           "91472dbee1daba8374b0018db1195cf6c7898bbb", "--", ".", ":!docs", ":!prompts"],
                          capture_output=True, text=True, check=True).stdout.split()
    # main.py is read by the harness (test_main_plumbing reads it with ast), not imported; it is listed as a file too
    inter = sorted(set(diff) & (files | {"main.py"}))
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07: Gate V0 (a), the parent's files the K0 workers loaded")
    lines.append(f"# workers with a bye: {n_workers}; distinct files of p loaded: {len(files)}")
    lines.append(f"# git diff --name-only 48cf87c0 91472dbe -- . ':!docs' ':!prompts': {diff}")
    lines.append(f"# files both changed and loaded: {inter}")
    lines.append(f"# verdict: {'HOLDS (nothing the harness imports is changed)' if not inter else 'FAILS'}")
    lines += sorted(files)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if not inter else 1


if __name__ == "__main__":
    sys.exit(main())
