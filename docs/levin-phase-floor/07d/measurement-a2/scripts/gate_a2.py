"""levin-phase-floor prompt 07d Part A' (section 9.2): the gate, read only. New in Part A'.

Checks and writes to OUT (refuses to overwrite):
- the worktree is at the amendment's SHA c2c57a36, clean, and its code is 07c's: git diff --stat 67b5af74 HEAD --
  AdaptiveLevin ComputeTargets LiouvilleGreen tools is empty; levin_quadrature.py's SHA-256 is d141a80b...;
- the main checkout's HEAD and status;
- Part A's records unchanged: every line of its KEPT_ON_MAC_STUDIO.txt re-hashed (SHA-256 and size);
- runregistrykit list: nothing running or stale;
- no other var/levin-phase-floor-07d/* directory but measurement/ and measurement-a2/.
Usage: PYTHONPATH=<tree> python -B gate_a2.py OUT
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import MAIN, PARTA, PY, ROOT, TREE, TREE_LQ, TREE_LQ_SHA, _git, header, sha256  # noqa: E402


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    L = header("levin-phase-floor 07d Part A': the gate", __file__, [PARTA / "KEPT_ON_MAC_STUDIO.txt"])
    ok = True
    head = _git(TREE, "rev-parse", "HEAD")
    L.append(f"worktree HEAD {head}")
    ok &= head.startswith("c2c57a36")
    st = _git(TREE, "status", "--short", "--ignored")
    L.append(f"worktree status --short --ignored: {'empty' if not st else st}")
    ok &= not st
    diff = _git(TREE, "diff", "--stat", "67b5af74", "HEAD", "--", "AdaptiveLevin", "ComputeTargets", "LiouvilleGreen", "tools")
    L.append(f"git diff --stat 67b5af74 HEAD -- AdaptiveLevin ComputeTargets LiouvilleGreen tools: {'empty' if not diff else diff}")
    ok &= not diff
    s = sha256(TREE_LQ)
    L.append(f"levin_quadrature.py sha256 {s} ({'equal' if s == TREE_LQ_SHA else 'DIFFERS'})")
    ok &= s == TREE_LQ_SHA
    L.append(f"main checkout HEAD {_git(MAIN, 'rev-parse', 'HEAD')}, status --short: {_git(MAIN, 'status', '--short') or 'empty'}")
    n = bad = 0
    for line in (PARTA / "KEPT_ON_MAC_STUDIO.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, size, rel = line.split(None, 2)
        p = MAIN / rel
        n += 1
        if not p.exists() or p.stat().st_size != int(size) or sha256(p) != h:
            bad += 1
            L.append(f"  Part A kept file differs: {rel}")
    L.append(f"Part A's KEPT_ON_MAC_STUDIO.txt: {n} files re-hashed, {bad} differ")
    ok &= bad == 0
    rr = subprocess.run([PY, "-B", "-m", "runregistrykit", "list"], capture_output=True, text=True, cwd=str(TREE))
    last = [x for x in rr.stdout.splitlines() if x.strip()][-1]
    # v2: the status is the second column of a run's row (v1 matched "running" inside a description)
    rows = [x.split() for x in rr.stdout.splitlines() if x.startswith("   ") and len(x.split()) > 2]
    live = [" ".join(x[:6]) for x in rows if x[0] == "!!" or x[1] in ("running", "stale", "foreign")]
    L.append(f"runregistrykit list: {last}; running or stale: {len(live)}")
    L += [f"  {x}" for x in live]
    ok &= not live
    others = sorted(p.name for p in (MAIN / "var" / "levin-phase-floor-07d").iterdir() if p.name not in ("measurement", ROOT.name))
    L.append(f"other var/levin-phase-floor-07d/* entries: {others or 'none'}")
    L.append(f"GATE {'HOLDS' if ok else 'FAILS'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
