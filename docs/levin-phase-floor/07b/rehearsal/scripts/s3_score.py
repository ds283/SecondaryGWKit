"""S3 -- consistency (prompt 07b section 5 item 4), R31 against R30, over every QuadSourceIntegral row.

levin-phase-floor 07b, rehearsal agent, R31. 07a's rehearsal agent's s3_score.py (f12ebafd...), with the store tags
29 -> 30 and 30 -> 31 and the reader's module (tabulate_qsi_r30_r31.py) changed, and this docstring. Reads both stores
with sqlite mode=ro through tabulate_qsi_r30_r31.py's reader (06g's, verbatim). The rule: every row with finite
total_abserr in both has |total_31 - total_30| <= total_abserr_30 + total_abserr_31, or is one section 9 predicts to
move further (section 9.5: S3 names no row). Holds if no row violates it. Evaluated exactly, in fractions.Fraction on
the doubles (no rounding); the ratio |d|/(e30 + e31) is printed as a float.

    python -B s3_score.py <R30 primary> <R31 primary>
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabulate_qsi_r30_r31 import read  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a_path, b_path = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    tq = Path(__file__).resolve().parent / "tabulate_qsi_r30_r31.py"
    print(f"# module {tq} sha256 {sha256(tq)}")
    for p in (a_path, b_path):
        prim = Path(p)
        for f in sorted(prim.parent.glob(prim.stem + "*.sqlite")):
            print(f"# input {f} sha256 {sha256(f)}")
    A, _ = read(a_path)
    B, _ = read(b_path)
    assert set(A) == set(B)
    n = scored = viol = 0
    not_finite = []
    worst = None
    rows = []
    for key in sorted(A):
        n += 1
        t30, t31 = A[key]["total"], B[key]["total"]
        e30, e31 = A[key]["total_abserr"], B[key]["total_abserr"]
        if not all(isinstance(x, float) and math.isfinite(x) for x in (e30, e31, t30, t31)):
            not_finite.append((key, e30, e31, t30, t31))
            continue
        scored += 1
        d = abs(Fraction(t31) - Fraction(t30))
        s = Fraction(e30) + Fraction(e31)
        ok = d <= s
        viol += not ok
        r = float(d / s) if s else (0.0 if d == 0 else math.inf)
        if worst is None or r > worst[0]:
            worst = (r, key, t30.hex(), t31.hex(), e30.hex(), e31.hex())
        if d:
            rows.append({"key": list(key), "ratio": r, "holds": ok})
    print(f"rows {n}; scored (finite total_abserr and total in both) {scored}; not scored {len(not_finite)}")
    for x in not_finite:
        print(f"  not scored: {x}")
    print(f"rows with total moved: {len(rows)}; violations {viol}")
    print(f"largest |total_31 - total_30| / (abserr_30 + abserr_31): {worst[0]!r} at {worst[1]}; total {worst[2]} -> {worst[3]}, abserr {worst[4]} / {worst[5]}")
    print(f"S3 {'HOLDS' if viol == 0 else 'FAILS'}")
    dst = Path(__file__).resolve().parents[1] / "out" / "s3_scores.json"
    dst.write_text(json.dumps({"holds": viol == 0, "rows_scored": scored, "violations": viol, "moved": rows}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
