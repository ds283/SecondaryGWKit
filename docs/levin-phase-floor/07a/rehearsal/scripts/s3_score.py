"""S3 -- consistency (prompt 07a section 5 item 4), R30 against R29, over every QuadSourceIntegral row.

levin-phase-floor 07a, rehearsal agent, R30. New. Reads both stores with sqlite mode=ro through tabulate_qsi_r29_r30.py's
reader (06g's, verbatim). The rule: every row with finite total_abserr in both has
|total_30 - total_29| <= total_abserr_29 + total_abserr_30. Holds if no row violates it. Evaluated exactly, in
fractions.Fraction on the doubles (no rounding); the ratio |d|/(e29 + e30) is printed as a float.

    ./venv/bin/python -I -B s3_score.py <R29 primary> <R30 primary>
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabulate_qsi_r29_r30 import read  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a_path, b_path = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    tq = Path(__file__).resolve().parent / "tabulate_qsi_r29_r30.py"
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
        t29, t30 = A[key]["total"], B[key]["total"]
        e29, e30 = A[key]["total_abserr"], B[key]["total_abserr"]
        if not all(isinstance(x, float) and math.isfinite(x) for x in (e29, e30, t29, t30)):
            not_finite.append((key, e29, e30, t29, t30))
            continue
        scored += 1
        d = abs(Fraction(t30) - Fraction(t29))
        s = Fraction(e29) + Fraction(e30)
        ok = d <= s
        viol += not ok
        r = float(d / s) if s else (0.0 if d == 0 else math.inf)
        if worst is None or r > worst[0]:
            worst = (r, key, t29.hex(), t30.hex(), e29.hex(), e30.hex())
        if d:
            rows.append({"key": list(key), "ratio": r, "holds": ok})
    print(f"rows {n}; scored (finite total_abserr and total in both) {scored}; not scored {len(not_finite)}")
    for x in not_finite:
        print(f"  not scored: {x}")
    print(f"rows with total moved: {len(rows)}; violations {viol}")
    print(f"largest |total_30 - total_29| / (abserr_29 + abserr_30): {worst[0]!r} at {worst[1]}; total {worst[2]} -> {worst[3]}, abserr {worst[4]} / {worst[5]}")
    print(f"S3 {'HOLDS' if viol == 0 else 'FAILS'}")
    dst = Path(__file__).resolve().parents[1] / "out" / "s3_scores.json"
    dst.write_text(json.dumps({"holds": viol == 0, "rows_scored": scored, "violations": viol, "moved": rows}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
