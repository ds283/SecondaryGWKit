"""S3 -- consistency (prompt 07c section 5 item 4), R32 against R31, over every QuadSourceIntegral row.

levin-phase-floor 07c, rehearsal agent, R32. 07b's rehearsal agent's s3_score.py (f6396428...), with the store tags
30 -> 31 and 31 -> 32 and the reader's module (tabulate_qsi_r31_r32.py) changed, section 9.5's named row added, and
this docstring (by gen/make_s3_score.py). Reads both stores with sqlite mode=ro through tabulate_qsi_r31_r32.py's
reader (06g's, verbatim). The rule: every row with finite total_abserr in both has
|total_32 - total_31| <= total_abserr_31 + total_abserr_32, or is one section 9 predicts to move further. Section 9.5
names one row, R31's record G+q+r at (k, q, r) = (1e5, 1e6, 1e6), z = 144627.7 (predicted to converge at 86
regions, with total_converged possibly false -> true); it is listed whether or not it is beyond the bound. Holds if no
row other than a named one violates it. Evaluated exactly, in fractions.Fraction on the doubles (no rounding); the
ratio |d|/(e31 + e32) is printed as a float.

    python -B s3_score.py <R31 primary> <R32 primary>
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabulate_qsi_r31_r32 import read  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


NAMED = {(100000.0, 1000000.0, 1000000.0, 144627.70674226896)}  # prompt 07c section 9.5 (S3's named row)


def main():
    a_path, b_path = sys.argv[1:3]
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    tq = Path(__file__).resolve().parent / "tabulate_qsi_r31_r32.py"
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
    named = []  # prompt 07c
    for key in sorted(A):
        n += 1
        t31, t32 = A[key]["total"], B[key]["total"]
        e31, e32 = A[key]["total_abserr"], B[key]["total_abserr"]
        if not all(isinstance(x, float) and math.isfinite(x) for x in (e31, e32, t31, t32)):
            not_finite.append((key, e31, e32, t31, t32))
            continue
        scored += 1
        d = abs(Fraction(t32) - Fraction(t31))
        s = Fraction(e31) + Fraction(e32)
        ok = d <= s
        if key in NAMED:  # prompt 07c: a named row is listed, and does not count as a violation
            named.append({"key": list(key), "ratio": float(d / s) if s else 0.0, "within_bound": ok, "t31": t31.hex(), "t32": t32.hex(), "e31": e31.hex(), "e32": e32.hex()})
        viol += (not ok) and key not in NAMED
        r = float(d / s) if s else (0.0 if d == 0 else math.inf)
        if worst is None or r > worst[0]:
            worst = (r, key, t31.hex(), t32.hex(), e31.hex(), e32.hex())
        if d:
            rows.append({"key": list(key), "ratio": r, "holds": ok})
    print(f"rows {n}; scored (finite total_abserr and total in both) {scored}; not scored {len(not_finite)}")
    for x in not_finite:
        print(f"  not scored: {x}")
    print(f"rows with total moved: {len(rows)}; violations {viol}")
    print(f"largest |total_32 - total_31| / (abserr_31 + abserr_32): {worst[0]!r} at {worst[1]}; total {worst[2]} -> {worst[3]}, abserr {worst[4]} / {worst[5]}")
    for x in named:  # prompt 07c
        print(f"  section 9.5's named row {x['key']}: |d|/(e31 + e32) {x['ratio']!r}, within the bound {x['within_bound']}; total {x['t31']} -> {x['t32']}, abserr {x['e31']} / {x['e32']}")
    print(f"S3 {'HOLDS' if viol == 0 else 'FAILS'}")
    dst = Path(__file__).resolve().parents[1] / "out" / "s3_scores.json"
    dst.write_text(json.dumps({"holds": viol == 0, "rows_scored": scored, "violations": viol, "moved": rows, "named_9_5": named}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
