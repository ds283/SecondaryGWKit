"""handover prompt 12, section 6.5 (supplement): QuadSourceIntegral values, R8 against R4.

    ./venv/bin/python -I var/handover-12/scripts/compare_qsi_extract.py <R4 csv dir> <R8 csv dir>

Reads only the readers' CSV output (extract_QuadSourceIntegral_data's integral_data.csv), never a
store. For every (k, q, r) triple that both extracts hold an integral_data.csv for (the readers
draw their triples unseeded, so the overlap is whatever it is), compares the value columns row by
row, matched on z_response. R4's store (handover 06) reproduced R3's fingerprint 39809dca, which
covers identities and value counts but not QuadSourceIntegral values, so this is the only record
of R3-era integral values on this host.
"""

import csv
import sys
from pathlib import Path

COLUMNS = ["total", "numeric_quad", "WKB_Levin", "analytic_rad"]


def triple(d: Path) -> str:
    return d.name.split("-k-serial")[0]


def load(path: Path):
    with open(path) as f:
        return {float(r["z_response"]): r for r in csv.DictReader(f)}


def main(r4: Path, r8: Path) -> int:
    old = {triple(d): d for d in r4.iterdir() if (d / "integral_data.csv").exists()}
    new = {triple(d): d for d in r8.iterdir() if (d / "integral_data.csv").exists()}
    common = sorted(set(old) & set(new))
    print(f"R4 triples with integral_data: {len(old)}; R8: {len(new)}; common: {len(common)}")
    for t in common:
        a, b = load(old[t] / "integral_data.csv"), load(new[t] / "integral_data.csv")
        zs = sorted(set(a) & set(b), reverse=True)
        print(f"{t}: R4 rows {len(a)}, R8 rows {len(b)}, common z_response {len(zs)}")
        for c in COLUMNS:
            identical = 0
            worst = (0.0, None)
            for z in zs:
                x, y = float(a[z][c]), float(b[z][c])
                if x == y:
                    identical += 1
                    continue
                rel = abs(y - x) / max(abs(x), abs(y))
                if rel > worst[0]:
                    worst = (rel, z)
            line = f"  {c:>13}: bit-identical {identical}/{len(zs)}"
            if worst[1] is not None:
                line += f"; largest |rel diff| {worst[0]:.3e} at z_response {worst[1]!r}"
            print(line)
            # the same differences scaled by the column's largest |value| over these z_response,
            # since a relative difference near a zero of the integral says little; five largest
            scale = max(max(abs(float(a[z][c])), abs(float(b[z][c]))) for z in zs)
            ranked = sorted(
                (
                    (abs(float(b[z][c]) - float(a[z][c])) / scale, z)
                    for z in zs
                    if scale > 0
                ),
                reverse=True,
            )[:5]
            if ranked and ranked[0][0] > 0:
                print(f"    max |value| over z_response: {scale:.6e}")
                for d, z in ranked:
                    print(
                        f"    z_response {z!r:>22}: R4 {float(a[z][c]): .10e}  R8 {float(b[z][c]): .10e}"
                        f"  |diff|/max|value| {d:.3e}"
                    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
