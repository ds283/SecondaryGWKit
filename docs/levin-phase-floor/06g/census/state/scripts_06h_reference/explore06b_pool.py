"""levin-phase-floor prompt 06c section 5: what 06b's census holds that Gate X0 and X1's rule need
to know before the run. New in 06c. Read-only: reads 06a's and 06b's checkpoints and rows.json.

- **X2-type calls in 06b** (prompt 06c section 5, X2's premise): a Levin call (group, ``calls``;
  or analytic block, ``other_calls``) that 06b reported unconverged, with no region at depth_max
  (``max_depth < 20``), no phase-limited region (``n_phase_limited_regions == 0``) and a finite
  abserr. Counted per rtol and per unit, inside and outside 06a's 720 stationary-point rows (a 1e-7
  row with any group call whose 06a scan has ``n_sign_changes > 0``; analyse06b.py's reading).
- **X1's first stratum, before the run:** the 1e-7 group calls of that kind whose 06a scan has
  ``n_sign_changes == 0``, by (band, k_key). Whether they moved is known only after the run.
- **X0's determinism unit, by rule:** of the 1e-7 rows outside the stationary-point rows, the one
  whose 06b unit has the most X2-type calls (group and analytic), ties broken by rows.json's
  order. Its 06b wall time is printed.

Usage: python -B explore06b_pool.py
"""

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06A, RUN06B, WIDER, header, sha256  # noqa: E402

ROWS_JSON = WIDER / "state" / "rows.json"
DEPTH_MAX = 20


def read_checkpoint(run_dir, rtol=None):
    recs = {}
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if rtol is not None and f'"rtol": "{rtol}"' not in line:
                continue
            r = json.loads(line)
            recs[r["unit"]] = r["data"]
    return recs


def x2_type(c):
    ab = float.fromhex(c["abserr"]["hex"])
    return (
        not c["converged"]
        and c["max_depth"] < DEPTH_MAX
        and c["n_phase_limited_regions"] == 0
        and math.isfinite(ab)
    )


def main():
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06c: 06b's X2-type calls, X1's first stratum, X0's determinism unit",
        [ROWS_JSON, RUN06A / "checkpoint.jsonl", RUN06B / "checkpoint.jsonl"],
    ):
        print(l)
    rows = json.loads(ROWS_JSON.read_text())["rows"]
    order = {f"{r['shard']}-{r['serial']}": i for i, r in enumerate(rows)}
    by_key = {f"{r['shard']}-{r['serial']}": r for r in rows}
    A7 = read_checkpoint(RUN06A, "1e-7")
    sp_rows = set()
    scan06a = {}
    for uid, d in A7.items():
        key = "-".join(uid.rsplit("-", 2)[-2:])
        scan06a[key] = {s["index"]: s for s in (d.get("scan") or []) if s}
        if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
            sp_rows.add(key)
    print(f"\n06a's stationary-point rows (1e-7 rows with a group call whose scan has a sign change): {len(sp_rows)}")
    del A7
    for rtol in ("1e-7", "1e-8"):
        B = read_checkpoint(RUN06B, rtol)
        n = collections.Counter()
        per_unit = []
        pool = collections.Counter()
        status = collections.Counter(d["status"] for d in B.values())
        for uid, d in B.items():
            key = "-".join(uid.rsplit("-", 2)[-2:])
            sp = key in sp_rows
            g = [c for c in d["calls"] if x2_type(c)]
            o = [c for c in d["other_calls"] if x2_type(c)]
            n[("group", "sp" if sp else "non-sp")] += len(g)
            n[("analytic", "sp" if sp else "non-sp")] += len(o)
            n[("all group calls", "")] += len(d["calls"])
            n[("all analytic calls", "")] += len(d["other_calls"])
            if not sp:
                per_unit.append((len(g) + len(o), len(g), len(o), order[key], uid, d["wall_s"]))
            if rtol == "1e-7":
                R = by_key[key]
                for c in g:
                    s = scan06a[key].get(c["index"])
                    if s is not None and s["n_sign_changes"] == 0:
                        pool[(R["band"], R["k_key"])] += 1
        print(f"\n== rtol {rtol}: 06b units by status {dict(status)}")
        for k_, v in sorted(n.items()):
            print(f"   {k_[0]:<20} {k_[1]:<7} {v}")
        per_unit.sort(key=lambda t: (-t[0], t[3]))
        print("   units outside the stationary-point rows with the most X2-type calls (total, group, analytic, rows.json order, unit, 06b wall s):")
        for t in per_unit[:8]:
            print(f"      {t}")
        if rtol == "1e-7":
            print(f"   X0's determinism unit, by rule: {per_unit[0][4]} ({per_unit[0][0]} X2-type calls; 06b wall {per_unit[0][5]:.2f} s)")
            print(f"\n   X1's first stratum before the run (1e-7 group calls, X2-type in 06b, 06a scan no sign change), by (band, k_key): total {sum(pool.values())}")
            for k_, v in sorted(pool.items()):
                print(f"      {k_}: {v}")
        del B


if __name__ == "__main__":
    main()
