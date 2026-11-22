"""levin-phase-floor 06a, after the run: two read-only cross-checks of the analysis. New in 06a.
(1) the rows deeper than 11 at 1e-8: how many are rows whose 1e-7 scan found an interior stationary
point; and where their deepest regions lie against the stationary point's abscissa (from the
pickles); (2) a sanity check of item 6's size: 05's own p units (B1's code, before 05's change)
against A3 v2's stored totals, from 05's run (read only), |p - stored| / stored total_abserr.
Usage: python -I -B checks_after.py RUN_DIR"""
import csv, gzip, hashlib, io, json, math, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import MAIN, WIDER, header

run = Path(sys.argv[1])
for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: cross-checks after the run"):
    print(l)
rows = list(csv.reader(l for l in gzip.open(WIDER / "out" / "census_rows.csv.gz", "rt") if not l.startswith("#")))
hdr = rows[0]; R = [dict(zip(hdr, r)) for r in rows[1:]]
sp = {r["name"] for r in R if r["rtol"] == "1e-7" and int(r["scan_calls_with_sign_change"]) > 0}
for rtol in ("1e-7", "1e-8"):
    deep = [r for r in R if r["rtol"] == rtol and r["h_depth"] not in ("", "None") and int(r["h_depth"]) > 11]
    print(f"rtol {rtol}: rows deeper than 11: {len(deep)}; of them with an interior stationary point at 1e-7: "
          f"{sum(1 for r in deep if r['name'] in sp)}")
# where the deepest regions lie, against the scan's sign-change abscissae (1e-7 and 1e-8)
units = {}
for line in open(run / "checkpoint.jsonl"):
    d = json.loads(line)["data"]; units[(d["rtol"], d["row"])] = d
def rec(d):
    with open(run / d["pickle"], "rb") as f:
        f.seek(d["pickle_offset"]); b = f.read(d["pickle_length"])
    assert hashlib.sha256(b).hexdigest() == d["pickle_record_sha256"]
    return pickle.loads(b)
dist = {"1e-7": [], "1e-8": []}
for rtol in ("1e-7", "1e-8"):
    for r in R:
        if r["rtol"] != rtol or r["h_depth"] in ("", "None") or int(r["h_depth"]) < 11 or r["name"] not in sp:
            continue
        s7 = rec(units[("1e-7", r["name"])])
        xs = [x for sc in s7["scan"] for x in (sc.get("sign_change_x") or [])]
        hs = [sc["h"] for sc in s7["scan"] if sc.get("n_sign_changes")]
        rr = rec(units[(rtol, r["name"])])
        for reg in rr["regions"]:
            m = reg["depth"] >= 11
            for a, b in zip(reg["start"][m], reg["end"][m]):
                mid = 0.5 * (a + b)
                dist[rtol].append(min(abs(mid - x) for x in xs) / max(hs) if xs else float("nan"))
for rtol, v in dist.items():
    v = np.array(v)
    print(f"rtol {rtol}: regions at depth >= 11 in stationary-point rows: {len(v)}; distance of the region's midpoint "
          f"from the nearest sign change, in grid spacings h: median {np.median(v):.3g}, 90th {np.percentile(v, 90):.3g}, max {v.max():.3g}")
# 05's p units against the stored totals
run05 = MAIN / "var" / "runs" / "levin-phase-floor-05-wider-20261026T020005"
q = []
for line in open(run05 / "checkpoint.jsonl"):
    r = json.loads(line); d = r["data"]
    if not r["unit"].startswith("p-1e-7-") or d.get("result") is None:
        continue
    p = float.fromhex(d["result"]["total"]["hex"]); s = float.fromhex(d["stored"]["total"]["hex"])
    e = float.fromhex(d["stored"]["total_abserr"]["hex"])
    if e > 0 and math.isfinite(e):
        q.append((abs(p - s) / e, d["row"], abs(p - s) / abs(s) if s else float("nan")))
q.sort(reverse=True)
print(f"05's p units (B1's code) against A3 v2's stored totals: {len(q)} rows with a finite stored error; "
      f"|p - stored|/stored_abserr median {np.median([x[0] for x in q]):.3g}, max {q[0][0]:.3g}; beyond 1x {sum(1 for x in q if x[0] > 1)}")
print("  largest:", [(n, f"{a:.3g}x", f"rel {b:.3g}") for a, n, b in q[:5]])
