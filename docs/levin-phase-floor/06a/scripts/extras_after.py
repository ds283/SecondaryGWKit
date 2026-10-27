"""levin-phase-floor 06a, after the run: three read-only extracts for issues.md. New in 06a.
(1) the GkSourcePolicyData whose Levin_z differs from A3 v2's; (2) +inf total_abserr at h against
stored; (3) total_converged at h against stored, 1e-7. Usage: python -I -B extras_after.py RUN_DIR"""
import collections, csv, gzip, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import WIDER, header
run = Path(sys.argv[1])
for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: extracts for issues.md"):
    print(l)
zs = {}
for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]:
    zs[(r["k_key"].split(":")[1], r["z_response_serial"])] = (r["k"], r["z_response"], r["band"])
seen = set()
for p in sorted(run.glob("events_1e-7-*.jsonl")):
    for line in open(p):
        e = json.loads(line)["event"]
        if e["kind"] != "Gk":
            continue
        a = e["summary"]["GkSourcePolicyData"]
        if a["rerun"]["Levin_z"] != a["stored"]["Levin_z"]:
            k, z, b = zs[(str(e["k_serial"]), e["z_response_serial"])]
            lz = lambda h: None if h is None else float.fromhex(h)
            print(f"Levin_z differs: k {float(k):.4g} z_response {float(z):.5g} [{b}]: h {lz(a['rerun']['Levin_z'])!r} "
                  f"stored {lz(a['stored']['Levin_z'])!r}; type {a['rerun']['type']} quality {a['rerun']['quality']}")
rows = list(csv.reader(l for l in gzip.open(WIDER / "out" / "census_rows.csv.gz", "rt") if not l.startswith("#")))
R = [dict(zip(rows[0], r)) for r in rows[1:]]
for rtol in ("1e-7", "1e-8"):
    c = collections.Counter()
    cv = collections.Counter()
    for r in R:
        if r["rtol"] != rtol:
            continue
        hi = r["total_abserr_hex"] not in ("", "None") and math.isinf(float.fromhex(r["total_abserr_hex"]))
        si = r["stored_total_abserr_hex"] not in ("", "None") and math.isinf(float.fromhex(r["stored_total_abserr_hex"]))
        c[(hi, si)] += 1
        cv[(r["total_converged"], r["stored_total_converged"])] += 1
    print(f"rtol {rtol}: (+inf at h, +inf stored): {dict(c)}")
    print(f"rtol {rtol}: (total_converged at h, stored): {dict(cv)}")
    inf_new = [r for r in R if r["rtol"] == rtol and r["total_abserr_hex"] not in ("", "None") and math.isinf(float.fromhex(r["total_abserr_hex"]))]
    print(f"  +inf rows at h by band {dict(collections.Counter(r['band'] for r in inf_new))}, by class {dict(collections.Counter(r['class'] for r in inf_new))}, "
          f"with a Levin call {sum(1 for r in inf_new if int(r['n_calls']) > 0)}")
