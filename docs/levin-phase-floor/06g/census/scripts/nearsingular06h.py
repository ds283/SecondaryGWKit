"""levin-phase-floor prompt 06h section 5, descriptive: does A3 v2 reach the near-singular class of the
LiouvilleGreen draws (a sign group of the three-Bessel integral whose |K| = |k + e_q q c_s + e_r r c_s| is
small against its coefficients)? state/predictions.md Part B: for each row, c_s = sqrt((1 - b)/(1 + b)/3)
with the row's b (each h 1e-7 unit's pickle record, result_full["b"], SHA-checked), K for the three groups
with a minus sign (+,+,-), (+,-,+), (+,-,-) (the (+,+,+) group cannot cancel), and
rel = |K| / (k + q c_s + r c_s). Reports its distribution over the rows, and every row with rel <= 1e-3,
with its analytic-block calls' convergence under 06e and h and its oracle's second-reading ratio
(out/census_rows_06h.csv.gz, out/h4_changed_calls_06h.json). New in 06h. Read only.
Usage: python -B nearsingular06h.py RUN_DIR"""
import collections
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header  # noqa: E402

THRESH = 1e-3


def q(vals, ps=(0.0, 0.001, 0.01, 0.1, 0.5, 1.0)):
    v = sorted(vals)
    return ", ".join(f"p{p*100:g} {v[min(len(v)-1, int(p*(len(v)-1)))]:.3g}" for p in ps)


def main():
    run_dir = Path(sys.argv[1])
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06h: the near-singular class in A3 v2",
                    [run_dir / "checkpoint.jsonl", WIDER / "out" / "census_rows_06h.csv.gz", WIDER / "out" / "h4_changed_calls_06h.json"]):
        print(l)
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]}
    bs = {}
    handles = {}
    with open(run_dir / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            r = json.loads(line)
            d = r["data"]
            if d["rtol"] != "1e-7":
                continue
            h = handles.get(d["pickle"])
            if h is None:
                h = handles[d["pickle"]] = open(run_dir / d["pickle"], "rb")
            h.seek(d["pickle_offset"])
            blob = h.read(d["pickle_length"])
            assert hashlib.sha256(blob).hexdigest() == d["pickle_record_sha256"]
            rec = pickle.loads(blob)
            bs[f"{d['shard']}-{d['serial']}"] = (float.fromhex(rec["result_full"]["b"]["hex"]), len(d["other_calls"]))
    for h in handles.values():
        h.close()
    print(f"\nrows read: {len(bs)}; distinct b: {sorted(collections.Counter(b for b, _ in bs.values()).items())[:10]}")
    rel = {}
    for key, (b, n_an) in bs.items():
        r = rows[key]
        k, qq, rr = float(r["k"]), float(r["q"]), float(r["r"])
        cs = math.sqrt((1.0 - b) / (1.0 + b) / 3.0)
        den = k + qq * cs + rr * cs
        ks = {"(+,+,-)": k + qq * cs - rr * cs, "(+,-,+)": k - qq * cs + rr * cs, "(+,-,-)": k - qq * cs - rr * cs}
        g, K = min(ks.items(), key=lambda t: abs(t[1]))
        rel[key] = (abs(K) / den, g, n_an)
    with_an = [v[0] for v in rel.values() if v[2] > 0]
    print(f"min over the three groups of |K|/(k + q c_s + r c_s), rows with analytic Levin calls ({len(with_an)}): {q(with_an)}")
    print(f"all rows ({len(rel)}): {q([v[0] for v in rel.values()])}")
    near = sorted((v[0], key, v[1], v[2]) for key, v in rel.items() if v[0] <= THRESH)
    print(f"rows with rel <= {THRESH:g}: {len(near)}")
    csvr = {}
    with gzip.open(WIDER / "out" / "census_rows_06h.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            csvr[row["unit"]] = row
    changed = json.loads((WIDER / "out" / "h4_changed_calls_06h.json").read_text())
    for rv, key, g, n_an in near:
        line = [f"   {key} rel {rv:.3g} group {g} analytic calls {n_an} band {rows[key]['band']}"]
        for rtol in ("1e-7", "1e-8"):
            u = csvr.get(f"g-{rtol}-{key}")
            ch = [c for c in changed if c["unit"] == f"g-{rtol}-{key}"]
            if u:
                ah, ae, eh, ee = (float.fromhex(u[x]) for x in ("analytic_rad_h", "analytic_rad_06e", "analytic_abserr_h", "analytic_abserr_06e_counterfactual"))
                line.append(f"{rtol}: |da|/(e_06e+e_h) {abs(ah-ae)/(eh+ee):.3g}, oracle converged h {u['analytic_converged_h']} 06e {u['analytic_converged_06e_counterfactual']}, "
                            f"calls changed {[(c['index'], 'lost' if c.get('lost') else 'gained') for c in ch]}")
        print("; ".join(line))
    key = "3-1072"
    print(f"\nthe H4 call's row {key}: rel {rel[key][0]:.3g} (group {rel[key][1]}); b {bs[key][0]!r}")


if __name__ == "__main__":
    main()
