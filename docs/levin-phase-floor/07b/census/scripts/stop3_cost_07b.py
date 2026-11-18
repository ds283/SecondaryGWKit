"""levin-phase-floor prompt 07b section 6, planned stop 3 (descriptive only): the census's wall, unit walls and regions
against 07a's census, the duplicate census calls among U8's ids, and the over-budget records. New in 07b. Read only.
Usage: python -B stop3_cost_07b.py RUN_DIR CALLS_JSONL U8_JSON OUT_JSON
"""
import collections
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN07A, header, sha256  # noqa: E402


def main():
    run_dir, calls_p, u8_p, out_p = map(Path, sys.argv[1:5])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b: planned stop 3, cost against 07a's census",
                 [calls_p, u8_p, run_dir / "checkpoint.jsonl"])
    out = {"header": hdr}
    for name, rd in (("07b", run_dir), ("07a", RUN07A)):
        man = json.loads((rd / "manifest.json").read_text())
        st = json.loads((rd / "status.json").read_text())
        t0 = datetime.datetime.fromisoformat(man["created"]).timestamp()
        t1 = datetime.datetime.fromisoformat(st["heartbeat"]).timestamp()
        walls = collections.Counter()
        n_ob, n = 0, 0
        for line in open(rd / "checkpoint.jsonl"):
            r = json.loads(line)
            d = r["data"]
            n += 1
            n_ob += d["status"] == "over budget"
            walls[d["rtol"]] += d.get("wall_s") or 0.0
        out[name] = {"created": man["created"], "last_heartbeat": st["heartbeat"], "wall_s": t1 - t0, "n_records": n,
                     "n_over_budget": n_ob, "unit_wall_sum_s": dict(walls)}
    reg = collections.Counter()
    for line in open(calls_p):
        c = json.loads(line)
        k = (c["rtol"], c["kind"])
        reg[k + ("07b",)] += c["num_regions"][0] or 0
        reg[k + ("07a",)] += c["num_regions"][1] or 0
    out["regions"] = {f"{a}|{b}": {"07b": reg[(a, b, "07b")], "07a": reg[(a, b, "07a")],
                                   "ratio": reg[(a, b, "07b")] / reg[(a, b, "07a")]}
                      for a, b in sorted({(k[0], k[1]) for k in reg})}
    tot = (sum(v for k, v in reg.items() if k[2] == "07b"), sum(v for k, v in reg.items() if k[2] == "07a"))
    out["regions_total"] = {"07b": tot[0], "07a": tot[1], "ratio": tot[0] / tot[1]}
    out["wall_ratio"] = out["07b"]["wall_s"] / out["07a"]["wall_s"]
    u8 = json.loads(u8_p.read_text())["calls"]
    by = collections.defaultdict(list)
    for sid, r in u8.items():
        by[(r["unit"], r["kind"], r["index"])].append(sid)
    out["u8_shared_census_calls"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in by.items() if len(v) > 1}
    out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
