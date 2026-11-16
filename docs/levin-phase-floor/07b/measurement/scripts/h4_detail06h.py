"""levin-phase-floor prompt 06h section 5: the calls whose convergence changed between 06e's census and
this one (out/h4_changed_calls_06h.json, written by analyse06h.py), in detail: each call's checkpoint
record in both runs (value, abserr, converged, regions, the error breakdown, the budget loop's counters,
E_share, cc_floor, and under h declared_06h), and each returned leaf from both runs' pickles (start, end,
depth, type, abserr, phase_err, total_err, phase_limited, trusted), with h's per-leaf declared terms
(declared_terms_06h) beside them. New in 06h. Read only.
Usage: python -B h4_detail06h.py RUN_DIR"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06E, WIDER, header  # noqa: E402

KEYS = ("index", "notify_label", "value", "abserr", "converged", "phase_limited", "num_regions", "max_depth", "rtol", "atol",
        "relerr", "abserr_resolution", "abserr_roundoff", "abserr_fallback", "abserr_truncation", "num_untrusted_regions",
        "n_phase_limited_regions", "E_share", "E_largest_share", "E_unwidened", "leaves", "budget", "solves_by_purpose",
        "cc_floor", "declared_06h")
LEAF = ("start", "end", "depth", "type", "abserr", "phase_err", "total_err", "phase_limited", "trusted", "declared_resolution",
        "abserr_truncation")


def read_one(run_dir, uid):
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{uid}"' in line:
                r = json.loads(line)
                if r["unit"] == uid:
                    return r["data"]
    raise KeyError(uid)


def load_rec(run_dir, d):
    with open(Path(run_dir) / d["pickle"], "rb") as f:
        f.seek(d["pickle_offset"])
        blob = f.read(d["pickle_length"])
    assert hashlib.sha256(blob).hexdigest() == d["pickle_record_sha256"]
    return pickle.loads(blob)


def hx(v):
    return float(v).hex()


def main():
    run_dir = Path(sys.argv[1])
    src = WIDER / "out" / "h4_changed_calls_06h.json"
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06h: the calls whose convergence changed, in detail",
                    [src]):
        print(l)
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads((WIDER / "state" / "rows.json").read_text())["rows"]}
    for ch in json.loads(src.read_text()):
        uid, idx = ch["unit"], ch["index"]
        key = "-".join(uid.rsplit("-", 2)[-2:])
        r = rows[key]
        print(f"\n==== {uid} call {idx} ({ch['kind']}, {ch['label']}); row {key}: k {r['k']} q {r['q']} r {r['r']} "
              f"z_response {r['z_response']} band {r['band']}; stationary-point row {ch['stationary_point_row']}")
        for name, rd in (("06e", RUN06E), ("h", run_dir)):
            d = read_one(rd, uid)
            c = next(x for x in d["calls"] + d["other_calls"] if x["index"] == idx)
            print(f"\n-- {name}: unit total {d['result']['total']['hex']} abserr {d['result']['total_abserr']['hex']}; "
                  f"analytic_rad {d['result']['analytic_rad']['hex']}; analytic_06h {json.dumps(d.get('analytic_06h'))}")
            for k in KEYS:
                if k in c:
                    print(f"   {k}: {json.dumps(c[k])}")
            rec = load_rec(rd, d)
            ent = next(x for x in rec["regions"] + rec["other_regions"] if x["index"] == idx)
            n = len(ent["start"])
            print(f"   leaves ({n}):")
            for i in range(n):
                print("      " + ", ".join(f"{k}={ent[k][i] if k in ('depth', 'type', 'phase_limited', 'trusted') else hx(ent[k][i])}"
                                         for k in LEAF if k in ent))
            dt = next((x for x in (rec.get("declared_terms_06h") or []) if x["index"] == idx), None)
            if dt is not None:
                print("   declared terms by leaf (h):")
                for i in range(len(dt["branch"])):
                    print("      " + ", ".join(f"{k}={dt[k][i]!r}" for k in dt if k not in ("index", "notify_label", "errors")))
                print(f"   errors {dt['errors']}")


if __name__ == "__main__":
    main()
