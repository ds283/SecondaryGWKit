"""Final review (levin-phase-floor 07c Part C, §C.2): my own compact extract of one census checkpoint. Read only but for
OUT_JSONL.

usage: python -I -B extract.py {07c|07b|07a|07} OUT_JSONL

Reads every unit through my index (final/index<census>.json; the checkpoint's SHA-256 asserted against the index, each
line's SHA-256 checked) in sorted key order and writes one line per unit: key, unit, status, rtol, row, the result's
total, total_abserr (hex) and total_converged, and per Levin call (kind "group" = data.calls, "analytic" =
data.other_calls): index, notify_label, x_span, value and abserr (hex), converged, num_regions, max_depth, depth_max,
n_at_depth_max, the two refusal counts (07b and later), the three chase counts (07c), and guard_07's locates and guards
(07 and later). The first line is a header with the script's and the index's SHA-256s.
"""
import hashlib, json, sys
from pathlib import Path

M = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
FIN = M / "var/levin-phase-floor-07c/review/final"
RUNS = {"07c": ("levin-phase-floor-07c-census-20261119T201419", "33d94ce21077cd23f29da65ef967b64676db6722c081b999946bbc91c822a38e"),
        "07b": ("levin-phase-floor-07b-census-20261117T010200", "396784bcf2dff82267f37caf3b547c786aaac86e7abacd37afab72596cd88253"),
        "07a": ("levin-phase-floor-07a-census-20261112T132744", "461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b"),
        "07": ("levin-phase-floor-07-census-20261108T171137", "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10")}
CALL_KEYS = ("num_regions", "max_depth", "depth_max", "n_at_depth_max", "num_range_rule_refusals", "num_edge_rule_refusals",
             "num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections")


def hx(v):
    return None if v is None else (v["hex"] if isinstance(v, dict) else float(v).hex())


def main():
    which, out = sys.argv[1], Path(sys.argv[2])
    run, want = RUNS[which]
    ixp = FIN / f"index{which}.json"
    ix = json.load(open(ixp))
    assert ix["sha256"] == want and ix["checkpoint"].endswith(run + "/checkpoint.jsonl")
    n = 0
    with open(M / "var/runs" / run / "checkpoint.jsonl", "rb") as f, open(out, "w") as o:
        o.write(json.dumps({"header": True, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "index": str(ixp), "index_sha256": hashlib.sha256(ixp.read_bytes()).hexdigest(),
                            "checkpoint_sha256": want, "census": which}) + "\n")
        for key in sorted(ix["units"]):
            e = ix["units"][key]
            f.seek(e[0]); raw = f.read(e[1])
            assert hashlib.sha256(raw).hexdigest() == e[2], key
            d = json.loads(raw)["data"]
            r = d.get("result") or {}
            rec = {"key": key, "unit": d["unit"], "status": d["status"], "rtol": d["rtol"], "row": d["row"],
                   "total": hx(r.get("total")), "total_abserr": hx(r.get("total_abserr")),
                   "total_converged": r.get("total_converged"), "calls": []}
            for kind, lst in (("group", d.get("calls") or []), ("analytic", d.get("other_calls") or [])):
                for c in lst:
                    g = c.get("guard_07") or {}
                    x = {"kind": kind, "index": c["index"], "label": c["notify_label"], "x_span": c.get("x_span"),
                         "value": hx(c["value"]), "abserr": hx(c["abserr"]), "converged": c["converged"],
                         "locates": g.get("locates"), "guards": g.get("guards")}
                    for k in CALL_KEYS:
                        x[k] = c.get(k)
                    rec["calls"].append(x)
            o.write(json.dumps(rec) + "\n")
            n += 1
    print(which, "units", n)


if __name__ == "__main__":
    main()
