"""levin-phase-floor prompt 07a section 6, W5 (new in 07a; read only): the forced solves whose V took the node-partition
sum (a window piece, or a window leaf's halves and quarters, all forced to the nested pair, whose V routes nothing), by
rtol, kind, the call's located points' kinds, and the solve's role (from the pickle's arrays); and the calls' scanned
sign changes. Usage: python -B w5_forced_detail_07a.py RUN_DIR OUT_JSON
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

UNITS = WIDER / "state" / "score_07a_units.json"


def main():
    import ray.cloudpickle as cp

    run, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: W5's forced solves on the node-partition sum", [UNITS])
    U = json.loads(UNITS.read_text())["units"]
    offs, off = {}, 0
    with open(run / "checkpoint.jsonl", "rb") as f:
        for line in f:
            offs[line[10:60].decode().split('"', 1)[0]] = (off, len(line))
            off += len(line)
    by, roles, calls_by = Counter(), Counter(), Counter()
    examples = []
    for u in U:
        cs = [c for c in u.get("calls") or [] if ((c.get("route") or {}).get("partition") or {}).get("forced")]
        if not cs:
            continue
        with open(run / "checkpoint.jsonl", "rb") as f:
            f.seek(offs[u["unit"]][0])
            d = json.loads(f.read(offs[u["unit"]][1]))["data"]
        with open(run / d["pickle"], "rb") as f:
            f.seek(d["pickle_offset"])
            blob = f.read(d["pickle_length"])
        assert hashlib.sha256(blob).hexdigest() == d["pickle_record_sha256"]
        rec = {e["index"]: e for e in cp.loads(blob)["route_07a"]}
        for c in cs:
            pk = "+".join(sorted(Counter(p["kind"] for p in c.get("points") or []))) or "none"
            calls_by[f"{u['rtol']}|{c['kind']}|points={pk}"] += 1
            r = rec[c["index"]]["record"]
            fl, ro = r["flags"], r["role"]
            for i in range(len(fl)):
                if (fl[i] & 1) and (fl[i] & 8):
                    roles[f"{r['roles'][ro[i]]}|made={bool(fl[i] & 2)}"] += 1
                    by[f"{u['rtol']}|{c['kind']}|points={pk}"] += 1
                    if len(examples) < 10:
                        examples.append({"unit": u["unit"], "index": c["index"], "a": float(r["a"][i]), "b": float(r["b"][i]),
                                         "V": float(r["V"][i]), "est_07": float(r["est_07"][i]), "role": r["roles"][ro[i]],
                                         "n_sign_changes_scan": (c.get("scan") or {}).get("n_sign_changes")})
    doc = {"header": hdr, "solves_by": dict(by), "solves_by_role": dict(roles), "calls_by": dict(calls_by),
           "n_solves": sum(by.values()), "n_calls": sum(calls_by.values()), "examples": examples}
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
