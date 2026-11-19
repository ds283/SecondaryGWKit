"""levin-phase-floor prompt 07b Part A (sections 2.1, 2.4): the census records every replay is compared with. New in 07b,
after 07a Part A's extract_census_refs_07a.py (a record), changed in what it reads: the wanted records are
state/census_wanted_07b.json's, each from 07a's census (``which`` "h", RUN07A, checkpoint 461df2ec...) or 07's (``"07"``,
RUN07, 03dfe786...), each checkpoint's SHA-256 checked before and after. Per record: the checkpoint's call record (every
key), the unit's budget and pickle pointer, and from the unit's pickle record (its SHA-256 checked) the call's
``regions`` entry and its ``guard_07`` entry.

Writes state/census_refs_07b.pkl ({sid: record}) and out/census_refs_07b.json (the index without the arrays).
Usage: PYTHONPATH=<tree>:<scripts> python -B extract_census_refs_07b.py
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07_SHA256, CHECKPOINT07A_SHA256, RUN07, RUN07A, WIDER, header, sha256  # noqa: E402

CENSUS = {"h": (RUN07A, CHECKPOINT07A_SHA256), "07": (RUN07, CHECKPOINT07_SHA256)}


def load_blob(path, off, ln, sha):
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(ln)
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{path}@{off}: sha256 differs")
    import ray.cloudpickle as cp

    return cp.loads(b)


def main():
    pk = WIDER / "state" / "census_refs_07b.pkl"
    js = WIDER / "out" / "census_refs_07b.json"
    for p in (pk, js):
        if p.exists():
            raise SystemExit(f"{p} exists")
    wf = WIDER / "state" / "census_wanted_07b.json"
    wanted = json.loads(wf.read_text())["wanted"]
    out, index = {}, {}
    for which, (run, want_sha) in CENSUS.items():
        ck = run / "checkpoint.jsonl"
        w = {s: v for s, v in wanted.items() if v["which"] == which}
        if not w:
            continue
        if sha256(ck) != want_sha:
            raise SystemExit(f"{ck}: another SHA-256")
        units = {v["unit"] for v in w.values()}
        recs = {}
        with open(ck) as f:
            for line in f:
                uid = line[10:60].split('"', 1)[0]
                if uid in units:
                    r = json.loads(line)
                    if r["data"].get("status") == "done":
                        if uid in recs:
                            raise SystemExit(f"{uid}: two done records")
                        recs[uid] = r["data"]
        missing = sorted(units - set(recs))
        if missing:
            raise SystemExit(f"{which}: units with no done record: {missing}")
        for sid, v in sorted(w.items()):
            d = recs[v["unit"]]
            key = "calls" if v["kind"] == "group" else "other_calls"
            c = [c for c in d[key] if c["index"] == v["index"]]
            if len(c) != 1:
                raise SystemExit(f"{sid}: {len(c)} call records")
            prec = load_blob(run / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
            rkey = "regions" if v["kind"] == "group" else "other_regions"
            reg = [x for x in prec[rkey] if x["index"] == v["index"]]
            g7 = [x for x in prec["guard_07"] if x["index"] == v["index"]]
            if len(reg) != 1 or len(g7) != 1:
                raise SystemExit(f"{sid}: regions {len(reg)}, guard_07 {len(g7)}")
            out[sid] = dict(v, call=c[0], budget_s=d["budget_s"], regions=reg[0], guard_07=g7[0],
                            pickle=[str(run / d["pickle"]), d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]])
            c0 = c[0]
            index[sid] = dict(v, value=c0["value"]["hex"], abserr=c0["abserr"]["hex"], num_regions=c0["num_regions"],
                              max_depth=c0["max_depth"], converged=c0["converged"], atol=c0["atol"], rtol=c0["rtol"],
                              elapsed=c0["elapsed"], notify_label=c0["notify_label"])
        if sha256(ck) != want_sha:
            raise SystemExit(f"{ck} changed")
    if set(out) != set(wanted):
        raise SystemExit("not every wanted record was found")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: the census records of the replays",
                 [wf, RUN07A / "checkpoint.jsonl", RUN07 / "checkpoint.jsonl"])
    pk.write_bytes(pickle.dumps({"header": hdr, "calls": out}, protocol=5))
    js.write_text(json.dumps({"header": hdr, "pickle": str(pk), "pickle_sha256": sha256(pk), "calls": index}, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(out)} records; wrote {pk} sha256 {sha256(pk)}; {js} sha256 {sha256(js)}")


if __name__ == "__main__":
    main()
