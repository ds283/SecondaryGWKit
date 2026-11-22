"""levin-phase-floor prompt 07d Part A, section 2.1: the 07c census records every replay is compared with. New in 07d,
after 07c Part A's extract_census_refs_07c.py (scripts/from_07c_measurement/, a record), reading 07c's census instead of
07b's and wanting what make_calls_07d.py lists.

Per wanted record: the checkpoint's call record (every key), the unit's budget, wall and pickle pointer, and from the
unit's pickle record (its SHA-256 checked) the call's ``regions`` entry (group) or ``other_regions`` entry (analytic) and
its ``guard_07`` entry. The checkpoint's SHA-256 is checked before and after.
Writes state/census_refs_07d[_NAME].pkl ({census sid: record}) and out/census_refs_07d[_NAME].json (the index).
Usage: PYTHONPATH=<tree>:<scripts> python -B extract_census_07d.py [NAME]
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07C_SHA256, RUN07C, WIDER, header, sha256  # noqa: E402


def load_blob(path, off, ln, sha):
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(ln)
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{path}@{off}: sha256 differs")
    import ray.cloudpickle as cp

    return cp.loads(b)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    suf = f"_{name}" if name else ""
    wf = WIDER / "state" / f"census_wanted_07d{suf}.json"
    pk = WIDER / "state" / f"census_refs_07d{suf}.pkl"
    js = WIDER / "out" / f"census_refs_07d{suf}.json"
    for p in (pk, js):
        if p.exists():
            raise SystemExit(f"{p} exists")
    wanted = json.loads(wf.read_text())["wanted"]
    ck = RUN07C / "checkpoint.jsonl"
    if sha256(ck) != CHECKPOINT07C_SHA256:
        raise SystemExit(f"{ck}: another SHA-256")
    units = {v["unit"] for v in wanted.values()}
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
        raise SystemExit(f"units with no done record: {missing}")
    out, index, pcache = {}, {}, {}
    for sid, v in sorted(wanted.items(), key=lambda kv: (recs[kv[1]["unit"]]["pickle"], recs[kv[1]["unit"]]["pickle_offset"])):
        d = recs[v["unit"]]
        key = "calls" if v["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == v["index"]]
        if len(c) != 1:
            raise SystemExit(f"{sid}: {len(c)} call records")
        pkey = (d["pickle"], d["pickle_offset"])
        if pkey not in pcache:
            pcache.clear()
            pcache[pkey] = load_blob(RUN07C / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
        prec = pcache[pkey]
        rkey = "regions" if v["kind"] == "group" else "other_regions"
        reg = [x for x in prec[rkey] if x["index"] == v["index"]]
        g7 = [x for x in prec["guard_07"] if x["index"] == v["index"]]
        if len(reg) != 1 or len(g7) != 1:
            raise SystemExit(f"{sid}: regions {len(reg)}, guard_07 {len(g7)}")
        out[sid] = dict(v, call=c[0], budget_s=d["budget_s"], unit_wall_s=d["wall_s"], regions=reg[0], guard_07=g7[0],
                        pickle=[str(RUN07C / d["pickle"]), d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]])
        c0 = c[0]
        index[sid] = dict(v, value=c0["value"]["hex"], abserr=c0["abserr"]["hex"], num_regions=c0["num_regions"],
                          max_depth=c0["max_depth"], converged=c0["converged"], atol=c0["atol"], rtol=c0["rtol"],
                          notify_label=c0["notify_label"], num_chase_stops=c0.get("num_chase_stops"),
                          num_chase_stopped_leaves=c0.get("num_chase_stopped_leaves"),
                          num_chase_noise_bisections=c0.get("num_chase_noise_bisections"))
    if sha256(ck) != CHECKPOINT07C_SHA256:
        raise SystemExit(f"{ck} changed")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07d Part A: 07c census records of the replays{suf}", [wf, ck])
    pk.write_bytes(pickle.dumps({"header": hdr, "calls": out}, protocol=5))
    js.write_text(json.dumps({"header": hdr, "pickle": str(pk), "pickle_sha256": sha256(pk), "calls": index}, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(out)} records from {len(recs)} units; wrote {pk} {sha256(pk)}; {js} {sha256(js)}")


if __name__ == "__main__":
    main()
