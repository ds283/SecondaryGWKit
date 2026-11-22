"""levin-phase-floor prompt 07c Part A (section 2.1): the census records every replay is compared with. New in 07c, after
07b Part A's extract_census_refs_07b.py (scripts/from_07b_measurement/, a record), changed in what it reads and wants:

- the wanted records (written here first, to state/census_wanted_07c.json, before any record is read):
  * ``chase``: the 198 calls of state/populations_07c.json ``chase`` (07b's ``newly_unconverged``), sid
    ``c-<rtol>-<shard>-<serial>-<index>``;
  * ``moved``: the 324 of ``moved_beyond_sum`` (R5 is drawn from the 306 unreferenced among them), sid
    ``m-<rtol>-<shard>-<serial>-<g|a><index>``;
  * ``parta``: 07b Part A's 187 calls (its state/census_wanted_07b.json, every entry whose ``which`` is "h", i.e. a unit
    of 07a's census, whose unit name 07b's census shares), sid ``a:<07b sid>``; ``i3-34599-07`` (07's capture, 07's
    census) is not a test at 07b's code and is left out;
- every record from 07b's census (RUN07B, checkpoint 396784bc..., SHA-256 checked before and after). Per record: the
  checkpoint's call record (every key), the unit's budget, wall and pickle pointer, and from the unit's pickle record
  (its SHA-256 checked) the call's ``regions`` entry and its ``guard_07`` entry.

Writes state/census_wanted_07c.json, state/census_refs_07c.pkl ({sid: record}) and out/census_refs_07c.json (the index
without the arrays). Usage: PYTHONPATH=<tree> python -B extract_census_refs_07c.py
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07B_SHA256, MAIN, RUN07B, WIDER, header, sha256  # noqa: E402

P07B = MAIN / "var/levin-phase-floor-07b/measurement/state/census_wanted_07b.json"


def load_blob(path, off, ln, sha):
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(ln)
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{path}@{off}: sha256 differs")
    import ray.cloudpickle as cp

    return cp.loads(b)


def unit_parts(unit):
    p = unit.split("-")
    return "-".join(p[1:-2]), p[-2], p[-1]


def main():
    wf = WIDER / "state" / "census_wanted_07c.json"
    pk = WIDER / "state" / "census_refs_07c.pkl"
    js = WIDER / "out" / "census_refs_07c.json"
    for p in (wf, pk, js):
        if p.exists():
            raise SystemExit(f"{p} exists")
    pops_f = WIDER / "state" / "populations_07c.json"
    pops = json.loads(pops_f.read_text())
    wanted = {}
    for c in pops["chase"]["calls"]:
        rt, sh, se = unit_parts(c["unit"])
        wanted[f"c-{rt}-{sh}-{se}-{c['index']}"] = {"set": "chase", "unit": c["unit"], "kind": c["kind"], "index": c["index"]}
    for c in pops["moved_beyond_sum"]["calls"]:
        rt, sh, se = unit_parts(c["unit"])
        wanted[f"m-{rt}-{sh}-{se}-{c['kind'][0]}{c['index']}"] = {"set": "moved", "unit": c["unit"], "kind": c["kind"],
                                                                  "index": c["index"], "referenced": c["referenced"]}
    w07b = json.loads(P07B.read_text())["wanted"]
    for sid, v in w07b.items():
        if v["which"] != "h":
            continue
        wanted[f"a:{sid}"] = {"set": "parta", "unit": v["unit"], "kind": v["kind"], "index": v["index"]}
    whdr = header(Path(__file__).resolve(), "levin-phase-floor 07c Part A: the census records wanted", [pops_f, P07B])
    wf.write_text(json.dumps({"header": whdr, "wanted": wanted}, indent=1))
    print(f"wanted {len(wanted)} records; wrote {wf} sha256 {sha256(wf)}", flush=True)
    ck = RUN07B / "checkpoint.jsonl"
    if sha256(ck) != CHECKPOINT07B_SHA256:
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
    out, index = {}, {}
    pcache = {}
    for sid, v in sorted(wanted.items()):
        d = recs[v["unit"]]
        key = "calls" if v["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == v["index"]]
        if len(c) != 1:
            raise SystemExit(f"{sid}: {len(c)} call records")
        pkey = (d["pickle"], d["pickle_offset"])
        if pkey not in pcache:
            pcache.clear()
            pcache[pkey] = load_blob(RUN07B / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
        prec = pcache[pkey]
        rkey = "regions" if v["kind"] == "group" else "other_regions"
        reg = [x for x in prec[rkey] if x["index"] == v["index"]]
        g7 = [x for x in prec["guard_07"] if x["index"] == v["index"]]
        if len(reg) != 1 or len(g7) != 1:
            raise SystemExit(f"{sid}: regions {len(reg)}, guard_07 {len(g7)}")
        out[sid] = dict(v, call=c[0], budget_s=d["budget_s"], unit_wall_s=d["wall_s"], regions=reg[0], guard_07=g7[0],
                        pickle=[str(RUN07B / d["pickle"]), d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]])
        c0 = c[0]
        index[sid] = dict(v, value=c0["value"]["hex"], abserr=c0["abserr"]["hex"], num_regions=c0["num_regions"],
                          max_depth=c0["max_depth"], converged=c0["converged"], atol=c0["atol"], rtol=c0["rtol"],
                          elapsed=c0["elapsed"], notify_label=c0["notify_label"],
                          num_edge_rule_refusals=c0.get("num_edge_rule_refusals"),
                          num_range_rule_refusals=c0.get("num_range_rule_refusals"),
                          budget_s=d["budget_s"], unit_wall_s=d["wall_s"])
    if sha256(ck) != CHECKPOINT07B_SHA256:
        raise SystemExit(f"{ck} changed")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c Part A: the census records of the replays (07b's census)", [wf, ck])
    pk.write_bytes(pickle.dumps({"header": hdr, "calls": out}, protocol=5))
    js.write_text(json.dumps({"header": hdr, "pickle": str(pk), "pickle_sha256": sha256(pk), "calls": index}, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(out)} records from {len(recs)} units; wrote {pk} sha256 {sha256(pk)}; {js} sha256 {sha256(js)}")


if __name__ == "__main__":
    main()
