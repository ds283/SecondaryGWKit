"""levin-phase-floor prompt 07c Part A (section 2.1, "every variant is run on every call of section 2.1, at both rtols"):
07b's census records of the other-rtol counterparts of section 2.1's integrands that the populations hold at one rtol
only. New in 07c; the reading is extract_census_refs_07c.py's (imported), the wanted set built here:

- ``chase``: each chase integrand (state/populations_07c.json ``chase``) present at one rtol only: its call at the other
  rtol, sid ``x-<rtol>-<shard>-<serial>-<index>``;
- ``r5``: each R5 integrand (state/selection_r5.json) whose drawn calls are at one rtol only: its call at the other rtol,
  same sid form.

Writes state/census_wanted_07c_<label>.json, state/census_refs_07c_<label>.pkl and out/census_refs_07c_<label>.json.
Usage: PYTHONPATH=<tree> python -B extract_census_refs_extra_07c.py LABEL (chase|r5)
"""

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07B_SHA256, RUN07B, WIDER, header, sha256  # noqa: E402
from extract_census_refs_07c import load_blob  # noqa: E402

OTHER = {"1e-7": "1e-8", "1e-8": "1e-7"}


def main():
    label = sys.argv[1]
    wf = WIDER / "state" / f"census_wanted_07c_{label}.json"
    pk = WIDER / "state" / f"census_refs_07c_{label}.pkl"
    js = WIDER / "out" / f"census_refs_07c_{label}.json"
    for p in (wf, pk, js):
        if p.exists():
            raise SystemExit(f"{p} exists")
    inputs = []
    by_int = {}
    if label == "chase":
        pops_p = WIDER / "state" / "populations_07c.json"
        inputs.append(pops_p)
        for c in json.loads(pops_p.read_text())["chase"]["calls"]:
            p = c["unit"].split("-")
            by_int.setdefault((p[-2], p[-1], c["kind"], c["index"]), []).append(c["rtol"])
    elif label == "r5":
        sel_p = WIDER / "state" / "selection_r5.json"
        inputs.append(sel_p)
        for k, cs in json.loads(sel_p.read_text())["calls"].items():
            for c in cs:
                p = c["unit"].split("-")
                by_int.setdefault((p[-2], p[-1], c["kind"], c["index"]), []).append(c["rtol"])
    else:
        raise SystemExit("chase or r5")
    wanted = {}
    for (sh, se, kind, idx), rts in sorted(by_int.items()):
        if len(set(rts)) == 2:
            continue
        rt = OTHER[rts[0]]
        wanted[f"x-{rt}-{sh}-{se}-{idx}"] = {"set": f"{label}~o", "unit": f"h-{rt}-{sh}-{se}", "kind": kind, "index": idx,
                                            "counterpart_of": f"h-{rts[0]}-{sh}-{se}"}
    whdr = header(Path(__file__).resolve(), f"levin-phase-floor 07c Part A: the counterparts wanted ({label})", inputs)
    wf.write_text(json.dumps({"header": whdr, "wanted": wanted}, indent=1))
    print(f"wanted {len(wanted)}; wrote {wf} sha256 {sha256(wf)}", flush=True)
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
    if units - set(recs):
        raise SystemExit(f"units with no done record: {sorted(units - set(recs))}")
    out, index = {}, {}
    for sid, v in sorted(wanted.items()):
        d = recs[v["unit"]]
        key = "calls" if v["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == v["index"]]
        if len(c) != 1:
            raise SystemExit(f"{sid}: {len(c)} call records")
        prec = load_blob(RUN07B / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
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
                          num_range_rule_refusals=c0.get("num_range_rule_refusals"))
    if sha256(ck) != CHECKPOINT07B_SHA256:
        raise SystemExit(f"{ck} changed")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07c Part A: census records of the counterparts ({label})", [wf, ck])
    pk.write_bytes(pickle.dumps({"header": hdr, "calls": out}, protocol=5))
    js.write_text(json.dumps({"header": hdr, "pickle": str(pk), "pickle_sha256": sha256(pk), "calls": index}, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(out)} records; wrote {pk} sha256 {sha256(pk)}; {js} sha256 {sha256(js)}")


if __name__ == "__main__":
    main()
