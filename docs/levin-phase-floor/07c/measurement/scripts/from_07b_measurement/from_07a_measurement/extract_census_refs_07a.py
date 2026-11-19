"""levin-phase-floor prompt 07a Part A (section 2.1): the census records every replay is compared with. New in 07a.

For every call of section 2.1 (V4's 25, the two pairs at both rtols, and the 396 class-(c) rows at 1e-8), reads
07's census g (``refs_common.CENSUS07``, checkpoint SHA-256 checked before and after) once, and keeps, per call:
the checkpoint's call record (every key; ``value``/``abserr`` by hex), the unit's budget and pickle pointer, and
from the unit's pickle record (its SHA-256 checked) the call's ``regions`` entry and its ``guard_07`` entry
(section 1.5's result entries in full). The calls:
- V4: ``state/v4_capture_selection.json`` of 07 (``g_unit``, ``index``, ``kind``), sid v00..v23 and cal_z12;
- the pairs: rows 3-14501 and 3-35006, group call index 1, at 1e-7 and 1e-8 (sid p14501_7, p14501_8, ...);
- the 396: ``state/xcheck_capture_selection.json`` of 07, each row's call at **1e-8** (``g-1e-8-<row>``, index 1),
  sid as 07's (x000..x395). 07 captured 390 of them at 1e-7 and 6 at 1e-8 (the integrand is the same at both
  rtols); section 2.1 item 3 names the calls at 1e-8, so the 1e-8 record is the one compared.

Writes state/census_refs.pkl ({sid: record}) and out/census_refs.json (the index without the arrays), each with
a provenance header. Read-only outside them. Usage: PYTHONPATH=<tree>:<this dir> python -B extract_census_refs_07a.py
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header, sha256  # noqa: E402

C07 = RC.MAIN / "var" / "levin-phase-floor-07" / "census" / "state"
V4SEL = C07 / "v4_capture_selection.json"
XSEL = C07 / "xcheck_capture_selection.json"
OUT = HERE.parent


def wanted():
    w = {}
    for e in json.loads(V4SEL.read_text()):
        w[e["sid"]] = {"set": "v4", "unit": e["g_unit"], "index": e["index"], "kind": e["kind"]}
    for row in ("3-14501", "3-35006"):
        for rt in ("1e-7", "1e-8"):
            w[f"p{row.split('-')[1]}_{rt[-1]}"] = {"set": "pairs", "unit": f"g-{rt}-{row}", "index": 1, "kind": "group",
                                                   "row": row}
    for e in json.loads(XSEL.read_text()):
        p = e["unit"].split("-")
        w[e["sid"]] = {"set": "x396", "unit": f"g-1e-8-{p[3]}-{p[4]}", "index": e["index"], "kind": e["kind"],
                       "captured_unit": e["unit"], "class": e["class"]}
    return w


def load_blob(path, off, ln, sha):
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(ln)
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{path}@{off}: sha256 differs")
    import ray.cloudpickle as cp

    return cp.loads(b)


def main():
    ck = RC.CENSUS07 / "checkpoint.jsonl"
    if sha256(ck) != RC.CENSUS07_CHECKPOINT_SHA256:
        raise SystemExit("07's census checkpoint has another SHA-256")
    w = wanted()
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
        raise SystemExit(f"units with no done record: {missing}")
    out, index = {}, {}
    for sid, v in sorted(w.items()):
        d = recs[v["unit"]]
        key = "calls" if v["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == v["index"]]
        if len(c) != 1:
            raise SystemExit(f"{sid}: {len(c)} call records")
        prec = load_blob(RC.CENSUS07 / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
        rkey = "regions" if v["kind"] == "group" else "other_regions"
        reg = [x for x in prec[rkey] if x["index"] == v["index"]]
        g7 = [x for x in prec["guard_07"] if x["index"] == v["index"]]
        if len(reg) != 1 or len(g7) != 1:
            raise SystemExit(f"{sid}: regions {len(reg)}, guard_07 {len(g7)}")
        out[sid] = dict(v, call=c[0], budget_s=d["budget_s"], regions=reg[0], guard_07=g7[0],
                        pickle=[d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]])
        c0 = c[0]
        index[sid] = dict(v, value=c0["value"]["hex"], abserr=c0["abserr"]["hex"], num_regions=c0["num_regions"],
                          max_depth=c0["max_depth"], converged=c0["converged"], atol=c0["atol"], rtol=c0["rtol"],
                          elapsed=c0["elapsed"], notify_label=c0["notify_label"],
                          n_windows=(c0.get("guard_07") or {}).get("n_windows"),
                          n_windows_stopped=(c0.get("guard_07") or {}).get("n_windows_stopped"))
    if sha256(ck) != RC.CENSUS07_CHECKPOINT_SHA256:
        raise SystemExit("07's census checkpoint changed")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a Part A: the census records of section 2.1's calls",
                 [ck, V4SEL, XSEL])
    pk = OUT / "state" / "census_refs.pkl"
    js = OUT / "out" / "census_refs.json"
    for p in (pk, js):
        if p.exists():
            raise SystemExit(f"{p} exists")
    pk.write_bytes(pickle.dumps({"header": hdr, "calls": out}, protocol=5))
    js.write_text(json.dumps({"header": hdr, "pickle": str(pk), "pickle_sha256": sha256(pk), "calls": index}, indent=1))
    for l in hdr:
        print(l)
    from collections import Counter

    print(Counter(v["set"] for v in w.values()), "calls;", len(units), "units")
    print(f"wrote {pk} sha256 {sha256(pk)}; {js} sha256 {sha256(js)}")


if __name__ == "__main__":
    main()
