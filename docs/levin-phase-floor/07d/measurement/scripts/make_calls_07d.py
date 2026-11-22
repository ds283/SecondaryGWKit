"""levin-phase-floor prompt 07d Part A, section 2.1: the replay manifest, and the census records it wants. New in 07d.

Entries (sid -> capture pointer, rtol override, the census record it must reproduce):
- ``d:<unit>|<g|a><index>``: section 2.1 item 1's calls (out/captures/captures_<label>.json, each capture equal to 07c's
  census record bit for bit), at their own rtol; and each one's counterpart at the other rtol, the same row and index
  (``d:<other unit>|...``), replayed from the same capture with the counterpart's census (rtol, atol) (07c Part A's
  deviation 7), set ``calls~o``;
- ``R:<07c sid>``: 07c Part A's 454 calls (its state/calls_07c.json and calls_07c_r5.json), the regression set, from
  their captures (SHA-256 checked when replayed), each with the 07c census record of the unit and index 07c Part A's
  census index names for it (out/census_refs_07c*.json; R2-28 is ``R:a:R2-28`` and ``R:a:R2-28~o``, and again as the
  chase calls ``R:c-1e-7-0-13757-5``, ``R:c-1e-8-0-13757-5``);
- ``R:r31``: R31's record (07c Part A's ``r31``), compared with R32's record of the same call as 07c's rehearsal agent
  captured it (var/levin-phase-floor-07c/rehearsal/out/refcap.json, ``R32``), R32 being the rehearsal at 07c's code.
Writes state/calls_07d.json (or --extra NAME: state/calls_07d_NAME.json from captures_NAME.json, sets NAME and NAME~o)
and state/census_wanted_07d[_NAME].json ({census sid: {unit, kind, index}}).
Usage: PYTHONPATH=<tree>:<scripts> python -B make_calls_07d.py [--extra NAME]
"""

import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, MEAS07C, WIDER, header, sha256  # noqa: E402

REFCAP = MAIN / "var" / "levin-phase-floor-07c" / "rehearsal" / "out" / "refcap.json"


def other(unit):
    p = unit.split("-")
    rt = "-".join(p[1:-2])
    return "-".join([p[0], "1e-8" if rt == "1e-7" else "1e-7", p[-2], p[-1]])


def main():
    extra = sys.argv[sys.argv.index("--extra") + 1] if "--extra" in sys.argv else None
    label = extra or "calls"
    man_p = WIDER / "state" / (f"calls_07d_{extra}.json" if extra else "calls_07d.json")
    want_p = WIDER / "state" / (f"census_wanted_07d_{extra}.json" if extra else "census_wanted_07d.json")
    for p in (man_p, want_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    idx = {r["sid"]: r for r in pickle.loads((WIDER / "state" / "census_index_07d.pkl").read_bytes())["rows"]}
    cap_p = WIDER / "out" / "captures" / f"captures_{label}.json"
    caps = json.loads(cap_p.read_text())
    calls, wanted = {}, {}
    bad = [c for c in caps if not c["capture_matches_census"]]
    if bad:
        raise SystemExit(f"{len(bad)} captures do not match the census: a stop (section 8)")
    for c in caps:
        sid = c["sid"]
        r = idx[sid]
        ptr = {"path": str(Path(c["dir"]) / c["pickle"]), "offset": c["offset"], "length": c["length"], "sha256": c["sha256"]}
        calls[f"d:{sid}"] = dict(ptr, set=label, override=None, census=sid, captured_unit=r["unit"])
        wanted[sid] = {"unit": r["unit"], "kind": r["kind"], "index": r["index"]}
        osid = f"{other(r['unit'])}|{sid.split('|')[1]}"
        o = idx[osid]
        if o["notify_label"] != r["notify_label"]:
            raise SystemExit(f"{osid}: the counterpart's label differs")
        calls[f"d:{osid}"] = dict(ptr, set=f"{label}~o", override={"rtol": float(o["rtol"]), "atol": None},
                                  census=osid, captured_unit=r["unit"], captured_as=sid)
        wanted[osid] = {"unit": o["unit"], "kind": o["kind"], "index": o["index"]}
    inputs = [cap_p]
    if extra:
        # (added at the top-up) an entry another manifest already holds (the same call, e.g. a first-sample call's
        # counterpart at the other rtol) is dropped here, with its census record, and counted
        held = set()
        for q in sorted((WIDER / "state").glob("calls_07d*.json")):
            if q != man_p:
                held |= set(json.loads(q.read_text())["calls"])
        dup = sorted(k for k in calls if k in held)
        for k in dup:
            del calls[k]
        wanted = {calls[k]["census"]: wanted[calls[k]["census"]] for k in calls}
        print(f"dropped {len(dup)} entries another manifest holds: {dup[:10]}{' ...' if len(dup) > 10 else ''}")
    if not extra:
        cidx = {}
        for f in ("census_refs_07c.json", "census_refs_07c_chase.json", "census_refs_07c_r5.json"):
            cidx.update(json.loads((MEAS07C / "out" / f).read_text())["calls"])
            inputs.append(MEAS07C / "out" / f)
        for f in ("calls_07c.json", "calls_07c_r5.json"):
            mp = MEAS07C / "state" / f
            inputs.append(mp)
            for s7, c in json.loads(mp.read_text())["calls"].items():
                ptr = {k: c[k] for k in ("path", "offset", "length", "sha256")}
                if c["census"].startswith("r31:"):
                    rc = [x for x in json.loads(REFCAP.read_text())["captures"] if x["label"] == "G+q+r"]
                    if len(rc) != 1:
                        raise SystemExit("refcap: not one G+q+r record")
                    x = rc[0]
                    if not x.get("reproduces_store"):
                        raise SystemExit("refcap's G+q+r record does not reproduce R32's store")
                    # v2: the call-level record (the call's own scale), not the store-scale R32 entry
                    r32 = {"value": x["value"], "abserr": x["abserr"], "converged": x["converged"], "regions": x["regions"],
                           **x["refusals_07b"], **x["chase_07c"], "store_R32": x["R32"]}
                    calls["R:r31"] = dict(ptr, set="R:" + c["set"], override=c["override"], census="r32:" + x["uid"],
                                          r32_record=r32, r31_record_07c=c.get("r31_record"))
                    inputs.append(REFCAP)
                    continue
                e = cidx[c["census"]]
                csid = f"{e['unit']}|{e['kind'][0]}{e['index']}"
                ov = c["override"]
                calls[f"R:{s7}"] = dict(ptr, set="R:" + c["set"],
                                        override=None if ov is None else {"rtol": ov["rtol"], "atol": None},
                                        census=csid, census_07c_sid=s7)
                wanted[csid] = {"unit": e["unit"], "kind": e["kind"], "index": e["index"]}
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07d Part A: the replay manifest ({label})", inputs)
    man_p.write_text(json.dumps({"header": hdr, "calls": calls}, indent=1))
    want_p.write_text(json.dumps({"header": hdr, "wanted": wanted}, indent=1))
    for l in hdr:
        print(l)
    from collections import Counter
    print(f"{len(calls)} manifest entries, by set: {dict(Counter(c['set'] for c in calls.values()))}; {len(wanted)} census records wanted")
    print(f"wrote {man_p} {sha256(man_p)}; {want_p} {sha256(want_p)}")


if __name__ == "__main__":
    main()
