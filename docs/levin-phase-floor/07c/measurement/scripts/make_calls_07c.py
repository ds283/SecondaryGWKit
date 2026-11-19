"""levin-phase-floor prompt 07c Part A, section 2.1: the replay manifest. New in 07c, after 07b Part A's make_calls_07b.py.

Usage: python -B make_calls_07c.py base | r5
- ``base`` writes state/calls_07c.json with
  * ``chase``: the 198 chase calls, each its own capture (out/captures/captures_chase.json; every capture must match 07b's
    census), census record ``c-...`` in census_refs_07c.pkl;
  * ``chase~o``: the other-rtol counterpart of each chase integrand at one rtol only, the chase capture replayed with the
    counterpart's (rtol, atol) (07b Part A's deviation 8), census record ``x-...`` in census_refs_07c_chase.pkl;
  * ``parta:<set>``: 07b Part A's 187 calls (its state/calls_07b.json, every entry but ``i3-34599-07``), with their captures
    and overrides unchanged, census record ``a:<census>`` (07b's census);
  * ``r31``: R31's record (07b's rehearsal capture ref-G+q+r-h-3-20363-1.pkl, df2d478f...), compared with the record 07b's
    rehearsal agent captured (out/refcap.json; reproduces R31's store), at its own rtol 1e-7 only.
- ``r5`` writes state/calls_07c_r5.json with R5's drawn calls (out/captures/captures_r5.json) and their counterparts
  (``r5~o``, census_refs_07c_r5.pkl).
replay_07c.py reads both manifests (a sid is in one only).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, WIDER, header, sha256  # noqa: E402

CALLS07B = MAIN / "var/levin-phase-floor-07b/measurement/state/calls_07b.json"
REFCAP = MAIN / "var/levin-phase-floor-07b/rehearsal/out/refcap.json"
R31_PKL = MAIN / "var/levin-phase-floor-07b/rehearsal/state/inputs/ref-G+q+r-h-3-20363-1.pkl"
R31_SHA = "df2d478f91034ad60f1ee125e7d760c7a8b7a8f2c659644b60eeb65c9ac3a249"


def cap_entries(cap_json, census_json, set_name):
    caps = json.loads(Path(cap_json).read_text())
    out = {}
    for e in caps:
        if not e["capture_matches_census"]:
            raise SystemExit(f"{e['sid']}: the capture does not match 07b's census (a section 8 stop)")
        out[e["sid"]] = {"set": set_name, "path": str(Path(e["dir"]) / e["pickle"]), "offset": e["offset"], "length": e["length"],
                         "sha256": e["sha256"], "override": None, "census": e["sid"]}
    return out


def counterparts(entries, extra_label, set_name):
    ex = json.loads((WIDER / "out" / f"census_refs_07c_{extra_label}.json").read_text())["calls"]
    out = {}
    for xsid, x in ex.items():
        p = x["counterpart_of"].split("-")
        rt, sh, se = "-".join(p[1:-2]), p[-2], p[-1]
        src = [s for s in entries if s in (f"c-{rt}-{sh}-{se}-{x['index']}", f"m-{rt}-{sh}-{se}-g{x['index']}")]
        if len(src) != 1:
            raise SystemExit(f"{xsid}: {len(src)} source captures")
        e = dict(entries[src[0]])
        e.update(set=set_name, override={"rtol": float(x["rtol"]), "atol": float(x["atol"])}, census=xsid,
                 census_pkl=f"census_refs_07c_{extra_label}.pkl", captured_as=src[0])
        out[xsid] = e
    return out


def main():
    which = sys.argv[1]
    out_p = WIDER / "state" / ("calls_07c.json" if which == "base" else "calls_07c_r5.json")
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    inputs = []
    calls = {}
    if which == "base":
        cj = WIDER / "out" / "captures" / "captures_chase.json"
        inputs += [cj, WIDER / "out" / "census_refs_07c_chase.json", CALLS07B, REFCAP]
        chase = cap_entries(cj, None, "chase")
        calls.update(chase)
        calls.update(counterparts(chase, "chase", "chase~o"))
        for sid, e in json.loads(CALLS07B.read_text())["calls"].items():
            if sid == "i3-34599-07":
                continue
            e = dict(e)
            e["set"] = "parta:" + e["set"]
            e["census"] = "a:" + e["census"]
            calls["a:" + sid] = e
        rc = [c for c in json.loads(REFCAP.read_text())["captures"] if c["uid"] == "h-1e-7-3-20363" and c["index"] == 1]
        if len(rc) != 1 or rc[0]["sha256"] != R31_SHA or not rc[0]["reproduces_store"]:
            raise SystemExit("R31's record: not found as expected in refcap.json")
        rc = rc[0]
        if sha256(R31_PKL) != R31_SHA:
            raise SystemExit("R31's capture has another SHA-256")
        calls["r31-3-20363-1"] = {
            "set": "r31", "path": str(R31_PKL), "offset": 0, "length": R31_PKL.stat().st_size, "sha256": R31_SHA,
            "override": None, "census": "r31:h-1e-7-3-20363-1",
            "r31_record": {"unit": rc["uid"], "notify_label": rc["notify_label"], "value": rc["value"], "abserr": rc["abserr"],
                           "num_regions": rc["regions"], "max_depth": 20, "converged": rc["converged"],
                           "num_edge_rule_refusals": rc["refusals_07b"]["num_edge_rule_refusals"],
                           "num_range_rule_refusals": rc["refusals_07b"]["num_range_rule_refusals"],
                           "store_R31": rc["R31"], "store_R30": rc["R30"]}}
    elif which == "r5":
        cj = WIDER / "out" / "captures" / "captures_r5.json"
        inputs += [cj, WIDER / "out" / "census_refs_07c_r5.json"]
        r5 = cap_entries(cj, None, "r5")
        calls.update(r5)
        calls.update(counterparts(r5, "r5", "r5~o"))
    else:
        raise SystemExit("base or r5")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07c Part A: the replay manifest ({which})", inputs)
    out_p.write_text(json.dumps({"header": hdr, "calls": calls}, indent=1))
    from collections import Counter

    print(Counter(e["set"] for e in calls.values()))
    print(f"{len(calls)} calls; wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
