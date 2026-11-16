"""levin-phase-floor prompt 07b Part A, sections 2.1 and 2.4: the replay manifest. New in 07b (after 07a Part A's
make_calls_07a.py, whose manifest state/calls_07a.json it reads for 07's and 07a Part A's captures).

Every replay is one captured call at one (rtol, atol), compared with one census record. The sids:
- **V4** ``v00``-``v23``, ``cal_z12``: 07's V4 inputs (07's state/v4/inputs, as 07a Part A replayed them), compared with
  07a's census (unit ``h-<rtol>-<row>``);
- **the pairs** ``p14501_7`` ... ``p35006_8``, and **the no-window calls** ``n0``-``n4`` (07a Part A's pairs captures,
  taken at 07's code; their integrands are the call's inputs, which 07a's code does not change), compared with 07a's census;
- **R1** ``R1-x###-8`` and ``R1-x###-7``: 07's cross-check capture of the row (taken at 1e-7 for 390 rows, at 1e-8 for 6),
  replayed at each rtol with the other rtol's (rtol, atol) where it was not captured there (``override``), compared with
  07a's census at that rtol. A capture that reproduces both rtols' records is the integrand of both;
- **section 2.1 item 3** ``i3-34599-07``: 07a's census capture of ``1-34599`` #2 at 1e-7 with the parent's (07's) module
  (capture_parent2), compared with **07's** census, to be replayed at 07's module (scripts/variants/levin_quadrature_07.py);
  ``i3-34599-h``: the capture at h (capture_h2), compared with 07a's census;
- **item 4 and R2/R3 (not the pairs)**: this Part's captures at h (out/captures/captures_h.json), sid as selected; and
  ``<sid>~o``, the same capture at the other rtol's (rtol, atol), compared with 07a's census record there.

Writes state/calls_07b.json (the manifest) and state/census_wanted_07b.json (the census records to extract).
Usage: PYTHONPATH=<tree>:<scripts> python -B make_calls_07b.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07A_SHA256, REC07, REC07A, REC07A_M, RUN07A, WIDER, header, sha256  # noqa: E402

CALLS07A = REC07A_M / "state" / "calls_07a.json"
V4SEL = REC07 / "state" / "v4_capture_selection.json"
SEL = WIDER / "state" / "selection_07b.json"
CAPS = WIDER / "out" / "captures" / "captures_h.json"
I3_PARENT = REC07A / "out" / "refs" / "captures_parent.json"
I3_H = REC07A / "out" / "refs" / "captures_h.json"


def atol_rtol(units):
    """{(unit, kind, index): (atol hex, rtol str)} from 07a's census checkpoint."""
    ck = RUN07A / "checkpoint.jsonl"
    if sha256(ck) != CHECKPOINT07A_SHA256:
        raise SystemExit("07a's checkpoint has another SHA-256")
    out = {}
    with open(ck) as f:
        for line in f:
            uid = line[10:60].split('"', 1)[0]
            if uid in units:
                d = json.loads(line)["data"]
                if d.get("status") != "done":
                    continue
                for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                    for c in d[key]:
                        out[(uid, kind, c["index"])] = (c["atol"], c["rtol"], c["notify_label"])
    return out


def main():
    out_m = WIDER / "state" / "calls_07b.json"
    out_w = WIDER / "state" / "census_wanted_07b.json"
    for p in (out_m, out_w):
        if p.exists():
            raise SystemExit(f"{p} exists")
    c07a = json.loads(CALLS07A.read_text())["calls"]
    v4sel = {e["sid"]: e for e in json.loads(V4SEL.read_text())}
    sel = json.loads(SEL.read_text())["calls"]
    caps = json.loads(CAPS.read_text())
    calls, wanted = {}, {}

    def h_of(g_unit):
        return "h-" + g_unit.split("-", 1)[1]

    def other(unit):
        p = unit.split("-")
        rt = "1e-8" if p[1] + "-" + p[2] == "1e-7" else "1e-7"
        return f"h-{rt}-{p[3]}-{p[4]}"

    def cap07a(sid):
        c = dict(c07a[sid])
        p = Path(c["path"])
        if not p.is_absolute():
            p = REC07A_M / p
        return {"path": str(p), "offset": c["offset"], "length": c["length"], "sha256": c["sha256"]}

    pending_override = []
    for sid, e in v4sel.items():
        calls[sid] = {"set": "v4", **cap07a(sid), "override": None, "census": sid}
        wanted[sid] = {"which": "h", "unit": h_of(e["g_unit"]), "kind": e["kind"], "index": e["index"]}
    pn = {sid: c for sid, c in c07a.items() if c["set"] in ("pairs", "nowindow")}
    kinds = atol_rtol({h_of(c["census_unit"]) for c in pn.values()})  # 07b: each call's kind from 07a's checkpoint
    for sid, c in pn.items():
        u = h_of(c["census_unit"])
        kind = [k for (uu, k, i) in kinds if uu == u and i == c["census_index"]]
        if len(kind) != 1:
            raise SystemExit(f"{sid}: kind not found")
        calls[sid] = {"set": c["set"], **cap07a(sid), "override": None, "census": sid}
        wanted[sid] = {"which": "h", "unit": u, "kind": kind[0], "index": c["census_index"]}
    for e in sel:
        if e["set"] != "R1":
            continue
        x = c07a[e["xcheck_sid"]]
        cap_unit = e["xcheck_unit"]  # the unit 07 captured it at
        cap_rt = cap_unit.split("-")[1] + "-" + cap_unit.split("-")[2]
        row = "-".join(cap_unit.split("-")[3:])
        for rt in ("1e-8", "1e-7"):
            s = f"{e['sid']}-{rt[-1]}"
            u = f"h-{rt}-{row}"
            calls[s] = {"set": "R1", **cap07a(e["xcheck_sid"]), "override": None if rt == cap_rt else "pending", "census": s,
                        "captured_unit": cap_unit}
            if rt != cap_rt:
                pending_override.append((s, u, e["kind"], e["index"], cap_unit))
            wanted[s] = {"which": "h", "unit": u, "kind": e["kind"], "index": e["index"]}
    for src, sid, which, variant in ((I3_PARENT, "i3-34599-07", "07", "07"), (I3_H, "i3-34599-h", "h", None)):
        r = [r for r in json.loads(src.read_text()) if r["sid"] == "r1-34599-1e-7-2"][0]
        if not r["capture_matches_census"]:
            raise SystemExit(f"{sid}: the capture did not match its census")
        p = Path(r["dir"])
        if not p.is_absolute():
            p = REC07A / p
        calls[sid] = {"set": "item3", "path": str(p / r["pickle"]), "offset": r["offset"], "length": r["length"],
                      "sha256": r["sha256"], "override": None, "census": sid, "replay_variant": variant}
        wanted[sid] = {"which": which, "unit": r["unit"] if which == "h" else r["g_unit"], "kind": "group", "index": 2}
    by_sid = {e["sid"]: e for e in sel}
    for r in caps:
        if not r["capture_matches_census"]:
            raise SystemExit(f"{r['sid']}: the capture did not match 07a's census")
        p = Path(r["dir"]) / r["pickle"]
        base = {"path": str(p), "offset": r["offset"], "length": r["length"], "sha256": r["sha256"]}
        st = r["set"]
        calls[r["sid"]] = {"set": st, **base, "override": None, "census": r["sid"]}
        wanted[r["sid"]] = {"which": "h", "unit": r["unit"], "kind": r["kind"], "index": r["index"]}
        s2 = r["sid"] + "~o"
        u2 = other(r["unit"])
        calls[s2] = {"set": st + "~o", **base, "override": "pending", "census": s2, "captured_unit": r["unit"]}
        wanted[s2] = {"which": "h", "unit": u2, "kind": r["kind"], "index": r["index"]}
        pending_override.append((s2, u2, r["kind"], r["index"], r["unit"]))
    ar = atol_rtol({p[1] for p in pending_override})
    for s, u, kind, idx, cap_unit in pending_override:
        if (u, kind, idx) not in ar:
            raise SystemExit(f"{s}: no census call {(u, kind, idx)}")
        atol, rtol, label = ar[(u, kind, idx)]
        calls[s]["override"] = {"rtol": float(rtol), "atol": float.fromhex(atol["hex"]) if isinstance(atol, dict) else float(atol),
                                "from_unit": cap_unit, "to_unit": u}
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: the replay manifest",
                 [CALLS07A, V4SEL, SEL, CAPS, I3_PARENT, I3_H, RUN07A / "checkpoint.jsonl"])
    out_m.write_text(json.dumps({"header": hdr, "calls": calls}, indent=1))
    out_w.write_text(json.dumps({"header": hdr, "wanted": wanted}, indent=1))
    from collections import Counter

    for l in hdr:
        print(l)
    print(Counter(c["set"] for c in calls.values()))
    print(f"wrote {out_m} sha256 {sha256(out_m)}; {out_w} sha256 {sha256(out_w)}")


if __name__ == "__main__":
    main()
