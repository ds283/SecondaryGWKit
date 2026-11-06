"""levin-phase-floor prompt 06l section 3 ("Against route 1's census"): of the calls route 1 converges and route B does
not, how many route 1 recovered only through a Levin parent with a Clenshaw-Curtis half (A's difference from route 1).
New in 06l. Read only.

Route 1's checkpoint (var/runs/levin-phase-floor-06k-census-route1-20261104T103009/checkpoint.jsonl, its SHA-256
checked against 36e3dd4c... in the same pass) records each firing's span, the parent's kind, phi_P, phi_L, phi_R
and delta1, but not the halves' kinds. Route B's records (out/partC/routeB.json's calls, from the census extract)
hold, for every leaf B left untrusted where route 1's yardstick would trust it ("withheld"), the halves' kinds. A
route 1 firing whose span equals a B-withheld leaf's span is a decision route 1 made and B did not; it is matched by
span, exactly, and gets B's kinds. For each matched firing, A's decision is computed from route 1's own numbers:
on a Clenshaw-Curtis parent A is route 1 (trusts); on a Levin parent A trusts iff delta1 <= phi_P + the Levin
halves' phi (in route 1's summation order), and since a firing has r < 1, the trust test is that alone.
Per call (outside the 720, and apart within it):
- ``A_withholds``: matched firings A would not make (Levin parents with a Clenshaw-Curtis half, delta1 above the
  Levin-only yardstick); ``A_keeps``: matched firings A would make (Clenshaw-Curtis parents, where B's signed
  yardstick and route 1's differ, and Levin parents whose Levin half covers delta1);
- the call's class: "only through Levin parents with a CC half" (every matched firing is one A withholds: route 1's
  recovery rests only on what A takes away, so A would not recover it the same way), "only through CC parents"
  (every matched firing is one A keeps: A would decide as route 1 on them), "both", or "no match" (B's refinement
  left route 1's tree before any firing it withholds; not classified).
Route 1's firings that B never reached (B's refinement diverged earlier) are counted, not classified.
Writes out/partC/against_A.json and prints the counts.
Usage: PYTHONPATH=<tree>:<scripts> python -B against_a_06l.py
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

RUN_R1 = RC.MAIN / "var" / "runs" / "levin-phase-floor-06k-census-route1-20261104T103009"
CK_R1 = "36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38"
RB_JSON = RC.ROOT / "out" / "partC" / "routeB.json"
EX_B = RC.ROOT / "state" / "extract" / "routeB.pkl"


def main():
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: route 1's recoveries B does not make, against A",
                 [RB_JSON, EX_B])
    hdr.append(f"# input {RUN_R1 / 'checkpoint.jsonl'} (SHA-256 checked in the pass below against {CK_R1})")
    res = json.loads(RB_JSON.read_text())
    import pickle

    with open(EX_B, "rb") as f:
        B = pickle.load(f)
    # every call route 1 converges and B does not, in and out of the 720, from B's extract and route 1's checkpoint
    out_calls = res["against_route1"]["route1_converges_B_not"]["calls_outside_720"]
    want = {(c["unit"], c["kind"], c["index"]): "outside_720" for c in out_calls}
    n_in = res["against_route1"]["route1_converges_B_not"]["in_720"]
    sha = hashlib.sha256()
    r1 = {}
    units_needed = {u for u, _, _ in want}
    # the in-720 calls: found in the same pass, by the same rule (route 1 converged, B not)
    sp = RC.scan06a_sp()
    with open(RUN_R1 / "checkpoint.jsonl", "rb") as fck:
        for line in fck:
            sha.update(line)
            head = line[:200].decode("utf-8", "replace")
            uid = head.split('"unit": "', 1)[1].split('"', 1)[0] if '"unit": "' in head else None
            row = "-".join(uid.rsplit("-", 2)[-2:]) if uid else None
            if uid is None or (uid not in units_needed and row not in sp):
                continue
            d = json.loads(line)["data"]
            for kind, lst in (("group", d.get("calls") or []), ("analytic", d.get("other_calls") or [])):
                for c in lst:
                    key = (uid, kind, c["index"])
                    if key in want or (row in sp and c.get("converged")):
                        r1[key] = {"converged": c.get("converged"), "firings": (c.get("route_06k") or {}).get("firings", [])}
    if sha.hexdigest() != CK_R1:
        raise SystemExit(f"route 1's checkpoint sha256 {sha.hexdigest()} is not {CK_R1}")
    for (uid, kind, idx), v in list(r1.items()):
        row = "-".join(uid.rsplit("-", 2)[-2:])
        if row in sp and (uid, kind, idx) not in want:
            cb = [c for c in B["units"][uid]["calls"] if c["kind"] == kind and c["index"] == idx][0]
            if v["converged"] and not cb["converged"]:
                want[(uid, kind, idx)] = "in_720"
    if sum(1 for z in want.values() if z == "in_720") != n_in:
        raise SystemExit(f"in-720 calls found {sum(1 for z in want.values() if z == 'in_720')}, not {n_in}")
    calls = []
    for (uid, kind, idx), zone in sorted(want.items()):
        cb = [c for c in B["units"][uid]["calls"] if c["kind"] == kind and c["index"] == idx][0]
        recs = [dict(zip(("kind", "depth", "at_depth_max", "start", "end", "is_direct", "is_direct_L", "is_direct_R", "r",
                          "delta1", "delta2", "phi_P", "phi_L", "phi_R", "phim_L", "phim_R", "E_P", "E_L", "E_R", "nu1",
                          "nuB", "trusted_g", "trusted_r1", "residual"), t)) for t in cb["routeB"]["records"]]
        withheld = {(x["start"], x["end"]): x for x in recs if x["kind"] == "untrusted" and x["trusted_r1"]}
        fires = r1[(uid, kind, idx)]["firings"]
        a_with, a_keep, unreached = [], [], 0
        for f in fires:
            x = withheld.get((f["start"], f["end"]))
            if x is None:
                unreached += 1
                continue
            if x["is_direct"]:
                a_keep.append({"parent": "CC", "halves": f"{'CC' if x['is_direct_L'] else 'Levin'}+{'CC' if x['is_direct_R'] else 'Levin'}",
                               "d1_over_phiP": f["delta1"] / f["phi_P"], "d1_over_nuB": x["delta1"] / x["nuB"], "depth": f["depth"]})
                continue
            nu_a = f["phi_P"]
            if not x["is_direct_L"]:
                nu_a = nu_a + f["phi_L"]
            if not x["is_direct_R"]:
                nu_a = nu_a + f["phi_R"]
            entry = {"parent": "Levin", "halves": f"{'CC' if x['is_direct_L'] else 'Levin'}+{'CC' if x['is_direct_R'] else 'Levin'}",
                     "d1_over_phiP": f["delta1"] / f["phi_P"], "d1_over_nuA": f["delta1"] / nu_a,
                     "halves_phi_over_phiP": (f["phi_L"] + f["phi_R"]) / f["phi_P"], "depth": f["depth"]}
            (a_with if f["delta1"] > nu_a else a_keep).append(entry)
        if not a_with and not a_keep:
            cls = "no match"
        elif a_with and not a_keep:
            cls = "only through Levin parents with a CC half"
        elif a_keep and not a_with:
            cls = "only through CC parents (or Levin parents A keeps)"
        else:
            cls = "both"
        calls.append({"unit": uid, "kind": kind, "index": idx, "zone": zone, "rtol": uid.split("-")[1] + "-" + uid.split("-")[2],
                      "group": cb["group"], "n_route1_firings": len(fires), "n_matched": len(a_with) + len(a_keep),
                      "n_unreached": unreached, "A_withholds": a_with, "A_keeps": a_keep, "class": cls,
                      "B_holdback": "untrusted leaf" if (cb.get("num_untrusted_regions") or 0) > 0 else ("node_limited" if cb.get("node_limited") else "other")})
    summ = {z: {"calls": sum(1 for c in calls if c["zone"] == z),
                "by_class": dict(collections.Counter(c["class"] for c in calls if c["zone"] == z)),
                "by_class_and_rtol": dict(collections.Counter(f"{c['rtol']}/{c['class']}" for c in calls if c["zone"] == z)),
                "matched_firings_A_withholds": sum(len(c["A_withholds"]) for c in calls if c["zone"] == z),
                "matched_firings_A_keeps": sum(len(c["A_keeps"]) for c in calls if c["zone"] == z),
                "A_keeps_by_parent": dict(collections.Counter(e["parent"] for c in calls if c["zone"] == z for e in c["A_keeps"])),
                "route1_firings_unreached_by_B": sum(c["n_unreached"] for c in calls if c["zone"] == z)}
            for z in ("outside_720", "in_720")}
    doc = {"header": hdr, "route1_checkpoint_sha256": sha.hexdigest(), "summary": summ, "calls": calls}
    p = RC.ROOT / "out" / "partC" / "against_A.json"
    p.write_text(json.dumps(doc, indent=1, default=str))
    print("\n".join(hdr))
    print(json.dumps(summ, indent=1))
    print(f"wrote {p} sha256 {RC.sha256(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
