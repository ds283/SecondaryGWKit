"""Orchestrator's own re-score of V2, V3 and V5 (levin-phase-floor 07 Part B, section B.3 check 7). Read only.

Usage:  python -I -B v235_rescore.py OUT_JSON

Independent of the census agent's score_07.py. Reads the census checkpoint (sha checked) and, for every unit with a
scanned call that the scan finds a sign change in, the unit's region record (sha checked against the line's own
pickle_record_sha256).

- **Scanned calls:** every scan_07 entry (group calls at both rtols, analytic at 1e-7), matched to its call by index.
- **V2:** a returned leaf with type 0 (Levin) holding a scan sign change. Recomputed here by the scan's "grid" test
  from the pickle's full scan (sign_change_x +- sign_change_halfwidth, both bracketing samples inside the leaf), on
  the region arrays (start, end, type, window_07, guarded_07). The scan's "ends" test needs theta' itself; its
  Levin listings are counted from the scan's own `listed` and reported apart. Also: my grid set equals the scan's
  own grid listings, leaf for leaf.
- **V3:** on scanned calls with no phase Rule A skipped: each sign change has a located `root` within h, and each
  located `root` lies within h of a sign change, compared exactly (Fractions of the hex). Points of kind `end` are
  reported apart. Calls with a skipped phase are set aside, counted, and those with a sign change listed.
- **V5:** any call, group or analytic, at either rtol, that locates a point or guards a leaf and has
  n_at_depth_max > 0; plus over-budget units.
"""

import hashlib
import json
import pickle
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

RUN = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-census-20261108T171137")
CK = "03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10"


def fh(x):
    return Fraction(float.fromhex(x["hex"] if isinstance(x, dict) else x))


def main():
    out_json = sys.argv[1]
    h = hashlib.sha256()
    with open(RUN / "checkpoint.jsonl", "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == CK
    c = Counter()
    v2_viol, v2_ends, v2_grid_mismatch, v2_nonlevin = [], [], [], Counter()
    v3_viol, v3_end_points, v3_setaside_sc = [], [], []
    v5_viol, over_budget, missing = [], [], Counter()
    for line in open(RUN / "checkpoint.jsonl"):
        d = json.loads(line)["data"]
        u = d["unit"]
        c["units"] += 1
        if d.get("status") not in ("done", "ok", None):
            c[f"status_{d.get('status')}"] += 1
        bud = d.get("budget_s")
        if d.get("wall_s") is not None and bud is not None and d["wall_s"] > bud:
            over_budget.append((u, d["wall_s"]))
        calls = {x["index"]: x for x in d["calls"] + d["other_calls"]}
        # V5
        for i, x in calls.items():
            g = x.get("guard_07")
            if g is None:
                missing["guard_07"] += 1
                continue
            act = g["locates"] or g["guards"]
            if act:
                c["v5_active_calls"] += 1
                nd = x.get("n_at_depth_max")
                if nd is None:
                    missing["n_at_depth_max_in_active"] += 1
                elif nd > 0:
                    v5_viol.append((u, i, x.get("group") or x.get("notify_label"), nd, x.get("converged")))
        # V3 (data line only)
        need_pickle = False
        for s in d["scan_07"] or []:
            c["scanned_calls"] += 1
            c[f"scanned_{s['kind']}_{d['rtol']}"] += 1
            x = calls[s["index"]]
            g = x["guard_07"]
            sc = [fh(v) for v in s["sign_change_x"]]
            if s["n_sign_changes"] != len(sc):
                missing["sign_change_x_truncated"] += 1
            if sc:
                need_pickle = True
            hh = fh(s["h"])
            if g["location_skipped_phases"]:
                c["v3_setaside"] += 1
                if sc:
                    v3_setaside_sc.append((u, s["index"], s["kind"], len(sc), g["guards"]))
                continue
            c["v3_scored"] += 1
            roots = [fh(p["x"]) for p in g["points"] if p["kind"] == "root"]
            ends = [p for p in g["points"] if p["kind"] == "end"]
            if ends:
                v3_end_points.append((u, s["index"], len(ends)))
            if sc or roots:
                c["v3_with_change_or_root"] += 1
            bad = [float(z) for z in sc if not any(abs(z - r) <= hh for r in roots)]
            bad += [float(r) for r in roots if not any(abs(z - r) <= hh for z in sc)]
            if bad:
                v3_viol.append((u, s["index"], s["kind"], bad))
        # V2 (needs the pickle)
        if not need_pickle:
            continue
        with open(RUN / d["pickle"], "rb") as f:
            f.seek(d["pickle_offset"])
            raw = f.read(d["pickle_length"])
        assert hashlib.sha256(raw).hexdigest() == d["pickle_record_sha256"], u
        p = pickle.loads(raw)
        c["pickles_read"] += 1
        ra = {r["index"]: r for r in p["regions"] + p["other_regions"]}
        sfull = {e["index"]: e["scan"] for e in p["scan_07"]}
        for s in d["scan_07"]:
            if not s["n_sign_changes"]:
                continue
            i = s["index"]
            r = ra[i]
            sf = sfull[i]
            assert len(sf["sign_change_x"]) == len(sf["sign_change_halfwidth"]) == s["n_sign_changes"], (u, i)
            assert [float(v).hex() for v in sf["sign_change_x"]] == [v["hex"] if isinstance(v, dict) else float(v).hex() for v in s["sign_change_x"]], (u, i)
            lo = r["start"].astype(float)
            hi = r["end"].astype(float)
            mine = set()
            for xc, hw in zip(sf["sign_change_x"], sf["sign_change_halfwidth"]):
                a, b = Fraction(xc) - Fraction(hw), Fraction(xc) + Fraction(hw)
                for k in range(len(lo)):
                    l, e = sorted((Fraction(float(lo[k])), Fraction(float(hi[k]))))
                    if l <= a and b <= e:
                        mine.add(k)
            scan_grid = {x["region_index"] for x in sf["listed"] if x["how"] == "grid"}
            if mine != scan_grid:
                v2_grid_mismatch.append((u, i, sorted(mine), sorted(scan_grid)))
            for k in mine:
                kind = "Levin" if int(r["type"][k]) == 0 else "CC"
                tag = (kind, "window" if int(r["window_07"][k]) == 1 else "", "guarded" if int(r["guarded_07"][k]) == 1 else "")
                if kind == "Levin":
                    v2_viol.append((u, i, k, tag))
                else:
                    v2_nonlevin[tag] += 1
            for x in sf["listed"]:
                if x["how"] == "ends":
                    v2_ends.append((u, i, x["region_index"], x["branch"]))
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": CK,
        "counts": dict(c),
        "missing": dict(missing),
        "over_budget": over_budget,
        "V2": {"levin_leaves_with_sign_change_grid": v2_viol, "n": len(v2_viol),
               "ends_listings": v2_ends, "ends_levin": [e for e in v2_ends if e[3] == "Levin"],
               "nonlevin_listed_by_tag": {str(k): v for k, v in v2_nonlevin.items()},
               "grid_set_mismatch_vs_scan": v2_grid_mismatch},
        "V3": {"violations": v3_viol, "n": len(v3_viol), "end_points": v3_end_points,
               "setaside_with_sign_change": v3_setaside_sc},
        "V5": {"violations": v5_viol, "n": len(v5_viol)},
    }
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print("# v235_rescore.py sha256", out["script_sha256"])
    print("counts", json.dumps(dict(c), indent=1))
    print("missing", dict(missing), " over budget", len(over_budget))
    print("V2: Levin leaves holding a sign change (grid):", len(v2_viol), v2_viol[:10])
    print("    'ends' listings:", len(v2_ends), "of them Levin:", len(out["V2"]["ends_levin"]))
    print("    non-Levin listed leaves by tag:", out["V2"]["nonlevin_listed_by_tag"])
    print("    my grid set differs from the scan's own grid listing in calls:", len(v2_grid_mismatch), v2_grid_mismatch[:5])
    print("V3: violations", len(v3_viol), v3_viol[:10])
    print("    calls with kind-'end' points:", len(v3_end_points), v3_end_points[:5])
    print("    set aside with a sign change:", len(v3_setaside_sc), "guarded among them:", sum(1 for x in v3_setaside_sc if x[4]),
          "by kind:", dict(Counter(x[2] for x in v3_setaside_sc)))
    print("V5: violations", len(v5_viol), "converged among them:", sum(1 for x in v5_viol if x[4]),
          "depth-max counts", sorted(x[3] for x in v5_viol), "by group", dict(Counter(x[2] for x in v5_viol)),
          "by rtol", dict(Counter(x[0].split("-")[1] + "-" + x[0].split("-")[2] for x in v5_viol)))


if __name__ == "__main__":
    main()
