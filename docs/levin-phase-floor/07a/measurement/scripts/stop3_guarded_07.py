"""levin-phase-floor prompt 07 section 6, planned stop 3: from the census checkpoint, read only, count every Levin call
by its guard_07 (locates, guards), and list each call that guards a leaf and locates nothing (the O9 case,
[07-c2-gate-threshold-was-never-measured]) with its unit, kind, call index, notify_label, rtol, Rule A's
location_skipped_phases, the counts, the returned leaves' flags, value, abserr and converged; also the statuses and any
unit in error or over budget. New in 07.
Usage: python -B stop3_guarded_07.py RUN_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402


def main():
    run_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    ck = run_dir / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: stop 3, calls by locate/guard, and the guard-only calls", [ck])
    counts, statuses, listed, bad = {}, {}, [], []
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            statuses[d["status"]] = statuses.get(d["status"], 0) + 1
            if d["status"] != "done":
                bad.append([r["unit"], d["status"], (d.get("error") or "")[-500:]])
            for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                for c in d.get(key) or []:
                    g = c.get("guard_07") or {}
                    if "error" in g or not g.get("present"):
                        k = (d["rtol"], kind, "guard_07 missing or error")
                    else:
                        k = (d["rtol"], kind, f"locates={g['locates']} guards={g['guards']}")
                    counts["|".join(k)] = counts.get("|".join(k), 0) + 1
                    if g.get("present") and g.get("guards") and not g.get("locates"):
                        listed.append({"unit": r["unit"], "rtol": d["rtol"], "kind": kind, "index": c["index"],
                                       "notify_label": c["notify_label"],
                                       "location_skipped_phases": g["location_skipped_phases"],
                                       "num_guarded_regions": g["num_guarded_regions"],
                                       "num_guard_at_depth_max": g["num_guard_at_depth_max"],
                                       "returned": g["returned"], "value": c["value"]["hex"], "abserr": c["abserr"]["hex"],
                                       "converged": c["converged"], "num_regions": c["num_regions"],
                                       "max_depth": c["max_depth"], "n_at_depth_max": c["n_at_depth_max"]})
    doc = {"header": hdr, "statuses": statuses, "units_not_done": bad, "calls_by_rtol_kind_locate_guard": dict(sorted(counts.items())),
           "n_guard_only": len(listed), "guard_only": listed}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("header", "guard_only")}, indent=1))
    for x in listed:
        sk = [(s["phase"], s["variation"]["repr"]) for s in x["location_skipped_phases"]]
        print(f"{x['unit']} {x['kind']} #{x['index']} rtol {x['rtol']} | {x['notify_label']} | skipped {sk} | guarded "
              f"{x['num_guarded_regions']} at-cap {x['num_guard_at_depth_max']} | regions {x['num_regions']} depth "
              f"{x['max_depth']} conv {x['converged']} | returned {x['returned']}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
