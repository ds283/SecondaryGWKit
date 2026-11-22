"""levin-phase-floor prompt 07b section 9.5, U8 ("the code is (h)"), at planned stop 3. New in 07b. Read only.

The calls are the 187 Part A replayed: var/levin-phase-floor-07b/measurement/state/census_wanted_07b.json, every id whose
``which`` is ``h`` (that is, less ``i3-34599-07``, replayed at 07's module); each id's census call is that file's (unit,
kind, index), read from this census's checkpoint (group calls in ``calls``, analytic in ``other_calls``, by ``index``).
(h)'s replay record is out/ext/replays/h/<id>.json's ``replay``; it is checked to equal the same id's row of
h_summary_all.json or h_summary_i3.json. Each census call is compared with it in value and abserr (``float.hex``),
``converged``, ``num_regions`` and ``max_depth``. U8 holds if all 187 are equal in all five.

Writes OUT_JSON (provenance, every call's two records and its verdict) and prints a summary. Exits 1 unless U8 holds.
Usage: python -B score_u8_07b.py RUN_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, header, sha256  # noqa: E402

PART_A = MAIN / "var" / "levin-phase-floor-07b" / "measurement"
WANTED = PART_A / "state" / "census_wanted_07b.json"
REPLAYS = PART_A / "out" / "ext" / "replays"
KEYS = ("value", "abserr", "converged", "num_regions", "max_depth")


def hx(v):
    return v.get("hex") if isinstance(v, dict) else v


def main():
    run_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    ck = run_dir / "checkpoint.jsonl"
    summ_all = REPLAYS / "h_summary_all.json"
    summ_i3 = REPLAYS / "h_summary_i3.json"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b: U8 at planned stop 3, the census against (h)'s replays",
                 [WANTED, summ_all, summ_i3, ck])
    for l in hdr:
        print(l, flush=True)
    wanted = {k: v for k, v in json.loads(WANTED.read_text())["wanted"].items() if v["which"] == "h"}
    excluded = sorted(k for k, v in json.loads(WANTED.read_text())["wanted"].items() if v["which"] != "h")
    srows = {**json.loads(summ_all.read_text())["rows"], **json.loads(summ_i3.read_text())["rows"]}
    units = {v["unit"] for v in wanted.values()}
    recs = {}
    with open(ck) as f:
        for line in f:
            u = line[10:60].split('"', 1)[0]  # {"unit": "<uid>", ...
            if u in units:
                r = json.loads(line)
                if r["unit"] in recs:
                    raise SystemExit(f"{r['unit']} twice in the checkpoint")
                recs[r["unit"]] = r["data"]
    rows = {}
    n_eq = 0
    for sid in sorted(wanted):
        w = wanted[sid]
        rp = json.loads((REPLAYS / "h" / f"{sid}.json").read_text())
        rep = rp["replay"]
        rep_t = [rep.get(k) for k in KEYS]
        summ_t = [srows.get(sid, {}).get("replay", {}).get(k) for k in KEYS]
        d = recs.get(w["unit"])
        row = {"unit": w["unit"], "kind": w["kind"], "index": w["index"], "replay": dict(zip(KEYS, rep_t)),
               "replay_file_sha256": sha256(REPLAYS / "h" / f"{sid}.json"), "replay_equals_summary": rep_t == summ_t}
        if d is None or d["status"] != "done":
            row.update(census=None, equal=False, why="unit missing or not done")
        else:
            key = "calls" if w["kind"] == "group" else "other_calls"
            cs = [c for c in d.get(key) or [] if c["index"] == w["index"]]
            if len(cs) != 1:
                row.update(census=None, equal=False, why=f"{len(cs)} calls with that index")
            else:
                c = cs[0]
                cen_t = [hx(c.get(k)) for k in KEYS]
                row["census"] = dict(zip(KEYS, cen_t))
                row["census_counts"] = [c.get("num_edge_rule_refusals"), c.get("num_range_rule_refusals")]
                row["differing"] = [k for k, a, b in zip(KEYS, cen_t, rep_t) if a != b]
                row["equal"] = not row["differing"] and row["replay_equals_summary"]
        n_eq += bool(row["equal"])
        rows[sid] = row
    holds = n_eq == len(wanted) == 187
    doc = {"header": hdr, "run_dir": str(run_dir), "n_calls": len(wanted), "excluded": excluded, "n_equal": n_eq,
           "n_replay_equals_summary": sum(1 for r in rows.values() if r["replay_equals_summary"]),
           "n_distinct_census_calls": len({(r["unit"], r["kind"], r["index"]) for r in rows.values()}),
           "u8_holds": holds, "failures": {k: r for k, r in rows.items() if not r["equal"]}, "calls": rows}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"calls {len(wanted)} (excluded {excluded}); distinct census calls {doc['n_distinct_census_calls']}; "
          f"replay record equals its summary row: {doc['n_replay_equals_summary']}; equal in all five: {n_eq}")
    for k, r in list(doc["failures"].items())[:20]:
        print(f"FAILS {k}: {r.get('why') or r.get('differing')} census {r.get('census')} replay {r['replay']}")
    print(f"U8: {'all equal' if holds else 'NOT all equal'}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(main())
