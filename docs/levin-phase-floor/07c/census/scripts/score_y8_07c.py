"""levin-phase-floor prompt 07c section 9.5, Y8 ("the code is (S)"), at planned stop 3. New in 07c, after 07b's
score_u8_07b.py (a record, not edited), whose form it keeps. Read only.

The calls (state/predictions.md part 4): every record Part A wrote of (S)'s replays,
var/levin-phase-floor-07c/measurement/out/replays/S/<id>.json (454), less ``r31-3-20363-1`` (R31's record, a rehearsal
row, not a census call): 453 ids. Each id's census call (unit, kind, index) is read from Part A's mappings:
out/census_refs_07c.json ``calls``, and state/census_wanted_07c.json, census_wanted_07c_chase.json and
census_wanted_07c_r5.json ``wanted`` (every id mapped, no id mapped two ways). The census call is read from this census's
checkpoint (group calls in ``calls``, analytic in ``other_calls``, by ``index``). (S)'s replay record is the file's
``replay``; it is checked to equal the same id's row of out/replays/S_summary_vplain.json. Each census call is compared
with it in value and abserr (``float.hex``), ``converged``, ``num_regions`` and ``max_depth``. Y8 holds if all 453 are
equal in all five. Where two ids name one census call, each is compared.

Writes OUT_JSON (provenance, every call's two records and its verdict) and prints a summary. Exits 1 unless Y8 holds.
Usage: python -B score_y8_07c.py RUN_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, header, sha256  # noqa: E402

PART_A = MAIN / "var" / "levin-phase-floor-07c" / "measurement"
MAPS = [(PART_A / "out" / "census_refs_07c.json", "calls"), (PART_A / "state" / "census_wanted_07c.json", "wanted"),
        (PART_A / "state" / "census_wanted_07c_chase.json", "wanted"), (PART_A / "state" / "census_wanted_07c_r5.json", "wanted")]
REPLAYS = PART_A / "out" / "replays"
SUMMARY = REPLAYS / "S_summary_vplain.json"
EXCLUDED = ("r31-3-20363-1",)
KEYS = ("value", "abserr", "converged", "num_regions", "max_depth")


def hx(v):
    return v.get("hex") if isinstance(v, dict) else v


def main():
    run_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    ck = run_dir / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: Y8 at planned stop 3, the census against (S)'s replays",
                 [p for p, _ in MAPS] + [SUMMARY, ck])
    for l in hdr:
        print(l, flush=True)
    m = {}
    for p, key in MAPS:
        for k, v in json.loads(p.read_text())[key].items():
            t = (v["unit"], v["kind"], v["index"])
            if k in m and m[k] != t:
                raise SystemExit(f"{k} mapped two ways: {m[k]} and {t}")
            m[k] = t
    ids = sorted(f.name[:-5] for f in (REPLAYS / "S").glob("*.json"))
    wanted = [i for i in ids if i not in EXCLUDED]
    unmapped = [i for i in wanted if i not in m]
    if unmapped:
        raise SystemExit(f"ids with no census call: {unmapped}")
    srows = json.loads(SUMMARY.read_text())["rows"]
    units = {m[i][0] for i in wanted}
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
    for sid in wanted:
        unit, kind, index = m[sid]
        fp = REPLAYS / "S" / f"{sid}.json"
        rp = json.loads(fp.read_text())
        rep = rp["replay"]
        rep_t = [rep.get(k) for k in KEYS]
        summ_t = [srows.get(sid, {}).get("replay", {}).get(k) for k in KEYS]
        d = recs.get(unit)
        row = {"unit": unit, "kind": kind, "index": index, "set": rp.get("set"), "replay": dict(zip(KEYS, rep_t)),
               "replay_module_sha256": (rp.get("header") or {}).get("module_sha256"),
               "replay_file_sha256": sha256(fp), "replay_equals_summary": rep_t == summ_t}
        if d is None or d["status"] != "done":
            row.update(census=None, equal=False, why="unit missing or not done")
        else:
            key = "calls" if kind == "group" else "other_calls"
            cs = [c for c in d.get(key) or [] if c["index"] == index]
            if len(cs) != 1:
                row.update(census=None, equal=False, why=f"{len(cs)} calls with that index")
            else:
                c = cs[0]
                cen_t = [hx(c.get(k)) for k in KEYS]
                row["census"] = dict(zip(KEYS, cen_t))
                row["census_counts"] = [c.get("num_chase_stops"), c.get("num_chase_stopped_leaves"),
                                        c.get("num_chase_noise_bisections")]
                row["differing"] = [k for k, a, b in zip(KEYS, cen_t, rep_t) if a != b]
                row["equal"] = not row["differing"] and row["replay_equals_summary"]
        n_eq += bool(row["equal"])
        rows[sid] = row
    holds = n_eq == len(wanted) == 453
    mods = sorted({str(r["replay_module_sha256"]) for r in rows.values()})
    doc = {"header": hdr, "run_dir": str(run_dir), "n_ids": len(ids), "excluded": list(EXCLUDED), "n_calls": len(wanted),
           "n_equal": n_eq, "replay_modules_sha256": mods,
           "n_replay_equals_summary": sum(1 for r in rows.values() if r["replay_equals_summary"]),
           "n_distinct_census_calls": len({(r["unit"], r["kind"], r["index"]) for r in rows.values()}),
           "n_acted": sum(1 for r in rows.values() if (r.get("census_counts") or [0])[0]),
           "y8_holds": holds, "failures": {k: r for k, r in rows.items() if not r["equal"]}, "calls": rows}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"ids {len(ids)}; calls {len(wanted)} (excluded {list(EXCLUDED)}); distinct census calls "
          f"{doc['n_distinct_census_calls']}; replay modules {mods}; replay record equals its summary row: "
          f"{doc['n_replay_equals_summary']}; equal in all five: {n_eq}; the stop acted (census) in {doc['n_acted']}")
    for k, r in list(doc["failures"].items())[:20]:
        print(f"FAILS {k}: {r.get('why') or r.get('differing')} census {r.get('census')} replay {r['replay']}")
    print(f"Y8: {'all equal' if holds else 'NOT all equal'}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(main())
