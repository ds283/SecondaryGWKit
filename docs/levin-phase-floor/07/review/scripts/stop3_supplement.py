"""Supplement to stop3_compare.py at planned stop 3 (levin-phase-floor 07 Part B). Read only.

Usage:  python -B stop3_supplement.py CENSUS_RUN_DIR STOP3_JSON OUT_JSON

stop3_compare.py left two per-call structures at UNIT level, compared positionally: the data's
deep_lines.lines (a list inside a dict) and the pickle's result_full.WKB_Levin_subintervals[].groups[] (the
store-level group records). In every unit with an ACTIVE call, this attributes both to calls and compares each QUIET
call's entries with 06m's:
- deep_lines.lines entries by their own `index`;
- group records flattened in order and matched to data.calls in order; the match is accepted only if every group
  record's `label` equals its call's `group` (and the counts agree), in both records.
A quiet call's group record must equal 06m's on every key except 07's four counts (stationary_points,
window_pieces, guarded_regions, guard_at_depth_max, absent at 06m) and `elapsed` (timing). Its deep lines must be
equal whole. It also lists every distinct unit-level key from STOP3_JSON, so none is left unread.
"""

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import hashlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stop3_compare as S  # noqa: E402

NEW4 = {"stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max"}


def flat_groups(p):
    out = []
    for s in (p.get("result_full") or {}).get("WKB_Levin_subintervals") or []:
        out.extend(s.get("groups") or [])
    return out


def main():
    run_dir, stop3_json, out_json = sys.argv[1:4]
    s3 = json.load(open(stop3_json))
    rows = {r["unit"]: r for r in s3["rows"]}
    idx = json.load(open(S.INDEX))["units"]
    all_ul = Counter()
    for r in s3["rows"]:
        for p in r["unit_level"]:
            all_ul[re.sub(r"\[\d+\]", "[i]", p)] += 1
    tot = Counter()
    bad = []
    for line in open(Path(run_dir) / "checkpoint.jsonl"):
        d = json.loads(line)["data"]
        u = d["unit"]
        r = rows[u]
        if not r["n_active"]:
            continue
        tot["active_units"] += 1
        quiet = set(c["index"] for c in d["calls"] + d["other_calls"]) - set(r["active"])
        e = idx[u]
        g = json.loads(S.load_bytes(S.RUN06M / "checkpoint.jsonl", e["offset"], e["length"], e["line_sha256"]))["data"]
        p6 = pickle.loads(S.load_bytes(S.RUN06M / g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]))
        pm = pickle.loads(S.load_bytes(Path(run_dir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]))
        # deep lines by index
        dm = S.rekey_list(d["deep_lines"]["lines"], lambda x: x["index"])
        d6 = S.rekey_list(g["deep_lines"]["lines"], lambda x: x["index"])
        for i in quiet:
            if i in dm or i in d6:
                tot["quiet_calls_with_deep_lines"] += 1
                if S.diff(dm.get(i), d6.get(i)):
                    bad.append((u, i, "deep_lines", S.diff(dm.get(i), d6.get(i))[:5]))
        # group records by order
        for name, dd, pp in (("mine", d, pm), ("06m", g, p6)):
            fg = flat_groups(pp)
            ok = len(fg) == len(dd["calls"]) and all(x.get("label") == c["group"] for x, c in zip(fg, dd["calls"]))
            if not ok:
                bad.append((u, None, f"group-record match failed in {name}", [len(fg), len(dd["calls"])]))
        fm, f6 = flat_groups(pm), flat_groups(p6)
        for c, a, b in zip(d["calls"], fm, f6):
            if c["index"] not in quiet:
                tot["active_group_records"] += 1
                continue
            tot["quiet_group_records"] += 1
            a2 = {k: v for k, v in a.items() if k not in NEW4 and k != "elapsed"}
            b2 = {k: v for k, v in b.items() if k != "elapsed"}
            extra06 = set(b) & NEW4
            if extra06:
                bad.append((u, c["index"], "06m group record carries a 07 count", sorted(extra06)))
            dd_ = S.diff(a2, b2)
            if dd_:
                bad.append((u, c["index"], "group record", dd_[:8]))
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "totals": dict(tot),
        "n_bad": len(bad),
        "bad": bad[:200],
        "all_unit_level_keys": dict(all_ul.most_common()),
    }
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    print("# stop3_supplement.py sha256", out["script_sha256"])
    print(json.dumps(out["totals"], indent=1))
    print("bad:", len(bad))
    for b in bad[:20]:
        print("  ", b)
    print("distinct unit-level keys:", len(all_ul))
    for k, v in all_ul.most_common():
        print(f"  {v:6d}  {k}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
