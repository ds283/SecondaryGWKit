"""Summaries of the R29 replay's table (out/replay_calls.csv) and checks, with Rule A's skipped phases.

levin-phase-floor 07, rehearsal agent, R29. New, after the stop-1 replay_summary.py. Reads the replay's CSV and
checks JSON and out/r29_vs_r28_records.csv; no store. Reports: the located calls (kinds, locations); the guarded
blocks; for every record that changed between R28 and R29, its V_i from the replay (location_skipped_phases); and
every replayed call that reports a skipped phase, by label, with its V range.

    ./venv/bin/python -I -B replay_summary_r29.py
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

Z = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tab, chk, chg = Z / "out" / "replay_calls.csv", Z / "out" / "replay_calls.checks.json", Z / "out" / "r29_vs_r28_records.csv"
    print(f"# script {Path(__file__).resolve()} sha256 {sha(__file__)}")
    for p in (tab, chk, chg):
        print(f"# input {p} sha256 {sha(p)}")
    rows = list(csv.DictReader(open(tab)))
    checks = json.loads(chk.read_text())
    key = lambda r: (float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"]), r["record"])
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    print(f"records in the table {len(by)}; all reproduce R29: {all(r['reproduces'] == 'True' for r in rows)}")
    loc = [r for r in rows if r["point_kind"]]
    print(f"located points {len(loc)}: by label {dict(Counter(r['label'] for r in loc))}; kinds {dict(Counter(r['point_kind'] for r in loc))}")
    gq = defaultdict(set)
    for r in loc:
        gq[(r["label"], r["k"], r["q"], r["r"])].add(round(float(r["location"]), 6) if r["label"] == "G+q+r" else float(f"{float(r['location']):.6g}"))
    for k, v in sorted(gq.items()):
        print(f"  {k}: locations {sorted(v)}")
    guarded = [v[0] for v in by.values() if v[0]["num_guarded_regions"] not in ("0", "", None)]
    print(f"records with a guarded region: {len(guarded)}")
    for r in guarded:
        print(f"  ({float(r['k']):.6g}, {float(r['q']):.6g}, {float(r['r']):.6g}, z {float(r['z_response']):.8g}) {r['record']}: guarded {r['num_guarded_regions']}, "
              f"skipped {r['location_skipped_phases']}")
    print("\nthe records that changed between R28 and R29, with V_i from the replay:")
    cc = list(csv.DictReader(open(chg)))
    vs = defaultdict(list)
    for c in cc:
        k = (float(c["k"]), float(c["q"]), float(c["r"]), float(c["z_response"]), c["record"])
        r = by[k][0]
        sk = json.loads(r["location_skipped_phases"]) if r["location_skipped_phases"] not in ("", "null") else None
        v = None if not sk else float.fromhex(sk[0]["variation"]["hex"])
        vs[c["label"]].append(v)
        print(f"  {'ok  ' if c['allowed'] == 'True' else 'FIND'} ({k[0]:.6g}, {k[1]:.6g}, {k[2]:.6g}, z {k[3]:.8g}) {c['record']} {c['label']}: "
              f"R29 points {r['num_stationary_points']}, guarded {r['num_guarded_regions']}; V = {v!r}" + (f" ({sk[0]['variation']['hex']})" if sk else ""))
    for lab, v in vs.items():
        vv = [x for x in v if x is not None]
        print(f"  {lab}: {len(v)} records, V from {min(vv)!r} to {max(vv)!r}; every one below SIX_PI = 18.84955592153876: {all(x < 18.84955592153876 for x in vv)}; missing {v.count(None)}")
    sk = checks["skipped_calls"]
    lab = defaultdict(list)
    for s in sk:
        l = s["notify_label"].split("group ")[-1] if "group " in s["notify_label"] else s["notify_label"]
        lab[l].append(float.fromhex(s["skipped"][0]["variation"]["hex"]))
    print(f"\nreplayed calls reporting a skipped phase: {len(sk)}; by label (count, V min, V max):")
    for l, v in sorted(lab.items()):
        print(f"  {l}: {len(v)}, {min(v)!r}, {max(v)!r}")
    print(f"  of them with points: {sum(1 for s in sk if s['num_stationary_points'])}; with a guarded region: {sum(1 for s in sk if s['num_guarded_regions'])}")


if __name__ == "__main__":
    main()
