"""[prompt 07b section 5 item 5, rehearsal agent, R31: R30's s2_score_r30.py (6fc3fb30), changed where marked "prompt 07b r31": the replay at h (replay07b_r31.py) and R31's records; the tag R30 read as R31, R31's call compared with R28's (stop 1) and R30's (07a's s2_scores.json); each call's refusals recorded; outputs under the 07b rehearsal.]

[prompt 07a section 5 item 5, rehearsal agent, R30: R29's s2_score_r29.py (e7430b57), changed where marked "prompt 07a r30": the replay at h (replay07a_r30.py) and R30's records; the tag R29 read as R30, with R30's call compared with R28's and R29's; outputs under the 07a rehearsal.]

S2 -- P1's two rows (prompt 07 section 5.6), re-scored on R29 against the standing p1-references readings.

levin-phase-floor 07, rehearsal agent, R29. New, after the stop-1 s2_score.py (whose rules and arithmetic it keeps).
Reads, read-only: the stop-1 records out/p1_references/calls.json (R27's call value and abserr, the pickle) and
out/p1_references/p1_row{1,2}.json (the four readings of run levin-phase-floor-07-p1-references-20261108T103132, taken
at R27's tree, which stand); the R29 replay (r29/state/capture/replay; the call reproducing R29's stored G+q+r group
bit for bit, r29/out/replay_calls.checks.json) for R29's call value, abserr and located point. The tolerance stays the
stop-1 one, min(abserr_R27, abserr_R28)/100; the count verdict is the tool's (scatter against it), and R29's abserr
against it is reported too.

    ./venv/bin/python -I -B s2_score_r29.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import mpmath as mp

R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
Z = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal")  # prompt 07b r31
Z30 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal")  # prompt 07b r31: R30's s2_scores.json, for R30's call
sys.path.insert(0, str(Z / "scripts" / "capture"))
import replay07b_r31 as RP  # noqa: E402  prompt 07b r31

P1 = {1: (9.2313499, 10211.3179), 2: (5.6204320, 275.00859)}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    cpath = R / "out" / "p1_references" / "calls.json"
    print(f"# input {cpath} sha256 {sha256(cpath)}")
    rows = json.loads((Z / "state" / "replay_rows.json").read_text())["rows"]
    units = RP.load_units(Z / "state" / "capture" / "replay")
    chk = {(tuple(c["key"]), c["path"]): c for c in json.loads((Z / "out" / "replay_calls.checks.json").read_text())["checks"]}
    out, holds = [], True
    s30p = Z30 / "out" / "s2_scores.json"  # prompt 07b r31
    print(f"# input {s30p} sha256 {sha256(s30p)}")
    s30 = json.loads(s30p.read_text())["rows"]
    for c in json.loads(cpath.read_text()):
        rp = Path(c["out"])
        print(f"# input {rp} sha256 {sha256(rp)}")
        doc = json.loads(rp.read_text())
        e = [e for e in rows if e["key"] == c["key"]][0]
        rec = [r for r in e["records"] if r["kind"] == "group" and r["label"] == "G+q+r"][0]
        u, _, _ = units[f"h-1e-7-{e['R31']['shard']}-{e['R31']['serial']}"]  # prompt 07b r31
        hits = [x for x in u["calls"] if x["notify_label"].endswith("group G+q+r") and RP.stored_form(x, e["key"][3]) == (rec["R31"]["value"], rec["R31"]["abserr"])]  # prompt 07b r31
        assert len(hits) == 1
        call = hits[0]
        g = call["guard_07"]
        xa, xb = float.fromhex(g["x_span"][0]["hex"]), float.fromhex(g["x_span"][1]["hex"])
        h = abs(xb - xa) / 4096
        pts = g["points"]
        xs, zs = P1[c["p1"]]
        x = float.fromhex(pts[0]["x"]["hex"]) if pts else None
        loc_ok = len(pts) == 1 and pts[0]["kind"] == "root" and abs(x - xs) <= h
        ref = mp.mpf(doc["value_30"])
        row = {"p1": c["p1"], "key": c["key"], "replay_reproduces_record": chk[(tuple(c["key"]), rec["path"])]["reproduces"],
               "x": None if x is None else pts[0]["x"]["hex"], "z": None if x is None else repr(math.exp(x) - 1.0),
               "abs_x_minus_x_s": None if x is None else repr(abs(x - xs)), "scan_spacing_x": repr(h), "location_ok": loc_ok,
               "counts": doc["counts"], "scatter": doc["scatter"], "tolerance": doc["tolerance"], "ref_hex": doc["value_hex"], "ref_30": doc["value_30"],
               "R28_value_stop1": c["R28_value"], "R31_equals_R28": call["value"]["hex"] == c["R28_value"] and call["abserr"]["hex"] == c["R28_abserr"],
               "R31_equals_R30": (call["value"]["hex"], call["abserr"]["hex"]) == tuple([r_ for r_ in s30 if r_["p1"] == c["p1"]][0]["R30"][k_] for k_ in ("value", "abserr")),
               "R30_ratio": [r_ for r_ in s30 if r_["p1"] == c["p1"]][0]["R30"]["ratio"], "refusals_07b": call.get("refusals_07b")}  # prompt 07b r31
        for tag, v_, e_ in (("R27", c["R27_value"], c["R27_abserr"]), ("R31", call["value"]["hex"], call["abserr"]["hex"])):  # prompt 07b r31
            v, ee = mp.mpf(float.fromhex(v_)), mp.mpf(float.fromhex(e_))
            d = abs(v - ref)
            row[tag] = {"value": v_, "abserr": e_, "abs_v_minus_ref": mp.nstr(d, 6), "ratio": mp.nstr(d / ee, 6), "honest": bool(d <= ee)}
        row["honesty_ok"] = row["counts"] and row["R31"]["honest"]  # prompt 07b r31
        holds &= loc_ok and row["honesty_ok"]
        out.append(row)
        print(f"row {c['p1']} {c['key']}: replay reproduces R31 {row['replay_reproduces_record']}; root at z {row['z']} (|x - x_s| {float(row['abs_x_minus_x_s']):.3g} "
              f"against {h:.4g}: {'within' if loc_ok else 'NOT within'}); readings count {row['counts']} (scatter {row['scatter']:.3g}, tolerance {row['tolerance']:.4g}); "
              f"ref {row['ref_hex']}; R31 value {row['R31']['value']} abserr {row['R31']['abserr']}: {row['R31']['ratio']} x abserr; "
              f"R27 {row['R27']['ratio']}; R30 {row['R30_ratio']}; R31 equals R28 {row['R31_equals_R28']}; R31 equals R30 {row['R31_equals_R30']}; "
              f"refusals {row['refusals_07b']}")  # prompt 07b r31
    print(f"\nS2 on R31 {'HOLDS' if holds else 'FAILS'}")  # prompt 07b r31
    dst = Z / "out" / "s2_scores.json"
    dst.write_text(json.dumps({"holds": holds, "rows": out}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
