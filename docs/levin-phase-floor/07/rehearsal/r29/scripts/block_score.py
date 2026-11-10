"""Score R27, R28 and R29 on the four blocks against block_reference.py's references.

levin-phase-floor 07, rehearsal agent, R29. New. Reads out/block_reference/*.json and out/blocks.json. For each block:
the three trees' captures replay bit for bit; their integrands are bit-identical at 1001 points (x, integrand, theta,
theta_mod_2pi, theta_deriv) with equal x_span and kw; each store's value and abserr (the call's own, from the replay of
its capture, equal to the harness's record which reproduces the store's block record); the reference, its stated
error and the GL sequence; |v - ref|/abserr per store in mpmath at 40 digits; V (the trapezoid on the 4,097-point scan,
and R29's location_skipped_phases) and the gate's span estimate on the root leaf.

    ./venv/bin/python -I -B block_score.py
"""

import hashlib
import json
from pathlib import Path

import mpmath as mp

Z = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29")
BLOCKS = [("J4", "0-10171", "0-20452", "0-157"), ("Y4", "0-10171", "0-20452", "0-157"),
          ("J4", "1-186", "1-10349", "1-10283"), ("Y4", "1-186", "1-10349", "1-10283")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    bj = Z / "out" / "blocks.json"
    print(f"# input {bj} sha256 {sha256(bj)}")
    blocks = json.loads(bj.read_text())["blocks"]
    out = []
    for blk, sp, so, sg in BLOCKS:
        docs = {}
        for t, s in (("p", sp), ("o", so), ("g", sg)):
            f = Z / "out" / "block_reference" / f"{blk}-{t}-{s}.json"
            print(f"# input {f} sha256 {sha256(f)}")
            docs[t] = json.loads(f.read_text())
        g = docs["g"]
        key = [b for b in blocks if b["tree"] == "g" and b["block"] == blk and b["uid"].endswith(sg)][0]["key"]
        same = docs["p"]["samples"] == docs["o"]["samples"] == g["samples"]
        same_kw = docs["p"]["x_span"] == docs["o"]["x_span"] == g["x_span"] and docs["p"]["kw"] == docs["o"]["kw"] == g["kw"]
        ref = mp.mpf(float.fromhex(g["reference"]["reference"]))
        row = {"key": key, "block": f"2pt5 {blk}", "integrands_bit_identical_3_trees": same, "x_span_kw_equal": same_kw,
               "x_span": g["x_span"], "root_x": g["roots"], "root_eta": g["root_eta"],
               "V_trapezoid_4097": g["V_trapezoid_4097"], "R29_location_skipped_phases": g["replay"]["location_skipped_phases"],
               "gate_span_root_leaf": g["gate_span_root_leaf"],
               "reference": g["reference"]["reference"], "reference_repr": g["reference"]["reference_repr"],
               "stated_error": g["reference"]["stated_error"], "stated_error_hex": g["reference"]["stated_error_hex"],
               "differences": g["reference"]["differences"], "QAGS_abserr": g["reference"]["QAGS"]["abserr"]}
        for t, tag in (("p", "R27"), ("o", "R28"), ("g", "R29")):
            dd = docs[t]
            v, e = mp.mpf(float.fromhex(dd["capture_record"]["value"])), mp.mpf(float.fromhex(dd["capture_record"]["abserr"]))
            d_ = abs(v - ref)
            row[tag] = {"value": dd["capture_record"]["value"], "abserr": dd["capture_record"]["abserr"],
                        "regions": dd["capture_record"]["num_regions"], "converged": dd["capture_record"]["converged"],
                        "replay_reproduces": dd["replay_reproduces_capture"], "ray_initialized": dd["ray_initialized"],
                        "abs_v_minus_ref": mp.nstr(d_, 6), "ratio": mp.nstr(d_ / e, 6), "honest": bool(d_ <= e),
                        "stated_error_over_abserr": mp.nstr(mp.mpf(g["reference"]["stated_error"]) / e, 4)}
        out.append(row)
        print(f"\n{key} 2pt5 {blk}: integrands bit-identical at 1001 points across R27/R28/R29 trees {same}; x_span and kw equal {same_kw}")
        print(f"  root x {g['roots']} (eta {g['root_eta']}); V (trapezoid, 4097-point scan) {g['V_trapezoid_4097']['repr']} "
              f"({g['V_trapezoid_4097']['hex']}); R29's location_skipped_phases {g['replay']['location_skipped_phases']}; "
              f"gate span on the root leaf {g['gate_span_root_leaf']['repr']} ({g['gate_span_root_leaf']['hex']}) against SIX_PI {g['gate_span_root_leaf']['SIX_PI']}")
        print(f"  reference {row['reference']} ({row['reference_repr']}), stated error {row['stated_error']} ({row['stated_error_hex']}); "
              f"differences {row['differences']}; QAGS own abserr {row['QAGS_abserr']}")
        for tag in ("R27", "R28", "R29"):
            x = row[tag]
            print(f"  {tag}: value {x['value']} abserr {x['abserr']} regions {x['regions']} converged {x['converged']} replay {x['replay_reproduces']}: "
                  f"|v - ref| {x['abs_v_minus_ref']} = {x['ratio']} x abserr ({'honest' if x['honest'] else 'NOT honest'}); stated error = {x['stated_error_over_abserr']} x abserr")
    dst = Z / "out" / "block_scores.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
