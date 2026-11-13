"""S4 -- O9's four blocks (prompt 07a section 5 item 6), scored on R30 against R29's block references.

levin-phase-floor 07a, rehearsal agent, R30. New, after R29's block_score.py (e62946d1), whose arithmetic it keeps.
Reads: R30's block replays at the tree h (out/block_reference/{J4,Y4}-h-*.json, block_reference_r30.py, each a replay
of the pass-2 capture that replay07a_r30.py's blockcheck checked against R30's store record, out/blocks.json); R29's
references and R29's own g-tree replays of the same blocks (var/levin-phase-floor-07/rehearsal/r29/out/block_reference/
{J4,Y4}-g-*.json, read only); R29's block_scores.json for R27/R28/R29's ratios (context only).

The rule (section 5 item 6): in R30 each block locates its root, guards no leaf, and is honest against R29's block
references. Read as: the replay reproduces the capture (which reproduces R30's store record); the integrand R30's
capture holds is bit-identical to R29's at its 1001 sample points (x, integrand, theta, theta_mod_2pi, theta_deriv),
with equal x_span and kw (so R29's reference is a reference for it); exactly one stationary point, a root, whose x
lies within one 4097-point scan spacing of the reference's root (brentq on 40,001 samples); num_guarded_regions 0;
and |v30 - ref| <= abserr30, in mpmath at 40 digits; with the reference's stated error reported against abserr30.

    ./venv/bin/python -I -B block_score_r30.py
"""

import hashlib
import json
from pathlib import Path

import mpmath as mp

Z = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal")
Z29 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29")
# (block, R30's shard-serial, R29's shard-serial)
BLOCKS = [("J4", "0-10129", "0-157"), ("Y4", "0-10129", "0-157"), ("J4", "1-20022", "1-10283"), ("Y4", "1-20022", "1-10283")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    bj = Z / "out" / "blocks.json"
    print(f"# input {bj} sha256 {sha256(bj)}")
    blocks = json.loads(bj.read_text())["blocks"]
    s29p = Z29 / "out" / "block_scores.json"
    print(f"# input {s29p} sha256 {sha256(s29p)}")
    s29 = json.loads(s29p.read_text())
    out, holds = [], True
    for blk, sh, sg in BLOCKS:
        fh = Z / "out" / "block_reference" / f"{blk}-h-{sh}.json"
        fg = Z29 / "out" / "block_reference" / f"{blk}-g-{sg}.json"
        for f in (fh, fg):
            print(f"# input {f} sha256 {sha256(f)}")
        h, g = json.loads(fh.read_text()), json.loads(fg.read_text())
        b = [x for x in blocks if x["block"] == blk and x["uid"].endswith(sh)]
        assert len(b) == 1
        b = b[0]
        print(f"#   tree of R30's replay: {h['header']['tree']} HEAD {h['header']['tree_head']} status {h['header']['tree_status_short']!r}; "
              f"levin_quadrature {h['header']['levin_quadrature_sha256']}; pickle {h['header']['pickle']} sha256 {h['header']['pickle_sha256']}")
        same = h["samples"] == g["samples"]
        same_kw = h["x_span"] == g["x_span"] and h["kw"] == g["kw"]
        ref = mp.mpf(float.fromhex(g["reference"]["reference"]))
        err = mp.mpf(g["reference"]["stated_error"])
        v, e = mp.mpf(float.fromhex(h["capture_record"]["value"])), mp.mpf(float.fromhex(h["capture_record"]["abserr"]))
        d_ = abs(v - ref)
        rp = h["replay"]
        xa, xb = float.fromhex(h["x_span"][0]), float.fromhex(h["x_span"][1])
        spacing = abs(xb - xa) / 4096
        root_ref = float.fromhex(g["roots"][0])
        pts = rp["points"]
        loc_ok = len(pts) == 1 and pts[0]["kind"] == "root" and abs(float.fromhex(pts[0]["x"]) - root_ref) <= spacing
        guard_ok = rp["num_guarded_regions"] == 0
        honest = bool(d_ <= e)
        ok = (h["replay_reproduces_capture"] and b["reproduces_store"] and same and same_kw and loc_ok and guard_ok and honest
              and not h["ray_initialized"])
        holds &= ok
        ctx = [r for r in s29 if r["block"] == f"2pt5 {blk}" and r["key"] == b["key"]][0]
        row = {"key": b["key"], "block": f"2pt5 {blk}", "replay_reproduces_capture": h["replay_reproduces_capture"],
               "capture_reproduces_R30_store": b["reproduces_store"], "integrand_bit_identical_to_R29_capture": same,
               "x_span_kw_equal_to_R29": same_kw, "points": pts, "root_reference": g["roots"][0],
               "abs_x_minus_root_ref": repr(abs(float.fromhex(pts[0]["x"]) - root_ref)) if pts else None,
               "scan_spacing_x": repr(spacing), "location_ok": loc_ok, "num_guarded_regions": rp["num_guarded_regions"],
               "location_skipped_phases": rp["location_skipped_phases"],
               "R30": {"value": h["capture_record"]["value"], "abserr": h["capture_record"]["abserr"],
                       "regions": h["capture_record"]["num_regions"], "converged": h["capture_record"]["converged"],
                       "phase_limited": rp["phase_limited"], "abs_v_minus_ref": mp.nstr(d_, 6), "ratio": mp.nstr(d_ / e, 6),
                       "honest": honest, "stated_error_over_abserr": mp.nstr(err / e, 4)},
               "reference": g["reference"]["reference"], "reference_repr": g["reference"]["reference_repr"],
               "stated_error": g["reference"]["stated_error"],
               "context_R27_R28_R29": {t: {k: ctx[t][k] for k in ("value", "abserr", "regions", "ratio")} for t in ("R27", "R28", "R29")},
               "R30_equals_R28": (h["capture_record"]["value"], h["capture_record"]["abserr"]) == (ctx["R28"]["value"], ctx["R28"]["abserr"]),
               "holds": ok}
        out.append(row)
        print(f"\n{b['key']} 2pt5 {blk}: replay reproduces capture {row['replay_reproduces_capture']}, capture reproduces R30's store {row['capture_reproduces_R30_store']}; "
              f"integrand bit-identical to R29's capture at 1001 points {same}; x_span and kw equal {same_kw}")
        print(f"  located {pts} (reference root {g['roots'][0]}; |dx| {row['abs_x_minus_root_ref']} against one spacing {spacing:.4g}: {'within' if loc_ok else 'NOT within'}); "
              f"guarded regions {rp['num_guarded_regions']}; skipped phases {rp['location_skipped_phases']}")
        print(f"  R30: value {row['R30']['value']} abserr {row['R30']['abserr']} regions {row['R30']['regions']} converged {row['R30']['converged']}: "
              f"|v - ref| {row['R30']['abs_v_minus_ref']} = {row['R30']['ratio']} x abserr ({'honest' if honest else 'NOT honest'}); "
              f"reference {row['reference']} ({row['reference_repr']}), stated error {row['stated_error']} = {row['R30']['stated_error_over_abserr']} x abserr30")
        print(f"  context (R29's scores): " + "; ".join(f"{t} {c['ratio']} x abserr (regions {c['regions']})" for t, c in row["context_R27_R28_R29"].items())
              + f"; R30 equals R28 bit for bit {row['R30_equals_R28']}")
    print(f"\nS4 {'HOLDS' if holds else 'FAILS'}: {sum(r['location_ok'] for r in out)} of 4 locate their root; "
          f"{sum(r['num_guarded_regions'] == 0 for r in out)} of 4 guard no leaf; {sum(r['R30']['honest'] for r in out)} of 4 honest")
    dst = Z / "out" / "block_scores.json"
    dst.write_text(json.dumps({"holds": holds, "blocks": out}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
