"""06f review, check 3 (the orchestrator's own reader): M1 re-scored from the agent's per-leaf tables
(out/replay/<unit>_<index>/leaves.json.gz), independently of replay06f.py's m1().

For each call named on the command line:
- every Clenshaw-Curtis leaf's floor is rebuilt from its recorded terms in the code's own order
  (levin_quadrature.py _cc_roundoff_floor: ((n+4)uS + u*arg) + (|E| + R_g + R_2 + R_c), then
  + declared, as _adaptive_levin_subregion_cc forms "phase_err"), and its leaf error from
  used_interval.total_err (max(resolution, phase_err) [+ truncation]), and both are checked
  bit for bit against the leaf's recorded phase_err and total_err;
- the sum of total_err over all leaves, left to right as _adaptive_levin sums it, is checked
  against the call's abserr (result.json);
- sum|E| (fsum), |sum E| (fsum), their ratio, and the signed-combination error: each CC leaf's
  error rebuilt with |E| taken out of its node term (everything else as recorded), summed over the
  leaves, plus |sum E|. Also the literal form abserr - sum|E| + |sum E|.
For Z2's call, sum|E| over its returned CC leaves is compared with the 06e checkpoint's
cc_floor.sums_all_cc_leaves.abs_E, bit for bit.

Usage: python -I orch_m1.py UNIT_INDEX [UNIT_INDEX ...]"""

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
V = MAIN / "var/levin-phase-floor-06f"
RUN06E = MAIN / "var/runs/levin-phase-floor-06e-census-20261030T081318"


def H(x):
    return None if x is None else float.fromhex(x["hex"])


def checkpoint_call(unit, index):
    with open(RUN06E / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{unit}"' not in line[:300]:
                continue
            d = json.loads(line)["data"]
            for c in (d.get("calls") or []) + (d.get("other_calls") or []):
                if c["index"] == index:
                    return c
    return None


def leaf_err(res, phase_err, trunc):
    if res is None:
        base = phase_err
    elif phase_err is None:
        base = res
    else:
        base = max(res, phase_err)
    return base if trunc is None else base + trunc


def one(tag):
    unit, index = tag.rsplit("_", 1)
    index = int(index)
    d = V / "out/replay" / tag
    leaves = json.load(gzip.open(d / "leaves.json.gz", "rt"))
    res = json.loads((d / "result.json").read_text())
    print(f"== {tag}  leaves.json.gz sha256 {hashlib.sha256((d / 'leaves.json.gz').read_bytes()).hexdigest()}")
    c = checkpoint_call(unit, index)
    abserr = float.fromhex(c["abserr"]["hex"])
    value = float.fromhex(c["value"]["hex"])
    request = max(c["atol"], c["rtol"] * abs(value))
    leaves = sorted(leaves, key=lambda L: H(L["a"]))
    bad_pe = bad_te = 0
    errs, errs_noE, E = [], [], []
    for L in leaves:
        resol = H(L["declared_resolution"]) if L["declared_resolution"] is not None else H(L["abserr"])
        trunc = H(L["abserr_truncation"])
        if L["branch"] == "cc":
            e = H(L["E"])
            E.append(e)
            node = abs(e) + H(L["R_g"]) + H(L["R_2"]) + H(L["R_c"])
            first2 = H(L["t_value_rounding"]) + H(L["t_argument"])
            pe = (first2 + node) + H(L["declared"])
            bad_pe += pe.hex() != H(L["phase_err"]).hex()
            te = leaf_err(resol, pe, trunc)
            node0 = H(L["R_g"]) + H(L["R_2"]) + H(L["R_c"])
            pe0 = (first2 + node0) + H(L["declared"])
            errs_noE.append(leaf_err(resol, pe0, trunc))
        else:
            te = leaf_err(resol, H(L["phase_err"]), trunc)
            errs_noE.append(te)
        bad_te += te.hex() != H(L["total_err"]).hex()
        errs.append(te)
    tot = 0.0
    for x in errs:
        tot = tot + x
    rest = 0.0
    for x in errs_noE:
        rest = rest + x
    s_abs = math.fsum(abs(x) for x in E)
    abs_s = abs(math.fsum(E))
    print(f"  leaves {len(leaves)}, CC {len(E)}; phase_err rebuild mismatches {bad_pe}; total_err rebuild mismatches {bad_te}")
    print(f"  sum of total_err (left to right) {tot!r} vs abserr {abserr!r}: equal {tot.hex() == abserr.hex()}; fsum {math.fsum(errs)!r}")
    print(f"  sum|E| {s_abs!r}  |sum E| {abs_s!r}  ratio {s_abs / abs_s if abs_s else None!r}  sqrt(N) {math.sqrt(len(E))!r}")
    print(f"  request max(atol, rtol|v|) {request!r}  declared error {abserr!r}")
    sc = rest + abs_s
    lin = abserr - s_abs + abs_s
    print(f"  signed-combination (leafwise) {sc!r}  /request {sc / request!r}  <= request {sc <= request}")
    print(f"  signed-combination (literal: abserr - sum|E| + |sum E|) {lin!r}  /request {lin / request!r}  <= request {lin <= request}")
    m = res.get("M1") or {}
    agent = {k: m.get(k) for k in ("sum_absE", "abs_sumE", "signed_combination_error", "signed_combination_error_linear")}
    mine = {"sum_absE": s_abs, "abs_sumE": abs_s, "signed_combination_error": sc, "signed_combination_error_linear": lin}
    print("  agent's result.json M1:", json.dumps(agent))
    print("  equal to mine bit for bit:", {k: (agent[k] is not None and float(agent[k]).hex() == mine[k].hex()) for k in mine})
    if unit == "g-1e-8-0-33995":
        ck = c["cc_floor"]["sums_all_cc_leaves"]["abs_E"]["hex"]
        print(f"  checkpoint cc_floor.sums_all_cc_leaves.abs_E {ck} vs my fsum {s_abs.hex()}: equal {ck == s_abs.hex()}")
    return s_abs, abs_s, sc


def main():
    print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print("# 06e checkpoint sha256", hashlib.sha256((RUN06E / "checkpoint.jsonl").read_bytes()).hexdigest())
    for t in sys.argv[1:]:
        one(t)


if __name__ == "__main__":
    main()
