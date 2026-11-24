"""levin-phase-floor prompt 07d Part A' (section 9.2 item 1, "the reproduction is checked against the tree before it is
used"): checkrepro_a2.check at T = 6 pi with the tree's fallback, against I6's record of the same check, bit for bit, on
the checked Levin leaves the rule below selects. New in Part A'. The rule is fixed here and in
state/repro_selection_rule.txt before this runs.

**The selection** (from out/replays/I6/, checks with is_direct False and forced False, ordered by (sid, seq)):
- group: the first such check of each call, in sid order, for the first 150 calls that have one;
- analytic: likewise, 150 calls;
- the first 100 checks whose leaf's phase_span (I6's made record) is at least 20 pi (the least-squares path, the new band);
- the first 50 checks a rate rule refused;
- the first 50 checks with a Clenshaw-Curtis solve among their six sub-solves.
The union, each check once. Kind: "group " in the call's notify_label, else analytic (Part A's rule).

**Compared, as float.hex or exactly:** V_P, V_L + V_R, Q, delta1, delta2, r, nu_B, trusted (after the rate rules), refused,
the rule, the residual, phase_err, and the six sub-solves' routes. Prompt section 9.3: a single difference is a stop.
Writes OUT (refuses to overwrite).
Usage: PYTHONPATH=<tree>:<scripts> python -B verify_repro_a2.py OUT
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, TREE_LQ_SHA, header, sha256  # noqa: E402

FIELDS = ["seq", "a", "b", "depth", "is_direct", "forced", "V_P", "halves_sum", "Q", "delta1", "delta2", "r", "nu_B",
          "trusted", "refused_by_rate_rule", "rule", "residual", "phase_err", "sub_direct"]


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import AdaptiveLevin.levin_quadrature as LQ
    import checkrepro_a2 as CR
    from leafref_v4 import load_capture, manifest

    assert sha256(LQ.__file__) == TREE_LQ_SHA
    man = manifest()
    docs = {}
    for p in sorted((ROOT / "out" / "replays" / "I6").glob("*.json")):
        d = json.loads(p.read_text())
        docs[d["sid"]] = d
    cand = []
    for sid in sorted(docs):
        d = docs[sid]
        kind = "group" if "group " in d["notify_label"] else "analytic"
        made = {m[0]: m for m in d["a2_records"]["made"]}
        for c in sorted(d["a2_records"]["checks"], key=lambda c: c[0]):
            rec = dict(zip(FIELDS, c))
            if rec["is_direct"] or rec["forced"]:
                continue
            ps = made[rec["seq"]][5]
            rec["phase_span"] = None if ps is None else float.fromhex(ps)
            cand.append((sid, kind, rec, d))
    sel, seen = [], set()

    def take(x):
        k = (x[0], x[2]["seq"])
        if k not in seen:
            seen.add(k)
            sel.append(x)

    for kind in ("group", "analytic"):
        calls = []
        for x in cand:
            if x[1] == kind and x[0] not in calls:
                calls.append(x[0])
                take(x)
            if len(calls) >= 150:
                break
    for pred, n in ((lambda r: r["phase_span"] is not None and r["phase_span"] >= 20 * math.pi, 100),
                    (lambda r: r["refused_by_rate_rule"], 50), (lambda r: any(r["sub_direct"]), 50)):
        k = 0
        for x in cand:
            if pred(x[2]) and k < n:
                take(x)
                k += 1
    rows, ctx, caps = [], {}, {}
    t0 = time.time()
    for sid, kind, rec, d in sel:
        if sid not in ctx:
            c = man[sid]
            r = load_capture(c)
            kw = dict(r["kw"])
            kw["rtol"] = float.fromhex(d["rtol"])
            ctx = {sid: CR.CallContext(LQ, r, kw)}
        C = ctx[sid]
        got = C.check(float.fromhex(rec["a"]), float.fromhex(rec["b"]), LQ.SIX_PI, "cc47")
        want = {"V_P": rec["V_P"], "halves_sum": rec["halves_sum"], "Q": rec["Q"], "delta1": rec["delta1"],
                "delta2": rec["delta2"], "r": rec["r"], "nu_B": rec["nu_B"], "trusted": rec["trusted"],
                "refused_by_rate_rule": rec["refused_by_rate_rule"], "rule": rec["rule"], "residual": rec["residual"],
                "phase_err": rec["phase_err"], "sub_routes": ["cc" if x else "levin" for x in rec["sub_direct"]]}
        have = {k: got.get(k) for k in want}
        if have.get("r") is not None:
            have["r"] = float(have["r"]).hex()
        diff = sorted(k for k in want if have[k] != want[k]) if got.get("checked") else ["not checked"]
        rows.append({"sid": sid, "kind": kind, "seq": rec["seq"], "depth": rec["depth"], "phase_span": rec["phase_span"],
                     "equal": not diff, "differs": diff, "want": want if diff else None, "have": have if diff else None})
    n_eq = sum(r["equal"] for r in rows)
    by = {k: (sum(1 for r in rows if r["kind"] == k), sum(1 for r in rows if r["kind"] == k and r["equal"]))
          for k in ("group", "analytic")}
    summ = {"selected": len(rows), "equal": n_eq, "by_kind_selected_equal": by,
            "phase_span_ge_20pi": sum(1 for r in rows if (r["phase_span"] or 0) >= 20 * math.pi),
            "refused": sum(1 for s in sel if s[2]["refused_by_rate_rule"]),
            "mixed_routes": sum(1 for s in sel if any(s[2]["sub_direct"])),
            "untrusted": sum(1 for s in sel if not s[2]["trusted"]), "wall_s": time.time() - t0,
            "verdict": "HOLDS" if n_eq == len(rows) and len(rows) >= 200 and min(v[0] for v in by.values()) > 0 else "FAILS"}
    doc = {"header": header("levin-phase-floor 07d Part A': the reproduction of _check_leaf against I6", __file__,
                            [Path(CR.__file__)]), "summary": summ, "rows": rows}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
