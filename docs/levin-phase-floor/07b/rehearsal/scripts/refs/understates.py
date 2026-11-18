"""levin-phase-floor 07b section 5 (Part C), the R31 rehearsal agent: which store understates, for the 7 group records
whose |v31 - v30| > e30 + e31 (the user's word at stop 1; descriptive, not scored against any rule). New. Scores by the
rule fixed in state/refs_plan_r31.md before any reading (its mtime and SHA-256 in state/refs_plan_r31.mtime.txt):

- the reference ref is the tool's FCC r2 (``value_30``); its stated error s is the four readings' scatter (largest minus
  smallest), recomputed here in mpmath from the readings' ``value_30`` and taken as the larger of that and the tool's own
  ``scatter``;
- the reference does not decide if s > min(e30, e31)/100: reported, no verdict;
- a store understates if |v - ref| - e > s; is honest if e - |v - ref| > s; otherwise undecided for that store;
- v and e are each store's call value and abserr, in the call's own units, from the bit-checked replays of the captured
  call (out/refs/refcap_replay_a07.json, R30's record at 07a's module; out/refs/refcap_replay_h.json, R31's at the tree).

Arithmetic in mpmath at 40 digits from the doubles (float.hex) and the 30-digit readings. Reads only; writes OUT_JSON and
OUT_TXT (refuses to overwrite). Usage: python -I -B understates.py OUT_JSON OUT_TXT
"""

import hashlib
import json
import sys
from pathlib import Path

from mpmath import mp, mpf, nstr

mp.dps = 40
Z = Path(__file__).resolve().parents[2]
READ = Z / "out" / "refs" / "readings"
H = Z / "out" / "refs" / "refcap_replay_h.json"
A07 = Z / "out" / "refs" / "refcap_replay_a07.json"
PLAN = Z / "state" / "refs_plan_r31.md"
PLAN_SHA = "2c2514b017ee1974decc6aeb295c541bfbc2c0d8eb5979a7057577c74baef096"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out_j, out_t = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (out_j, out_t):
        if p.exists():
            raise SystemExit(f"{p} exists")
    assert sha256(PLAN) == PLAN_SHA, "the plan changed after it was fixed"
    lines = [f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}",
             f"# plan {PLAN} sha256 {sha256(PLAN)} (the rule, fixed before any reading)"]
    for p in [H, A07] + sorted(READ.glob("*.json")):
        lines.append(f"# input {p} sha256 {sha256(p)}")
    h = {Path(c["pickle"]).stem: c for c in json.loads(H.read_text())["calls"]}
    a = {Path(c["pickle"]).stem: c for c in json.loads(A07.read_text())["calls"]}
    rows = []
    for p in sorted(READ.glob("*.json")):
        sid = p.stem
        d = json.loads(p.read_text())
        ch, ca = h[sid], a[sid]
        assert ch["reproduces"] and ca["reproduces"] and ch["key"] == ca["key"]
        rd = {f"{r['impl']}_r{r['res']}": r for r in d["readings"]}
        vals = {k: mpf(r["value_30"]) for k, r in rd.items()}
        ref = mpf(rd["fcc_r2"]["value_30"])
        assert d["value_30"] == rd["fcc_r2"]["value_30"]
        s_mp = max(vals.values()) - min(vals.values())
        s = max(s_mp, mpf(d["scatter"]))
        direct = {k: sum(1 for g in (r.get("segments") or []) if isinstance(g, dict) and g.get("direct")) for k, r in rd.items()}
        stores = {}
        for tag, c in (("R30", ca), ("R31", ch)):
            v, e = mpf(float.fromhex(c["got"]["value"])), mpf(float.fromhex(c["got"]["abserr"]))
            dv = abs(v - ref)
            if dv - e > s:
                verdict = "understates"
            elif e - dv > s:
                verdict = "honest"
            else:
                verdict = "undecided"
            stores[tag] = {"value": c["got"]["value"], "abserr": c["got"]["abserr"], "regions": c["got"]["regions"],
                           "converged": c["got"]["converged"], "abs_v_minus_ref": nstr(dv, 8), "ratio": nstr(dv / e, 8),
                           "ratio_low": nstr((dv - s) / e, 8), "ratio_high": nstr((dv + s) / e, 8), "verdict": verdict}
        e_min = min(mpf(float.fromhex(ca["got"]["abserr"])), mpf(float.fromhex(ch["got"]["abserr"])))
        decides = s <= e_min / 100
        if not decides:
            answer = "the reference does not decide"
        else:
            u = [t for t in ("R30", "R31") if stores[t]["verdict"] == "understates"]
            und = [t for t in ("R30", "R31") if stores[t]["verdict"] == "undecided"]
            answer = ("both understate" if len(u) == 2 else f"{u[0]} understates" if u else "neither understates")
            if und:
                answer += f" ({' and '.join(und)} undecided)"
        row = {"sid": sid, "key": ch["key"], "record": ch["record"], "label": ch["label"], "rtol": ch["rtol"],
               "notify_label": ch["notify_label"], "pickle_sha256": ch["sha256"],
               "reference": {"value_hex": rd["fcc_r2"]["value_hex"], "value_30": rd["fcc_r2"]["value_30"],
                             "readings_30": {k: r["value_30"] for k, r in rd.items()},
                             "readings_hex": {k: r["value_hex"] for k, r in rd.items()},
                             "stated_error": nstr(s, 6), "scatter_tool": d["scatter"], "scatter_mp": nstr(s_mp, 6),
                             "tolerance_min_e_over_100": nstr(e_min / 100, 6), "counts_tool": d["counts"],
                             "stated_error_over_min_e": nstr(s / e_min, 4), "route": d["route"],
                             "direct_segments": direct, "wall_s": {k: r.get("wall_s") for k, r in rd.items()},
                             "method": "tools/filon_reference.py reference(): QAWO and FCC at resolutions 1 and 2; ref FCC r2"},
               "R30": stores["R30"], "R31": stores["R31"],
               "dv_over_e30_plus_e31": nstr(abs(mpf(float.fromhex(ch["got"]["value"])) - mpf(float.fromhex(ca["got"]["value"])))
                                            / (mpf(float.fromhex(ch["got"]["abserr"])) + mpf(float.fromhex(ca["got"]["abserr"]))), 6),
               "decides": decides, "answer": answer}
        rows.append(row)
        lines.append(f"{ch['key']} {ch['label']} ({sid}): ref {row['reference']['value_hex']} (FCC r2), stated error {row['reference']['stated_error']} "
                     f"= {row['reference']['stated_error_over_min_e']} x min(e); direct segments {direct}")
        for t in ("R30", "R31"):
            st = stores[t]
            lines.append(f"    {t}: v {st['value']} e {st['abserr']} regions {st['regions']} converged {st['converged']}: |v - ref| = "
                         f"{st['ratio']} e (with the reference's error {st['ratio_low']} to {st['ratio_high']}): {st['verdict']}")
        lines.append(f"    |v31 - v30|/(e30 + e31) {row['dv_over_e30_plus_e31']}; ANSWER: {answer}")
    summary = {}
    for r in rows:
        summary[r["answer"]] = summary.get(r["answer"], 0) + 1
    lines.append(f"SUMMARY over {len(rows)} records: {summary}")
    out_j.write_text(json.dumps({"header": lines[: 2 + len(list(READ.glob('*.json'))) + 2], "rows": rows, "summary": summary}, indent=1))
    out_t.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_j} sha256 {sha256(out_j)} and {out_t} sha256 {sha256(out_t)}")


if __name__ == "__main__":
    main()
