"""levin-phase-floor prompt 07a section 6: write state/predictions.md, after Gate W0 and before the census is launched.
New in 07a. The prompt's text is copied from the prompt file in the tree h by line range, word for word, and checked to
begin and end where it should; V4's 33 calls are copied from 07's records (docs/levin-phase-floor/07/census/out/v4/
v4_scores.json, in the tree h); the rest is this agent's, marked as such. Refuses to overwrite.

Usage: python -B write_predictions_07a.py OUT_MD
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_TREE, header, sha256  # noqa: E402

PROMPT = H_TREE / "prompts" / "levin-phase-floor" / "07a-window-stop-and-phase-variation.md"
V4 = H_TREE / "docs" / "levin-phase-floor" / "07" / "census" / "out" / "v4" / "v4_scores.json"
RANGES = {  # (first, last) line, 1-based inclusive, and the text each must start with
    "s6": (525, 556, "**Predictions, fixed before the run**", "- **Cost:** regions and wall against 07's."),
    "s95": (716, 731, "### 9.5 Additions to §6's predictions", "    changed. The threshold is 57 times that."),
    "s104": (823, 825, "**Into §6's descriptive items:**", "abserr under each code, for that issue."),
}

OWN = r"""
## 4. This agent's additions, descriptive and not scored (the dispatch brief)

- **§10.4's item** (part 3), as the census lists it: every call that goes from converged (07) to unconverged (h) with a
  solve whose route changed and no window at 07 or at h, with its V, 07's estimate, and the abserr under each code. "A
  region whose route changed" is listed under both readings of part 5's question 1 (an own solve; any solve).
- **The largest |theta| sampled at a gated region's end**, by kind (group, analytic) and rtol: the largest of
  |theta(a)|, |theta(b)| over every non-forced solve's builds (section 6's diagnostics, ``theta_end_max_nonforced``),
  with its ulp.
- **The smallest margins |V - 6 pi| and |V - 20 pi|** among solves whose route and LU choice did not change (the
  diagnostics' ``min_margin_6pi_same`` and ``min_margin_20pi_same_levin``, the latter among solves on the Levin route),
  by kind and rtol, with their V and 07's estimate.

## 5. How each rule is read (this agent's, fixed here before the census; the open questions are the user's)

**The diagnostics** (worker.py, ``route_07a``; out/w0/ shows they change nothing). For every subregion solve the driver
makes in a call (each call of ``_adaptive_levin_subregion``), they record its span, its role, whether it was forced to
the nested Clenshaw-Curtis pair, and, for each build of its Levin data, V (h's), 07's estimate mean|theta'| * width from
the same returned theta_prime_Cheb and grid by the parent's expression, and whether V took the node-partition sum. A
solve's **decision differs** if at some build its C2 route differs (V < 6 pi against the estimate < 6 pi), or both
routes are Levin and its LU choice differs (> 20 pi). A forced solve (a window piece, a window leaf's halves and
quarters, a guarded leaf at depth_max) makes no route decision and is never counted as differing. Each solve is
classed by **role**:
- ``own``: its span is a region the driver made (a ``used_interval``: returned, or made and later bisected), so its
  value and error became, at some point, a region's own;
- otherwise by the description the driver gave it: ``quarter`` (a trust-check quarter: "trust check: ..."),
  ``comparison_half`` (a Levin leaf's step-(4) halves, "Levin left/right-comparison region", data_L/data_R),
  ``fallback_half`` (a fallback leaf's halves, solved when it is chosen), ``other``.
The harness's own re-solves after the timed section (06b's W3, 06c's deep lines, 06e's floor terms, 06h's declared terms
and its analytic counterfactual) are not the call's and are tallied apart (``route_07a_post``).

**Question 1 (open; the user's): what W1's "region whose route or LU choice differs" and W7's counts.** R30's 49
abserr-only records each had exactly one changed solve, never a region's own: a trust-check quarter or a comparison half.
Such a solve's value enters the call's declared error (the trust check's delta1 and delta2, a Levin leaf's residual), not
its value. Two readings, both scored:
- **reading A:** every solve the driver makes counts (own and auxiliary): W1 covers only calls in which no solve of any
  role changed;
- **reading B:** only a call's own region solves count: W1 also covers calls whose only changes are auxiliary solves,
  and predicts them bit-identical (R30 says they are not, in abserr).
The same choice applies to W7's "no region whose route or LU choice differs". The census's comparison classes every call
without a window as ``untouched`` (no solve changed), ``aux_only`` (only auxiliary solves changed) or ``own`` (an own
solve changed), and reports W1 and W7 under each reading. This agent does not choose.

**Question 2 (open; the user's): what W1's "equals 07's call bit for bit" compares.** Every call is diffed against 07's on
everything 07's comparison takes as the call's, and each differing path is put in one category:
- ``call``: the call's own outputs (value, abserr, converged, regions, leaves, its solve counts, its trust checks' and
  premise checks' records, guard_07's windows and counts, its scans);
- ``location``: guard_07's ``location_skipped_phases``, ``unlocated_minima`` and ``n_unlocated_minima``: the location's
  own report. §1.1 item 1 changes it by construction for every call whose phase Rule A skipped at 07: 07's entry
  carries "variation", h's (only for a rate identically zero) "reason", and a phase Rule A skipped is now scanned, so it
  can newly report an unlocated minimum even where it locates nothing and makes no window;
- ``harness``: the harness's own re-solves after the timed section (06e's floor terms, 06h's declared terms, 06c's and
  06e's deep lines, 06b's W3), which re-solve spans the driver may not have visited, under h's routing.
W1 is scored three ways: **strict** (no differing path), **as written for phase_span** (the ``location`` and ``harness``
categories set aside as phase_span is: each is a record, not a computed value of the call, and each differing call's
``route_07a_post`` or location record is shown to explain it), and the counts in each category. This agent does not
choose. (``phase_span`` itself is not a key of any record of the harness at 07 or at h: compare_07a.py counts its
occurrences in every unit compared, and sets it aside by showing it is absent.)

**W1's rows.** A unit all of whose calls are of W1's classes (under the reading) is scored on ``result.total`` and
``result.total_abserr``, hex, as written; its whole-unit diff is reported with its categories (the unit's 06h analytic
counterfactual is ``harness``).

**W2** is 07's V2 on the census's scan_07 (every group call at both rtols, every analytic call at 1e-7), as 07 scored it.

**W3** is 07's V3 on the scanned calls, as 07 scored it, with 07 §10's set-aside replaced by 07a's: a phase is set aside
only if its entry in ``location_skipped_phases`` says ``reason: zero_rate``; each such call is listed and its rate is
checked to be exactly zero on the scan (scan_07: no sign change, and min |theta'| / max |theta'| of 0 or no finite
non-zero sample). A skipped phase of any other kind fails W3.

**W4** is scored by 07's rule (score_v4_07.py's) on 07's readings, with "the before" now 07's census and "the new" h's:
the tolerance min(abserr_07, abserr_h)/100, the readings' agreement test unchanged in form, honest iff |v_h - ref| <=
abserr_h. A call whose readings do not agree under the new tolerance is reported, not scored, and its verdict under 07's
own tolerance is shown beside it. §9's amendment adopted no rule that leaves a call unconverged (§9.3), so W4's escape
clause "or is unconverged under a rule of §9's amendment" admits no call. The prediction of §9.5 (fails on exactly v08,
v09, v12, v17, v22 at 3.16, 2.21, 3.08, 2.38, 2.53 to three figures; holds on the other 28) is scored as written: to three
significant figures.

**W5:** the three counts are made on every solve of every call (forced included), each broken down by role, and the
node-partition count is also given for the gated (non-forced) solves alone, which is how Part B's review counted it
("493,516 regions gated"). "A Clenshaw-Curtis region with a half sent to the Levin solve": a non-forced solve that took
the nested pair (is_direct) and a visited non-forced solve of one of its halves (its break point (a + b)/2) that took
the Levin solve. The harness's own re-solves are not regions of the call and are not counted. ``num_guarded_regions`` is
guard_07's, as 07 recorded it.

**W6** is the cross-check's r_g at h, r_g = |v_h(1e-7) - v_h(1e-8)| / e_h(1e-7) (state/xcheck_rule_20261109.md's
part A, 07's values replaced by h's), on 07's 2,484 pairs (the calls 07 marked as locating or guarding), matched as 07
matched them. "A member unconverged under §9's rule": none can be, as for W4. The pairs h itself marks (newly located
calls included) are reported beside it, not scored.

**W7** is scored on the calls with a window at 07 and at h (guard_07 ``n_windows`` > 0 in both), no differing decision
(under each reading of question 1), and no newly located phase (a phase in 07's ``location_skipped_phases`` that h
located): |v_h - v_07| <= 1e-5 abserr_07 (exactly, as Fractions) and converged_h == converged_07.

**The over-budget pass:** a unit over budget in both passes is counted as not scored for W1 and W7 and listed.
"""


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    lines = PROMPT.read_text().splitlines()
    parts = {}
    for k, (a, b, start, end) in RANGES.items():
        seg = lines[a - 1:b]
        if not seg[0].startswith(start) or not seg[-1].startswith(end):
            raise SystemExit(f"{k}: lines {a}-{b} do not start/end as expected: {seg[0][:60]!r} ... {seg[-1][:60]!r}")
        parts[k] = "\n".join(seg)
    v4 = json.loads(V4.read_text())
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a section 6: the census's predictions, fixed before the census",
                 [PROMPT, V4])
    md = [l[2:] if l.startswith("# ") else l for l in hdr]
    out = ["# levin-phase-floor 07a §6 — the census's predictions, fixed before the census", ""]
    out += ["Provenance (the writer's header):", "", "```"] + md + ["```", ""]
    out += ["Written after Gate W0 (all four clauses: out/w0/, out/k0/compare_w0b.json, out/w0c3/compare_instr.json, "
            "out/w0c4/compare_det.json) and before the census is launched. Not to be rewritten: a correction is a new file "
            "and a deviation.", "",
            f"Source: `{PROMPT}` (file SHA-256 {sha256(PROMPT)}). Part 1 is lines {RANGES['s6'][0]}-{RANGES['s6'][1]} (§6, the "
            f"predictions and the descriptive items); part 2 is lines {RANGES['s95'][0]}-{RANGES['s95'][1]} (§9.5, the "
            f"additions: W4, W6 on r_g, W7); part 3 is lines {RANGES['s104'][0]}-{RANGES['s104'][1]} (§10.4's descriptive item). "
            "Each is copied unchanged, word for word, by line range. Part 4 (descriptive additions) and part 5 (how each rule "
            "is read, with the two questions left open for the user) are this agent's. Part 6 fixes V4's 33 calls.", ""]
    out += ["## 1. Prompt §6, the predictions and the descriptive items (verbatim)", "", parts["s6"], ""]
    out += ["## 2. Prompt §9.5, the additions to §6's predictions (verbatim)", "", parts["s95"], ""]
    out += ["## 3. Prompt §10.4, the descriptive item (verbatim)", "", parts["s104"], ""]
    out += OWN.strip("\n").splitlines() + [""]
    out += ["## 6. V4's 33 calls, fixed by name (07's set, from 07's records; W4 is scored on 07's readings)", "",
            f"From `{V4}` (SHA-256 {sha256(V4)}), its `calls` (the 24) and `z` (z00, z12 and Z3's seven). The unit is 07's "
            "name (`g-...`); at h it is `h-...`. Each call is the unit's Levin call of that index (group: `calls`; analytic: "
            "`other_calls`), matched at h by index, label and span as compare_07a.py matches calls.", "",
            "| call | unit | rtol | kind | index | 07's basis | 07's ref (30 digits) | 07's |v-ref|/abserr | 07's verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in v4["calls"]:
        out.append(f"| {c['sid']} | {c['unit']} | {c['rtol']} | group | {c['index']} | {c['basis']} | "
                   f"{c.get('readings_30', {}).get(c.get('ref_kind'), c.get('ref'))} | {c['ratio_g']:.4g} | {c['verdict']} |")
    for c in v4["z"]:
        out.append(f"| {c['sid']} | {c['unit']} | {c['rtol']} | {c.get('kind')} | {c['index']} | {c['basis']} | "
                   f"{c.get('ref')!r} | {c['ratio_g']:.4g} | {c['verdict']} |")
    out += ["", f"{len(v4['calls'])} + {len(v4['z'])} = {len(v4['calls']) + len(v4['z'])} calls.", ""]
    out_p.write_text("\n".join(out) + "\n")
    print("\n".join(out[:12]))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
