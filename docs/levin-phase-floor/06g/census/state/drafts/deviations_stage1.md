# Census agent, stage 1 (2026-11-01, 11:05-11:20Z): deviations so far (draft for log.md)

Nothing computed: no store hash, no fingerprint, no unit, no Gate V, no tree import. Commands in out/cmd/00-05.

1. prov.py: trees p = -lpf06g-4f4f9ed (4f4f9ed), g = -lpf06g-1c5336e (1c5336e); 06h's run, records, rows and
   checkpoint SHA-256 (RUN06H, REC06H, ROWS06H, CHECKPOINT06H_SHA256); 06h's trees kept by name. STRUCTURALLY REQUIRED.
2. harness.py: job_name "levin-phase-floor-06g" (the brief); stored, never read. STRUCTURALLY REQUIRED.
3. run_census.py: docstring only. STRUCTURALLY REQUIRED.
4. launch.py: prompt 06g, purpose, scope, imports. STRUCTURALLY REQUIRED.
5. worker.py, tree evidence: 06g's markers and the premise's signature; the worker refuses a signature other than
   06h's or 06g's. STRUCTURALLY REQUIRED (the brief: check the signature the wrapper forwards).
6. worker.py, the premise pass-through forwards keywords (split_could_meet_budget=), and records per check the
   result, the budget clause as passed, the halving test alone (the pure original called again with no keyword)
   and the three floors. Forwarding: STRUCTURALLY REQUIRED (06h's wrapper would raise TypeError at g on every
   premise check, a unit error). The detail: IMPLEMENTATION CHOICE, allowed by the brief.
7. worker.py, every checked leaf: a pass-through at used_interval.__init__ (class attribute, timed section only,
   restored and asserted) keeps every region the driver makes; checked leaves (delta1 set), returned or bisected,
   recorded after the call. Inside the timed section, as 06h's premise and solve counters are. IMPLEMENTATION
   CHOICE, allowed by the brief (prompt section 5's per-checked-leaf record; 06g's test 3 counted leaves later
   bisected too, through a recorder on used_interval).
8. worker.py, per call: node_error_signed_sum, node_error_abs_sum, abserr_node_signed, node_limited, has_cc_leaf,
   abserr_eprime_sum and abserr_reassembles, values_06g, premise_06g, checked_06g (errors under error_06g).
   Allowed by the brief.
9. worker.py, returned-leaf arrays: value, value_own, value_kind, phase_err_declared, phase_err_roundoff,
   node_error_signed, phase_err_less_node, budget_err. Allowed by the brief.
10. worker.py, _budget_record parses 06g's record (16 arguments: 06h's 13 plus sum E, sum |E|, node-limited); 06h's
    parser kept it only as raw text. STRUCTURALLY REQUIRED.
11. worker.py, _cc_floor_terms and _declared_terms_06h keep the first _cc_roundoff_floor capture of each re-solve:
    06g's fallback solve calls it twice (the floor, then the floor less |E|), and the second call overwrote the
    capture, so every fallback leaf at g would have recorded "not reproduced". At p (one call) nothing changes.
    STRUCTURALLY REQUIRED.
12. gate_v.py (from gate_h.py): the before is 06h's census; the comparison at p also compares declared_terms_06h
    (pickle) and analytic_06h's own oracle (h: value, abserr, converged) and n_captured, which 06h's gate compared
    only between its two G workers; the brief's list printed call by call in hex; 06g's new pickle keys compared
    between G1 and G2, and every pickle key accounted for; the determinism unit g-1e-7-0-14164 with a 06g
    qualification test in the check (a CC call with sum E non-zero, a leaf returning its quarters); 06h's
    counterfactual check made descriptive. IMPLEMENTATION CHOICE.
13. gate_v_clause1.py (new, from gate_h_clause1.py): clause 1 is the prompt's (c1ec3b2..3ae6f3a against the rule,
    with the import list); clause 2 (c1ec3b2..4f4f9ed, CLAUDE.md or AST-equal: 06h's rule for its parent) and
    clause 3 (3ae6f3a..1c5336e empty) added. IMPLEMENTATION CHOICE.
14. scripts_equal.py (state/V), check_predictions.py (prompt 06g lines 401-422), checks_after_06g.py (06g's
    per-call checks), static_names.py (new). IMPLEMENTATION CHOICE.
15. 06h's oracle counterfactual (06e's BesselPhaseGroup, executed from 2c325f5) is still computed at g, unchanged:
    at g it is 06e's group phase under 06g's driver, used by no 06g score; it costs post-timed time, outside
    wall_s. Kept to change nothing that need not change. IMPLEMENTATION CHOICE.
16. out/cmd/00 was recorded by an inline equivalent of rec.sh (rec.sh is copied by that command), as 06h's 00.
    IMPLEMENTATION CHOICE.
17. 04b and 04c ran static_names.py while it held only its docstring (a BSD sed '1i' error when writing it); each
    printed nothing and exited 0, and is annotated; 04d is the check. UNINTENDED DRIFT (no record but the two
    command files touched).
18. Scripts are not black-formatted; they are records. IMPLEMENTATION CHOICE.
19. Text edits to the copied scripts were made with /opt/local/bin/python3 (3.14.8) string replacement, and the
    static check runs under it; neither imports any tree module. The parent's levin_quadrature.py was read via
    `git show 4f4f9ed:...` into the session scratchpad (not a record). IMPLEMENTATION CHOICE.

Observations (not acted on):
- W3 (1e-7 units) still evaluates 06b's premise formula on whole floors (halves' phase_err >= 1/2 own), not 06g's
  premise (round-off parts and the budget clause); kept as 06h's record. 06g's premise is in premise_06g.
- E_share (06c) still attributes the whole floor, |E| included; budget_err is recorded beside it.

# Stage 2 (11:36-11:43Z): Gate V, predictions, launch

20. gate_v.py's P-unit lines still print "vs 06e", "06e deep lines", "CC floor-term leaves 06e": labels left from
    06h's text; the comparison is with RUN06H (06h's census), as the "IDENTICAL in every value 06h recorded" line and
    the code say. UNINTENDED DRIFT (cosmetic, out/cmd/20).
21. out/cmd/11 piped rec.sh's output through sed, so the lock's listing shows that sed as a process of /tmp/ray-sgk
    (as 06h's deviation 18). 29 and the two ray-processes listings (12, 30: empty, none) do not. UNINTENDED DRIFT.
22. The smoke ledger's summary (22b) is scripts/smoke_summary.py; a first inline attempt failed on shell quoting
    before rec.sh ran and wrote nothing. IMPLEMENTATION CHOICE.
23. Part B's V8 cause classification was extended before the launch (the coordinator's message after R24): the
    ending (node_limited / depth cap / other) is read beside (b), giving five classes mapped onto Part A's three
    causes. The stage-1 draft is kept as state/drafts/predictions_partB_draft_stage1.md. IMPLEMENTATION CHOICE.
24. The determinism unit g-1e-7-0-14164 qualified (2 CC calls with sum E non-zero; 107 leaves returning their
    quarters), so the re-choice rule was not used.

# Stage 3 (17:23-17:28Z): after the run, V8 only

25. New scripts v8_06g.py (V8 by Part B rules 1-3; its loaders from analyse06h.py) and v8_breakdown.py
    (descriptive). The breakdown section of out/v8.md was appended by hand after the scoring run, marked as such.
    IMPLEMENTATION CHOICE.
26. The fingerprint after the run (42) was taken at g only, as 06h's was. IMPLEMENTATION CHOICE.
