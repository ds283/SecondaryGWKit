# Orchestrator prompt — Prompt 15 (`PrimitivePhase` explicit rate)

You are orchestrating prompt 15 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

This is not a workstream — it is a single interim prompt, dispatched at the user's request ahead
of Workstream E, to close `[10-primitive-phase-leading-rate-is-hardcoded]`: `PrimitivePhase`
hard-wires its closed-form leading derivative to `k/H`, right for the Green's function's `tau` but
wrong by `1/c_s` for the transfer function's `cs_tau`; prompt 10 worked around it with a
`_SoundHorizonRate` adapter rather than editing `primitive_phase.py`, which was outside its file
list. Prompt 15 adds an explicit `rate` callable and drops the adapter.

## What to read

The prompt: [`../15-primitive-phase-explicit-rate.md`](../15-primitive-phase-explicit-rate.md) —
read it in full before dispatch, it is short. `../README.md` §3 (the row-15 note), §5;
`../IMPLEMENTATION_STATE.md` — rows 09 and 10, M15, and the
`[10-primitive-phase-leading-rate-is-hardcoded]` entry in §3; `orchestrator/README.md`;
`logs/09-…` and `logs/10-…` "State handed to the next prompt".

## Preconditions

`git status` clean; board rows 09 and 10 show ✅ or ⚠️. **The prompt file cites line numbers in
`ComputeTargets/primitive_phase.py` and `ComputeTargets/TkSourceFunctions.py`** (written against
the tree at commit `121de53`) — before dispatch, confirm they still hold:

```bash
sed -n '273,291p' ComputeTargets/primitive_phase.py
sed -n '98,103p;203,232p;393,402p' ComputeTargets/TkSourceFunctions.py
```

If an intervening commit moved or changed these regions, do not silently hand the subagent a stale
prompt — update the prompt file's line references yourself in a planning commit first, or note the
drift explicitly in the dispatch.

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `15-primitive-phase-explicit-rate`.
Model: **Sonnet**.

## Reviewing prompt 15

Structural checks; allowed files: `ComputeTargets/primitive_phase.py`,
`ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/tests/test_primitive_phase.py`,
`ComputeTargets/tests/test_tk_source_functions.py`, log, board, `docs/OPEN_ISSUES.md`.
**`ComputeTargets/GkSourcePolicyData.py` untouched** — `git diff HEAD~1 -- ComputeTargets/GkSourcePolicyData.py`
must be empty; that file's `PrimitivePhase` call takes no `rate` argument and must need none.

1. One new commit, message per README §5 rule 2.
2. `logs/15-primitive-phase-explicit-rate.md` exists, follows README §5.1, and classifies every
   deviation.
3. `IMPLEMENTATION_STATE.md` row 15, M15, and §3/§4 updated in that commit; `docs/OPEN_ISSUES.md`
   with it — the `[10-primitive-phase-leading-rate-is-hardcoded]` row deleted and the header count
   corrected.
4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_primitive_phase ComputeTargets.tests.test_tk_source_functions ComputeTargets.tests.test_gk_source_primitive_phase ComputeTargets.tests.test_gk_source_policy -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
5. `grep -n "_SoundHorizonRate" ComputeTargets/TkSourceFunctions.py` empty; `grep -n "H_eff"
   ComputeTargets/TkSourceFunctions.py` empty (the docstring is rewritten, not just the code).
6. **`PrimitivePhase.__init__` gained a keyword-only `rate` parameter defaulting to `None`**, and
   every existing default-path caller (`GkSourcePolicyData.py`, and every `test_primitive_phase.py`
   / `test_gk_source_primitive_phase.py` call site that does not pass `rate`) is unchanged —
   read the prompt's §3.1 item 1 assertion, don't take "tests pass" alone as proof the default path
   is untouched; check the diff shows no edited tolerance in those cases.
7. **A test exists that would have caught the original defect**: an explicit `rate` different from
   `1/Hubble` changes `theta_deriv`, and `model_functions.Hubble` read off a `PrimitivePhase` still
   returns the genuine Hubble rate even when `rate` differs from it (prompt §3.1 items 2–3).
8. **The reordered `T_k` arithmetic is measured, not assumed equal**: the log quotes the difference
   between the old (`k / (H/sqrt(c_s^2))`) and new (`k * (sqrt(c_s^2)/H)`) expressions on the
   existing `test_omega_matches_phase_derivative` fixture, and it is far below the 1e-9/1e-11
   window that test already asserts.
9. `black --check` clean on both edited modules.
10. `git diff HEAD~1 --stat` touches only the files listed above.

## Continue or stop

Stop on the campaign-wide conditions in `orchestrator/README.md`, on any of checks 5–8 failing, or
on `GkSourcePolicyData.py` appearing in the diff.

## Completion criterion

Row 15 ✅/⚠️. Report: "Prompt 15 complete; the tree is at `<SHA>`.
`[10-primitive-phase-leading-rate-is-hardcoded]` is resolved. Ready for Workstream E (11, 12)."
Include the final `PrimitivePhase.__init__` signature, where the positive-`c_s^2` check now lives,
and the measured difference from check 8.
