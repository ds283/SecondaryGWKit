# Orchestrator — prompt 04b, landing the convergence block

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../04b-regenerate-the-convergence-block.md`](../04b-regenerate-the-convergence-block.md)
**Model:** Opus. **Production code changed:** none. **Test-tree code changed:**
`ComputeTargets/tests/wkb_reference_data.json`, `ComputeTargets/tests/test_background_tau.py` and
`ComputeTargets/tests/test_background_cs_tau_friction.py`, **and only those**, under README §7 **D5**
as widened by **D8**. `docs/gktk-remedial/residual_convergence.py` is the fourth carve-out file and
is not under `ComputeTargets/`.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt finishes what 04 measured.** Prompt 04 stopped one step short of writing its own
answer into the tree because one of the two tests that would then fail was outside its grant; the
user removed that boundary on 2026-09-18 (**D8**) and settled **D3** in the same exchange. So 04b
lands the fixture and 05 is unblocked behind it.

**Precondition:** prompt 04 has landed (`0b28158`) and `docs/tolerance-convergence/ORDER-AUDIT.md`
is in the tree. **Do not dispatch 04b if `ORDER-AUDIT.md` is absent or if the `convergence` block
already reads a 2026-09-17-or-later `generated` stamp** — in the second case something has already
written it and this prompt's premise is gone.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the four baselines. At `0b28158` they are **491**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5), **39** (`CosmologyModels`), **32** (`test_convergence_reference`) and
**41** (the three block readers). `ComputeTargets` must not fall below **452**, the campaign floor.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_tau ComputeTargets.tests.test_background_cs_tau_friction ComputeTargets.tests.test_phase_residual 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

**Record the block's identity, because this time it must move:**

```bash
./venv/bin/python -c "import json;d=json.load(open('ComputeTargets/tests/wkb_reference_data.json'));c=d['convergence'];print(c['generated'],c['campaign'],c['decision']['recommended_scheme'],c['decision']['N_tau'],c['decision']['N_cs_tau'],c['decision']['N_F'],c['decision']['N_rho'])"
```

At `0b28158` that still reads `2026-09-10`, `prompts/GkTk-remedial (prompt 02)`, `branch+knots`,
`4 4 4 4` — prompt 04 did not write it.

**And record the two constants**, so that you can check the move rather than the endpoint:

```bash
grep -n "^QCD_FLOOR_FACTOR\|^QCD_BREAK_POINT_ALIGNMENT_TOL" ComputeTargets/tests/test_background_tau.py ComputeTargets/tests/test_background_cs_tau_friction.py
```

At `0b28158`: `QCD_FLOOR_FACTOR = 3.0` in both modules, `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04`.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `04b-regenerate-the-convergence-block`,
model **Opus**. The standing parameter-freeze sentence applies unchanged — every constant this
prompt moves is in the test tree, so rule 8 is not strained. Add this sentence, and no more:

> **README §7 D5, as widened by D8 on 2026-09-18, gives you four files outside the usual
> boundary — `residual_convergence.py`, `wkb_reference_data.json`, `test_background_tau.py` and
> `test_background_cs_tau_friction.py` — and exactly four.** `test_phase_residual.py` reads the
> same block and is **not** among them: it must pass unedited. Do not widen the grant yourself.

**Do not tell the agent what the thresholds should become.** Prompt §3 gives it the rule and the
two floors; the review below turns on whether the number it picked follows from the arithmetic it
wrote down. An orchestrator who supplies the number has removed the only thing worth reviewing.

**Do not reassure it that the orders will come back 4.** They did six days ago on the same tree and
they almost certainly will again — which is exactly why its own §2.2 stop condition has to be a
live check rather than a formality.

## 3. The review — six checks

1. **The block moved, and the generator moved it.**

   ```bash
   ./venv/bin/python -c "import json;c=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))['convergence'];print(c['generated'],c['campaign'],c['decision']['recommended_scheme'],c['decision']['N_tau'],c['decision']['N_cs_tau'],c['decision']['N_F'],c['decision']['N_rho'])"
   ```

   `generated` and `campaign` must both have moved from §1's reading. If they have not and the
   block's contents changed, the agent hand-edited it, which prompt §2.1 forbids and which no other
   check will catch. The four orders must read `4 4 4 4` and the scheme **`branch`**.

2. **Only the `convergence` key.**

   ```bash
   ./venv/bin/python -c "
   import json,subprocess
   old=json.loads(subprocess.run(['git','show','HEAD~1:ComputeTargets/tests/wkb_reference_data.json'],capture_output=True,text=True).stdout)
   new=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))
   print(sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k)))"
   ```

   Must print `['convergence']`, or `['convergence', 'generated']` if the top-level stamp moved with
   it. **Anything else is a stop.**

3. **`branch+knots` survived as a control.**

   ```bash
   ./venv/bin/python -c "import json;c=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))['convergence'];print(c['schemes']);print({m:list(c['models'][m]) for m in c['models']})"
   ```

   The key must still be present and populated. Two modules index it by name, and prompt 04
   measured what it buys (nothing) — that is a reason to demote it, not to delete it.

4. **The thresholds are bounds, not multiples of the fixture's own error.**

   ```bash
   git diff HEAD~1 HEAD -- ComputeTargets/tests/test_background_tau.py ComputeTargets/tests/test_background_cs_tau_friction.py
   ```

   Read the new assertions and their comments. **A raised `QCD_FLOOR_FACTOR` is a fail**, however
   well justified in prose: prompt §1 names it as the repair that looks right and is wrong, and §3
   gives the rule it violates. What must be there instead is an absolute bound with (i) the measured
   production figure (2.254e-15 / 2.212e-15), (ii) the measured accumulation floor (2.16e-16 /
   3.30e-16) with `ORDER-AUDIT.md` §§3.1, 5 cited, and (iii) the headroom stated as a factor.
   **Check the arithmetic that picked it, not just the number.** A bound sitting four decades above
   the measurement asserts nothing and is the defect this campaign exists to undo; a bound with no
   headroom is a tripwire.

   Check too that the history in those comment blocks was **appended to, not deleted** — four
   prompts of `qcd-background-audit` recorded why the constant moved three times and predicted this
   repair, and that record is evidence.

5. **The alignment tolerance.**

   ```bash
   grep -n -B 4 "^QCD_BREAK_POINT_ALIGNMENT_TOL" ComputeTargets/tests/test_background_tau.py
   ```

   Just above **1.421085e-14**, with the three per-break offsets (3.55e-15, 7.11e-15, 1.421085e-14)
   in the comment. Ten orders is the right size of move; an agent that softened it to something
   "safer" has not applied the measurement.

6. **The suites, the diff and the bookkeeping.**

   ```bash
   git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'
   ```

   Must list **at most** the three named test-tree files. Re-run all three suites and the three
   block readers. `ComputeTargets` **must not fall below 452** and should read 491; `CosmologyModels`
   **39**; `test_convergence_reference` **32**; the three readers **41**, with
   `test_phase_residual.py` showing **zero** lines changed:

   ```bash
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/test_phase_residual.py
   ```

   Board row 04b, item **T14**, the `qcd-background-audit` board entry, and `docs/OPEN_ISSUES.md` in
   the **same commit** with its count and date corrected — two issues close here, so the count falls
   by two. `black --check` clean. `ORDER-AUDIT.md` gained a **§12** and §§1–11 are byte-unchanged:

   ```bash
   git diff --numstat HEAD~1 HEAD -- docs/tolerance-convergence/ORDER-AUDIT.md
   ```

   should show additions and **no deletions** (README §5 rule 7).

## 4. What a good outcome looks like

- A `convergence` block that is current, written by its generator, recommending a scheme production
  can actually execute, with `branch+knots` retained as the measured control.
- Two tests that assert something true about the production table, with the floor under them named
  and the headroom stated — and a comment that says why the factor form was abandoned rather than
  raised, because that is the question the next reader will have.
- `QCD_BREAK_POINT_ALIGNMENT_TOL` at ~1.42e-14, closing a loop four prompts of another campaign
  left open.
- Two issues closed, on both boards.

**What a good outcome does *not* look like:** a green suite bought by raising `QCD_FLOOR_FACTOR` to
8, or by writing the old floor into the new block. Both leave the tree green and the defect intact.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **Any recommended order comes back other than 4**, or the scheme comes back `branch+knots`.
  Relay the measurement; it contradicts prompt 04 on the same tree and that is not something to
  resolve by re-running until it agrees.
- **The agent proposes to edit `test_phase_residual.py`**, any production module, or any file
  outside the four D5/D8 names.
- **The agent proposes to raise `QCD_FLOOR_FACTOR`** rather than replace the construction, and
  argues for it. Relay the argument verbatim (rule 5); it may be right, but it is a charter change.
- **The agent asks you to choose a threshold, a headroom factor or a tolerance.** Relay verbatim;
  do not pick one.
- **A moved source-grid digest**, or a top-level key other than `convergence` differing.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and sits on it — observed at 0.0601–0.0612 across
consecutive runs at an unchanged commit, passing roughly one run in three. It is unrelated to this
prompt's diff. Confirm by re-running that module alone before attributing anything to the commit.

## 6. After it lands

**Hand back to the user briefly, then continue.** Unlike 04, this prompt settles nothing that is
the user's to settle — **D3 and D8 were both settled on 2026-09-18** and this prompt only executes.
Report:

1. The block's identity before and after, and the two floors the threshold tests now read.
2. What each threshold became, and the arithmetic behind it.
3. `QCD_BREAK_POINT_ALIGNMENT_TOL`, old and new.
4. The two issues closed, and the counts on both boards.

Then **prompt 05 is next and is not written yet.** Campaign README §3.5 fixes its charter and §6.2
its acceptance rows; **D1 closed 2026-09-17 and D3 settled 2026-09-18**, so nothing gates it. Write
it after the user has read this. Do not dispatch it in the same turn.
