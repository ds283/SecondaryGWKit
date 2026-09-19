# Orchestrator — prompt 05b, the order an object actually carries

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../05b-make-the-recorded-order-the-order-used.md`](../05b-make-the-recorded-order-the-order-used.md)
**Model:** Opus. **Production code changed: yes, but narrowly** — three compute classes, one
integrator plumbing file, three factories' `build()` paths. **No schema, no number, and `main.py`
not at all.**

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt runs before 05a** (campaign README §7 **D10**, 2026-09-18, the user's decision), and
the letter therefore records insertion rather than run order. That is a departure from the 02a/03a
convention and the campaign README says so; do not "correct" the ordering to match the letters.

**Why it exists.** Prompt 05's review passed all seven checks and the agent itself opened
`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` — a `order=` keyword by
which a residual table can be built at one order and persisted at another. Asked whether the fix was
a one-liner, the orchestrator found a second and larger half the issue does not name: **a rehydrated
`BackgroundModel` reassembles its cumulative tables at the current module constant, never consulting
the stored column.** The user's instruction on 2026-09-18 was that this is a correctness issue and
must be made correct — set up, computed, persisted, rehydrated — whatever that costs downstream.

**Precondition:** prompt 05 has landed (`90d0114`). **Do not dispatch 05b if the six factories'
columns are not exactly as 05 left them** — this prompt changes no schema and a schema difference
before it starts means something else moved.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the baselines. At `90d0114` they are **508**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5) and **39** (`CosmologyModels`). `ComputeTargets` must not fall below
**508** and **should rise**: this prompt adds tests.

**Record the schema, because this time nothing in it may move:**

```bash
./venv/bin/python -c "
import re,pathlib
for f in ('BackgroundModel','GkWKBIntegration','TkWKBIntegration','GkSource','GkNumericIntegration','TkNumericIntegration'):
    t=pathlib.Path('Datastore/SQL/ObjectFactories/%s.py'%f).read_text()
    cols=re.findall(r'sqla\.Column\(\s*\"([a-z_A-Z0-9]+)\"',t)
    print('%-22s %s'%(f,[c for c in cols if 'tol' in c or 'gauss' in c or c=='break_point_kind']))"
```

At `90d0114`: `BackgroundModel` `['tau_gauss_order','cs_tau_gauss_order','friction_F_gauss_order']`,
both WKB factories `['rho_gauss_order']`, `GkSource` `[]`, both numeric factories
`['atol_serial','rtol_serial','break_point_kind']`. **Every one of these six must read identically
afterwards.** A changed column list is check 1's stop.

**And the two things that must not move:**

```bash
git rev-parse HEAD:config/defaults.py
git rev-parse HEAD:main.py
```

`config/defaults.py` is `a5ef9ed193c0f8e74a67cb5f28a7bf7cade00982` and has been since `bc6dc97`.
Note `main.py`'s blob too — §3 check 4 compares against it, and **zero lines** is the bar.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_gauss_order_key ComputeTargets.tests.test_numeric_break_point_key 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

16 + 11 = **27**, OK.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` =
`05b-make-the-recorded-order-the-order-used`, model **Opus**. The standing parameter-freeze sentence
**applies unchanged** — D9 lifted it for 05a alone, and this prompt changes no number. Add this
sentence, and no more:

> **This is a correctness prompt.** The defect has zero impact today and that is not an argument
> against fixing it; do not offer it as one. But it is also not a licence to widen the change: the
> schema is prompt 05's and settled, the tolerances are 05a's, and `main.py` is nobody's here.

**Do not tell the agent whether the `order=` parameter keeps its default.** Prompt §2 poses that as
its one design question and §4 gives the constraint that bears on it — the six doc scripts that must
still compile. Check 3 turns on the justification. An orchestrator who answers it has removed the
only judgement in the prompt.

**Do not tell it how the stored order reaches a rehydrated object.** Prompt §3 gives it the
asymmetry (`_metadata` is `None` on the build path) and the tree's own precedent, and asks it to
work out the rest.

## 3. The review — six checks

1. **Nothing in the schema moved.** Re-run §1's reader. All six lists identical. A column added,
   renamed or dropped is a **stop** — including a "harmless" one on `GkSource`.

2. **Both new tests fail at `90d0114` and pass at the commit.** This is the whole prompt. Check out
   the parent, run the two tests prompt §6 requires, and confirm they **fail**; then run them at
   the commit and confirm they pass. The log must report the same. **A test that passes at the
   parent is testing nothing**, and every other check here would still be green — this is 05's
   check 2 in a new costume, and it is the one that is easiest to pass by accident.

   ```bash
   git stash list && git worktree list
   ```

   Use a worktree rather than checking out the parent in place; the campaign's venv symlink
   convention is in `CLAUDE.md`.

3. **The design question was answered, not dodged.** Read the log's account of the `order=` default
   and of how the stored order reaches the constructor. Then the constraint:

   ```bash
   ./venv/bin/python -m py_compile docs/qcd-background-audit/grid_density_criterion.py docs/qcd-background-audit/consumer_knot_scheme_scan.py docs/gktk-remedial/verify_production_path.py docs/tolerance-convergence/order_audit.py docs/tolerance-convergence/inventory.py
   ```

   All must compile, and none may appear in the diff. An agent that made `order` required and then
   edited five doc scripts across three campaigns to suit has done exactly what prompt §4 and §8
   forbid.

4. **No number moved, and `main.py` did not move at all.**

   ```bash
   git diff HEAD~1 HEAD -- config/defaults.py main.py
   ```

   Must be **empty**, both blob hashes equal to §1's. All four orders still 4.

5. **No computed value moved.** The prompt claims bit-identity on the production path and §9 makes
   the agent say how it checked. Verify the digests yourself:

   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid 2>&1 | grep -E "^(OK|FAILED|Ran )"
   ```

   and read the log's argument. "The orders are all 4 so nothing can have changed" is the right
   answer only if the agent also checked that the rehydration path now passes the stored 4 rather
   than the constant 4 — same value, different provenance. A moved digest is a **stop**.

6. **The suites, the diff and the bookkeeping.** Re-run both suites. `ComputeTargets` **must not
   fall below 508** and a fall is a stop even above it; `CosmologyModels` **39**.
   `test_numeric_break_point_key.py` shows **zero** lines changed:

   ```bash
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/test_numeric_break_point_key.py
   ```

   Board row 05b, item **T15**, §3/§4, and `docs/OPEN_ISSUES.md` in the **same commit** with its
   count and date corrected — `[05-rho-gauss-order-…]` closes, so the count falls by one unless the
   prompt opened something. `black --check` clean.

## 4. What a good outcome looks like

- An object that can be asked what order it was built at and answers from its own state, on both
  paths, rather than from whatever the module currently says.
- A `BackgroundModel` read back from a row whose order is not the current constant, reassembling
  its primitives at the row's order — which is impossible today and is the half the issue never
  named.
- `build()` still filtering on the current constant, with a line saying why that is right: a row
  computed at another order is not a miss to be repaired, it is a different row.
- Two tests that fail at `90d0114`.
- Six doc scripts across three campaigns still compiling, untouched.

**What a good outcome does *not* look like:** a green suite, a passing `order=` parameter and a
property that still returns the module constant on the rehydration path. That is the defect intact
with a test beside it.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **The agent reports that the defect is not real** — prompt §8's last stop. Relay the argument
  verbatim (rule 5); do not talk it out of the position and do not accept an empty commit.
- **The agent proposes a schema change**, or to change what `build()` filters on. Both are D3 and
  the user settled D3 on 2026-09-18.
- **The agent proposes to edit `main.py`**, `config/defaults.py`, a numeric target, or the five
  doc-script call sites of prompt §4.
- **Any computed value or source-grid digest moves.**
- **`ComputeTargets` falls at all**, even above 508.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and sits on it — observed at 0.0601–0.0612 across
consecutive runs at an unchanged commit, passing roughly one run in three. Confirm by re-running
that module alone before attributing anything to the commit.

## 6. After it lands

**Hand back to the user briefly, then continue.** This prompt settles nothing that is the user's to
settle — D10 was settled 2026-09-18 — so the hand-back is a report. Report:

1. The mechanism by which a rehydrated object now gets its order, and the agent's answer to the
   `order=` default question.
2. The two tests, and their failure messages at `90d0114`.
3. Confirmation that no schema, no number, no `main.py` line and no computed value moved.
4. `[05-rho-gauss-order-…]` closed, both halves, and the counts on the board and the index.

Then **prompt 05a is next and is not written yet.** Its charter is campaign README §3.5a and its
board items are **T8** and **T10**. Write it against what 05 and 05b together left — the
`object_get` inventory in log 05's "State handed to the next prompt" still stands, because 05b
touches no line of `main.py`. Do not dispatch it in the same turn.
