# Workstream D — housekeeping (prompts 07–08) — **GATED**

**Read [`README.md`](README.md) in this directory first.** The orchestrator rules, the three checks
and the dispatch template are there and are not repeated here.

**Prompts:** 07 (Sonnet), 08 (Sonnet).
**Gate:** README §7 **D3**. **Do not start without the user's explicit go-ahead.**

---

## 0. What this workstream is, and why it is gated

Two orphaned one-liners that no campaign owns and that this one happens to be adjacent to. Neither
is this campaign's subject.

- **07** — `sqla_QCDCosmology_factory.inventory()` does not report the `T_z_representation` column,
  which is part of the lookup key, so rows differing only in their representation render as
  indistinguishable duplicates to the only tool that inspects a datastore.
  `[03-qcd-inventory-does-not-report-the-representation]`, declined on scope by two prompts of
  `qcd-background-audit`.
- **08** — `CosmologyModels/tests/test_wPerturbations.py:34-41` describes a "500-point spline" and
  quotes two figures for a representation replaced twice since. Opened by prompt 01 as
  `[01-agreement-threshold-comment-predates-the-representation]`.

**"No, leave them indexed" is a perfectly good answer and costs nothing** — the rows stay in
`docs/OPEN_ISSUES.md` where they are, with their next steps intact. Put the gate to the user as a
genuine question, not as a formality. 07 in particular widens the campaign's file set into
`Datastore/`, which nothing else here touches.

The two are independent of each other and of the A→B→C chain. 08 needs prompt 01's log. Either may
be authorised without the other; ask about them separately if the user's answer is partial.

## 1. Before dispatching

The usual baselines, plus — for 07 — find out **which suite covers `Datastore/`** and record its
count. There may not be one; if there is not, that is itself the answer to prompt 07 §2 item 4 and
you should expect the subagent to ship the change with a recorded gap rather than build a test
harness.

```bash
git rev-parse HEAD
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .     # 452, OK (449 pre-07, 447 pre-09)
ls Datastore/tests 2>/dev/null || echo "no Datastore test package"
```

## 2. Prompt 07 — the inventory column

Dispatch with the template. Model: **Sonnet**.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Nothing that writes is in the diff** | `build()` must not appear. `git diff HEAD~1 HEAD -- Datastore/` and read it |
| 2 | The lookup key is unchanged | No change to what is compared, only to what is reported |
| 3 | The column name is the one in the table, not one inferred from the constant | Check it against the factory's table definition yourself |
| 4 | Two rows differing only in the representation render distinguishably | The log must demonstrate it, not assert it |
| 5 | If no test was written, the reason is recorded and an issue opened | Prompt §2 item 4 permits this outcome explicitly; an unrecorded gap does not |
| 6 | `ComputeTargets` **449 → 452** at prompt 07 (its three new tests), **452** at 08, `CosmologyModels` unchanged | the three checks |
| 7 | `[03-…]` closed on the **`qcd-background-audit`** board's §4, row deleted from the index | It is that board's issue |

### Stop and report if

- The column is absent or differently named. Report what is there.
- The agent built new datastore test infrastructure. The prompt forbids it.
- A second reporting site with the same gap was **fixed** rather than recorded.

## 3. Prompt 08 — the agreement threshold

Dispatch with the template. Model: **Sonnet**.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Only the comment and the constant are in the diff** | `git diff HEAD~1 HEAD -- CosmologyModels/tests/test_wPerturbations.py`. No assertion body, no change to `PureRadiationEOS` or `lambdaCDM_gstar` — three modules import those |
| 2 | Both figures were **re-taken**, not inherited from log 01 | The prompt's whole content is that a number in a comment must be reproducible by its reader; the log must give the command |
| 3 | `AGREEMENT_RTOL` tightened or unchanged — **never loosened** | If the agent loosened it, that is a stop and a finding |
| 4 | The comment describes the representation at this commit | Segmented entropy factor, 3,000 nodes, order 5, ramp in closed form. Read it |
| 5 | The sentence explaining what the constant is *for* survived | It is still true and it is why the constant exists |
| 6 | §2 item 4's regime statement is present | Rounding floor or interpolation error — that distinction is the most useful thing this prompt records |
| 7 | Both suites unchanged in count, OK | the three checks |

### Stop and report if

- The measurement says the threshold must loosen.
- An assertion body changed.
- Another module was touched.

## 4. Completion criterion

Whatever subset the user authorised is ✅ on the board, with its issue closed on the right board and
its row out of `docs/OPEN_ISSUES.md` with the count corrected. Anything not authorised stays
indexed and is recorded on the board as **not taken, by the user's decision, with the date** — the
same treatment `qcd-background-audit` gave the wavenumber-set half of
`[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]`.

Then update `IMPLEMENTATION_STATE.md` §5's close-out to state D3's outcome, and report to the user.
