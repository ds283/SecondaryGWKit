# Workstream B — what the redshifts feed, and the file's stray probe (prompts 03–04)

**Read [`README.md`](README.md) in this directory first.** The orchestrator rules, the three checks
and the dispatch template are there and are not repeated here.

**Prompts:** 03 (Opus), 04 (Sonnet).
**Precondition:** workstream A complete, with its completion criterion met.

---

## 0. What this workstream is, and why it is the riskiest

Prompt 03 measures and documents something the audit could not see: `main.py` recomputes both
equality redshifts from the same closed forms and forces them into the **production source grid**,
whose content digest is a `BackgroundModel` lookup-key column. So the two numbers `AUDIT.md` §2.1
calls "two banner lines" are, by a different route, production sample locations inside a datastore
identity (`RECONCILIATION.md` §5).

**Nothing in prompt 03 changes behaviour.** Its entire output is a measurement, a corrected
docstring sentence, one test and a decision put to the user. That is deliberate: the obvious tidy
here invalidates eight object types, and the campaign's job is to make sure nobody discovers that
by doing it.

Prompt 04 is unrelated and low-risk: it moves a test-only root solve out of a production class.

## 1. Baselines — take these before dispatching prompt 03

```bash
git rev-parse HEAD
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .     # 447, OK
```

And **the grid digest**, which is the object of prompt 03's byte-identity claim. Build the
production source grid for `QCD_Cosmology` and for `LambdaCDM(Planck2018)` using
`ComputeTargets/tests/test_main_plumbing.load_main_py_functions` (`CLAUDE.md`: `main.py` cannot be
imported) and record the digest and element count for each. If you cannot produce it in a few
minutes, **say so in your dispatch** and require prompt 03 to produce the `HEAD~1` value itself as
part of its own §2.3 — but then you are reviewing a number the subagent generated on both sides,
which is weaker, and your report must say so.

## 2. Prompt 03 — what the equality redshifts feed

Dispatch with the template. Model: **Opus**.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Zero executable lines of `main.py` in the diff** | `git diff HEAD~1 HEAD -- main.py` must show only docstring lines. Not one change to `cosmology_feature_redshifts`'s body |
| 2 | The corrected sentence quotes measured figures and names the commit | Read it |
| 3 | §2.1's table is complete | Three models × two pairs × every column present, 17 digits. A table with one model is not the deliverable |
| 4 | **The grid digest is identical** | Against your §1 baseline, both models |
| 5 | The new test reproduces `main.py`'s expression rather than importing it | Read it; a test that imports `main.py` will not run |
| 6 | The three options are **priced**, not listed | Each must carry its cost. Option (iii)'s cost is "a regeneration of eight object types" and must say so |
| 7 | The agent did **not** decide | If the log recommends and the diff unifies anything, that is a stop |
| 8 | `ComputeTargets` 447 | the three checks |

### Then stop, whatever the review says

**Prompt 03 ends in README §7 D2 and you put it to the user.** Do not dispatch prompt 04 until you
have. Report:

- the three sites, what each feeds, and the chain from `feature_z` to the lookup key;
- §2.1's table, or at least the QCD row of it;
- the three options with their costs;
- the campaign's recommendation — **(i) leave all three and document, with the new test as the
  standing guard** — and the reason: the sites agree to the floor, the duplication is already
  documented as deliberate at `main.py:522-528`, and the downside of getting it wrong is a
  datastore.

If the user chooses (ii) or (iii), **that is a new prompt, not an amendment to 03.** Say so.

### Stop and report if

- The grid digest moved. Something upstream is wrong and nothing else should happen until it is
  understood.
- The three closed-form sites disagree by more than 1 ulp. That is a bigger finding than this
  workstream is scoped for.
- Any executable line of `main.py` is in the diff.

## 3. Prompt 04 — relocate the crossing probe

Dispatch with the template. Model: **Sonnet**. This one may be dispatched once D2 has been put to
the user, whether or not the user has answered — it touches none of the same files.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **The `root_scalar` line is character-identical** | `git show HEAD~1:CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py \| sed -n '869p'` against the moved line. `xtol=1e-15, rtol=1e-15` unchanged |
| 2 | The docstring came across whole | It is long and it carries the entire argument for why bisection, not a bracket, locates a crossing. A summarised docstring is a stop |
| 3 | `grep -rn "self\._temperature_crossing_log1pz"` returns nothing | run it |
| 4 | **No suite count fell** | `CosmologyModels` unchanged, `ComputeTargets` **447** |
| 5 | `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` passes with its **present deltas** | Run it verbosely and read the three measured steps in the log: 1.97e-3, 1.38e-4, < 1e-8 |
| 6 | If option (a): the import works from **both** discovery roots | Run both `discover` commands yourself |
| 7 | If option (b): the agent justified it with measurement, not preference | Read the `IMPLEMENTATION CHOICE` entry |
| 8 | The cross-references at `LambdaCDM_GenericEOS.py:732` and in the three test modules are re-pointed | `grep -rn "_temperature_crossing_log1pz"` and read each hit |
| 9 | `[08-temperature-crossing-solver-is-test-only]` closed on the **`qcd-background-audit`** board's §4, row deleted from the index | It is that board's issue, not this one's |

### Stop and report if

- The agent **deleted** the method instead of moving it. The prompt says to stop and ask first.
- Any assertion delta in `test_numeric_break_points.py` changed. It should not need to.
- A suite count fell.

## 4. Completion criterion for workstream B

- 03 and 04 are ✅ with SHAs and logs.
- The grid digest is identical to the §1 baseline on both models.
- README §7 **D2** has been put to the user and their answer is recorded on the board.
- `_temperature_crossing_log1pz` is not a member of `LambdaCDM_GenericEOS`, and both suites are at
  or above their baselines.
- `[08-…]` is on the `qcd-background-audit` board's §4 and out of `docs/OPEN_ISSUES.md`, count
  corrected.

Report to the user: D2's answer as recorded, the corrected `main.py` figure, and the three measured
$H$ steps from the relocated probe's test. Then start workstream C.
