# Workstream C — cost and close-out (prompts 05–06)

**Read [`README.md`](README.md) in this directory first.** The orchestrator rules, the three checks
and the dispatch template are there and are not repeated here.

**Prompts:** 05 (Opus), 06 (Opus).
**Precondition:** workstreams A and B complete, with their completion criteria met.

---

## 0. What this workstream is

Prompt 05 hoists four loop-invariant `_outward` calls out of three hot-path `__call__` methods and
re-takes the `T_photon` cost row that `qcd-background-audit` left as a confirmed miss. **The code
change is trivial and the measurement is not.** Prompt 06 writes the provenance for all three root
solves in the file, amends the `tolerance-convergence` board, and closes the campaign.

## 1. Before dispatching prompt 05 — the machine

**This is a timing prompt.** The figure it replaces — 2.596 µs mean, range 2.505–2.671 over five
runs — was taken on a **quiet machine**, and a comparison against a loaded one is worthless.

Before dispatching, establish that the machine is quiet: no builds, no other agents, no test suites
running in the background. If you cannot, **tell the user and hold the prompt.** An unquiet number
is worse than no number, and the prompt's decision rule turns on a 4 % difference.

Baselines:

```bash
git rev-parse HEAD
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .     # 449, OK (447 pre-09)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py
```

**Keep the whole output of the last one.** Its §5 prose (the sentence prompt 05 corrects), its §6
cost table (all four rows) and its §2–§4 output are what you score prompt 05 against. Run it
**three times** and record the spread of the §6 rows — that spread is how you will judge whether a
control row "moved".

## 2. Prompt 05 — hoist the range logic

Dispatch with the template. Model: **Opus**.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Bit-identity** | The log must demonstrate it by `float.hex()` over a probe set spanning each class's full tabulated range, with the count quoted. "The change is numerically null by construction" is an argument, not a demonstration, and the prompt asks for the demonstration |
| 2 | **Message identity** | Both bounds errors from each of the three classes, character-identical. Trigger one yourself from `TemperatureRepresentation` and diff it against `HEAD~1` |
| 3 | `_outward` itself is untouched | `git diff HEAD~1 HEAD -- ComputeTargets/spline_wrappers.py` must not show the function body at `:14` |
| 4 | The comparison bounds and the message bounds were **both** hoisted | Four values per class, not two. Read `__init__` |
| 5 | **Ten runs, five per tree** | Mean, range and ratio. A single run either side is not the protocol |
| 6 | The three control rows moved by less than your §1 spread | If one did, the machine was not quiet and the measurement is void |
| 7 | §4's decision rule was applied as written | ≤ 2.5 µs closes `[06-…]`; above it **escalates**. If the log shows an extra optimisation attempted to get under the line, that is a stop |
| 8 | The §5 prose now prints the intersection with the **declared** set (0), and §2–§4 and §6 output are otherwise unchanged | Diff your §1 capture against the new run |
| 9 | `[07-…]` and `[09-…]` closed on the **`qcd-background-audit`** board's §4, rows deleted from the index | They are that board's issues |
| 10 | `ComputeTargets` **449**, `T_Z_REPRESENTATION_VERSION` 6 | the three checks |

### Then, if the mean is above 2.5 µs, stop and put README §7 D4 to the user

Report: the mean and range at both trees, the ratio, how much of the residual is the order-5
`BSpline.__call__`, and the statement that order 5 is not optional (a cubic needs ~25,000 nodes for
the required p90). The two honest options are that `qcd-background-audit` README §6.2's target was
set ~4 % too tight, or that the cost is accepted. **Do not let prompt 06 proceed as if the row
were closed when it is not**; the board must say "narrowed" and the index row must carry the new
figure.

### Stop and report if

- Any returned value differs by a bit, or any message by a character.
- A control row moved by more than the spread.
- The agent optimised anything beyond the hoist. Inlining `bisect_right`, caching on $z$, lowering
  the order — each is a change to a representation two campaigns settled, and §6 of the prompt
  forbids all of them.

## 3. Prompt 06 — provenance and close-out

Dispatch with the template. Model: **Opus**. **Do not dispatch until 05 is reviewed and D4, if it
arose, has been put to the user.**

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Zero production and zero test files in the diff** | `git diff --name-only HEAD~1 HEAD`. Only `PROVENANCE.md`, two files under `prompts/tolerance-convergence/`, this campaign's own docs and `docs/OPEN_ISSUES.md`. This rule **is** the review |
| 2 | `PROVENANCE.md` has **three** entries | One per `root_scalar` site. A document about `_find_rho_equality` alone misses the point: `tolerance-convergence` prompt 02 has to inventory all three |
| 3 | Every field is present in every entry | Including "choosing measurement **and its commit**" — a figure without its tree is not a measurement |
| 4 | The `_find_rho_equality` entry says **diagnostic** *and* points at `RECONCILIATION.md` §5 / log 03 | An entry that says "diagnostic" without that pointer is the exact mistake the campaign exists to stop being made twice |
| 5 | The `tolerance-convergence` amendment touched only the `:1008` bullet | `git diff HEAD~1 HEAD -- prompts/tolerance-convergence/`. The `[11-stop-point-root-tolerance]` and `QuadSourceIntegral.py:1550` bullets must be untouched |
| 6 | The agent **re-ran** the close-out checks rather than quoting the logs | §4 of the prompt requires it; the log must show the commands and their output |
| 7 | The suite counts **reconcile across every log** | Add them up yourself. A discrepancy is a finding, and the prompt forbids fixing it silently |
| 8 | `measure_rho_equality.py` was run **unedited** | `git diff HEAD~1 HEAD -- prompts/background-solver-robustness/measure_rho_equality.py` is empty. Its §2.2 evaluation counts will have changed (secant → Brent) and its roots must not |
| 9 | `AUDIT.md` §5's claim is stated as held or not held, with four pieces of evidence | the four redshifts, the grid digest, the version at 6, `ComputeTargets` at 447 through prompt 04 and **449** from prompt 09 |
| 10 | Board §5 close-out is written, and README §6's acceptance table has a measured value in every row | Read both |

### Stop and report if

- Any production or test file is in the diff, for any stated reason.
- A suite count does not reconcile.
- A prompt's acceptance threshold has no measured value anywhere and prompt 06 took the measurement
  itself to fill the gap. That makes it the reviewer of its own evidence; the honest outcome is
  that the earlier prompt is not ✅.
- `AUDIT.md` §5's claim did not hold. That is the campaign's central statement.

## 4. Completion criterion for the campaign

- 01–06 all ✅ with SHAs and logs.
- `PROVENANCE.md` exists with three complete entries.
- `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3 records the settled result.
- `CosmologyModels` risen and OK; `ComputeTargets` **447** at every commit through prompt 04 and
  **449** from prompt 09 on; `black --check` clean.
- `T_Z_REPRESENTATION_VERSION` **6** at every commit.
- `docs/OPEN_ISSUES.md` reconciles: three issues opened by the campaign closed or narrowed, four
  adopted issues closed on the `qcd-background-audit` board, the count and date correct.
- README §7's four decisions each marked **taken** or **outstanding** on the board.

**Report to the user, in this order:** whether `AUDIT.md` §5's claim held and on what evidence; the
four equality redshifts across the whole campaign; D2's and D4's answers; what remains open and who
owns it; and whether to open workstream D (README §7 **D3**).
