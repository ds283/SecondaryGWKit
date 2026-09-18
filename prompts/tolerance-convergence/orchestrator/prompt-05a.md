# Orchestrator — prompt 05a, the campaign's only parameter change

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../05a-decouple-the-tolerances-that-are-real.md`](../05a-decouple-the-tolerances-that-are-real.md)
**Model:** Opus. **Production code changed: yes** — `config/defaults.py`, `main.py` at twelve
`object_get` sites plus the tolerance block, `CosmologyConcepts/wavenumber.py`, six `extract_*.py`
readers and the `ast` guard. **No schema, and no file under `Datastore/SQL/ObjectFactories/`.**

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This is the last prompt before 06, and the only one that may move a number** (§5 rule 8, as §7
**D9** rewrote it). Every other prompt in the campaign has run under a parameter freeze; this one
lifts it, and the dispatch template's standing sentence must therefore be **replaced** rather than
sent. §2 has the replacement.

**Why it runs now.** D1 closed on 2026-09-17 with all four pairs accepted, so no number here is open.
D9 split the old §3.5 so that the schema landed first and this prompt decouples four targets rather
than eight; D10 put 05b in between, so the four order-governed targets are finished and will not move
again. Board items **T8** (the constants) and **T10** (the plumbing and the guard).

**Precondition:** 05 (`90d0114`) and 05b (`a351a50`) have landed. **Do not dispatch 05a if
`ComputeTargets` is below 516, or if the six factories' column lists differ from `a351a50`** — this
prompt changes no schema and a difference before it starts means something else moved.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the baselines. At `a351a50` they are **516**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5) and **39** (`CosmologyModels`). `ComputeTargets` must not fall below
**516** and **should rise**: this prompt adds tests to the guard.

**Record `config/defaults.py` as it stands, because this is the prompt that changes it** — and it is
the last chance to record the pre-change state:

```bash
git rev-parse HEAD:config/defaults.py
grep -n "^DEFAULT_ABS_TOLERANCE\|^DEFAULT_REL_TOLERANCE\|^DEFAULT_TK_NUMERIC_ABS_TOLERANCE\|^DEFAULT_QUADRATURE_ATOL\|^DEFAULT_QUADRATURE_RTOL\|^DEFAULT_FLOAT_PRECISION\|^DEFAULT_REDSHIFT_RELATIVE_PRECISION" config/defaults.py
```

The blob is `a5ef9ed193c0f8e74a67cb5f28a7bf7cade00982` and has been **since `bc6dc97`, across four
campaigns and two rebases**. It stops being byte-identical at this commit and that is correct; note
the hash anyway, because §3 check 1 diffs against it and needs to see *only* the intended lines move.

**Record the schema, which must not move at all:**

```bash
./venv/bin/python -c "
import re,pathlib
for f in ('BackgroundModel','GkWKBIntegration','TkWKBIntegration','GkSource','GkNumericIntegration','TkNumericIntegration'):
    t=pathlib.Path('Datastore/SQL/ObjectFactories/%s.py'%f).read_text()
    cols=re.findall(r'sqla\.Column\(\s*\"([a-z_A-Z0-9]+)\"',t)
    print('%-22s %s'%(f,[c for c in cols if 'tol' in c or 'gauss' in c or c=='break_point_kind']))"
```

**And the digests, because this prompt can move them.** The exit-time tolerance feeds `z_init`,
which is a root-solve output the grid tag digests:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid ComputeTargets.tests.test_gauss_order_key ComputeTargets.tests.test_numeric_break_point_key 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` =
`05a-decouple-the-tolerances-that-are-real`, model **Opus**. **The standing parameter-freeze sentence
must be replaced** — it names prompts 01–06 *except* 05a, and sending it would forbid this prompt's
entire purpose. Use this in its place, verbatim:

> **README §5 rule 8 names 05a as the one prompt in this campaign that may change a parameter, and
> this is that prompt.** `config/defaults.py` is yours. But **every number in it is already settled**
> — D1 closed on 2026-09-17 with all four pairs accepted by the user — so you are shipping values
> with their provenance, not choosing them. If you believe one of them is wrong, stop and say so
> rather than shipping a different number. **§5 rule 9 applies in full**: a number without its five
> provenance fields is an unfinished prompt.

Add this sentence, and no more:

> **You change no schema.** No column, no factory, no Gauss order: prompts 05 and 05b finished the
> four order-governed targets and they must not move again. A retuned tolerance changes which
> `tolerance` row a key points at, not what the key is made of.

**Do not tell the agent what to do about `DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE`.**
Prompt §3 poses that as its one design question — after the switch no object type keys on either,
and what is left of `DEFAULT_ABS_TOLERANCE` is seven float comparisons in modules the prompt may not
edit. Check 4 turns on the justification, not on which way it went. This is 05's mechanism question
and 05b's default question in the same slot; an orchestrator who answers it has removed the only
judgement in the prompt.

**Do not hint at the guard's new shape.** Prompt §4 gives it the defect — `endswith("Integration")`
matches four class names out of nine, and has never guarded `QuadSourceIntegral` because that name
ends in "Integral" — and asks it to design the enumeration.

## 3. The review — seven checks

1. **`config/defaults.py` moved, and only where it should.**

   ```bash
   git diff HEAD~1 HEAD -- config/defaults.py
   ```

   Six constants at the §2 values — `DEFAULT_GK_NUMERIC_ABS_TOLERANCE` 1e-10,
   `DEFAULT_GK_NUMERIC_REL_TOLERANCE` 1e-8, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` 1e-13 (already there),
   `DEFAULT_TK_NUMERIC_REL_TOLERANCE` 3e-11, `DEFAULT_HEXIT_ABS_TOLERANCE` 1e-10,
   `DEFAULT_HEXIT_REL_TOLERANCE` 1e-9 — and **`DEFAULT_ABS_TOLERANCE` still 1e-10 and
   `DEFAULT_REL_TOLERANCE` still 1e-8 in value**. A changed *value* on either is a **stop**: §6 of
   the prompt turns on it, because seven float comparisons outside the file list move with it.
   `DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL` must be untouched.

   Then read the comments. **Rule 9 is an acceptance condition**: each constant carries the
   measurement that chose it, the floor it competes against, and its cost, to the standard
   `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` already sets. A bare `= 3e-11` with no comment fails this
   check however green the suite is.

2. **The three qualifications are present.** Grep the file and read them. `3e-11` must say it is a
   **measured setting and not a bound** (`[03a-tk-numeric-excursion-is-sporadic-in-rtol]` is open and
   the excursion is non-monotone); `DEFAULT_HEXIT_ABS_TOLERANCE` must say it is **coupled** — `xtol`
   floors the pair at ~1e-10 relative at $u\approx37.6$, so `rtol` below ~2.6e-12 does nothing; and
   the exit-time pair must say it was accepted on the **guarantee** reading, not the achieved one.
   These are the three ways a later reader mis-reads these numbers, and the prompt exists partly to
   prevent that.

3. **No schema moved.** Re-run §1's reader; all six lists identical. And:

   ```bash
   git diff --name-only HEAD~1 HEAD -- Datastore/
   ```

   must be **empty**. A factory in the diff is a stop.

4. **The design question was answered, not dodged.** Read the log's §3 decision on the two shared
   names. Either answer is acceptable; an unargued one is not, and a rename that reached any of the
   seven float-comparison sites is a stop. Check:

   ```bash
   git diff --name-only HEAD~1 HEAD -- ComputeTargets/GkSource.py ComputeTargets/numeric_with_phase_cut.py ComputeTargets/WKBtools.py
   ```

   must be **empty**.

5. **Every site moved, and the guard can see them.** This is the check the prompt exists for, and a
   missed site does not crash — it silently recomputes a sector at full cost. Run the guard, then
   verify it is actually stronger:

   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_main_plumbing 2>&1 | grep -E "^(OK|FAILED|Ran )"
   grep -n "endswith(\"Integration\")" ComputeTargets/tests/test_main_plumbing.py
   ```

   The `grep` must find **nothing** — that predicate is the defect T10 names. Read the replacement
   and confirm it enumerates every class `main.py` looks up, that `QuadSourceIntegral` is now
   guarded, that `QuadSource` is classified as carrying none, and that `OneLoopIntegral` is
   represented as *out of scope pending the user* rather than as settled. Then satisfy yourself the
   guard would **fail** on a missed switch: the log should say what it would have caught that the old
   one would not.

6. **The readers followed, and the two failure modes are distinguished.** All six compile:

   ```bash
   ./venv/bin/python -m py_compile extract_Gk_data.py extract_GkWKB_data.py extract_TkWKB_data.py extract_GkSource_data.py extract_tensor_source_data.py extract_QuadSourceIntegral_data.py
   ```

   Then read the log. Every one of the six queries `wavenumber_exit_time` in a `create_k_exit_work`
   helper, and `extract_TkWKB_data.py`'s `TkNumericIntegration` lookup is the open issue that closes
   here. **The log must separate the miss cases from the exit-time case**: for the equality-keyed
   targets a stale reader finds nothing, but `wavenumber_exit_time`'s lookup takes the loosest row at
   least as tight as the request, so a stale reader **silently reuses** the new tighter row and
   reports the stored pair. An agent that treats all six lookups as one kind of change has not
   understood what it did.

7. **The digests, the suites and the bookkeeping.** **Check the digests first** — this is the one
   prompt whose parameter change can move them, through `z_init`:

   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid ComputeTargets.tests.test_gauss_order_key ComputeTargets.tests.test_numeric_break_point_key 2>&1 | grep -E "^(OK|FAILED|Ran )"
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/test_numeric_break_point_key.py ComputeTargets/tests/test_gauss_order_key.py
   ```

   Both test modules show **zero** lines changed. Then both suites: `ComputeTargets` **must not fall
   below 516** and a fall is a stop even above it; `CosmologyModels` **39**. Board rows 05a, items
   **T8** and **T10**, §3/§4, and `docs/OPEN_ISSUES.md` in the **same commit** with its count and
   date corrected — two issues close, three named in prompt §8 row 12 stay open, so the count falls
   by two unless the prompt opened something. `black --check` clean.

## 4. What a good outcome looks like

- Four targets whose tolerance is their own, each constant carrying the measurement that chose it at
  the point of use, so that the next reader does not have to find a campaign log.
- A guard that enumerates nine object types instead of pattern-matching four, and that fails on a
  class it has never seen — the failure mode being a *new* target added later without a tolerance
  decision, which is how this whole defect arrived.
- `QuadSourceIntegral` guarded for the first time, having been invisible to the old predicate for the
  entire campaign because its name ends in "Integral".
- Six readers that agree with `main.py` about which constant keys which target, and a test that says
  so — the thing that would have caught
  `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` on the day it landed rather than two
  campaigns later.
- `DEFAULT_ABS_TOLERANCE` still 1e-10, whatever it is now called, and seven float comparisons
  untouched.

**What a good outcome does *not* look like:** six new constants, a green suite, and a guard that
still pattern-matches on the class name. The numbers would be right and the next missed site would be
as silent as the last one.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **The agent disputes one of §2's numbers.** D1 is closed and the values are the user's. Relay the
  argument verbatim (rule 5); do not let it ship a different number and do not talk it out of the
  objection.
- **A source-grid digest moves.** A stop under README §4.3 even here, and *especially* here: this is
  the prompt entitled to move the tolerance that feeds `z_init`, so a moved digest is a real
  measurement rather than an accident. Relay the displacement.
- **The agent proposes to decouple `OneLoopIntegral`**, to touch the seven float-comparison sites, or
  to change a value of `DEFAULT_ABS_TOLERANCE` / `DEFAULT_REL_TOLERANCE` / `DEFAULT_QUADRATURE_*`.
- **The agent proposes a schema change**, or finds a fifth target that should be retuned.
- **The agent cannot build §7's reader cross-check without a datastore** and says so — that is a
  legitimate outcome and the user should hear it, not a failure to paper over.
- **`ComputeTargets` falls at all**, even above 516.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and sits on it — observed at 0.0601–0.0612 across
consecutive runs at an unchanged commit, passing roughly one run in three. Confirm by re-running that
module alone before attributing anything to the commit.

## 6. After it lands

**Hand back to the user, and this one is a real stopping point** in README §4.1's sense — not because
a number is open, but because the campaign's production work is finished with it and prompt 06 is a
different kind of prompt. Report:

1. The six constants and their values, and confirmation that the two shared ones did not move in
   value.
2. The agent's §3 decision on the shared names, and its argument.
3. The site inventory against log 05's table, and the `QuadSource` pair removed.
4. The guard's new predicate and what it would have caught.
5. The readers, and which failure mode each was in.
6. The digest check, with numbers.
7. Which issues closed and which stayed open, and the counts.

Then **prompt 06 is next and is not written yet** — deliberately, since 2026-09-16. Its charter is
campaign README §3.6 and §1.2, its board items are **T11** and **T12**, and it assembles
`docs/TOLERANCE-PROVENANCE.md` from the campaign's logs while measuring `QuadSourceIntegral`
read-only under §0.4. Write it against what 05a's log hands over — in particular its list of which
accuracy parameters are now decoupled and measured, which are inherited and unmeasured, and which
remain unexplained. **Do not dispatch it in the same turn**, and note that §0.4 makes 06 the one
prompt that must measure a target it may not edit.
