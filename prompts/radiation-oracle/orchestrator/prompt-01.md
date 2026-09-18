# Orchestrator — prompt 01, the Kohri–Terada radiation oracle

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../01-kohri-terada-radiation-oracle.md`](../01-kohri-terada-radiation-oracle.md) ·
**Audit:** [`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
**Model:** Opus. **Production code changed: no.** One new module, new tests under
`ComputeTargets/tests/`, this campaign's log and board, and `docs/OPEN_ISSUES.md`. **No existing
`.py` file may be modified.** `config/defaults.py` must be **byte-identical**.

Read [`README.md`](README.md) first — the seven binding rules and the suite checks are there and are
not repeated here.

**Why the review is unusual.** The measurement is done: the audit found $N=-9/8$ on all nine $b=0$
cases at `2033cfc`, so a correct implementation will pass, and so will several incorrect ones. The
paper carries three errata that each yield a smooth, plausible function, and tests 1–5 of the prompt
compare two of the agent's own transcriptions that share the Green's function, the measure and the
source. **A green suite is therefore not evidence.** What is evidence is (a) the log's record of the
implementation broken deliberately, trap by trap, with the test that caught each; and (b) test 6,
the only comparison with an object in it — the pipeline's `total` — that nobody in this campaign
wrote. Checks 3 and 4 are the review; the rest is hygiene.

**Precondition:** `prompts/tolerance-convergence` prompt 06a has landed (`300d964`, 2026-09-18) —
see [`README.md`](README.md) on ordering. **Do not dispatch if `ComputeTargets` is not 521, or if
`config/defaults.py` is not at blob `76bab78…`** — this prompt changes no number and a difference
before it starts means something else moved.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD`, the two suite counts, and **the `ComputeTargets` wall time** —
prompt §3 says test 6 costs minutes, and the user should be told what it actually cost.

```bash
git rev-parse HEAD:config/defaults.py          # must be 76bab78d9e9374263a91da87d86b509b2ed019d0
time (PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . > /tmp/ct-before.txt 2>&1); grep -E "^(Ran|OK|FAILED)" /tmp/ct-before.txt
```

(Use your scratchpad directory rather than `/tmp` if you have one.)

**Record the audit's own reference numbers, from the audit and not from memory**, because the log
will be scored against them: §5's **3.545e-15** (eq. 22 against quadrature), §6's **0.9999861197**
of $2x^2/9$ at $x=0.01$, §6.1's **0.1391877548292** at the resonance to **3.99e-16**, §7's
**3.09e-13** and §7.1's **3.18e-10**. You do **not** need to re-run `kt_verification.py`; if you
choose to, it takes minutes and its output must match the audit's §4 onwards digit for digit.

**Record the open-issues count** — `docs/OPEN_ISSUES.md` header, **82 open** at `300d964` — and
confirm `[01-general-w-normalisation-is-predicted-not-measured]` is **already** indexed at §1.9:

```bash
grep -c "01-general-w-normalisation-is-predicted-not-measured" docs/OPEN_ISSUES.md   # 1
```

The prompt's header says it "opens" that issue. The audit commit already did, on the board and in
the index. **The agent must not add a second row**, and the count must not rise on its account.

## 2. Dispatch

Launch one subagent, model **Opus**, with **exactly** this context:

> You are the implementation agent for the one prompt in a campaign. Read, in this order:
> `prompts/radiation-oracle/README.md`, `prompts/radiation-oracle/IMPLEMENTATION_STATE.md`, then
> your prompt `prompts/radiation-oracle/01-kohri-terada-radiation-oracle.md` and every file its
> "Read first" list names, in full where it says so. Execute the prompt exactly. Follow README §4
> and the repository's `CLAUDE.md` for the commit, the log, the board and `docs/OPEN_ISSUES.md`.
> Other commits may land on this branch while you work: make exactly one commit, and do not amend,
> reset or rebase anything you did not create — if you need to change a commit you already made and
> it is no longer `HEAD`, stop and say so rather than rewriting. **The baseline is `300d964`, with
> `ComputeTargets` at 521 and `CosmologyModels` at 39**, and `config/defaults.py` at blob
> `76bab78…`. **You modify no existing `.py` file.** If you find yourself needing to edit anything
> on README §2's out-of-scope list — including `test_quadsource_integral.py`, which you import from
> — stop and say so rather than editing it. **A number without its reference's own error beside it
> is not a measurement** (README §4). When you finish, reply with: the commit SHA; the **Result**
> line from your log; the "State handed to the next prompt" section verbatim; both suite counts at
> the parent and at your commit, and the number of test methods you added; the `ComputeTargets`
> wall time before and after; where you put the module and the one-sentence reason; the deliberate-
> breakage table from your log (which trap, what you broke, which tests failed); the bound test 6
> asserts and which $b=0$ cases it runs; and every deviation with its classification tag.

Add this sentence, and no more:

> **`prompts/tolerance-convergence` prompt 06a landed at `300d964` and that campaign is closed**, so
> README §2's last out-of-scope bullet is satisfied and is not a reason to stop; its documents —
> `docs/tolerance-convergence/`, `docs/TOLERANCE-CONVERGENCE.md`, `docs/TOLERANCE-PROVENANCE.md` —
> remain read-only to you, including where they call `analytic_rad` the tree's only closed form.

**Do not tell the agent where to put the module.** Prompt §1 makes the choice and its justification
part of the deliverable.

**Do not hand the agent the Ci/Si pairing, the small-$x$ limit or the `Cin` identity.** The prompt
already states all four traps and points at `kt_verification.py` for the arguments. Restating them
adds nothing the agent lacks and makes it likelier to copy your sentence than read the source.

**Do not suggest how to break the implementation.** Prompt §7 asks the agent to find out which of its
tests would catch each trap. If you name the mutations, the log records the ones you thought of.

## 3. The review — eight checks

1. **Nothing existing was modified; only additions and the bookkeeping.**

   ```bash
   git diff --name-status HEAD~1 HEAD
   git diff --diff-filter=MDR --name-only HEAD~1 HEAD
   ```

   The second must list **only** `prompts/radiation-oracle/IMPLEMENTATION_STATE.md` and
   `docs/OPEN_ISSUES.md`. Every `.py` file in the first must be status **`A`**. Then:

   ```bash
   git diff --stat HEAD~1 HEAD -- ComputeTargets/QuadSourceIntegral.py ComputeTargets/QuadSource.py ComputeTargets/phase_groups.py AdaptiveLevin/ ComputeTargets/tests/test_quadsource_integral.py config/defaults.py main.py Datastore/ 'extract_*.py' docs/spec/ docs/radiation-oracle/ docs/tolerance-convergence/ docs/TOLERANCE-CONVERGENCE.md docs/TOLERANCE-PROVENANCE.md
   ```

   must be **empty** — a single line is a **stop**. `docs/radiation-oracle/` is on this list
   although README §2 does not name it: the audit is a verification document and CLAUDE.md
   invariant 6 makes those additive, and `kt_verification.py` is what the agent *productionises*,
   not what it edits. Confirm the `config/defaults.py` blob is unchanged from §1.

2. **It is an oracle, not a pipeline component.** Find the new module's import name from the diff,
   then:

   ```bash
   grep -rn "<module import name>" --include='*.py' . | grep -v -e '^./venv/' -e '/tests/'
   ```

   must return nothing but the module itself. And the tests must import **the new module**, not
   `docs/radiation-oracle/kt_verification.py` — a module that re-exports the audit script, or tests
   that import it directly, have landed nothing and leave the oracle in a directory the suite does
   not own. Ask if you find either.

   Read the log's justification of the location (prompt §1). Any location is acceptable if it is
   argued; an unargued one is an incomplete log, not a wrong choice.

3. **The deliberate-breakage record — the check this prompt exists for.** Prompt §7 requires the log
   to say which of §2's four traps the tests would have caught, *established by breaking the
   implementation and recording what failed*. Read that table and hold it to three things:

   - **All four traps appear**, each with the specific mutation, the tests that failed and the
     tests that passed. A trap marked "caught" without a failing test named is an assertion, not a
     record.
   - **Trap 1, the inverted Ci/Si pairing, is caught by more than test 1.** The audit says the
     inverted form misses by O(1) at every $x$ and returns $-12.08$ rather than $0$ as $x\to0$, so
     tests 1, 2 and 6 should all fail against it. If only one test catches it, the suite's coverage
     of the most dangerous erratum is one assertion thick — report that; it is not a stop.
   - **Trap 4: test 4 fails against the unregularised form.** This is acceptance item 2 and it is a
     hard requirement. The log must show the failure — `inf`, `nan`, or a quadrature mismatch —
     not merely say it happened. If the log says test 4 *passed* against the unregularised form,
     the test is not evaluating at the resonance, and that is a **stop**.

   Then read test 4 itself and confirm it evaluates **exactly** at $u=v=\sqrt3/2$ — not at
   $u+v=\sqrt3+\epsilon$, and not with the resonance nudged inside `I_RD`. `math.sqrt(3)/2` doubled
   is `math.sqrt(3)` exactly in binary floating point, so a test written that way lands on the
   resonance; a test written with a literal `0.866` or `0.8660254` does not. **A guard that raises,
   returns `nan` or perturbs the argument is prompt §2 item 4's stated failure.**

   **Trap 2, the other direction.** Confirm no test asserts the paper's $x^2/2$:

   ```bash
   git diff HEAD~1 HEAD -- '*.py' | grep -nE "x[[:space:]]*\*\*[[:space:]]*2[[:space:]]*/[[:space:]]*2([^0-9]|$)|0\.5[[:space:]]*\*[[:space:]]*x[[:space:]]*\*\*[[:space:]]*2"
   ```

   A hit is fine only inside a comment or docstring that says the paper's remark is wrong. The grep
   is a net, not the check — read the small-$x$ test's assertion.

4. **Test 6 scores the pipeline against eq. (22), at the reference pair, against $-9/8$.** Read the
   test and confirm each of these; each failure is a question for the user, not a thing to repair.

   - **It compares against eq. (22) with the head subtracted**, not only against a quadrature of
     KT's integrand (prompt §3.6). A test that only uses the quadrature would pass with eq. (22)
     transcribed wrongly, which is exactly the object being landed.
   - **It asserts $N=-9/8$, not that $N$ is constant.** A test that checks the spread of $N$ across
     cases, or compares each case with the mean, passes with a wrong factor of two in the chain —
     the thing board §5 note 1 says this test regression-guards.
   - **The tolerance pair is the reference `(1e-45, 1e-12)`**, the pair `kt_verification.py` ran.
     Prompt §3 permits fewer cases, never a looser pair.
   - **The bound is defensible against 3.18e-10.** The audit's worst case against eq. (22) is
     3.18e-10, on the `q-smooth` shape. A bound near 1e-9 is a margin of ~3 and needs one sentence
     in the log. **A bound looser than 1e-8 must be argued from a named reference's own error, not
     from a margin; if it is not, stop and relay.** A bound tighter than 3.18e-10 that passes means
     the agent's head subtraction is better conditioned than the audit's — worth reporting, not a
     problem.
   - **The fixture is imported, not copied.** `Case`, `SHAPES` and `X_RESP_VALUES` come from
     `ComputeTargets.tests.test_quadsource_integral`. A local copy of the fixture is `UNINTENDED
     DRIFT` unless the log shows why the import could not work.
   - **The mapping keeps $a_0$ absorbed.** Read `total_from_I_RD` (or its equivalent) and confirm it
     is written in $1/k_{\rm phys}^2$ and says why (campaign README §4, spec 04 §0(1)). A docstring
     that says $a_0$ "is set to 1" is the convention error CLAUDE.md names, even if the arithmetic is
     right.
   - **The docstring carries the Levin paragraph** and its three limits (prompt §3.6): it bounds the
     combination, it does not reach `_three_bessel_Levin`, and it is the exact flavour only. The
     second limit is the one a later reader will get wrong.

   If the agent reduced the number of cases, the log says which it kept and why. Report the subset.
   **Ask** if it drops the `q-smooth` shape entirely — that is where the eq.-(22) figure is worst, so
   dropping it removes the case that sets the bound.

5. **The errata are in the code at the point of use** (acceptance item 3). Read the docstrings; do
   not grep for keywords and call it done.

   - `I_RD` states the Ci/Si pairing, says the rendered PDF invites the inversion, and names the
     ar5iv `alttext` as the authority — so that a reader comparing against the PDF finds out *from
     the code* that the PDF is wrong.
   - The small-$x$ statement, wherever the module documents limits, says $2x^2/9$ and says the
     paper's $x^2/2$ is wrong.
   - `f_RD` says it is built from eq. (16), not eq. (20), and why. If eq. (20) is implemented at all
     it says it is unusable below $x\approx0.1$ and is used only by test 5.
   - The `Cin` regularisation is documented where it is applied, with the identity, and `Cin` has a
     small-$z$ series branch.

   Every public function names its KT equation (prompt §1).

6. **The numbers carry their references' errors.** The log's verification section quotes each
   figure beside the error of whatever it was scored against — `quad`'s declared error for test 1,
   the audit figure for tests 4 and 6. Compare to §1's recorded audit numbers. A test-1 worst
   materially above **3.5e-15**, or a resonance figure off **0.1391877548292** beyond ~1e-15
   relative, is a question: the agent's transcription differs from the audit's, and one of them is
   wrong.

7. **The suites.** `ComputeTargets` = **521 + the number of test methods added**, every one
   accounted for; `CosmologyModels` **39**. Run the new test module alone as well, and record its
   count and wall time:

   ```bash
   time PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.<new test module> 2>&1 | tail -20
   ```

   Record the full-suite wall time against §1's. `black --check` clean on every `.py` in the diff.

8. **The bookkeeping, in the same commit.** Board row 01 filled (✅ or ⚠️, commit, log); item **R1**
   closed; the board's status line updated. The log exists at
   `prompts/radiation-oracle/logs/01-kohri-terada-radiation-oracle.md` with README §4.1's front
   matter and sections, and classifies every deviation. `[01-general-w-normalisation-is-predicted-not-measured]`
   is still in board §3 and still indexed **once** in `docs/OPEN_ISSUES.md` — `grep -c` returns 1.
   The index's count equals §1's **82** plus exactly the issues the log says it opened; the **Last
   updated** date is current. Any new §3 issue has its index row under §1.9.

   **Nothing on another campaign's board moved.** In particular
   `[06-analytic-rad-is-computed-at-the-callers-tolerance]` on `qsi-phase-groups` is **not** closed
   or narrowed: the oracle scores `total`, not `analytic_rad`'s `_three_bessel_Levin`, and board §5
   note 5 says so. An agent that closes it has misread what its own test 6 reaches.

## 4. What a good outcome looks like

- A deliberate-breakage table in which **every trap is caught by a named failing test**, and the
  resonance test demonstrably fails on the unregularised form.
- Test 6 against eq. (22), at the reference pair, asserting $-9/8$ at a bound argued from 3.18e-10 —
  run on all nine $b=0$ cases, or on a stated subset that keeps `q-smooth`.
- Docstrings a later reader could not "correct" back to the PDF without reading why they should not.
- `ComputeTargets` up by exactly the tests added, and a wall time the user can weigh.

**What a good outcome does *not* look like:** six green tests, a tidy module and a log that says the
traps were "verified". Every one of those can be produced by an implementation with the Ci/Si
arguments inverted and a test 6 that scores against its own quadrature — which is precisely the
suite the prompt was written to prevent.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s rules, prompt §6's four stops, restated for you:

- **The oracle and the pipeline disagree** on a $b=0$ case by a margin the agent cannot attribute to
  its own quadrature. **Compare against the right figure:** 3.18e-10 if scored against eq. (22)
  with the head subtracted, as test 6 must be; 3.09e-13 only if scored against a quadrature between
  the pipeline's limits. **Exceeding 3.09e-13 against eq. (22) is expected and is not a stop.**
  Relay the case and both numbers verbatim.
- **The agent needs to edit a file on README §2's out-of-scope list**, or any existing `.py` file.
- **$N=-9/8$ does not reproduce.** The audit measured it at `2033cfc`; if it has moved, something in
  the prefactor chain moved between there and `300d964`, and that is a report about the tree.
- **The agent concludes the `Cin` regularisation is wrong or insufficient.** Relay what it found. A
  better-conditioned form is a legitimate finding; a guard that avoids the resonance is not one.

And, for this prompt specifically:

- **Check 1's guarded diff is non-empty**, or any `.py` file in the diff has status other than `A`.
- **Test 4 passes against the unregularised form**, or does not evaluate exactly at the resonance.
- **Test 6 is missing, scores only against a quadrature, asserts constancy rather than $-9/8$, or
  runs at a looser pair than `(1e-45, 1e-12)`.** Prompt §3 says not to drop it for cost; a missing
  or weakened test 6 leaves the campaign with no check on the kernel at all.
- **The agent edits `docs/spec/`**, or proposes to. The finding on `cross-spec-check.md` §3 item 9
  is reported, not written in (prompt §4). This includes adding a "verified" marker.
- **`ComputeTargets` falls**, or rises by a number that does not match the tests added.
- **The agent reports `PARTIAL` or `BLOCKED`, or keeps an `UNINTENDED DRIFT`.**

**The known flake is not a stop** — see [`README.md`](README.md).

## 6. After it lands

**This closes the campaign** (campaign README §5). Report to the user:

1. Where the module lives and the agent's reason, in one line.
2. The deliberate-breakage table: each trap, the mutation, which tests caught it — and any trap
   caught by only one test.
3. The numbers, each beside its reference's error and the audit's figure: test 1's worst against
   `quad`, the small-$x$ ratio, the resonance value, test 6's worst deviation from $-9/8$ against
   eq. (22) and the bound it asserts.
4. What test 6 runs — all nine cases or a subset — and the `ComputeTargets` wall time before and
   after. If the suite grew materially, that is the user's cost to weigh; the agent was entitled to
   keep all nine.
5. Both suite counts at `300d964` and at the commit, and how many tests were added.
6. Which issues opened, closed and stayed open, and the index count. Expect
   `[01-general-w-normalisation-is-predicted-not-measured]` still open: it is the campaign's standing
   limit, not unfinished work.
7. **The item for another campaign's owner.** `docs/spec/cross-spec-check.md` §3 item 9 lists the
   $h_{ij}$ normalisation relative to Kohri & Terada under "no spec allows a check", and
   `docs/spec/05-one-loop.md` §539 records it as transcribed and unchecked. It is now checked, by a
   test in the tree, to the bound test 6 asserts. **The spec is `prompts/spec-transcription`'s and
   its sign-off is the author's** — relay this as a finding for them, with the test's name, and do
   not propose the edit.
