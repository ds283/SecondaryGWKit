# Orchestrator — prompt 01, the Domènech general-$b$ oracle

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you and the
suite baselines. **You do not write code.**

**The prompt:** [`prompts/handover/01-domenech-general-b-oracle.md`](../01-domenech-general-b-oracle.md)
**Board item:** A1 · **Closes:** `[01-general-w-normalisation-is-predicted-not-measured]`, on the
**`radiation-oracle`** board's §4, not this campaign's.

## 0. What makes this prompt unusual

The subagent is transcribing an equation whose published form is **wrong by an overall sign**. A
faithful transcription of the page produces a smooth function with the right scaling, the right
limits in $x$, and the wrong sign. **Every test that compares the module against itself will pass.**

So the review is not "do the tests pass". It is:

- does **test 5** — the tie to `kohri_terada`, the one object in the suite this campaign did not
  write — actually exercise the sign?
- does the **deliberate-breakage record** show that flipping the combination back to the printed
  order fails something?

If either is missing, the prompt has not done its job however green the suite is.

---

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`.** The campaign runs on **`handover-remedial`**, which starts
   at `d6afa05`; expect that or later. Record the actual SHA; the agent must be told it, and told that `HEAD` is **not** its own.
2. **Take the baselines** of [`README.md`](README.md) and record them. `ComputeTargets` **530**,
   `CosmologyModels` **39**, `LiouvilleGreen` **148** (`skipped=1`), all OK — measured at
   `b80b0f5` and unchanged at `d6afa05`, which moves only markdown. The full tree is ~2.8 minutes
   since `main` was merged, so take all three.
3. **Confirm the inputs exist**: `docs/handover/DOMENECH-KERNEL-RECON.md` (862 lines, §11 present),
   `docs/handover/sources/SOURCES.md`, the two `-src/` directories with their `.tex`, and
   `ComputeTargets/tests/kohri_terada.py`. Run
   `cd docs/handover/sources && shasum -a 256 -c CHECKSUMS` — **8 of 8 must be OK**. If a `.tex`
   line has moved, the recon's `file:line` citations are stale and the agent will be misled.
4. **Confirm `IMPLEMENTATION_STATE.md` does not exist.** Prompt §8 has the agent create it. If it
   does exist, someone landed work you do not know about: stop and report.
5. `git status` clean.

## 2. Dispatch

One fresh-context subagent. Give it:

- the prompt file, and **only** that prompt;
- the SHA of `HEAD`, with "this is not your commit; do not assume anything at `HEAD` is yours";
- the baselines;
- the campaign `README.md`, and the files the prompt's "Read first" list names. Nothing else.

Tell it plainly: **one commit**; the log at `logs/01-domenech-general-b-oracle.md`; the board created
per §8; `docs/OPEN_ISSUES.md` in the same commit; `black` before committing; and that §6's stop
conditions mean *stop and ask*, not *loosen and continue*.

## 3. The review — nine checks

Run these yourself, at the subagent's commit.

1. **Suites.** `ComputeTargets` must **rise by exactly the number of test methods added** — count
   them in the diff — and must not fall. `CosmologyModels` must read **39**. The flake in
   `test_tk_wkb_phase.TestCost.test_wall_time_per_object` is not a stop; re-run that module alone
   to confirm.
2. **The deliberate-breakage record exists** and covers all four breakages of prompt §3: the
   combination sign, the off-cut factor 2, `I_target` for `I_quadrature` in test 8, and a
   `type=2`/`type=3` swap. For each, the log must name **which tests failed**. "All tests failed" is
   a weak record; "test 5 and test 3 failed, tests 1, 2, 6 did not" is a strong one.
   **A breakage that caught nothing, with no test added in response, is a stop.**
3. **The sign is the corrected one.** Read the module. The combination must be
   $\big(Y_{b+1/2}(x)\,\mathcal I_J - J_{b+1/2}(x)\,\mathcal I_Y\big)$. If it matches the review's
   printed (4.10), the prompt has been transcribed rather than implemented — stop.
4. **The off-cut factor 2 is present** — $2\frac{b+2}{b+1}$ against $\frac{b+2}{b+1}$ on the
   on-cut branches. If it has been "tidied" to match, stop.
5. **Test 5 ties to `kohri_terada` and `kohri_terada` is unmodified.**
   `git diff HEAD~1 HEAD -- ComputeTargets/tests/kohri_terada.py` must be empty, as must the diff of
   `ComputeTargets/tests/test_quadsource_integral.py`.
6. **Test 8 scores against `I_quadrature`, not `I_target`.** Prompt §2 item 4. If it uses
   `I_target`, the $N$ it reports is contaminated by the non-uniform $O(1/x)$ and is not a
   measurement of the pipeline.
7. **$N$ is reported with its spread and with the reference's own error.** A bare "$N = -1.1942$"
   does not meet README §5 rule 8. The log must also say whether the test asserts on **constancy**
   or on the value — prompt §3 test 8 requires it to say.
8. **Scope.** `git diff --name-only HEAD~1 HEAD`. Permitted: the new module and its test module
   under `ComputeTargets/`, `prompts/handover/IMPLEMENTATION_STATE.md`,
   `prompts/handover/logs/01-*.md`, `docs/OPEN_ISSUES.md`. Anything else — especially `main.py`,
   `config/`, `QuadSourceIntegral.py`, `docs/spec/`, `docs/radiation-oracle/` — is a stop.
9. **Bookkeeping.** Board created in the shape §8 describes, with prompt 00's landed row present and
   its "no log" noted. `docs/OPEN_ISSUES.md`: the closed issue's row **deleted**, count and date
   corrected, §1.9 left with a line rather than a bare heading. `black --check` clean. Commit
   message in `CLAUDE.md`'s form with the `Co-Authored-By` naming the model that did the work.

## 4. What a good outcome looks like

- $N$ constant over the nine $b = 0.2$ cases, at $-1.194214876\ldots$, with a spread quoted beside
  the reference's own error — and the log saying which of the two it asserts on.
- Test 5's $b=0$ tie at $\le 4.5\times10^{-15}$, except at $u = 0.01$ where it floors around
  $4\times10^{-10}$ on eq. (22)'s own rounding, **with both quoted**.
- A breakage record in which the sign flip is caught by test 5 and *not* by the module's internal
  consistency tests. That asymmetry is the evidence the suite has an external anchor.
- A §3 issue for the $y>1$ continuation if it was implemented on the recon's inferred rule.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- $N$ **drifts** with $(u,v,x)$.
- $N$ is constant and is **not** $-\frac{(3+2b)^2}{2(2+b)^2}$.
- The recon's §9 brief does not construct.
- Test 5 fails.
- The agent asks whether to read Gervois–Navelet for the $y>1$ continuation.
- The agent proposes editing `test_quadsource_integral.py`, `kohri_terada.py`, or anything in
  `docs/radiation-oracle/`.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report to the user: the commit, the suite counts before and after with the rise attributed to named
tests, the measured $N$ and spread with its reference error, the four breakage outcomes, any §3
issue opened, and the `docs/OPEN_ISSUES.md` count before and after.

Then stop. **Do not dispatch prompt 02 without being asked.** It is independent work and the user
may want to read this one first.
