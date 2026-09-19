# Orchestrator — prompt 02, the realistic-flavour large-$x$ harness

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you and the
suite baselines. **You do not write code.**

**The prompt:** [`prompts/handover/02-realistic-flavour-large-x-harness.md`](../02-realistic-flavour-large-x-harness.md)
**Board item:** A2 · **Closes:** nothing. It is an instrument.

## 0. What makes this prompt unusual

It closes no issue and adds no test. Its output is **two numbers per cell** — the clamp term and the
representation term — and the entire value of the prompt is whether those two numbers are honestly
separated and honestly bounded. A suite that stays green tells you nothing about that.

So the review is not "does it run". It is:

- does the **control cell** (`exact` × gap-closed) reproduce KT §8 Table 8.1 to the digits printed?
  If not, nothing downstream of it means anything.
- is the **gap the fixture opens comparable to production's**, in grid-step units? If not, the clamp
  term measured is not the clamp term that matters.
- does the document say **what it does not cover**, in KT §8's closing manner?

This prompt is also the one most likely to produce a *negative* result — the realistic flavour may
not drive at large $x$ at all. That is a finding, not a failure, and prompt §7 says so. Do not let a
subagent convert it into a fallback to the exact flavour.

---

## 1. Before you dispatch

1. **Confirm `HEAD`** on `claude/integration-handover-review-f4bfb4`. Record the SHA; the agent must
   be told it, and told that `HEAD` is **not** its own.
2. **Take the baselines.** For this prompt they are a *ceiling and a floor*: both suites must read
   **exactly** what they read before. Record them at the actual `HEAD`, not from this file — prompt
   01 may have landed and raised `ComputeTargets`.
3. **Confirm `IMPLEMENTATION_STATE.md` exists.** Prompt 01 creates it. If it does not, prompt 01 has
   not landed: either dispatch that first, or tell the agent to create the board and say so —
   **ask the user which**, do not decide.
4. **Confirm the inputs**: `docs/radiation-oracle/large_x.py`, `eq22_rounding.py`,
   `KOHRI-TERADA-ORACLE.md` with §8 present, and `ComputeTargets/tests/test_quadsource_integral.py`.
5. **Run `large_x.py` yourself, unedited, and keep its output.** ~70 s. It is the control the
   subagent's §2 cell must reproduce, and having taken it yourself means you are not checking the
   agent's claim against the agent's own run.
6. `git status` clean.

## 2. Dispatch

One fresh-context subagent. Give it the prompt file and only that prompt, the SHA, the baselines,
the campaign `README.md`, and the files its "Read first" list names.

Tell it plainly: **one commit**; log at `logs/02-realistic-flavour-large-x-harness.md`; board and
`docs/OPEN_ISSUES.md` in the same commit; `black` before committing; **suite counts unchanged, not
risen**; and that §7's stop conditions mean *stop and ask*.

## 3. The review — eight checks

1. **Suites unchanged.** `ComputeTargets` and `CosmologyModels` must read **exactly** the baseline
   figures. A *rise* is as much a failure as a fall: prompt §5 says this adds no test. If the agent
   added one, that is a scope breach — report it, do not accept it as a bonus.
2. **The control cell.** Compare the agent's `exact` × gap-closed rows against the `large_x.py`
   output **you** took in §1.5, row by row. Digits printed. Any disagreement is a stop.
3. **One command, no Ray, no datastore.** Run it yourself from the repository root with
   `PYTHONPATH=.`. Record the wall time and compare it against what the log claims.
4. **The gap is production-shaped.** The document must state the gap the fixture opens in units of
   the fixture's own grid step, against production's median 1.2e-02 and maximum 2.2e-02 in
   $\log(1+z)$ (about one mean source-grid step). A document that reports a clamp term without this
   comparison has measured something whose relevance is unestablished — stop.
5. **The two terms are separated and each carries its reference's error.** README §5 rule 8. Check
   the 2×2 is actually reported as a 2×2: if only two of the four cells were run, the difference
   attributed to "the clamp" is confounded with the flavour.
6. **The 50-digit path is used on `q-smooth`.** At $u = 0.01$, `kt_verification.I_RD` rounds at
   ~3e-10 and `kohri_terada` at ~1e-10 (KT §7.2). If the document's `q-smooth` figures are at that
   level, the agent is reporting eq. (22)'s rounding as a pipeline result.
7. **Scope.** `git diff --name-only HEAD~1 HEAD`. Permitted: `docs/handover/realistic_large_x.py`,
   `docs/handover/REALISTIC-LARGE-X.md`, `prompts/handover/IMPLEMENTATION_STATE.md`,
   `prompts/handover/logs/02-*.md`, `docs/OPEN_ISSUES.md`, and — only if the log justifies it as a
   narrowing — another campaign's board §3 entry. **`docs/radiation-oracle/` must be untouched**;
   that campaign is closed and its documents are its record.
8. **The document has a §0 and a closing "what this does not cover".** KT §0 is the model for the
   first and KT §8's last paragraph for the second. At minimum the closing must name: $b = 0$ only;
   the fixture grid is not the production grid; production's $x \approx 4\times10^{12}$ is still out
   of reach.

## 4. What a good outcome looks like

- The control cell reproduces Table 8.1 exactly.
- The clamp term and the representation term separate cleanly, with the clamp term at the
  gap-closed-to-gap-open difference and the representation term at the exact-to-realistic
  difference, each per shape and per $x$, each with an error beside it.
- A statement about how each scales with $x$ that is **measured**, and that says so where it
  disagrees with the prompt's own guess.
- An honest closing section.

A result in which the two terms **do not** separate — a large cross term — is also a good outcome,
provided it is reported as such. It would mean the clamp and the representation interact, which
changes what B1 can be expected to buy, and that is exactly the kind of thing this instrument exists
to find.

## 5. Stop and ask the user

Relay verbatim.

- The control cell does not reproduce KT §8 Table 8.1.
- The realistic flavour will not drive at large $x$.
- The 2×2 is not additive.
- The fixture gap cannot be made comparable to production's.
- The agent proposes editing `docs/radiation-oracle/`, `test_quadsource_integral.py`, or any
  production file.
- The agent proposes adding a test to `ComputeTargets/tests/`.
- Any check in §3 fails.

## 6. After it lands

Report: the commit; both suite counts before and after (**unchanged** is the expected result); the
control-cell comparison; the clamp term and the representation term with their errors; the gap in
grid-step units against production's; the script's wall time; any §3 issue opened; and any narrowing
note added to another campaign's board.

Then stop. The next thing in the dependency graph is **B1**, which changes production code and has
no prompt written. Do not start it.
