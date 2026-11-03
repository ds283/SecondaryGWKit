# Orchestrator — prompt 06i, Filon references for Z3's calls

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**: the references are the agent's, and your own checks are scratch readers.

**The prompt:** [`prompts/levin-phase-floor/06i-filon-references.md`](../06i-filon-references.md)
**Board item:** 06i · **Closes:** nothing. **Model:** Opus.

A measurement on the laptop: no code change, no Ray, no store, no lock. One agent, reviewed here;
the records commit is the orchestrator's. Written 2026-11-01 at dispatch, on the user's word:
*"If 06i is fast, then yes, please write the orchestration note and dispatch now."* 06h's Part B
runs on the Mac Studio meanwhile; 06i touches nothing there.

## 0. Facts, checked by the orchestrator, 2026-11-01

- **The inputs:** Z3's 13 pickles, copied by `scp -p` from the Mac Studio's
  `var/levin-phase-floor-06e/census/references/inputs/` into the laptop's
  `var/levin-phase-floor-06i/inputs/`. All 13 equal `06e/KEPT_ON_MAC_STUDIO.md`'s SHA-256, recorded
  in `inputs/SHA256SUMS.expected`.
- **The integrand's tree:** the worktree `../SecondaryGWKit-lpf06i-6d81fad`, detached at `6d81fad`
  (06e's Part B tree, code `f4d8e17`'s), with no venv. Loaded with
  `PYTHONPATH=<worktree>:var/levin-phase-floor-06e/census/scripts`, `PYTHONDONTWRITEBYTECODE=1` and
  the main checkout's `./venv/bin/python -B`, `s08` and `s01` unpickle, and `theta_mod_2pi` and both
  amplitudes evaluate. The worktree stays clean.
- **What the pickles hold:** `x_span`, `f` (two callables), `theta` (a dict of callables), `kw`, the
  census's `value`, `abserr`, `num_regions`, `max_depth` and `converged` (06c's capture; 06e's Z3
  re-scored them, `out/z3_scores.json`). For the analytic blocks, `theta`'s entries are bound methods
  of the `BesselPhaseGroup` object, so `__self__` gives `phases`, `coefficients`, `signs`, `K` and
  `C`. For the cosmological groups they are closures from `ComputeTargets/phase_groups.py`.
- **The 06e Z3 table**, by sample (route, counted?): 1 (B, dropped), 2 (B, counted), 3 (B,
  counted), 8 (A, counted), 9 (A, counted), 10 (A, dropped), 13 (B, dropped), 14 (B, counted; the
  same call as 2 at `1e-8`), 15 (B, counted), 22, 23, 25 and 27 (A, dropped).
- **Tools:** scipy 1.16.3 and mpmath 1.3.0 are pinned in the venv.

## 1. Before you dispatch

1. `git status --short` is empty; nothing is `running` or stale on the laptop.
2. Commit this note and its README row, and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad; record the inputs' SHA-256, the four stashes' commits and
   `var/levin-phase-floor-07/wip/`'s SHA-256.

## 2. Dispatch

One fresh-context agent, on **Opus**, in the main checkout. It is given the prompt, the dispatch SHA
and §0, and it does not read `orchestrator/`. Tell it plainly:
- **Where it writes:** only under `var/levin-phase-floor-06i/`, as `scripts/`, `out/` and `log.md`,
  `verification.md` and `issues.md`. The prompt's `docs/levin-phase-floor/06i/` is the
  orchestrator's copy of `scripts/` and `out/` at the records commit.
- **No commit,** and no git command that writes. It never runs `git stash` and never touches
  `var/levin-phase-floor-07/`, the inputs or the worktree, which it uses only as `PYTHONPATH`.
- **No code change** anywhere in the repository. No Ray, no store, no Mac Studio.
- **The predictions** R0–R4 are copied into `log.md` before any sample is run, and its record shows
  when.
- **Every deviation is classified**, and it **stops and asks** at prompt §8's conditions.
- **§4, the optional item, is not run.** It is reported as available if R1 holds, for the user to
  decide.

## 3. The review — on the laptop

1. **Scope:** nothing under the repository changed outside `var/levin-phase-floor-06i/`; no file
   under `var/` newer than the marker outside it; the inputs, the stashes and `wip/` unchanged by
   SHA-256; the worktree clean; `git log` shows no new commit.
2. **R0, re-checked:** recompute two closed-form cells yourself in `mpmath` at 50 digits, at
   $\omega = 10^6$ and $10^9$, and compare with the agent's two implementations.
3. **One reference re-made independently:** on sample 8 (Route A), and on sample 2 if Route B is
   scored, recompute the reference with your own scratch reader. Use a method different in at least
   one respect: another panelling, another $N$, or QAWO against the `mpmath` rule. It must agree
   with the agent's to within the scatter.
4. **Determinism:** one of the agent's Route A readings re-run in a fresh process, bit-identical.
5. **R1–R4 re-scored** by your own reader from the per-sample tables and `z3_scores.json`, against
   the rules as written.
6. **Route B's $t(x)$:** the construction and its bound read against prompt §0.3, if Route B was
   run.
7. **The records:** README §5.1's headings, every deviation classified; the predictions before the
   numbers; provenance in every output; references as hex doubles.

Then report to the user with R0–R4 and what each implies for 06g. **On the user's word,** make the
records commit:
- `docs/levin-phase-floor/06i/`, from `scripts/` and `out/`, without the inputs;
- the log in `logs/`;
- `VERIFICATION.md` §6i;
- the board, and `docs/OPEN_ISSUES.md` if an issue is narrowed or opened;
- the scripts' directory listed in `CLAUDE.md`'s formatting paragraph.

## 5. Stop and ask the user

At any of prompt §8's conditions, or when a check above fails: report the specific check and what
the log says about it. Do not repair. **Never touch 07's stashes or saved work, or anything on the
Mac Studio.**

## 6. Addendum, 2026-11-01: the stop, the run, the review and the records

**The stop.** The agent stopped at Gate R0, before any sample. The closed forms were met, but the
integrand rule could not hold: it compared against the census tree's declaration, which 06h found
incomplete, and the cosmological groups declare none. Relayed verbatim, with the orchestrator's
view that the prompt was at fault; the user: *"Yes, (A), and your two additions"* (prompt §9
point 1).

**The run.** All 52 readings ran on the laptop, at most 6 processes, none over 4 minutes. At the
user's request the agent moved its remaining readings into one background job, to stop interim
wake-ups. R1, R2 and R4 hold; R3 fails on sample 3 alone, 1.241. The agent asked about sample 8's
implementation check; the user: *"I accept sample 8."*

**The review** (§3), run by the orchestrator, passed:
1. **Scope:** no commit, no code change, the inputs, stashes and `wip/` unchanged, the worktree
   clean.
2. **R0:** two cells at $\omega = 10^6$ and $10^9$ against the orchestrator's own antiderivative,
   every implementation's error as recorded.
3. **One reference re-made:** sample 8 by the orchestrator's own closed form from the captured
   inputs, all four readings within $3\times10^{-25}$. For Route B, sample 2's Filon reference
   agrees with 06e's independent Gauss–Legendre one to $10^{-4}$ of the allowance.
4. **Determinism:** sample 9's FCC reading bit-identical in a fresh process.
5. **R1–R4** re-scored by the orchestrator's reader, the same verdicts.
6. **Route B's $t(x)$:** read against prompt §0.3; its fit-noise bound is below $1/100$ of the
   minimum abserr on every sample.
7. **The records:** README §5.1's headings, 15 deviations classified, the predictions before the
   numbers, provenance headers.

**The records commit**, on the user's word (*"Yes, please make 06i's records commit"*), carries
`docs/levin-phase-floor/06i/`, the log, `VERIFICATION.md` §6i, the prompt's §9, the board and
`docs/OPEN_ISSUES.md`. The scripts are `black`-clean, so `CLAUDE.md`'s formatting paragraph does not
list them. The 17 never-referenced Z3 samples go to 06g.

**The integrand worktree removed, 2026-11-03.** On the user's word (*"Please remove those
worktress."*), the laptop removed `../SecondaryGWKit-lpf06i-6d81fad` with `git worktree remove`. It
was clean with `--ignored`. `6d81fad` is in the history, so the tree can be remade.
