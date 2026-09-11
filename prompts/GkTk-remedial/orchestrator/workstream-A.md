# Orchestrator prompt — Workstream A (prompts 01, 02)

You are orchestrating Workstream A of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`, cut from `main` at or after
`9ff59d5` — record the base you find). You do not write code yourself. You dispatch one
fresh-context subagent per prompt, review what it produced against fixed criteria, and either
continue or stop and report to the user.

Workstream A changes **no production code**. It builds the references every later prompt is scored
against, prototypes the design, and measures the two things that could change it: the interval
accessor's throughput (prompt 01) and whether a fixed Gauss order converges on `QCD_Cosmology`
(prompt 02). So the review here is about **independence of the references and completeness of the
measurements**.

## What to read

- `prompts/GkTk-remedial/README.md` — §0–§0.4, §1, **§2 (all eight design facts)**, §3, §4,
  §4.3, §5, §5.1, **§6 (acceptance table and error definitions)**, §7.
- `prompts/GkTk-remedial/RECONCILIATION.md` — all of it.
- `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` — the board, §3, §5.
- `prompts/GkTk-remedial/orchestrator/README.md` — campaign-wide stop conditions.
- `docs/gk-wkb-review-fable-2026-09-09.md` §7, §13.3.

Read `01-reference-harness-and-prototype.md` only when about to dispatch it; likewise 02. **Do not
read prompts 03–13.**

## Preconditions

`git status` clean; board rows 01, 02 ⬜. Environment:

```bash
PYTHONPATH=. ./venv/bin/python -c "import scipy, numpy, mpmath; print(scipy.__version__, numpy.__version__, mpmath.__version__)"
```

Expect `1.15.2 2.2.4 1.3.0`. Baseline suite runs:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
```

Record the count and the time. Also check the in-flight sibling: `git branch -a | grep -i
transfer\|workstream` and note in your report which `transfer-remedial` prompts have landed on
which branch (`../README.md` §4.2) — nothing to decide yet, but Workstream D will need it.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/GkTk-remedial/README.md`, `prompts/GkTk-remedial/RECONCILIATION.md`,
> `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/GkTk-remedial/NN-<name>.md` and the sections of
> `docs/gk-wkb-review-fable-2026-09-09.md` it cites. Execute the prompt exactly. Do not read any
> other prompt under `prompts/GkTk-remedial/`. You may read the "State handed to the next prompt"
> sections of the logs your prompt names, and only those. Follow README §5 for the commit, the log,
> the board update and `docs/OPEN_ISSUES.md`. Other commits may land on this branch while you
> work: make exactly one commit, and do not amend, reset or rebase anything you did not create —
> if you need to change a commit you already made and it is no longer `HEAD`, stop and say so
> rather than rewriting. When you finish, reply with: the commit SHA, the
> **Result** line from your log, the "State handed to the next prompt" section verbatim, and every
> deviation with its classification tag.

Model: **01 → Opus**, **02 → Opus**. Run 01 → 02.

## Reviewing prompt 01

1. One new commit, message per README §5 rule 2.
2. `logs/01-reference-harness-and-prototype.md` exists, follows §5.1, and its "State handed to the
   next prompt" lists **verbatim** the module's public names and signatures, the JSON schema, the
   reference method and floor per model, the prototype's full measurement table, the two baselines.
3. Board row 01 and M23 updated; `docs/OPEN_ISSUES.md` unchanged unless §3 changed.
4. Tests pass when you run them, quickly and without `mpmath`:
   ```bash
   time PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_wkb_reference -v
   grep -n "mpmath" ComputeTargets/tests/wkb_reference.py ComputeTargets/tests/test_wkb_reference.py
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
   The grep must be empty (mpmath belongs in the generator under `docs/`).
5. `git diff HEAD~1 --stat`: only the allowed new files, the log, the board. No production module.
6. **Independence:**
   `grep -n "WKB_phase_function\|phase_spline\|cumulative_table\|phase_residual\|primitive_phase" ComputeTargets/tests/wkb_reference.py`
   must be empty.
7. **The numbers.** Read the prototype table: order 4 on LambdaCDM $\le2\times10^{-14}$ at the
   nodes; the double-double vs single-double short-baseline rows differ by the predicted
   $\sim10^{-6}$–$10^{-3}$ relative; the baselines are within 5 % of 13.9 rad and 2.01 rad. The
   **throughput row**: if `delta` costs more than 50 µs per off-grid call on `QCDModel`, or on-grid
   calls are not free of Hubble evaluations, **stop** — a design question (README §4.3).
8. The QCD references' recorded floor is stated; if orders 4 and 8 disagree on QCD by more than
   $10^{-13}$, note it in your report (prompt 02 decides; not a stop).

## Reviewing prompt 02

1–3 as above (M9 updated).
4. Tests: `test_wkb_reference.py` still passes with the extended JSON; the convergence script runs:
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/residual_convergence.py 2>&1 | tail -20
   ```
   (may take minutes; record the time).
5. Allowed files only (no production module).
6. **The decision is written down**: `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` and the log give
   $N_\tau$, $N_{\tau_s}$, $N_F$, $N_\rho$ as integers with the measurement that chose each, and
   the fallback flag. The LambdaCDM $\tau$ row reproduces review §7 (order 4 at the floor). The
   radiation controls are asserted ($\rho_G$ exactly zero; $\rho_T=1/x_i-1/x$).
7. **If the fallback flag is set** (no order $\le16$ converges for $\rho$ on QCD), **stop** and
   report the offending intervals and the proposed adaptive rule's cost. This is expected to be
   rare but it is the review's named risk (§11).
8. The predicted $\varphi$ spline error (item 4 of the prompt) is present — prompts 09/10 need it.

## Continue or stop

Continue when both logs are `COMPLETE`-class with only `IMPLEMENTATION CHOICE` deviations and
every check passes. Stop on any campaign-wide condition or on the two design questions above
(throughput, fallback flag).

## Completion criterion

Rows 01, 02 ✅/⚠️; two logs; the suite passing. Report: "Workstream A complete; the tree is at
`<SHA>`. Nothing in production has changed. Decision D1 (`[00-tau-storage-decision]`) is already confirmed by the user (new
`BackgroundModelValue` columns). Also report the
Gauss orders, the throughput figures and the `transfer-remedial` branch state."
