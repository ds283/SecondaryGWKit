# Prompt 06 — Provenance for the file's three solves, and close-out

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(i)** · **Implements:** `AUDIT.md` §3.3, §7
**Depends on:** 01–05, all landed. This prompt closes the campaign.
**Recommended model:** **Opus** — provenance prose for three solves, a cross-campaign board
amendment, and the close-out verification. **It may not touch production code, and that rule is
the review.**

**Files you may touch:** `prompts/background-solver-robustness/PROVENANCE.md` (**new**),
`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` (**§3 only**, the `:1008` bullet),
`prompts/tolerance-convergence/README.md` (**§1.2 or wherever it lists this site as unexplained** —
one sentence), plus this campaign's README §6 acceptance table, log, board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** any file under `CosmologyModels/`, `ComputeTargets/`, `CosmologyConcepts/`,
`Datastore/`, `LiouvilleGreen/`, `Quadrature/`, `main.py`, or `docs/`. **Zero production files and
zero test files in the diff.** If a test needs changing for this prompt to pass, the campaign has a
defect and you stop rather than change it.

**Read first:** `AUDIT.md` §3.3 and §7; README §0.4 and §0.6; every log in
[`logs/`](logs/) — **all five, in full; this prompt is a synthesis and it may not re-derive
anything**; `prompts/tolerance-convergence/README.md` §1.2 and §0.5;
`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3's "Recorded by the rebase, not owned
here" block; and `LambdaCDM_GenericEOS.py:569-583` and `:840-870`, the two comments that already
carry their own provenance.

---

## 1. Why this prompt exists

`prompts/tolerance-convergence` README §1.2 requires that when that campaign closes, **no accuracy
parameter in the pipeline is unexplained**, and its prompt 06 creates
`docs/TOLERANCE-PROVENANCE.md`. That file does not exist yet, and that campaign has not started.
`AUDIT.md` §7:

> If a fix lands before `tolerance-convergence` prompt 02 runs, prompt 02 records a settled
> provenance entry and `docs/TOLERANCE-PROVENANCE.md` can state it. If it lands after, the note
> records *unestablished* and the fix becomes a follow-up amendment. … the cheap moment is now.

So this prompt writes the provenance **in the shape that file will want**, as a campaign document,
and points that campaign's board at it. It does **not** create `docs/TOLERANCE-PROVENANCE.md`:
creating a project-wide document that another planned campaign's prompt 06 is chartered to create
would collide, and this campaign does not own it.

## 2. `PROVENANCE.md`

One entry per `root_scalar` site in `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`. **Three
entries**, not one — two of them already have their reasoning in code comments, and transcribing
them into the same shape is most of what makes the document useful to `tolerance-convergence`
prompt 02, which has to inventory all three.

Each entry carries **every** one of these fields. A field with no answer says so explicitly; it
does not go missing.

| Field | Meaning |
|---|---|
| **Site** | file, symbol and line at the campaign's final commit |
| **Value** | the `xtol`/`rtol` pair as shipped |
| **Method** | Brent, and whether bracketed |
| **What it sets** | the quantity, and whether that quantity is computed, stored, keyed, or diagnostic |
| **Call count** | per model construction, per run |
| **Choosing measurement** | what was measured, by whom, **and at which commit** — README §5 rule 9 |
| **Competing floor** | what limits the answer independently of this tolerance (Brent's $4\varepsilon$; the representation; the closed form) |
| **Cost** | extra evaluations × call count, in the units the sector counts in |
| **Citation** | campaign, prompt, log |

The three entries:

1. **`_solve_T_z`** — `xtol=1e-300, rtol=1e-14`, bracketed, ~3,000 calls per model construction,
   chosen by `qcd-background-audit` prompt 04 (which measured the pre-change node scatter at
   2.496e-05 and the p90 it set), reasoning already in the code at `:569-583`. **Transcribe; do not
   re-measure and do not re-word the code comment.**
2. **`_temperature_crossing_log1pz`** — `xtol=1e-15, rtol=1e-15`, bracketed, **zero production
   calls**, chosen by `qcd-background-audit` prompts 06/07, and **as of prompt 04 of this campaign
   it is test machinery in `CosmologyModels/tests/T_z_reference.py`**. The entry must record the
   move and that the value did not change. The "what it sets" field is where the trap belongs: a
   bracketing solver on a function with no root reports `converged` and returns a point whose
   offset (~1.126e-12 in $u$) depends on its tolerances.
3. **`_find_rho_equality`** — the campaign's own. `xtol=1e-300, rtol=1e-14`, **now bracketed**,
   two calls per model construction, chosen by prompt 02 of this campaign against the measurements
   in `AUDIT.md` §3.1 and log 02. The "what it sets" field must say **diagnostic** and must then
   point at `RECONCILIATION.md` §5 / log 03 — the *quantity* is a production sample location by a
   different route, and a provenance entry that says "diagnostic" without that pointer is the
   mistake this campaign was created to stop being made twice.

Head the document with a short §0 saying what it is, that it is written to be lifted into
`docs/TOLERANCE-PROVENANCE.md`, and that the code comments at the point of use are the primary
record — this document is the index, and if the two disagree the code is right.

## 3. The cross-campaign amendment

In `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3, under "Recorded by the rebase, **not
owned here**", replace the `LambdaCDM_GenericEOS.py:1008` bullet with a settled statement: what it
now is, the line it is now at, that its provenance is
`prompts/background-solver-robustness/PROVENANCE.md`, and that prompt 02 of that campaign should
record it rather than re-derive it. **Leave the other two bullets in that block alone** —
`[11-stop-point-root-tolerance]` and the `QuadSourceIntegral.py:1550` comment are not this
campaign's, and README §0.5 says so.

If `prompts/tolerance-convergence/README.md` lists this site among the parameters "nobody has ever
chosen" (§1.2, and `:394` names the line), amend that sentence too. **One sentence.** Do not
restructure that campaign's README; `be21f5c` wrote it four commits ago and it is not yours.

## 4. Close-out verification

**Run these yourself and quote the output.** Do not take them from the logs.

1. Both suites at the campaign's final commit: `CosmologyModels` and `ComputeTargets`. Compare with
   §10 of `RECONCILIATION.md` (30 and 447 at the pre-campaign baseline `f023eb8`; `ComputeTargets`
   is **449** from prompt 09 on) and with each prompt's recorded count. **A count that does not add
   up across the logs is a finding, not an arithmetic slip to fix silently.**
2. `PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py` —
   **unedited**. Its §2.2 table is now measuring a bracketed Brent solve rather than a secant, so
   the evaluation counts will differ and the roots must not. Quote the whole table and say which
   columns moved.
3. `black --check` over every file the campaign touched.
4. `git diff --stat f023eb8..HEAD` — the whole campaign in one table, in the log.
5. **The audit's own claim**, re-stated against the shipped tree: `AUDIT.md` §5's *"fixing this
   changes no computed quantity in the pipeline"*. Say whether it held, with the evidence: the four
   equality redshifts across the campaign, the grid digest from log 03, `T_Z_REPRESENTATION_VERSION`
   at 6 throughout, and `ComputeTargets` at 447 at every commit through prompt 04 and **449**
   from prompt 09 on.

## 5. The close-out section of the board

Write `IMPLEMENTATION_STATE.md` §5 ("Close-out") containing:

- the campaign's result — prompts landed, of how many, with the workstream D decision stated;
- the acceptance table of README §6 with every threshold's **measured** value beside it;
- what remains open, with each row's owner — including anything workstream D was not authorised to
  take;
- README §7's four decisions, each marked **taken** (with the answer and who took it) or
  **outstanding**;
- the one-paragraph statement a later reader needs: what this campaign changed, what it did not,
  and what it proved.

## 6. Acceptance

| Check | Threshold |
|---|---|
| Production and test files in the diff | **zero**; `git diff --name-only` quoted in the log |
| `PROVENANCE.md` | three entries, every field present in each |
| `tolerance-convergence` board | the `:1008` bullet replaced; the other two untouched, demonstrated by diff |
| Both suites | re-run by you, counts quoted, reconciled against every log |
| `measure_rho_equality.py` | run **unedited**, full output quoted |
| `AUDIT.md` §5's claim | stated as held or not held, with the four pieces of evidence |
| README §6 acceptance table | every row has its measured value |

## 7. Stop conditions

- **A suite count does not reconcile across the logs.** Report it. Do not fix it.
- **A prompt's acceptance threshold has no measured value in its log.** That prompt is not ✅.
  Record it as such and report; do not take the measurement yourself to fill the gap — that would
  make this prompt the reviewer of its own evidence.
- **`AUDIT.md` §5's claim did not hold.** That is the campaign's central statement and the user
  must hear it plainly. Stop and report.
- **You want to touch a production file.** You may not, for any reason.

## 8. Deliverables

1. `prompts/background-solver-robustness/PROVENANCE.md`.
2. The two amendments to `prompts/tolerance-convergence/`.
3. `logs/06-provenance-and-close-out.md` per README §5.1, with §4's five verifications quoted in
   full.
4. Board row 06, item row (i), §5 close-out, and `docs/OPEN_ISSUES.md` brought to its final state
   for this campaign — count and date corrected.
5. One commit, README §5 rule 2.
