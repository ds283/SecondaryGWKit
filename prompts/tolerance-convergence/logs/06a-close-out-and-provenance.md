# Log 06a — the close-out and the provenance note

**Prompt:** `prompts/tolerance-convergence/06a-close-out-and-provenance.md`
**Commit:** *(this prompt's own commit)* — "Close out the tolerance and convergence campaign"
**Model:** Opus (Claude Sonnet 5 in this run)
**Date:** 2026-09-18
**Result:** **DONE.** `docs/TOLERANCE-CONVERGENCE.md` is the narrative close-out and
`docs/TOLERANCE-PROVENANCE.md` is the campaign's durable deliverable (README §1.2): one entry per
accuracy parameter in the pipeline, covering all **44 rows** of
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5.4, assembled entirely from the eleven
campaign logs and five campaign documents with **no re-measurement**. Eight parameters are recorded
as unestablished, in those words; `BESSEL_ORDER_CHECK_TOL` and the Bessel phase construction budget
are correctly *not* among them. This closes the campaign: board item **T12** is done, and every item
T1–T15 is now ✅.

## What shipped

- **`docs/TOLERANCE-CONVERGENCE.md`** (new). §1: why the campaign existed and how its own plan
  mis-stated its subject twice (README §0.1's "only one of six" corrected to two; eight keyed object
  types corrected to nine). §2: the eight targets — `wavenumber_exit_time` (changed),
  `GkNumericIntegration` (`unchanged`), `TkNumericIntegration` (changed), `BackgroundModel`
  (`unchanged`, three orders), both WKB sectors' $N_\rho$ (`unchanged`) plus
  `RESIDUAL_WKB_REGION_MARGIN` (`unchanged` under a different rule), `GkSource` (drop),
  `QuadSourceIntegral` (`unchanged`, read-only) — each with its outcome, the §6.1 rule that produced
  it, and a citation rather than a re-tabulation. §3: the three `unchanged` results read together as
  the campaign's central finding — the prior "the error is set by `rtol`" held only in the one
  50-object sector it was cleanly measured in, and every real-object-count sector answered "spend
  nothing". §4: the plan's self-corrections. §5: what the campaign built beyond the eight targets —
  the convergence facility, the grid-buildability fix, the schema replacement, the block
  regeneration. §6: what is still open, itemised against `docs/OPEN_ISSUES.md` §1.5. §7: suite
  counts and digests for the record.

- **`docs/TOLERANCE-PROVENANCE.md`** (new). One entry per parameter under the inventory's own seven
  group letters (A–G), each carrying README §1.2's five fields — value and what it keys; what
  measurement chose it, with the reference's drift and the source-grid generation where one applies;
  the competing floor; the cost at the setting chosen and one step either side, times the object
  count; and the campaign/prompt/log/date. A closing section reconciles the 44-row count against the
  document's own headings (some headings combine two or three inventory rows that already read "as
  above" or share a value; the reconciliation is explicit) and lists the eight parameters whose
  provenance cannot be established, in README §1.2's own words, plus the two that are argued rather
  than swept and are correctly not on that list.

- **`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md`**: header status line moved to `CLOSED`;
  the 06a board row filled in (`✅`, log link); item-table row **T12** filled in with the acceptance
  figures; a narrative paragraph inserted for prompt 06a alongside the other prompts' paragraphs; one
  new §3 issue, `[06a-readme-and-config-comments-state-the-superseded-quadrature-rtol-regime]`.

- **`docs/OPEN_ISSUES.md`**: the §1.5 intake paragraph corrected to record the campaign's closure;
  the new issue added to the §1.5 table; the header count moved from 81 to 82 and the date left at
  2026-09-18 (already current).

## Deviations from the prompt

**None that change scope.** Two points worth recording as choices:

- **IMPLEMENTATION CHOICE** — the prompt's §1 table gives the coverage checklist as "44 rows", and
  the inventory's own §5.4 groups several closely related constants into a single table row (for
  example, `DERIVATIVE_SPLINE_ORDER` and `STORED_SAMPLE_SPLINE_ORDER`, whose inventory entries read
  "as above" against each other almost verbatim). This document sometimes gives such a group one
  heading rather than one heading per inventory row, to avoid repeating an identical five-field
  entry three times. The "Coverage and the closing rule" section at the end of
  `TOLERANCE-PROVENANCE.md` states the reconciliation explicitly (11+5+6+5+6+6+5 = 44) so that a
  reader counting headings against the acceptance condition is not left to work it out.
- **IMPLEMENTATION CHOICE** — the prompt names two "standing notes that supersede statements still
  written elsewhere" (§4): note 26 (the `rtol`/`atol` regime inversion) and note 27 (`analytic_rad`
  is not a fixed oracle). The prompt's instruction to "leave a §3 issue behind for the rest" is
  written directly under note 26 and not repeated under note 27. On inspection, note 27 does not
  correct a statement anywhere else in the tree — it is a fact about `QuadSourceIntegral.py` that
  nothing had recorded before prompt 06 measured it, and `[06-analytic-rad-is-computed-at-the-callers-tolerance]`
  already exists on `qsi-phase-groups`' own board to carry it. So this prompt reflects note 27 in
  both documents (as the prompt's §4 lead sentence requires of "both documents") but did **not**
  open a second §3 issue for it, there being nothing left uncorrected to assign one to. If this
  reading is wrong, the fix is a one-line issue naming the same fact a second time on this board.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check` → `inventory.py: TOLERANCE-INVENTORY.md is up to date`. Nothing this prompt did touches the generated block or
  any file it reads.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` →
  `Ran 39 tests in 0.781s` / `OK`.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` →
  `Ran 521 tests` / `OK` (run in the background because of its wall-clock length; the result is
  quoted in full below). Both suite counts match the acceptance floor exactly (§6 items 6): 521 and
  39, neither having moved, since this prompt's diff contains no production file, no test file, and
  no file under `docs/tolerance-convergence/`.
- `git status --short` before committing: only `docs/OPEN_ISSUES.md`,
  `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` modified, and
  `docs/TOLERANCE-CONVERGENCE.md`, `docs/TOLERANCE-PROVENANCE.md`,
  `prompts/tolerance-convergence/logs/06a-close-out-and-provenance.md` added. `config/defaults.py`
  does not appear in the diff, confirmed both by its absence from `git status` and by not having
  opened it for editing at any point in this prompt.
- `black --check` was not run because there is no `.py` file in the diff (acceptance item 10 is
  satisfied vacuously).
- The three published source-grid digests (`3bef2c06`, `60a3205a`, `21ffc126`) are unchanged because
  nothing in this prompt's diff can move them; not independently re-measured, since doing so would
  be the re-measurement §5 of the prompt forbids.

## Observations not acted on

- **README §3.6a is now stale in three ways this prompt's §1 already names** (ten logs vs. eleven,
  four campaign documents vs. five, "39 plus six" vs. "44 already including six"). The prompt
  explicitly forbids correcting §3.6a (§5: "It does not correct README §6.2 or §3.6a… rewriting the
  charter is not [this prompt's job]"), so it stands uncorrected. A future reader of README §3.6a
  alone, rather than of this log or the close-out, would be misled by the stale counts; recorded here
  rather than fixed.
- **README §6.2's table itself is stale beyond the one row named in §4 of the prompt.** Its
  `wavenumber_exit_time` and `TkNumericIntegration` rows still show the "Target" column as an
  in-progress recommendation rather than the settled, shipped value, and its `GkSource` row says
  "—" where the settled answer is `drop`. None of this is wrong — the table's own header says "only
  the 'now' column can be filled in today" and the acceptance section afterwards records what was
  decided — but a reader who stops at §6.2 without reading §7's decisions gets a partial picture.
  Not corrected, for the same reason as above: rewriting README §6.2 is explicitly out of this
  prompt's scope.
- **`TOLERANCE-INVENTORY.md`'s own "Owned by" column for the four decoupled solver-reaching targets
  still reads "settled (D1, closed 2026-09-17)" rather than naming the six constants by name.** This
  is accurate but terser than `TOLERANCE-PROVENANCE.md`'s own entries for the same parameters. Not a
  defect — the inventory is a different document with a different job, and prompt 06a may not edit
  it — but a reader comparing the two side by side will notice the difference in granularity.

## State handed to the next prompt

**There is no next prompt in this campaign.** This section instead records what a reader of
`docs/TOLERANCE-PROVENANCE.md` still cannot learn from it, and which campaign owns each gap, per the
prompt's own §8 instruction.

- **Whether `OneLoopIntegral` should be decoupled, dropped, or left as it is** — the ninth keyed
  object type, computing nothing, opened by prompt 02 and never assigned. The provenance note lists
  its two columns nowhere, because `TOLERANCE-INVENTORY.md` §5.4 does not carry them (`OneLoopIntegral`
  is outside the inventory's own 44-row table, which is derived from the eight/nine targets' own
  columns and the hard-coded literals, not from every foreign key in the schema). A reader wanting
  its provenance must go to `TOLERANCE-INVENTORY.md` §2.1 directly. **Owner: the user, per prompt 02
  §8's stop condition** — unchanged by this prompt.
- **Whether the source-grid density criterion should apply over a horizon-limited band** —
  `[01-density-criterion-imposed-outside-the-wkb-region]`, measured three times over the campaign
  (prompt 01's first estimate, prompt 02a's narrower and then corrected reading, prompt 04b's final
  figure) and still a recommendation rather than a decision. **Owner: unassigned**, candidate for
  board item T7 or the source grid's own campaign, `qcd-background-audit`.
- **Whether `QuadSourceIntegral`'s decoupled pair should itself be retuned** rather than merely
  measured — README §7 D4, explicitly not reopened by this campaign's `unchanged` finding.
  **Owner: `prompts/levin-refactor` / `prompts/qsi-phase-groups`**, per README §0.4.
- **What the numeric $G_k$ consumer spline should do about its own dominant error near the
  hand-over** — `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`, the floor
  that makes `GkNumericIntegration` `unchanged`. **Owner: the hand-over campaign**
  (`docs/OPEN_ISSUES.md` §1.1), assigned there on the user's 2026-09-17 D1 acceptance.
- **Three stale comments naming the superseded `atol=1e-25` quadrature regime** — README §6.2's last
  row, `config/defaults.py:163`, `ComputeTargets/QuadSourceIntegral.py:1550` — none of which this
  prompt could edit under its own file grant. **Owner: whoever next touches any of the three files**,
  per the new §3 issue this prompt opened.

`docs/TOLERANCE-PROVENANCE.md` itself is the record of what *is* now known; this section is
deliberately the residue, not a summary of the note.
