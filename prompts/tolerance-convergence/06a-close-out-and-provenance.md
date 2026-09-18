# Prompt 06a — the close-out and the provenance note

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T12**, and with it the campaign. **Depends on:** prompt 06 (`2033cfc`).
Every other prompt has landed.
**Recommended model:** **Opus** — the work is assembly, not measurement, and the failure mode is a
plausible sentence where the record has none.

**This is the deliverable the campaign exists for** (README §1.2): `docs/TOLERANCE-PROVENANCE.md`,
one entry per accuracy parameter in the pipeline, so that no constant is left for the next reader to
change by guesswork. It is written last because only now is every number in it measured.

**It changes no production code, no schema and no number.** §5 rule 8's parameter freeze applies in
full: `config/defaults.py` is byte-identical at the end of this prompt. It writes two documents and
the campaign's bookkeeping, and nothing else.

**Files you may create or touch:**
`docs/TOLERANCE-CONVERGENCE.md` — new, the close-out document;
`docs/TOLERANCE-PROVENANCE.md` — new, the provenance note;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** any production module; `config/defaults.py`; `main.py`; any factory, schema or
test; `docs/tolerance-convergence/*` — the five campaign documents and `inventory.py` are **read,
never edited** (§5 rule 7, and the inventory's generated block is under a `--check` gate);
`docs/spec/`; and anything belonging to `prompts/radiation-oracle`, which is a different campaign
and lands after this one.

**Read first:** README §1.2 in **full** — it is the specification and this prompt adds nothing to
it; §1, §2 (b), (c), (e), (f), (i); §5 rules 5, 6, 7 and 8; §6.1 the target rule; §6.2's table;
§7 **D11**; board item **T12** and the board's **§5 standing notes, all of them**;
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5.4 and §6;
and the campaign's **eleven** logs in `logs/`, whose "State handed to the next prompt" sections are
your working set.

---

## 1. What you are assembling from, and it is more than the plan says

README §3.6a was written before prompt 06 landed and forecasts the inputs. **The tree has more than
it forecasts, and three of its numbers are now wrong.** Take these, not §3.6a's:

| §3.6a says | The tree has | Why |
|---|---|---|
| "the campaign's **ten** logs" | **eleven** — `logs/` lists 01, 02, 02a, 03, 03a, 04, 04b, 05, 05a, 05b, 06 | 06 added its own |
| "the **four** campaign documents" | **five** — `GK-NUMERIC-SWEEP.md`, `ORDER-AUDIT.md`, `TK-NUMERIC-AND-EXIT-TIME.md`, `QUADSOURCE-READONLY.md`, `TOLERANCE-INVENTORY.md` | 06 added `QUADSOURCE-READONLY.md` |
| "§5.4's **39 rows** … **plus** the six constants prompt 05a shipped" | **44 rows**, and **the six are already among them** | 06 regenerated the inventory; adding them again double-counts |

**The 44 rows are the coverage checklist.** Work down them. A row you cannot account for is a gap in
this prompt, not a row to drop.

## 2. The two documents

### 2.1 `docs/TOLERANCE-CONVERGENCE.md` — the close-out

The eight targets, the parameter each ended at, the evidence, and the floor each is now limited by.
It is the campaign's narrative record and it is **shorter than the sum of its inputs**: each target
gets its outcome, the §6.1 rule that produced it, and a citation to the document holding the tables.
Do not re-tabulate what the five campaign documents already hold.

**Three of the campaign's results are `unchanged` and that is the campaign's main finding**, not an
absence of one: prompt 03's consumer spline dominating by ×631–×37,700, prompt 04's accumulation
floor by ×48.4–×3.03e5, prompt 06's representation floor by ×3.46e+04–×2.15e+11. Two parameters
moved — `TkNumericIntegration`'s `rtol` to 3e-11 and `wavenumber_exit_time`'s to 1e-9, both accepted
2026-09-17. Say so plainly and in that proportion.

### 2.2 `docs/TOLERANCE-PROVENANCE.md` — the note

**README §1.2 is the specification. Follow it exactly**: for every parameter, the five fields — the
value and what it keys; what measurement chose it, with the reference's own drift and the grid
generation; the competing floor; the cost in evaluations at the setting and one step either side,
times the object count of the sector; and the campaign, prompt, log and date.

It covers the parameters this campaign does **not** set as well as those it does — §1.2 names the
inherited ones explicitly and the point is that *no* accuracy parameter is unexplained when the
campaign closes.

**It is a summary with citations, not a second copy of the measurements.** Each entry points at the
document holding the tables. `config/defaults.py`'s comments stay the primary record at the point of
use.

## 3. The one acceptance condition diligence cannot satisfy

§1.2's closing rule: **where the provenance of a constant cannot be established from the record, the
note says so in those words.** An invented justification is worse than the admission, and it is the
one failure this prompt cannot recover from — a fabricated provenance is indistinguishable from a
real one to every later reader.

**You assemble; you do not investigate.** If the record does not contain a choice, the answer is
that the record does not contain a choice. Do not read a code comment's *reasoning* as a measurement,
and do not reconstruct one from a constant's magnitude.

Prompt 06's log already determined this for the parameters it touched, and its verdicts are inputs,
not suggestions:

- **Cannot be established** — `DEFAULT_LEVIN_MAX_DEPTH = 20` (the comment at
  `AdaptiveLevin/levin_quadrature.py:153` gives only the geometric reading "1/2^20 is roughly 1E-6",
  which is **not** a measurement) and `limit = 100` (`Quadrature/simple_quadrature.py:92`, no comment
  and no campaign document).
- **Can be established, and must not be mis-filed as unestablished** — `BESSEL_ORDER_CHECK_TOL = 1e-3`
  is *argued* rather than swept, in the comment at `QuadSourceIntegral.py:79-92`: nine orders above
  the reconstruction floor and two below the smallest defect it must catch. That is a provenance.
- **At risk, and the ones to check most carefully** — `DEFAULT_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE`,
  `DEFAULT_HEXIT_TOLERANCE`, `DEFAULT_LEVIN_THRESHOLD` and `find_phase_extremum`'s pair. Inventory
  §5.4 already marks these "never chosen": for each the record holds a *use* and no *choice*.

## 4. What the record now contradicts, and must not be copied forward

Prompt 06 left two standing notes that supersede statements still written elsewhere in the tree.
**Both documents you write must reflect the superseding version**, and the close-out says which was
superseded and when — additively, per §5 rule 7.

- **Board standing note 26.** `source-remediation` log 12's "`rtol` does not bind — `1e-8 → 1e-11`
  bit-identical on 159 items" was measured at `atol = 1e-25`. At `1e-32` the regime has inverted:
  `atol` is inert over twenty-eight decades and `rtol` is the only lever, moving `total` by ×274 over
  those same three decades. **README §6.2's last row still states the superseded version**, as do
  `config/defaults.py:163` and `QuadSourceIntegral.py:1550`. You may edit neither file and you may not
  rewrite §6.2; record the supersession in your documents and leave a §3 issue behind for the rest.
- **Board standing note 27.** `analytic_rad` is computed at the caller's own `atol`/`rtol`, so a
  stored value depends on its row's `atol_serial` and `rtol_serial` and is not a fixed oracle.

## 5. What this prompt does not do

- **It does not re-measure anything.** If a number you need is not in the record, that is §3's
  answer, not a reason to run a sweep.
- **It does not edit the five campaign documents or `inventory.py`.** They are §5 rule 7 verification
  documents and the inventory is under a `--check` gate.
- **It does not correct README §6.2 or §3.6a.** Both are stale in ways §1 and §4 name. Recording the
  staleness is this prompt's job; rewriting the charter is not (§5 rule 7).
- **It does not touch `config/defaults.py`**, however tempting it is to fix the comment at `:163`.
  Open an issue.

## 6. Acceptance

1. `docs/TOLERANCE-PROVENANCE.md` exists and carries README §1.2's **five fields** for every
   parameter, covering **all 44 rows** of inventory §5.4 and the inherited constants §1.2 names.
2. Every parameter whose provenance cannot be established says so **in §1.2's own words**, and
   `BESSEL_ORDER_CHECK_TOL` is **not** among them.
3. `docs/TOLERANCE-CONVERGENCE.md` exists and reports the eight targets with the §6.1 rule that
   produced each outcome, including the three `unchanged` results with their dominating factors.
4. §4's two supersessions are reflected, with what they supersede named.
5. No production file, no `docs/tolerance-convergence/*` file and no `docs/spec/` file is in the
   diff; `config/defaults.py` byte-identical at blob `76bab78…`.
6. `ComputeTargets` **must not fall below 521**; `CosmologyModels` **39**. Both OK.
7. The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`.
8. `PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check` still passes —
   you changed nothing it reads, and the check is cheap.
9. Board row 06a, item **T12**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date corrected —
   all in the **same commit**.
10. `black --check` clean on every `.py` in the diff (there should be none).

## 7. Stop conditions

- **A parameter's provenance cannot be established and you are tempted to supply one.** Stop and
  say which. This is §3 and it is the prompt's whole point.
- **Inventory §5.4 does not read 44 rows**, or a row names a constant no log accounts for. That is a
  report about the tree.
- **You need to edit a production file, `config/defaults.py`, a campaign document or `docs/spec/`.**
- **A campaign document contradicts a log** on a number you must quote. Report both and which you
  used; do not silently pick.
- **`ComputeTargets` falls at all.**

## 8. The log

`logs/06a-close-out-and-provenance.md`, on the campaign's template. Beyond it:

- **The coverage walk**: all 44 rows, and for each whether its provenance was established,
  established-by-argument, or **cannot be established**. This list is the campaign's real output.
- **What you could not establish**, and what in the record you looked at before concluding it.
- **What §3.6a's stale counts cost you**, if anything — the next campaign planning against a
  forecast rather than a tree should be able to read what that is worth.
- **"State handed to the next prompt"** — there is no next prompt in this campaign. Say what a
  reader of `docs/TOLERANCE-PROVENANCE.md` still cannot learn from it, and which campaign owns each
  gap.
