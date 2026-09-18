# Log 04b — Regenerate the convergence block, and repair the two tests that read it

**Prompt:** `prompts/tolerance-convergence/04b-regenerate-the-convergence-block.md`
**Commit:** *(this prompt's own commit)* — "Land the regenerated convergence block"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE.** The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` is
**regenerated and landed** — `generated` 2026-09-10 → **2026-09-18**, `campaign`
`prompts/GkTk-remedial (prompt 02)` → **`prompts/tolerance-convergence (prompt 04, board item
T7)`**, `recommended_scheme` `branch+knots` → **`branch`**, all four orders back **4 4 4 4**, and
`branch+knots` retained as a populated **control**. Only the `convergence` key differs from
`HEAD~1`. The two assertions that then failed are **rebuilt rather than loosened**: each bounds the
production quantity absolutely, at **1.0e-14**, against the double-precision accumulation floor
`ORDER-AUDIT.md` §§3.1 and 5 measure (2.16e-16 and 3.30e-16) and **not** against the fixture's own
agreement with a converged reference; `QCD_FLOOR_FACTOR` is gone from both modules.
**`QCD_BREAK_POINT_ALIGNMENT_TOL` goes 1.5e-04 → 3.0e-14**, ten orders, against a worst measured
offset of 1.421085e-14 — four ulp of $u$ — exactly as four comment blocks of another campaign
predicted. `ComputeTargets` **491, OK**; `CosmologyModels` **39, OK**;
`test_convergence_reference` **32, OK**; the three block readers **41, OK** with
`test_phase_residual.py` **unedited**. The three published source-grid digests are unmoved.

**Nothing of prompt 04's measurement is re-derived.** Every figure the landed block carries
reproduces §9's dry run to the digits that document prints, which is the check that what landed is
what prompt 04 measured.

`[01-convergence-block-has-a-separate-generator]` and
`[04-convergence-floor-used-as-a-test-threshold]` are both **closed**. Board item **T14**.

---

## What shipped

Four files under `ComputeTargets/tests/` and `docs/`, plus this campaign's log and board, the
`qcd-background-audit` board entry and `docs/OPEN_ISSUES.md`. **No production module is touched**:
`git diff HEAD~1 HEAD -- . ':!prompts' ':!docs' ':!ComputeTargets/tests'` is empty.

### `ComputeTargets/tests/wkb_reference_data.json` — the `convergence` key, written by its generator

Written by `docs/gktk-remedial/residual_convergence.py`, run with **no flags** so that `write_json`
replaces that one key, in 538.4 s (Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2). **Not hand-edited at
any point**, and the check the prompt names is satisfied: `generated` and `campaign` both moved.

| Field | before | after |
|---|---|---|
| `generated` | `2026-09-10` | **`2026-09-18`** |
| `campaign` | `prompts/GkTk-remedial (prompt 02)` | **`prompts/tolerance-convergence (prompt 04, board item T7)`** |
| `schema_version` | 1 | **2** |
| $N_\tau$ / $N_{c_s\tau}$ / $N_F$ / $N_\rho$ | 4 / 4 / 4 / 4 | **4 / 4 / 4 / 4** |
| `recommended_scheme` | `branch+knots` | **`branch`** |
| `rho_adaptive_fallback_required` | `false` | `false` |
| `rho_fixed_order_works_without_subdivision` | `true` | **`false`** (nothing reads it — observation 1) |
| `schemes` | `plain`, `branch`, `branch+knots` | unchanged; `branch+knots` is `control_scheme`, still populated under `schemes` and under `models.QCDModel` |
| QCD `branch+knots` `tau` `json_vs_reference_max_rel` | 1.878541e-14 | **3.223619e-16** |
| QCD `branch+knots` `cs_tau` `json_vs_reference_max_rel` | 1.886653e-14 | **4.354138e-16** |
| QCD `branch+knots` `friction` `json_vs_reference_max_rel` | 0 | **2.013295e-16** |
| `runtime_seconds` | 328.3 | 538.4 |

**Only `convergence` differs.** Checked on the parsed JSON, key by key, against the file at
`HEAD~1`: `schema_version`, `generated`, `generator`, `campaign`, `environment`, `schema`,
`k_values`, `k_keys`, `rho_anchor_efolds_subh`, `models` and `baselines` at the **top level** are
equal; `git diff -U0`'s first hunk is at line 1521 and `"convergence"` opens at line 1520.

### `ComputeTargets/tests/test_background_tau.py` — one threshold and one tolerance

`QCD_FLOOR_FACTOR = 3.0` is **removed**; `QCD_NODE_REL_TOL = 1.0e-14` replaces it, and
`test_qcd_nodes_against_adaptive_reference` asserts both its figures against it.
`QCD_BREAK_POINT_ALIGNMENT_TOL` goes **1.5e-04 → 3.0e-14**. Both comment blocks keep their history
in full and gain the resolution underneath it, with the arithmetic in the file.

### `ComputeTargets/tests/test_background_cs_tau_friction.py` — one threshold (README §7 D8)

`QCD_FLOOR_FACTOR = 3.0` removed, `QCD_CS_TAU_REL_TOL = 1.0e-14` in its place, and
`test_qcd_checkpoints` asserts against it. **Nothing else in that module is touched** beyond the
docstring bullet that described the old construction (deviation 1).

### `docs/tolerance-convergence/ORDER-AUDIT.md` — a new §12, additive

What the block says now, what the two thresholds became and the arithmetic that chose them, the
alignment tolerance with its three per-break offsets, and the verification. §§1–11 are untouched
and §1's account of the stop stands. The header blockquote gains a four-line forward pointer
(deviation 5).

---

## Deviations from the prompt

### 1. The two modules' docstring bullets describing the old threshold were rewritten — `IMPLEMENTATION CHOICE`

Each module's docstring carries a "Floors" bullet stating, in terms, that *the threshold is three
times* the JSON's own recorded agreement — `test_background_tau.py:14-19` and
`test_background_cs_tau_friction.py:28-33` as they now read. The prompt grants "the threshold of §3" in each module
and, for `test_background_cs_tau_friction.py`, "nothing else in that module". Leaving those two
bullets would have left a **false statement** about the assertion this prompt rewrote, so each is
replaced by a description of the new construction, naming
`[04-convergence-floor-used-as-a-test-threshold]`. No other line of either docstring is touched and
no figure elsewhere in them is changed.

### 2. `QCD_FLOOR_FACTOR` is removed rather than kept in a new role — `IMPLEMENTATION CHOICE`

Prompt §3 allows either, and requires that a constant not be left meaning something it no longer
means. With the assertion an absolute bound there is nothing for a factor to multiply, so the name
is gone from both modules and each new constant carries the arithmetic that chose it. Verified by
`grep`: no reference to `QCD_FLOOR_FACTOR` survives anywhere in the tree.

### 3. Both tests still **read and print** the reference's own floor — `IMPLEMENTATION CHOICE`

The assertion no longer uses it, but each test still reads
`convergence.models.QCDModel["branch+knots"][q]["json_vs_reference_max_rel"]` and prints it beside
the bound, labelled *not asserted against*. Two reasons: the two numbers are most informative side
by side, which is the whole content of the issue being closed; and prompt §2.2 requires the
`branch+knots` key to stay populated because test modules index it, which is a weaker claim if
nothing reads it any more.

### 4. The block's `campaign` string names prompt **04**, not 04b — `STRUCTURALLY REQUIRED`

`residual_convergence.py` hard-codes `prompts/tolerance-convergence (prompt 04, board item T7)`.
Hand-editing the JSON is forbidden (prompt §2.1) and the generator may be touched "only if the run
needs it"; the run did not need it. The string is also **right**: prompt 04 took the measurement
and 04b landed it unchanged. Recorded here and in `ORDER-AUDIT.md` §12.1 so that a reader tracing
the block to a log is not sent to the wrong one.

### 5. `ORDER-AUDIT.md`'s header blockquote gains an addendum — `IMPLEMENTATION CHOICE`

Prompt §5 acceptance 7 requires §§1–11 untouched and §1's description of the stop left as written;
both hold. The blockquote above §0 says "the fixture, `test_background_tau.py` and every production
module are exactly as this prompt found them", which is no longer true of the tree. Rather than
rewrite it (README §5 rule 7), a four-line addendum below it points at §12 and says the paragraph
is left as prompt 04 wrote it.

### 6. `QCD_BREAK_POINT_ALIGNMENT_TOL` is 3.0e-14, ×2.11 above the worst offset — `IMPLEMENTATION CHOICE`

Prompt §4 asks for "just above 1.421085e-14" with the arithmetic that picks the headroom. The three
offsets are exactly 1, 2 and 4 ulp of their own $u$ ($\mathrm{ulp}(27.485) = 3.552714$e-15), so the
constant is set at 8.4 ulp — a factor of 2.11. The argument for so little headroom is in the file:
every real instance of the failure this assertion catches showed at 1e-06 to 1e-04, ten orders
above, so nothing measurable is bought by going looser and the assertion is what is lost.

---

## Verification performed

**Both suites, on the tree as committed.**

| | baseline (`19bebcb`) | after |
|---|---|---|
| `ComputeTargets` | 491, OK (184.9 s) | **491, OK** (183.8 s) |
| `CosmologyModels` | 39, OK | **39, OK** |
| `ComputeTargets.tests.test_convergence_reference` | 32, OK | **32, OK** |
| `test_background_tau` + `test_background_cs_tau_friction` + `test_phase_residual` | 41, OK (old block) | **41, OK** (new block) |

No test is added or removed. **The failure was reproduced before it was repaired**: with the block
written and the two modules untouched, those 41 tests gave **2 failures** —
`test_qcd_nodes_against_adaptive_reference` at 2.254220593813745e-15 against 9.670856463e-16, and
`test_qcd_checkpoints` at 2.2119104448437696e-15 against 1.3062415232e-15. Both numerators are
prompt 04's figures to every digit, and both are `qcd-background-audit` prompt 06's.

**`black --check`** clean on both test modules (nothing to reformat).

**The three published source-grid digests are unmoved**, re-checked through `order_audit`'s own
precondition rather than asserted: `RadiationModel` 2,306 / `3bef2c06` at
`z_init = 44523947729.77296`, `LambdaCDMModel` 1,778 / `60a3205a` at `2.0636395964161516e+16`,
`QCDModel` 2,034 / `21ffc126` at `3.30033444460513e+16`.

**The alignment offsets were measured, not taken from §9.4**: the three declared crossings against
the landed block's branch boundaries are 3.552714e-15 (`T_LO`), 7.105427e-15 (`EOS_T_LO`) and
1.4210854715202004e-14 (`T_120_MEV`), i.e. 1, 2 and 4 ulp of $u$ — reproducing §9.4 exactly.

**`test_phase_residual.py` is unedited and green.** It pins `RHO_GAUSS_ORDER == decision.N_rho`
and `decision.rho_adaptive_fallback_required`, and the landed block still says `4` and `false`.

**No stop condition fired.** All four orders came back 4, `recommended_scheme` came back `branch`,
no top-level key other than `convergence` moved, no test outside the three named modules changed
its verdict, no source-grid digest moved, and nothing outside the prompt's file list was edited.
The 16 `NOT CONVERGED` cells `ORDER-AUDIT.md` §9.3 records are present and unchanged in kind.

---

## Observations not acted on

1. **`rho_fixed_order_works_without_subdivision` is now `false`.** Prompt §2.2 says to note it and
   not act. Nothing in the tree reads the field; under `plain` no fixed order up to 16 reaches the
   1e-7 rad target on QCD any more, which strengthens the case for `branch` rather than weakening
   it (log 04, observation 2, measured there).

2. **`test_background_tau.test_qcd_break_points`'s print string is now stale, and this commit is
   what made it stale.** It prints *"the convergence block still records 2411 knots, from before
   prompt 06"*; the regenerated block records the **current** count, which is also 2411, so the
   number is right and the characterisation is wrong. Nothing asserts on it — it is one f-string in
   a `print` — and it is neither of the two constants this prompt's grant covers in that module, so
   it is left. Opened as `[04b-break-point-print-string-describes-a-superseded-block]`.

3. **`ORDER-AUDIT.md` §§2–11 are `order_audit.py`'s stdout and §12 is not.** Anyone re-running that
   script and pasting its output must paste it *above* §12. Said in §12's own first paragraph.

4. **The generator's `decision.knots_control` now records that the demoted control is worse, not
   merely redundant**: on QCD's $\tau$ floor `branch` reaches 1.658e-16 against `branch+knots`'s
   3.224e-16 (ratio 0.51), and on `friction` the control reaches exactly 0, which is what
   `decision.zero_best_floor_primitives` exists to handle (log 04, deviation 3). Prompt 04 already
   drew the conclusion — splitting at the knots buys nothing — and this is the landed block saying
   the same thing.

---

## State handed to the next prompt

**Prompt 05 is next, and three things have changed under it.**

1. **The `convergence` block is current.** It was generated 2026-09-18 against the corrected $T(z)$
   representation and the 3-point break set, it recommends a scheme production can execute
   (`branch`), and it records $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$ — which is the evidence
   every `*_GAUSS_ORDER` comment in the tree cites, and it is no longer two representations out of
   date. **Prompt 05 does not re-run the generator** and has no reason to touch the fixture.

2. **`QCD_FLOOR_FACTOR` no longer exists**, in either module. The two QCD checkpoint assertions are
   `QCD_NODE_REL_TOL = 1.0e-14` (`test_background_tau.py`) and `QCD_CS_TAU_REL_TOL = 1.0e-14`
   (`test_background_cs_tau_friction.py`), each an absolute bound on the production quantity with
   its measured floor named beneath it. A prompt that changes `TAU_GAUSS_ORDER` or
   `CS_TAU_GAUSS_ORDER` — D3's schema work does **not**, it only puts the existing values in the
   key — would move the quantity those bounds hold, and the bounds are stated so that such a move
   is visible rather than absorbed.

3. **`QCD_BREAK_POINT_ALIGNMENT_TOL = 3.0e-14`**, and it is now a genuine assertion about the
   agreement of two independent solves of the same crossing rather than a record of a stale
   fixture. It will fail if the $T(z)$ representation moves again without the block being
   regenerated, which is the behaviour the constant was always supposed to have.

**`[01-convergence-block-has-a-separate-generator]` is closed** — first assigned to
`qcd-background-audit` prompt 08, declined on scope by two prompts of that campaign and by prompt
04 of this one, and closed here. **`[04-convergence-floor-used-as-a-test-threshold]` is closed** by
the repair the user chose. Prompt 04's note that "prompt 05 also inherits the stop" (log 04, State
handed) is **discharged**: the sequence it fixed — thresholds repaired in the same commit as the
block, alignment tolerance with them — happened here, and prompt 05 inherits none of it.

**Provenance (README §1.2).** This prompt ships **no parameter**: the three constants it moves are
test-tree bounds, not accuracy parameters of the pipeline, and `config/defaults.py` and `main.py`
are untouched. The five provenance fields for $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ and
`RESIDUAL_WKB_REGION_MARGIN` are in **log 04's** "State handed to the next prompt" and are **not
restated here**, as prompt §7 requires. What this commit adds to them is that the evidence they
cite is now *in the tree* rather than in a dry run: `wkb_reference_data.json`'s `convergence` block,
generated 2026-09-18 by `prompts/tolerance-convergence (prompt 04, board item T7)`.
