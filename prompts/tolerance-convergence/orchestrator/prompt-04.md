# Orchestrator — prompt 04, the order-governed targets

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../04-order-governed-targets.md`](../04-order-governed-targets.md)
**Model:** Opus. **Production code changed:** none. **Test-tree code changed:**
`ComputeTargets/tests/wkb_reference_data.json` and `ComputeTargets/tests/test_background_tau.py`,
**and only those**, under README §7 **D5**. `docs/gktk-remedial/residual_convergence.py` is the third
carve-out file and is not under `ComputeTargets/`.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This is the first prompt in the campaign that writes a fixture other prompts' tests read**, and
that — not the measurement — is what the review is for. Prompts 01 through 03a were additive or
document-only; this one rewrites `wkb_reference_data.json`'s `convergence` block, which **three**
test modules read and only **one** of which the agent may edit.

**Precondition:** prompts 01, 02, 02a, 03 and 03a have landed and **D1 is closed** (`8b8809c`). D1
closing does not gate this prompt technically — no tolerance here is one D1 settled — but it means
prompt 05 is now waiting on **D3**, which is this prompt's recommendation, so a weak §6 answer
blocks the campaign rather than merely disappointing it.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the three baselines. At `8b8809c` they are **491**
(`ComputeTargets`), **39** (`CosmologyModels`) and **32** (`test_convergence_reference`).
`ComputeTargets` must not fall below **452**, the campaign floor, and should be at 491 here.

**Take a fourth baseline this time**, because a fixture rewrite can move a count without touching a
test file:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_tau ComputeTargets.tests.test_background_cs_tau_friction ComputeTargets.tests.test_phase_residual 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

Those are the three readers of the `convergence` block (prompt §2.2). Record the count and the
verdict. If any of them is not green **before** the dispatch, stop — the prompt's §11 stop
conditions are written assuming they start green, and an agent that inherits a red test will spend
its run on someone else's defect.

**Record the block's identity too**, so that you can tell a regeneration from an edit:

```bash
./venv/bin/python -c "import json;d=json.load(open('ComputeTargets/tests/wkb_reference_data.json'));c=d['convergence'];print(c['generated'],c['campaign'],c['decision']['recommended_scheme'],c['decision']['N_tau'],c['decision']['N_cs_tau'],c['decision']['N_F'],c['decision']['N_rho'])"
```

At `8b8809c` that reads `2026-09-10`, `prompts/GkTk-remedial (prompt 02)`, `branch+knots`, and
`4 4 4 4`.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `04-order-governed-targets`, model
**Opus**. **The template's standing parameter-freeze sentence applies unchanged** — the four orders
and `RESIDUAL_WKB_REGION_MARGIN` live in `ComputeTargets/BackgroundModel.py` and
`ComputeTargets/phase_residual.py`, which are production modules, so rule 8 covers them without
amendment. **D5 is a carve-out for three named files and the template does not mention it**; the
prompt's own header does, which is where it belongs. Add this sentence, and no more:

> **README §7 D5 gives you three files outside the usual boundary — `residual_convergence.py`,
> `wkb_reference_data.json` and `test_background_tau.py` — and exactly three.** Your prompt §2
> explains why the third one is not enough and what to do when it is not. Do not widen it yourself.

**Do not tell the agent what you expect the orders to be.** The old block says 4, the production
constants say 4, and the likeliest outcome is `unchanged` four times over — which is exactly why
saying so would hand the agent its answer. Prompt §7 rule 4 already tells it that `unchanged` is a
result; that is as much as it may hear.

**Do not tell it what you expect the regenerated floors to do either.** The review below turns on
whether it noticed the floor-tightening hazard *by measurement*; an orchestrator who warns it that
`test_background_cs_tau_friction` is likely to fail has pre-empted the one check that distinguishes
a careful run from a lucky one.

## 3. The review — seven checks

1. **Did it look before it wrote?**

   Prompt §2.4 requires the four recommended orders to be determined **before** the JSON is written,
   and the fixture left alone if any is not 4. Read the log for the order of operations, not just
   the outcome. If the block was written and a recommended order is not 4, the tree is red and the
   agent has broken its own stop condition — that is a **stop**, and do not let a green suite
   reassure you, because `test_phase_residual` pins `RHO_GAUSS_ORDER == decision.N_rho` and would
   have caught only $N_\rho$.

   ```bash
   ./venv/bin/python -c "import json;c=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))['convergence'];print(c['generated'],c['campaign'],c['decision']['recommended_scheme'],c['decision']['N_tau'],c['decision']['N_cs_tau'],c['decision']['N_F'],c['decision']['N_rho'])"
   ```

   Compare with §1's reading. `generated` and `campaign` must have moved if the block was written at
   all; if they did not, the agent hand-edited the JSON instead of regenerating it, which prompt
   §2.3 forbids and which no other check will catch.

2. **Only the `convergence` key.**

   ```bash
   ./venv/bin/python -c "
   import json,subprocess
   old=json.loads(subprocess.run(['git','show','HEAD~1:ComputeTargets/tests/wkb_reference_data.json'],capture_output=True,text=True).stdout)
   new=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))
   print(sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k)))"
   ```

   Must print `['convergence']`, or `['convergence', 'generated']` if the top-level stamp moved with
   it, or `[]` if §2.4 applied. **Anything else is a stop**: `models`, `baselines`, `k_values`,
   `k_keys` and `rho_anchor_efolds_subh` are other campaigns' evidence.

3. **The `branch+knots` key survived.**

   ```bash
   ./venv/bin/python -c "import json;c=json.load(open('ComputeTargets/tests/wkb_reference_data.json'))['convergence'];print(c['schemes']);print([m for m in c['models']],[list(c['models'][m]) for m in c['models']])"
   ```

   Two test modules index that key by name and one of them is outside the carve-out (prompt §2.2
   (ii)). The key must still be there and still populated. **A recommendation of `branch+knots` is
   also wrong** — production cannot perform that split, `integration_break_points` having stopped
   returning knots at `qcd-background-audit` prompt 07 — so check `recommended_scheme` is `plain` or
   `branch` and that the document says what the knot split still buys.

4. **Is $N_\rho$ measured at fifty wavenumbers, or at prompt 02's three?**

   ```bash
   grep -n "1e5\|1e7\|3e8\|K_VALUES\|fifty\|50" docs/tolerance-convergence/order_audit.py | head -30
   ```

   The whole question of prompt §8 item 2 is whether `GkTk-remedial` prompt 02 was right at
   $10^5$, $10^7$ and $3\times10^8$ and lucky at the other forty-seven. A sweep that inherits those
   three wavenumbers has not asked it. Check the $\rho$ tables are per-$k$ and that $\tau$,
   $c_s\tau$ and $F$ are **not** — they are $k$-independent and fifty identical rows would be a sign
   the agent did not understand what it was measuring.

5. **The radiation oracle, and $\rho_G \equiv 0$ in particular.**

   Four of the five integrands have a closed form on `RadiationModel` and `rho_G` is identically
   zero there, which makes that column a pure quadrature-error measurement with no reference drift
   in it at all. If the document measures $N_\rho$ only by self-convergence, it has left the best
   evidence in the prompt on the table. Check too that the two models without an oracle carry
   `reference_drift` beside every figure and that unresolved cells are marked rather than quietly
   included (README §5 rule 5).

6. **The alignment tolerance, and the non-monotone column.**

   `QCD_BREAK_POINT_ALIGNMENT_TOL` must come back to **1.4e-05** with the three per-break figures
   behind it, or the log must say in rule-6-like words that it will not and what binds it. **A
   constant chosen to sit comfortably above whatever was measured is the defect this prompt exists
   to undo**, so check the arithmetic that picked it, not just the number.

   Check also that a non-monotone order column, if there is one, is reported as such. The old block
   has one — `plain` `tau_errors_rel` goes 3.44e-07 at order 2, 4.46e-08 at 12, back up to 1.03e-07
   at 16 — and an agent that reports "the first order that clears" from a curve that later rises has
   made prompt 03a's mistake without prompt 03a's caveat.

7. **The suites, the diff and the bookkeeping.**

   ```bash
   git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'
   ```

   Must list **at most** `ComputeTargets/tests/wkb_reference_data.json` and
   `ComputeTargets/tests/test_background_tau.py`. Re-run all three suites and the three block
   readers of §1. `ComputeTargets` **must not fall below 452** and should read 491 or more;
   `CosmologyModels` must read **39**; `test_convergence_reference` **32**. Board row 04, item **T7**,
   the `qcd-background-audit` board entry for `[01-convergence-block-has-a-separate-generator]`, and
   `docs/OPEN_ISSUES.md` in the **same commit** with its count and date corrected. `black --check`
   clean on every `.py` in the diff.

   `[01-convergence-block-has-a-separate-generator]` and
   `[01-density-criterion-imposed-outside-the-wkb-region]` are the two issues this prompt can
   legitimately close or narrow. For the first, check the claim is against the regenerated block and
   that the `qcd-background-audit` board — not only this campaign's — records the close. For the
   second, check that what is claimed is a **measurement of the band against the region**, since
   that issue has been carrying a single 69 % figure as its only evidence and prompt 02a's census was
   withdrawn from it once already.

## 4. What a good outcome looks like

- A regenerated `convergence` block that reproduces or supersedes every figure the 2026-09-10 one
  carried, **with the reference drift and grid generation the old one lacked**, written by the
  generator and touching no other key.
- An answer to "are the orders converged at 4?" that is fifty wavenumbers wide for $N_\rho$ and says
  which of the two 2026-09-10 causes — the corrected `T(z)`, the knots leaving the contract — moved
  what.
- `QCD_BREAK_POINT_ALIGNMENT_TOL` back at 1.4e-05, or a plain statement of what stops it.
- A `RESIDUAL_WKB_REGION_MARGIN` answer that is a measurement or rule 6's words, and with it the
  first real measurement of how much of `residual_node_range`'s band lies outside the region the
  density criterion protects.
- A **D3 recommendation the user can decide from** — three options, costed in tables and rows, with
  the reader breakage named.

**What a good outcome does *not* look like:** four `unchanged`s asserted from the fact that the old
block and the production constants agree. They agree because the constants were set from the block;
that is one source, not two, and re-deriving it is the entire point of the prompt.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **`test_background_cs_tau_friction.test_qcd_checkpoints` fails on a regenerated floor.** This is
  the prompt's most likely stop and it is a real collision, not an agent error: the assertion is
  *production error ≤ 3 × the block's recorded floor*, the corrected background is four orders more
  accurate, and the module is outside D5's three files. Relay both floors and both errors. **Do not
  authorise the edit yourself** — widening D5 is the user's, exactly as D5 itself was.
- **Any recommended order is not 4.** The repair is prompt 05's either way. Relay the measurement.
- **The agent proposes to edit a production module**, `config/defaults.py`, `main.py`, the source
  grid, the band, or any test module beyond `test_background_tau.py`.
- **The agent proposes to drop the `branch+knots` key** rather than keep it as a control.
- **`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` was overwritten.** README §5 rule 6; it is another
  campaign's published document and the re-run's tables belong in `ORDER-AUDIT.md`.
- **A reference did not converge** at any (model, $k$, order). Prompt 17's error.
- **A moved source-grid digest.**
- **The agent asks you to choose an order, a floor, a margin or a schema option.** Relay verbatim
  (rule 5); do not pick one, and in particular do not tell it which D3 option the README calls the
  user's stated target — prompt §6 already puts that in front of it in the right register.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

## 6. After it lands

**Hand back to the user**, because **D3 is theirs and prompt 05 is waiting on it**. Report, in this
order:

1. **Whether the block was written**, and if so what moved in it — the four orders, the recommended
   scheme, and the floors the two threshold tests read.
2. **Are the orders converged at 4?** Per knob, with the per-$k$ answer for $N_\rho$ and the factor
   by which the floor dominates where the answer is `unchanged`.
3. **`QCD_BREAK_POINT_ALIGNMENT_TOL`** — back at 1.4e-05, or not, with the binding break.
4. **`RESIDUAL_WKB_REGION_MARGIN`** — a value with its floor, or rule 6's words — and with it the
   band-versus-region measurement and what it means for
   `[01-density-criterion-imposed-outside-the-wkb-region]`.
5. **The D3 recommendation**, with the schema-churn cost of each of the three options, stated so the
   user can decide from your report.

Then stop. **Prompt 05 is next and is not written yet**; campaign README §3.5 fixes its charter and
§6.2 its acceptance rows. It may be written once the user has settled **D3**, and not before — D1 is
already closed, so D3 is the only gate left on it. Write it after the user has read this. Do not
dispatch anything further.
