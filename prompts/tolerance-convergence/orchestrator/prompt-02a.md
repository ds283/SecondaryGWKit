# Orchestrator — prompt 02a, make the source grid buildable

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../02a-source-grid-density-guard.md`](../02a-source-grid-density-guard.md)
**Model:** Opus. **Production code changed:** `main.py`, `source_grid_spacing_profile` **only**.
**Test code changed:** `wkb_reference.py` anchor constants, `test_source_grid.py` assertions.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt is an insertion, and it is the first in the campaign to change production code.** The
boundary that permits it is campaign README §7 **D6**, settled 2026-09-17, and it is deliberately
narrow: `main.source_grid_spacing_profile` and nothing else in `main.py`. Two of this orchestrator
prompt's checks exist only to hold that line.

**It is not a stopping point** (campaign README §4.1). 02a recommends nothing and decides nothing.
When it lands and the checks pass, dispatch prompt 03 — which by then must have been written, since
prompt 02's hand-back is what it is written from.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and both suite counts (**452** and **39**).

**Then take the numerical baseline, and take it yourself.** This prompt's acceptance is a
bit-identity claim against two published digests, and it cannot be reconstructed after the diff
lands. Run this at the pre-dispatch `HEAD` and keep the output:

```bash
PYTHONPATH=. ./venv/bin/python -c "
import numpy as np
from Units import Mpc_units
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyConcepts import redshift_grid_digest
from ComputeTargets.tests.wkb_reference import SOURCE_GRID_V2, source_grid
K = np.logspace(np.log10(1e5), np.log10(3e8), 50)
u = Mpc_units()
for name, cos, anchor in (
    ('QCD   @ LambdaCDM anchor', QCD_Cosmology(store_id=0, units=u, params=Planck2018(), max_z=1e20), 2.0636395964161516e16),
    ('LCDM  @ own anchor      ', LambdaCDM(store_id=0, units=u, params=Planck2018()), 2.0636395964161516e16),
    ('QCD   @ own anchor      ', QCD_Cosmology(store_id=0, units=u, params=Planck2018(), max_z=1e20), 3.30033444460513e16),
):
    try:
        z = source_grid(SOURCE_GRID_V2, anchor, 0.1, 100, cosmology=cos, k_inv_Mpc=K).z_values
        print(f'{name}: {len(z)} samples, digest {redshift_grid_digest(z)}')
    except ValueError as e:
        print(f'{name}: RAISE {str(e)[:110]}')
" 2>&1 | grep -E "samples|RAISE"
```

At `b41da73` this must print **1996 / `4849552b`**, **1778 / `60a3205a`**, and a **RAISE** on the
third line. If the third line does not raise, **stop and tell the user**: the premise of the whole
prompt has changed under it and the charter needs re-reading before anything is dispatched.

**Precondition:** prompts 01 and 02 have landed. 02a's entire test surface is prompt 01's named grid
generations, and it cites prompt 02's inventory for the anchor tolerance it is forbidden to touch.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `02a-source-grid-density-guard`, model
**Opus**, with **one amendment to the template's standing text**. The template says:

> **No prompt before 05 changes a parameter** (README §5 rule 8): if you find yourself needing to
> edit `config/defaults.py` or `main.py` to make your prompt pass, stop and say so rather than
> editing it.

For this prompt only, replace that sentence with:

> **You may change `main.source_grid_spacing_profile`, and nothing else in `main.py`** (README §7
> D6). You may **not** edit `config/defaults.py`, `CosmologyConcepts/wavenumber.py`,
> `ComputeTargets/phase_residual.py`, or any `Datastore/` file. A new `SOURCE_GRID_*` constant
> belongs beside the others it is named with; if placing it there means editing a file this prompt
> may not touch, stop and say so rather than editing it.

Do not paraphrase D6's carve-out more loosely than that. An agent told it may "fix the source grid"
will reorder the crossing mask, which §2 of its prompt forbids for a reason the agent cannot
rediscover on its own — reordering changes which nodes are evaluated, hence `usable`, hence the
log-interpolation fit, so it can move a published grid, and nothing has measured whether it does.

## 3. The review — six checks

1. **Is the production diff exactly one function?**

   ```bash
   git diff HEAD~1 HEAD -- main.py
   git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'
   ```

   The first must touch `source_grid_spacing_profile` and nothing else in `main.py` — check the
   hunk headers, not the agent's summary. The second must list only `main.py`,
   `ComputeTargets/tests/wkb_reference.py` and `ComputeTargets/tests/test_source_grid.py`. Any
   other production file is a **stop** under D6, and so is a second function in `main.py` however
   small the change.

2. **Do both published digests still hold?** Re-run §1's command at the agent's commit. It must now
   print all three lines, with **1996 / `4849552b`** and **1778 / `60a3205a`** *unchanged* and the
   third line building.

   **A moved digest is a stop** (campaign README §4.3), and it is a stop even if the agent argues
   the new grid is better, even if it is longer, and even if every test it wrote passes. The claim
   02a exists to make is that the guard is inert on everything already measured; a changed digest
   falsifies that claim and nothing in the log can repair it. Do not dispatch a follow-up agent to
   restore it (rule 6) — report the digest, the sample count, and what the log says.

   **The likely cause, if it happens.** The guard has one degree of freedom the prompt does not
   pin: whether the `try` wraps all four `dphi_du` calls together or each one separately. Wrapping
   them together drops the whole node and reproduces these digests; wrapping them individually
   keeps partial stencils and does not. Check which the agent wrote before you read anything else.

3. **Does the guard catch `ValueError` and only `ValueError`?** Read the `except` clause. A bare
   `except:`, an `except Exception:`, or a catch that also swallows `TypeError`, `KeyError` or
   `ZeroDivisionError` is a **stop**: those are defects, not region boundaries, and a guard that
   hides them is worse than the raise it replaces. Check too that the guarded branch leaves
   `d4[i] = 0.0` and `usable[i] = False` rather than writing some fallback value into `d4`.

4. **Does it count, report and refuse?** Three separate things, and a prompt that did two of them is
   incomplete:
   - the guarded nodes are **counted** across the whole `(k, sector)` loop, not per call;
   - the count is **surfaced** — `main.py:963`'s existing criterion line is the natural place;
   - a **refusal** fires above a measured fraction of the band, with the count, the band size and
     the constant in the message.

   Verify the refusal by running the agent's own test for it. Check that its threshold was **chosen
   from a measurement with a stated margin**, not set to a round number: the log must say what
   fraction the production cases actually reach and why the constant sits where it does. A
   threshold with no measurement behind it is the campaign's own subject matter done wrong, and
   README §1.2's "never chosen" rule applies to it as much as to a tolerance.

5. **Is the guarded-node count reported in the form T7 can use?** Per cosmology **and per sector**,
   with the band size beside it, in "State handed to the next prompt". This is not bookkeeping: it
   is the evidence `[01-density-criterion-imposed-outside-the-wkb-region]` has been waiting for, and
   prompt 04 is written from it. A single aggregate number is not enough.

6. **The suites and the bookkeeping.** `ComputeTargets` **must not fall below 452** and should rise,
   since the prompt adds assertions; `CosmologyModels` must read **39** — 02a touches nothing in
   that package, so any movement is unintended. `test_source_grid.py`'s existing **38** tests must
   still pass, and the four its docstring names as the ones that must not be weakened must be
   untouched — `git diff` that file and read what changed rather than trusting the count.

   Board row 02a, item row **T13**, `[01-v2-density-raises-at-the-qcd-production-anchor]` moved to
   **§4**, `[01-density-criterion-imposed-outside-the-wkb-region]` narrowed but **left open**, and
   `docs/OPEN_ISSUES.md` in the **same commit** with the count and date corrected — one row deleted,
   so **75 → 74**. `black --check` clean on every `.py` in the diff.

## 4. What a good outcome looks like

- A four-line guard, a counter, a threshold with a measurement behind it, and two digests that did
  not move.
- QCD buildable at its own anchor, with the sample count and guarded-node count recorded per sector.
- A per-cosmology anchor constant in `wkb_reference.py`, and `PRODUCTION_Z_INIT`'s false comment
  corrected — it currently claims one value serves **both** production cosmologies.
- "Observations not acted on" carrying the crossing-mask ordering defect, unfixed and explained.
- A log that says plainly that the band was not touched, and hands T7 the numbers rather than an
  opinion about them.

**What a good outcome does *not* look like:** a better grid. If the log argues that the new QCD grid
at its own anchor is an improvement on anything, read §3.2 again — the prompt's acceptance is
bit-identity, and "improvement" is the vocabulary of prompt 04's decision, not this one's.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list — noting that its "an agent proposes to change
`BREAK_POINT_KIND`, the source grid, or `RESIDUAL_WKB_REGION_MARGIN`'s value" bullet is **amended
by D6** for this prompt only, to the carve-out of §2 above:

- **Either published digest moved.** Check 2. Always a stop.
- **The agent proposes to touch the band, `residual_node_range`, `RESIDUAL_WKB_REGION_MARGIN`,
  `build_z_sample`, `_solve_horizon_exit`, or the grid digest.** Each is out of scope by name, each
  has a home (T7/prompt 04, T6/prompt 03, prompt 05), and each will look to the agent like the
  right fix — because in the long run it is. That does not make it 02a's.
- **The agent reordered the crossing mask**, or argues it should. Observation only; §2 says why.
- **The refusal threshold has no measurement behind it**, or the agent asks you to choose it. Relay
  the question verbatim (rule 5); do not pick a number.
- **The guarded-node count is far from 53 at QCD's own anchor**, in either direction. The probe
  behind the charter measured 53 with 2034 samples. A large discrepancy means the guard is placed
  differently from the probe, and the user should see both numbers before 02a is built on.
- **The third line of §1's baseline did not raise before dispatch.** Stop before dispatching at all.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

## 6. After it lands

Not a hand-back. Confirm the two digests to the user in one line, with the QCD sample and
guarded-node counts and the refusal threshold, then continue to prompt 03 — which must exist by
then, written against prompt 02's hand-off (campaign README §4). If prompt 03 has not been written,
**that** is the stop, and it is the user's to resolve, not yours.
