# Prompt 04 — Relocate `_temperature_crossing_log1pz` out of the production class

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(g)**
**Closes:** `[08-temperature-crossing-solver-is-test-only]` on the
[`qcd-background-audit` board](../qcd-background-audit/IMPLEMENTATION_STATE.md) §3
**Depends on:** 02 (same file; sequencing only — no logical dependency)
**Recommended model:** **Sonnet** — a relocation with an exhaustive file list and a numerically
null acceptance. The one open choice is written out below with both options and the criterion that
decides it, so nothing is left to invent.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` (remove the method
and update the two docstring cross-references), `CosmologyModels/tests/T_z_reference.py`,
`CosmologyModels/tests/test_T_z_representation.py` (prose references only),
`ComputeTargets/tests/test_numeric_break_points.py` (the one call site at `:514`), plus this
campaign's log, both boards and `docs/OPEN_ISSUES.md`.

**Do not touch:** `_bisect_temperature_crossing_log1pz` (`:759-826`) — **that is the production
locator and it stays exactly where it is**; `integration_break_points`; `_solve_T_z`;
`_find_rho_equality`; `ComputeTargets/BackgroundModel.py`; any tolerance anywhere.

**Read first:** `LambdaCDM_GenericEOS.py:828-880` (the method and its docstring, which is long and
carries the whole argument — **it moves with the method, not summarised**);
`:759-826` (`_bisect_temperature_crossing_log1pz`, the production locator, for the contrast);
`ComputeTargets/tests/test_numeric_break_points.py:493-526` (the one caller);
`CosmologyModels/tests/T_z_reference.py:40-60` and `:180-195`;
`CosmologyModels/tests/test_T_z_representation.py:430-470`
(`test_a_segment_edge_bisected_and_one_root_found_disagree`, the standing demonstration the
docstring names); and the index row for `[08-temperature-crossing-solver-is-test-only]` in
`docs/OPEN_ISSUES.md` §1.7.

---

## 1. What is wrong

`LambdaCDM_GenericEOS._temperature_crossing_log1pz` (`:828`) has had **no production caller** since
`qcd-background-audit` prompt 07 took `integration_break_points` onto the bisected
`_break_point_crossings_log1pz`. Its two remaining callers are tests. Its own docstring opens

> **Nothing in production calls this, and nothing may put it back on the break-point path.**

That docstring is excellent and must survive verbatim. But a private method on a production class
whose only callers are tests is a trap: the next reader sees a `root_scalar` in a cosmology model
and reasonably assumes something evaluates it.

**It is kept, not deleted**, because it is two things at once: the probe
`test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` uses to find a neighbourhood of a
crossing, and the documented illustration of the trap `qcd-background-audit` README §2 (b) is
about — a bracketing solver reporting `converged` on a function with no root, landing ~1.126e-12
above the jump.

**Its tolerances are not in scope.** `xtol=1e-15, rtol=1e-15` were chosen by `qcd-background-audit`
prompts 06 and 07 and are sound. Move the method; do not change a character of the `root_scalar`
line.

## 2. The change

### 2.1 Where it goes

Move it to `CosmologyModels/tests/T_z_reference.py`, as a **module-level function** taking the
cosmology as its first argument:

```python
def temperature_crossing_log1pz(cosmology, T: float, u_lo: float, u_hi: float) -> Optional[float]:
```

`T_z_reference.py` is the right home: it already exists to hold the independent scoring machinery
for the $T(z)$ representation, it already discusses this method at `:45`, and
`test_T_z_representation.py` already imports from it.

Bring the **entire docstring**, amended in exactly two ways:

1. The opening sentence changes from "Nothing in production calls this" to a statement that it
   **is** test machinery and why it is kept rather than deleted — naming both uses (the
   neighbourhood probe, and the illustration of the trap).
2. Add one sentence recording that it lived on `LambdaCDM_GenericEOS` until this campaign, with the
   campaign name, so that a `git log -S` is not the only way to find out.

Everything else — the ~1.126e-12 figure, the reference to
`test_a_segment_edge_bisected_and_one_root_found_disagree`, the `expm1` / `CLAUDE.md` note — is
carried across unchanged.

### 2.2 The one open choice, and how to decide it

`ComputeTargets/tests/test_numeric_break_points.py:514` is the caller, and it lives in a **different
package's** test directory. Nothing in the tree currently imports `CosmologyModels.tests.*` from
`ComputeTargets/tests/`, so this would be the first such edge.

| Option | What it means | When it is right |
|---|---|---|
| **(a)** `test_numeric_break_points.py` imports `temperature_crossing_log1pz` from `CosmologyModels.tests.T_z_reference` | One definition, cross-package test import | If the import works under `PYTHONPATH=. python -m unittest discover -s ComputeTargets/tests -t .` **and** under `-s CosmologyModels/tests` |
| **(b)** the `ComputeTargets` test uses `cosmology.integration_break_points(...)` to locate the crossing instead, and the moved function serves only `CosmologyModels/tests` | No new package edge | If (a) does not work, **or** if the probe's assertion survives unchanged under (b) |

**Prefer (a).** It keeps one definition and one docstring. Verify the import under **both**
discovery roots before committing — a test module that imports fine from the repository root and
not from its own package's discovery is a trap of the same kind you are removing.

If you take **(b)**, the criterion is that
`test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`'s three assertions **still pass
with their present deltas** (1.97e-3 and 1.38e-4 to within 5 %, and < 1e-8 at `EOS_T_LO`). That
test probes $H$ at $u(1\pm10^{-12})$; the bisected crossing and the root-solved one differ by
~1.126e-12 **absolute** in $u$, while the probe offset at $u\approx17$ is ~1.7e-11 absolute, so
both straddle the step. Confirm that by running it, not by repeating this paragraph.

**Whichever you take, say which and why in the log as an `IMPLEMENTATION CHOICE`, with the
measured evidence.**

### 2.3 The cross-references

Four places name the method. Update each to point at its new home; do not leave a reference to
`LambdaCDM_GenericEOS._temperature_crossing_log1pz`, which will no longer exist:

- `LambdaCDM_GenericEOS.py:732` — the `_build_break_point_crossings_log1pz` docstring's contrast
  between bisection and a `root_scalar` bracket. **This sentence is the reason the production
  locator is what it is; keep the argument and re-point the reference.**
- `CosmologyModels/tests/T_z_reference.py:45`.
- `ComputeTargets/tests/test_background_tau.py:91` and `:98` — **prose only.** Read them; if they
  read correctly without naming the method's owner, leave them. Say either way in the log.
- `CosmologyModels/tests/test_T_z_representation.py:558`, if it names the owner.

## 3. Acceptance

| Check | Threshold |
|---|---|
| `_temperature_crossing_log1pz` | is not a member of `LambdaCDM_GenericEOS`; `grep -rn "self\._temperature_crossing_log1pz"` returns nothing |
| The `root_scalar` line | **character-identical** to `:869` today, including `xtol=1e-15, rtol=1e-15` |
| `CosmologyModels` suite | count unchanged, OK — **every test that was passing still passes** |
| `ComputeTargets` suite | **447 → 447**, OK |
| `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` | passes, and its three measured steps are quoted in the log |
| Import works under both discovery roots (option a) | demonstrated by running both commands |
| `T_Z_REPRESENTATION_VERSION` | **6** before and after |
| Any production number | **unchanged** — this prompt is numerically null and must be shown to be |
| `black --check` | clean |

## 4. Stop conditions

- **A suite count falls.** Something did not move cleanly. Stop.
- **Any assertion in `test_numeric_break_points.py` needs its delta changed.** It does not; if it
  seems to, you have changed which locator is used in a way that matters, which is a finding. Stop
  and report the numbers.
- **You are tempted to delete the method rather than move it.** Both uses are real and the index
  row names the choice as "move **or** delete and have that one test bisect" — deleting means
  option (b) plus removing the illustration, which loses
  `qcd-background-audit` README §2 (b)'s worked example. If you believe deletion is right, **stop
  and ask**; do not take it unilaterally.

## 5. Deliverables

1. `temperature_crossing_log1pz` in `CosmologyModels/tests/T_z_reference.py`, docstring carried
   across and amended per §2.1.
2. The method removed from `LambdaCDM_GenericEOS`, and §2.3's cross-references re-pointed.
3. The one caller updated, per whichever of §2.2 you took.
4. `logs/04-relocate-the-crossing-probe.md` per README §5.1, with the §2.2 choice classified and
   evidenced, and the three measured steps from the probe test quoted.
5. Board row 04, item row (g), and `[08-temperature-crossing-solver-is-test-only]` moved to the
   **`qcd-background-audit` board's §4** (it is that board's issue) with its row **deleted** from
   `docs/OPEN_ISSUES.md` — count and date corrected, in the same commit.
6. One commit, README §5 rule 2.
