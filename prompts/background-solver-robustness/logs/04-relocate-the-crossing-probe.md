# Log 04 — relocate the crossing probe out of the production class

**Prompt:** prompts/background-solver-robustness/04-relocate-the-crossing-probe.md
**Commit:** *(this commit)* — Relocate the crossing probe to test machinery
**Model:** Claude Sonnet 5
**Date:** 2026-09-16
**Result:** COMPLETE

---

## What shipped

`LambdaCDM_GenericEOS._temperature_crossing_log1pz` (`:828-879` before this commit) has had no
production caller since `qcd-background-audit` prompt 07; its only callers were tests. It is moved,
whole, out of the production class.

| File | Before → after |
|---|---|
| `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` | The method (formerly `:828-879`) is removed entirely. `_build_break_point_crossings_log1pz`'s docstring (the sentence spanning what was `:731-735`) is re-pointed: "a `root_scalar` bracket on `T_photon(z) - T_break` there" becomes "... in what is now `CosmologyModels.tests.T_z_reference.temperature_crossing_log1pz` (moved out of this class by `prompts/background-solver-robustness/` prompt 04)". Nothing else in the file changed. `Optional` and `root_scalar` remain imported and used elsewhere (`_solve_T_z`, `_find_rho_equality`), so no import went unused. |
| `CosmologyModels/tests/T_z_reference.py` | New module-level function `temperature_crossing_log1pz(cosmology, T: float, u_lo: float, u_hi: float) -> Optional[float]`, inserted between `jump_locations` and the "probe geometry" section. Its body, including the `root_scalar(q, bracket=(u_lo, u_hi), xtol=1e-15, rtol=1e-15)` line, is character-identical to the removed method's, with `self` replaced by `cosmology` throughout (`self.T_photon` → `cosmology.T_photon`, `self._units.GeV` → `cosmology._units.GeV`) and the `RuntimeError` prefix changed from `"LambdaCDM_GenericEOS._temperature_crossing_log1pz:"` to `"T_z_reference.temperature_crossing_log1pz:"`, matching this module's existing convention (`accurate_T`'s error uses `"T_z_reference.accurate_T:"`). The module docstring's own reference to `_temperature_crossing_log1pz` (`:45`) is updated to the new, unprefixed name. |
| `ComputeTargets/tests/test_numeric_break_points.py` | New import `from CosmologyModels.tests.T_z_reference import temperature_crossing_log1pz`. The one caller, `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` (`:514-516`), changes from `self.cosmology._temperature_crossing_log1pz(T_in_GeV * self.units.GeV, u_lo, u_hi)` to `temperature_crossing_log1pz(self.cosmology, T_in_GeV * self.units.GeV, u_lo, u_hi)`. |

**No new public symbol other than the moved function.** Its signature is
`temperature_crossing_log1pz(cosmology, T: float, u_lo: float, u_hi: float) -> Optional[float]`.

`T_Z_REPRESENTATION_VERSION` is **6** before and **6** after; nothing in this prompt reads or
writes it.

### The docstring amendments (§2.1 of the prompt)

Carried across unchanged except the two amendments the prompt permits:

1. The opening sentence no longer reads "Nothing in production calls this, and nothing may put it
   back on the break-point path." It now opens "**This is test machinery, not a production solver:
   nothing in production calls it, and nothing may put it back on the break-point path.** It is
   kept, not deleted, because it is two things at once -- the neighbourhood probe
   `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` uses ..., and the documented
   illustration of the trap README §2 (b) of `prompts/qcd-background-audit/` is about." — naming
   both uses in the opening sentence, as the prompt asks, before the paragraph continues with the
   unchanged "Until prompt 07 ... standing demonstration" text.
2. A trailing sentence is added after the "T_photon(z) is monotone ..." paragraph: "It lived as a
   private method on `LambdaCDM_GenericEOS` until `prompts/background-solver-robustness/` prompt 04
   moved it here, character-identical in its `root_scalar` call."

Everything else in the docstring — the `~1.126e-12` figure, the reference to
`test_a_segment_edge_bisected_and_one_root_found_disagree`, the `expm1` / `CLAUDE.md` redshift-
arithmetic note, and the "It survives as a measurement probe" paragraph — is carried across
unchanged, character for character.

---

## The two equality redshifts

Mandatory per README §5.1, though this prompt's file list does not include `_find_rho_equality` or
its callers, and nothing in this prompt's diff touches them. Recomputed at this commit to confirm
the obvious — **no bit moved** — rather than assert it from the diff:

| Model | Pair | Before (unchanged tree) | After (this commit) | Move |
|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499511` (`0x1.a9d5683cafad0p+11`) | `3406.6689742499511` (`0x1.a9d5683cafad0p+11`) | **none** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299640738` (`0x1.36b4870e4a718p-2`) | `0.30342303299640738` (`0x1.36b4870e4a718p-2`) | **none** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279457` (`0x1.a963640e40f2cp+11`) | `3403.1059638279457` (`0x1.a963640e40f2cp+11`) | **none** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.30342303299640738` (`0x1.36b4870e4a718p-2`) | `0.30342303299640738` (`0x1.36b4870e4a718p-2`) | **none** |

Reproduced with the same construction `CosmologyModels/tests/test_rho_equality.py` uses
(`QCD_Cosmology(store_id=10, ..., max_z=1e12)` and
`LambdaCDM_GenericEOS(store_id=11, eos=PureRadiationEOS(units, lambdaCDM_gstar(params.Neff)), ...,
max_z=1e6)`), calling `_find_rho_equality` directly with the same guesses `__init__` uses. Every
figure matches board standing note 8's table to the last bit. The two printed banner lines at
`LambdaCDM_GenericEOS.py:516-517` are untouched by this prompt's diff and therefore
character-identical by construction.

---

## Deviations from the prompt

### D1 — which option of §2.2 was taken: (a), not (b) — IMPLEMENTATION CHOICE

The prompt's §2.2 leaves the cross-package import question open, with a stated preference for (a).
(a) was taken: `ComputeTargets/tests/test_numeric_break_points.py` imports
`temperature_crossing_log1pz` from `CosmologyModels.tests.T_z_reference`.

**Evidence the import works under both discovery roots**, per the prompt's own verification
requirement:

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
  → Ran 38 tests in 0.703s, OK

PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
  → Ran 447 tests in 170.456s, OK
```

Both pass cleanly with the new cross-package import in place. `ComputeTargets/tests/__init__.py`
and `CosmologyModels/tests/__init__.py` both exist, making each a proper package, and both
discovery commands use `-t .` (the repository root as the top-level directory and hence the
`sys.path` entry unittest adds), so the import is not sensitive to which of the two directories is
being discovered. `ComputeTargets/tests/test_numeric_break_points.py` already imported
`CosmologyModels.GenericEOS.*` (production code) before this change, confirming the same
cross-package resolution was already exercised in this tree; this commit is the first to import
`CosmologyModels.tests.*` from `ComputeTargets/tests/`.

(a) was preferred over (b) for the reason the prompt gives: it keeps one definition and one
docstring, and the import demonstrably works.

### None else

No other deviation. The `root_scalar` call is character-identical to the one removed
(`root_scalar(q, bracket=(u_lo, u_hi), xtol=1e-15, rtol=1e-15)`), `_bisect_temperature_crossing_log1pz`
and `integration_break_points` were not touched, and no tolerance anywhere in the file changed.

---

## Verification performed

**Ran, not reasoned about, unless stated otherwise.**

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` →
  **Ran 38 tests in 0.703s. OK.** (Also confirmed at `HEAD` before this prompt's edits, by
  `git stash` / re-run / `git stash pop`: identically **38, OK** — the count the prompt requires be
  unchanged really is unchanged, not merely asserted.)
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` →
  **Ran 447 tests in 170.456s. OK.** (447 → 447, the campaign's standing invariant, board note 3.)
- `PYTHONPATH=. ./venv/bin/python -m unittest
  ComputeTargets.tests.test_numeric_break_points.TestDeclaration.test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink
  -v` → **ok.**
- The same test's three measured steps, recomputed directly against the moved function (not merely
  inferred from "ok"):

  | Temperature | `u` (moved-function root) | measured step in H | test's comparison |
  |---|---|---|---|
  | `T_LO` (1e-5 GeV) | `17.565806941870036` | **1.969955e-03** | `assertAlmostEqual(step/1.97e-3, 1.0, delta=0.05)` — passes |
  | `EOS_T_LO` (0.002 GeV) | `23.197460552819646` | **9.272151e-11** | `assertLess(step, 1e-8)` — passes |
  | `T_120_MEV` (0.12 GeV) | `27.485391822044242` | **1.377111e-04** | `assertAlmostEqual(step/1.38e-4, 1.0, delta=0.05)` — passes |

  These are the same three deltas the board's issue entry and `ComputeTargets/tests/
  test_background_tau.py`'s comments record for this cosmology (1.97e-3, 1.38e-4), confirming the
  relocation changed no behaviour.
- `grep -rn "self\._temperature_crossing_log1pz"` (repository-wide) → **no matches.** The method is
  not a member of `LambdaCDM_GenericEOS`.
- `./venv/bin/python -m black --check CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
  CosmologyModels/tests/T_z_reference.py ComputeTargets/tests/test_numeric_break_points.py` →
  **all three unchanged, clean.**
- The two equality redshifts: recomputed and quoted in the section above — no bit moved.

**Not run:** `docs/qcd-background-audit/` reproduction scripts (out of this prompt's file list) and
`LiouvilleGreen/tests` (no prompt in this campaign touches that package, per board note in
`RECONCILIATION.md` §10).

---

## Observations not acted on

1. **`_entropy_segment_edges_log1pz`'s docstring (`LambdaCDM_GenericEOS.py`, around what is now
   `:698-701`) carries a similar historical sentence** — "They were once found two different ways
   -- here by bisection, there by a root solve on `T_photon(z) - T_break` -- and that is exactly the
   arrangement README §2 (b) warns against ... See `:meth:`_bisect_temperature_crossing_log1pz``." —
   which describes the same historical fact as the one sentence this prompt was asked to re-point
   (`_build_break_point_crossings_log1pz`'s docstring). The prompt's §2.3 lists exactly four places
   to update and this method's docstring is not one of them; its own `:meth:` reference points at
   `_bisect_temperature_crossing_log1pz` (which stays), not at the removed method, so it does not
   dangle. Left unchanged, per README §5 rule 5 ("do not fix things the prompt did not ask for").
   No board issue opened: it is not a defect, only an observation that the same historical prose
   appears in two docstrings and only one was in scope here.
2. **`ComputeTargets/tests/test_background_tau.py:91` and `:98`** were read as the prompt asked.
   Both are comments describing, in prose, the history of `QCD_BREAK_POINT_ALIGNMENT_TOL`'s
   successive loosenings; both name `_temperature_crossing_log1pz` without naming
   `LambdaCDM_GenericEOS` as its owner ("the crossing `_temperature_crossing_log1pz` finds", "and
   `_temperature_crossing_log1pz` solves T_photon(z) - T_break"). Per the prompt's own instruction
   ("if they read correctly without naming the method's owner, leave them"), both are left
   unchanged.
3. **`CosmologyModels/tests/test_T_z_representation.py:558`** (inside
   `test_the_production_segment_edges_are_the_declared_jumps`'s docstring) references
   `test_a_segment_edge_bisected_and_one_root_found_disagree` by name, not the moved method itself,
   and does not name `LambdaCDM_GenericEOS` as anyone's owner. Left unchanged, per the prompt's own
   conditional.

---

## State handed to the next prompt

- `_temperature_crossing_log1pz` no longer exists as a method; it is
  `CosmologyModels.tests.T_z_reference.temperature_crossing_log1pz(cosmology, T, u_lo, u_hi)`,
  imported by `ComputeTargets/tests/test_numeric_break_points.py`.
- `_bisect_temperature_crossing_log1pz`, `integration_break_points`, `_find_rho_equality` and every
  tolerance in `LambdaCDM_GenericEOS.py` are untouched by this commit.
- `[08-temperature-crossing-solver-is-test-only]` is closed on the `qcd-background-audit` board's
  §4 (resolved issues) and its row is deleted from `docs/OPEN_ISSUES.md` (72 → 71 open). This
  campaign's own §3.1 "adopted from other boards" table marks the row closed rather than deleting
  it, so the campaign's own history of what it adopted and why stays legible.
- Suite counts at this commit: `CosmologyModels` **38**, `ComputeTargets` **447**, both unchanged
  from immediately before this prompt (confirmed by `git stash` / re-run / `git stash pop` before
  making any edit). `T_Z_REPRESENTATION_VERSION` **6**.
- Prompt 05 (workstream C, "Hoist the range logic") touches the same file
  (`LambdaCDM_GenericEOS.py`) next; nothing in this prompt's diff overlaps the range-logic methods
  `[07-t-photon-range-logic-recomputes-its-bounds]` names, so there is no merge concern.
