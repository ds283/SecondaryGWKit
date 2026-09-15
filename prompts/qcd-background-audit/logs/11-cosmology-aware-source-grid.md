# Log 11 — a source grid that knows the cosmology it samples

**Prompt:** [`prompts/qcd-background-audit/11-cosmology-aware-source-grid.md`](../11-cosmology-aware-source-grid.md)
**Commit:** *(this commit)* — Build the source grid around the features the cosmology declares
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Parent:** `b580a78` (prompt 10) · **Campaign base:** `2a5e0fa` (numerically `e8f746d`)
**Result:** COMPLETE WITH DEVIATIONS

**The one-sentence version.** The production source grid now carries the points `QCD_Cosmology`
declares — a pair straddling each of the three equation-of-state crossings and the eleven grid
intervals around it refined by two, plus the two equality redshifts — which is **41 extra samples
in 1,732 (2.37 %)** and takes the two consumer rows prompt 09 recorded as a miss from
**34.11 → 1.61 ulp** ($G_k$, 3.8296e-07 rad) and **1876.61 → 65.78 ulp** ($T_k$, 4.9012e-07 rad),
both inside prompt 10 §5's 1e-06 rad target and both better than the campaign base. **The
prompt's §2 item 2 — two straddling samples per break — is necessary and, on its own, buys 1.96×
and 1.98× and leaves both rows outside the target**; what carries them is the neighbourhood, and
the pair is worth a further 2 % on top of it at the standoff that was scored rather than assumed.
LambdaCDM's grid is **bit-identical, element for element**. The grid tags now carry a digest of the
grid, which closes the collision the audit names and **invalidates every stored object of seven
types**; §5 below is the bill, in numbers.

---

## What shipped

**`T_Z_REPRESENTATION_VERSION` is 5 before and after.** No `CosmologyModels/` file is in the diff,
no compute target is in the diff, and no background value moves: `LambdaCDM`'s grid is unchanged
and `QCD_Cosmology`'s *cosmology* is untouched — only which redshifts are sampled.

### 1. `CosmologyConcepts/wavenumber.py` — the grid builder

New public names:

* `SOURCE_GRID_BREAK_STANDOFF = 0.25` — the straddling standoff, **as a fraction of a base grid
  interval**, so the pair goes at `z_break -+ standoff * (10**dlog10z - 1) * (1 + z_break)`. On the
  production grid that is 5.82e-03 relative in $(1+z)$.
* `SOURCE_GRID_BREAK_HALF_WIDTH = 5`, `SOURCE_GRID_BREAK_REFINEMENT = 2` — prompt 10's measured
  neighbourhood, named as such in the comment block.
* `SOURCE_GRID_MIN_SEPARATION = 10 * DEFAULT_REDSHIFT_RELATIVE_PRECISION` (1e-06) and
  `SOURCE_GRID_MESH_GUARD = 0.25` — the two degeneracy guards (§2 item 3).
* `class SourceGrid(NamedTuple)`: `z_values`, `protected_z`, `breaks`, `features` — all descending
  `float` arrays.
* `build_z_sample(z_init, z_end, samples_per_log10z, *, break_z=(), feature_z=(), standoff=...,
  half_width=..., refinement=...) -> SourceGrid` — takes **values**, not a cosmology, and imports
  no equation-of-state module, following `_cosmology_break_points`'s duck-typed precedent.
* `wavenumber_exit_time.populate_source_grid(...) -> SourceGrid` — the z_init resolution that
  `populate_z_sample` always did, now returning the whole grid record.

`wavenumber_exit_time.populate_z_sample(..., **kwargs)` (`:250` before → `:414` after) is now a
one-line delegation returning `populate_source_grid(...).z_values`; **called with no keyword
extras it returns exactly the `numpy.logspace` it returned before**, by construction rather than by
accident — `build_z_sample` computes `base = logspace(log10(z_init), log10(z_end), num=num)` with
the unchanged `num` arithmetic and returns that array object untouched when nothing is declared.

`CosmologyConcepts/__init__.py` re-exports `SourceGrid`, `build_z_sample`, the three break
constants and `redshift_grid_digest`.

### 2. `CosmologyConcepts/redshift.py` — the winnow and the digest

* `redshift_array.winnow(sparseness)` → `winnow(sparseness, protect=None)` (`:226` → `:260`). The
  stride is untouched; `protect` is an iterable of `redshift` matched **on `store_id`**, so no
  recovered redshift is ever compared for equality and the result is a subset of the array by
  construction — a protected point that is not in the array is ignored.
* `redshift_grid_digest(z_values, chars=8) -> str` and `redshift_array.digest(chars=8)`, new:
  `blake2b` over the exact `struct.pack("<d", ...)` bytes of the grid's values in order.
  `REDSHIFT_GRID_DIGEST_CHARS = 8`.

### 3. `main.py` — the grid-construction and tag hunks only

Two new module-level functions, placed before `run_pipeline` so that
`ComputeTargets/tests/test_main_plumbing.load_main_py_functions` can extract and **execute** them
(`main.py` cannot be imported, `CLAUDE.md`):

* `cosmology_feature_redshifts(cosmology, z_end, z_init) -> (break_z, feature_z)` — the policy
  layer. `break_z` is `_cosmology_break_points(cosmology, z_end, z_init)` converted out of
  $u=\log(1+z)$; `feature_z` is matter–radiation and matter–$\Lambda$ equality from the public
  `omega_m` / `omega_r` / `omega_cc`. **Both are empty when the cosmology declares no break
  points** (deviation 2).
* `build_grid_tag_labels(source_z_values, response_z_values) -> (source_label, response_label)` —
  `SourceRedshiftGrid_{len}_{digest}` and `ResponseRedshiftGrid_{len}_{digest}`.

In `run_pipeline` (`:520-590` before → `:603-651` after): `populate_z_sample` → `populate_source_grid`
with `break_z` / `feature_z`; a one-line report of what was declared; a second
`convert_to_redshifts` call recovering the protected points as `redshift` objects;
`winnow(sparseness=...)` → `winnow(sparseness=..., protect=z_protected_sample)`; and the two
`store_tag` labels now come from `build_grid_tag_labels`. The two f-string tag literals are gone
from `main.py` and a test asserts they are. Nothing else in `main.py` is touched — no Bessel, WKB
or numeric stage, and the other seven tags are unchanged.

`main.py` gains one import, `from ComputeTargets.BackgroundModel import _cosmology_break_points`,
and `redshift_grid_digest` from `CosmologyConcepts`.

### 4. Tests

* **`ComputeTargets/tests/test_source_grid.py`** (new, 19 tests, 0.25 s, no Ray, no datastore).
  Five classes: the bit-identity of an undeclared grid (three tests, including a control that the
  test is not vacuous); the protected set on production QCD (six); the response grid and the
  winnow (four, including one that asserts a *blind* stride does drop protected points, so the fix
  is not vacuous); the grid tag (five); and the refused standoff (one).
* **`ComputeTargets/tests/test_main_plumbing.py`**: `load_main_py_functions` gains an optional
  `extra_globals` parameter so a function whose *body* needs a name from `main.py`'s import list
  can be executed with the real object. No existing behaviour changes; the existing 15 tests are
  untouched.
* **`docs/qcd-background-audit/source_grid_consumer_check.py`** (new, ~32 s): the consumer
  measurement (deviation 3).

### 5. What the tag change invalidates — the bill, in numbers

This is the consequence least visible from the diff and the prompt makes it the headline of the
log, so it is stated here in full rather than by reference.

**The mechanism.** Eight `pool.object_get` call sites in `main.py` filter on
`SourceZGridSizeTag` and/or `ResponseZGridSizeTag`, across **seven stored object types**:

| type | tags it filters on | call sites |
|---|---|---|
| `TkNumericIntegration` | source | 1 |
| `TkWKBIntegration` | source | 1 |
| `QuadSource` | source | 1 |
| `GkNumericIntegration` | source + response | 1 |
| `GkWKBIntegration` | source + response | 2 |
| `GkSource` | source + response | 1 |
| `QuadSourceIntegral` | source + response | 1 |

Their factories treat the sample grid as *"a target rather than a selection criterion"*
(`Datastore/SQL/ObjectFactories/GkNumericIntegration.py:200`), so **the tag is the only thing that
records which grid an object was computed on**. Change the label and every stored object of those
seven types is unfindable. `GkSourcePolicyData` carries no grid tag but is keyed on `GkSource`, so
it goes with them: **eight object types in total**. `BackgroundModel`, `wavenumber_exit_time`,
`redshift`, `bessel_phase`, `tolerance`, `IntegrationSolver` and `store_tag` are not filtered on
these tags and survive.

**The cost**, from the only measured production-shaped run in the tree
(`docs/gktk-remedial-verification.md` §4.2 — LambdaCDM, 5 source and 5 response wavenumbers, a
1,584-node grid, one model):

| type | objects | wall |
|---|---|---|
| `TkNumericIntegration` | 5 | 1.15 s |
| `TkWKBIntegration` | 5 | 0.669 s |
| `QuadSource` | 15 | 0.714 s |
| `GkNumericIntegration` | 2,455 | 54.9 s |
| `GkWKBIntegration` | 7,920 | 3 m 59.9 s |
| `GkSource` | 660 | 1 m 18.2 s |
| `GkSourcePolicyData` | 660 | 19.2 s |
| `QuadSourceIntegral` | 3,300 | **stopped after ~3 h, incomplete** |
| **total** | **15,020 objects** | **6 m 35 s, plus an unfinished `QuadSourceIntegral` stage** |

Production is **50 source and 50 response wavenumbers and two models** — ten times that run in
each wavenumber sample. The per-$k$ stages scale ×10; `QuadSource` runs over unordered pairs,
$\binom{51}{2} = 1{,}275$ against 15, so ×85; `QuadSourceIntegral` scales with the pairs *and* the
response wavenumbers, of order ×850 on a stage that already failed to finish 3,300 objects in
three hours. **That multiplier is an extrapolation from a measured run and is labelled as one**;
the defensible statement is that a pre-prompt-11 datastore is regenerated from
`TkNumericIntegration` downwards for both models, and that the `QuadSourceIntegral` stage is what
the bill is made of.

**Half of it is avoidable and the user should be told so.** The QCD half was invalidated by this
commit anyway: its grid is 1,773 samples where it was 1,732, so the old `SourceRedshiftGrid_1732`
would not have matched on **length alone** even without the digest — and prompt 03's
`T_z_representation` key had already invalidated its cosmology row for an unrelated reason. **The
LambdaCDM half is invalidated by the tag change and by nothing else**, because its grid is
bit-identical. `store_tag` is keyed on its label and is a replicated table, so a datastore holding
**only** LambdaCDM objects can be carried across by relabelling rather than recomputing:

```sql
UPDATE store_tag SET label='SourceRedshiftGrid_1732_0960e169'  WHERE label='SourceRedshiftGrid_1732';
UPDATE store_tag SET label='ResponseRedshiftGrid_145_69050b4c' WHERE label='ResponseRedshiftGrid_145';
```

**Do not run that on a datastore that also holds pre-prompt-11 QCD objects.** They share the
`SourceRedshiftGrid_1732` row, and relabelling it would re-validate QCD objects computed on a grid
that no longer exists — exactly the silent staleness the digest exists to prevent. Drop them first.

The four production tag labels, for the record:

| | old | new |
|---|---|---|
| LambdaCDM source | `SourceRedshiftGrid_1732` | `SourceRedshiftGrid_1732_0960e169` |
| LambdaCDM response | `ResponseRedshiftGrid_145` | `ResponseRedshiftGrid_145_69050b4c` |
| QCD source | `SourceRedshiftGrid_1732` | `SourceRedshiftGrid_1773_303f9ce7` |
| QCD response | `ResponseRedshiftGrid_145` | `ResponseRedshiftGrid_156_197b46de` |

---

## Deviations from the prompt

### 1. `STRUCTURALLY REQUIRED` — the standoff is a fraction of a grid interval, not an absolute relative number, and two straddling samples are **not** the remedy

Prompt §2 item 2 asks for "a pair straddling each break at a relative standoff in $(1+z)$", chosen
"by measurement against what prompt 10's consumer spline needs". Both halves of what shipped differ
from the obvious reading, and both differences are measurements rather than preferences.

**(a) Two straddling samples alone do not deliver the improvement, and the prompt's acceptance
table expects them to.** Measured on the production geometry with the pair and no neighbourhood
refinement: QCD $k=10^5$ goes 34.11 → **17.43 ulp** ($G_k$) and 1876.61 → **949.24 ulp** ($T_k$) —
**1.96× and 1.98×**, both still far outside prompt 10 §5's 1e-06 rad target, and the same factor
prompt 10 measured for four extra samples *inside* the break's own interval before it stalled. So
the shipped design is the pair **plus** prompt 10's measured ±5-interval, 2× neighbourhood, and the
log says plainly that the pair is the smaller half: refinement alone reaches 1.64 / 74.60 ulp, and
the pair adds a further 2 % / 12 % on top of that (1.61 / 65.78). Prompt §4's row "Samples
straddling each break: 2 per break, at a justified standoff" is met; the row above it is met by the
neighbourhood, which the prompt does not name.

**(b) The standoff is 1/4 of a grid interval, not an absolute constant.** The prompt's language and
the `BREAK_POINT_STANDOFF = 1e-12` precedent both suggest an absolute relative number; an absolute
number is measurably wrong here. A pair a distance $d$ apart implies a slope carrying the
consumer's storage-granularity uncertainty $\sim 2\,\mathrm{ulp}/d$
(`[02-consumer-phi-below-the-storage-granularity]`, still open and still real), and a cubic through
it propagates that to the neighbouring intervals. Scored from 1/2 of an interval down to 1e-4 of
one (§3 below), the consumer degrades monotonically below ~1/32 and saturates **1.5×–1.7× worse
than no pair at all**. Tying the standoff to the grid spacing keeps it above that floor at *any*
density, which an absolute constant cannot do and which matters directly to prompt 12.

Alternatives considered: an absolute 1e-5 relative (scored: 2.52 / 123.45 ulp, i.e. worse than no
pair); an absolute 1e-3 (scored: on the production grid this is ~1/23 of an interval, 2.53 / 123.65
— also worse, because at the shipped density it is below the floor); and reusing
`BREAK_POINT_STANDOFF = 1e-12` (refused by `build_z_sample`, and a test asserts the refusal, since
at 1e-12 the two samples are the *same datastore row* — see deviation 4).

### 2. `STRUCTURALLY REQUIRED` — a cosmology that declares no break points gets no equality samples either, because §2 item 1 and §3 item 1 cannot both hold

Prompt §2 item 1 says the protected set is `_cosmology_break_points(...)` "plus the two equality
redshifts the model already computes", and that "a cosmology declaring nothing produces **exactly**
the grid it produces today", which §3 item 1 and §4 make an acceptance test *for LambdaCDM at the
production parameters*. **LambdaCDM declares no break points and does have equality redshifts**, so
those two requirements contradict each other: injecting the equalities for every model makes the
LambdaCDM production grid 1,734 samples where it is 1,732, and the grid is then not bit-identical.

It is decided in favour of bit-identity, because README §0.5 and §2 (g) make "every LambdaCDM,
`RadiationModel` and stand-in number bit-identical" a campaign-wide **stop condition**, README §4
lists it among the orchestrator's stop conditions, and no measurement says an equality sample buys
anything — the background is perfectly smooth at either equality. So
`cosmology_feature_redshifts` returns `([], [])` when the break list is empty, and the whole
cosmology-aware path is gated on the cosmology declaring some non-smoothness.

The cost of the decision, so a later reader can reverse it knowingly: LambdaCDM gets no sample at
$z=3403$ or $z=0.3034$. Reversing it is one `if` in `main.py` and costs 2 samples in 1,732 (0.12 %),
plus the bit-identity.

**`build_z_sample` itself is not gated** — handed features and no breaks it places them, and
`test_source_grid.py` covers that path. The gate is policy and lives in `main.py`, which is where
the cosmology is consulted.

### 3. `IMPLEMENTATION CHOICE` — a new script under `docs/qcd-background-audit/`, which is not on the prompt's file list

Prompt §3 item 6 asks for the consumer's interpolation error "at and around each break on the new
grid, against the old", and §5 for a dated section in `docs/qcd-background-verification.md`; the
measurement has to come from somewhere. `docs/qcd-background-audit/source_grid_consumer_check.py`
imports prompt 10's `build_cases` and `resolution_ladder` **unchanged** — the orchestrator's
instruction was to reuse prompt 10's tool rather than build a new scorer — and adds only the
ability to score an arbitrary extra sample set, which the fixed ladder rungs cannot do. Its scoring
closure is a transcription of `resolution_ladder`'s and `check_against_ladder` asserts on **every
run** that the two agree on the shipped consumer bit for bit before anything is printed; the
"±5 × 2 refinement alone" column reproduces prompt 10's 1.64 / 74.60 ulp exactly, which is the
second check.

Editing `consumer_knot_scheme_scan.py` to take an `extra_sites` parameter was the alternative and
would have avoided the ~25 duplicated lines; it was rejected because that file is prompt 10's
record and is not on this prompt's list either. The precedent for adding a script to this directory
is prompts 08, 09 and 10, each of which did.

### 4. `IMPLEMENTATION CHOICE` — two separate degeneracy guards, and a hard refusal rather than a clamp

`SOURCE_GRID_MIN_SEPARATION = 1e-06` relative in $(1+z)$ keeps any two samples apart by ten times
`DEFAULT_REDSHIFT_RELATIVE_PRECISION`, below which
`Datastore/SQL/ObjectFactories/redshift.py` returns the *same row* for both and the straddle would
silently cease to exist. `SOURCE_GRID_MESH_GUARD = 0.25` additionally keeps anything at least a
quarter of the standoff from a pair member, which bounds the local mesh ratio at 8 whatever the
accident of where a break falls inside its interval — without it, a break a quarter of an interval
from a base sample would put a pair member on top of that sample.

They are separate because they guard different things, and an earlier single tolerance sized by the
standoff was measurably wrong: at 1/8 of an interval it discarded the midpoint of the break's own
interval and cost 1.86 ulp where 1.61 was available, and applied to a *feature* redshift (which has
no pair and needs no mesh protection) it discarded a base sample of the LambdaCDM grid.

A standoff below the separation is **refused with a `ValueError`** rather than clamped, because a
clamp would leave a caller believing it had a tighter pair than it has. On the production grid
neither guard fires: no base sample is displaced, and the closest a base sample comes to a pair
member is **0.0755 of an interval** against the 0.0625 the mesh guard permits.

### 5. `IMPLEMENTATION CHOICE` — the tag is a digest of the grid's values, not of its construction parameters

Prompt §2 item 4 suggests "a short digest of the construction parameters and the protected set".
The digest is taken over the exact bits of the grid's own values instead. It cannot fall out of
step with the grid the way a parameter list can (a later prompt changing `SOURCE_GRID_BREAK_*`
would have to remember to add it to the parameter list; it does not have to remember anything
here), it distinguishes two grids differing in any sample including the protected set alone, and it
is 8 hex characters either way. The cost is that it is opaque: a reader cannot tell from
`SourceRedshiftGrid_1773_303f9ce7` what produced it. The four production labels are tabulated above
for that reason.

### 6. `IMPLEMENTATION CHOICE` — `populate_z_sample` keeps its signature and a new method carries the record

`populate_z_sample` still returns a bare `ndarray`, so every existing caller — including
`ComputeTargets/tests/wkb_reference.py`'s `production_source_z_values`, which transcribes its
arithmetic — is unaffected. `populate_source_grid` returns the `SourceGrid`, and only `main.py`
needs it. Returning a tuple from `populate_z_sample` was the alternative and would have broken
every caller for one of them.

Nothing is tagged `UNINTENDED DRIFT`. No README §2 design fact was touched: the redshift rule
(§2 (i)) is honoured — the only $u\to z$ conversion is a sample *location*, never an equality
comparison, `winnow` matches on `store_id`, and the standoff clears the recovery granularity by
twelve orders; no author convention (§2 (h)) is in the diff; and no equation-of-state module is
imported into `CosmologyConcepts/`.

---

## Verification performed

Machine load averages are quoted with every wall time; this machine has been under erratic external
load (above 140 earlier in the campaign). No conclusion below rests on a clock.

### 1. The grid

`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py`, load
average 6.34, 31.9 s.

| | LambdaCDM | `QCD_Cosmology` |
|---|---|---|
| declared break points in range | 0 | **3** (`17.565806941870026`, `23.197460552819653`, `27.485391822044257` in $u$) |
| feature redshifts | 0 (gated, deviation 2) | **2** ($z = 3406.668974$, $0.30342303$) |
| source grid | **1,732, bit-identical** | 1,732 → **1,773** (+41, +2.37 %) |
| protected samples | 0 | **8** |
| response grid, `winnow(12)` | **145**, unchanged | 148 blind → **156** |

The 41 are $3\times(2+11)+2$. `set(base) ⊆ set(new)` — **no base sample is displaced** — the grid
is strictly descending with no duplicates, and the closest approach between neighbouring samples is
**1.028e-03** relative in $(1+z)$, four orders above the datastore's 1e-07.

The two equality redshifts are the closed form from the public Omegas, not the model's own root
solve, which `LambdaCDM_GenericEOS` performs at `rtol=1e-4` in a private method this prompt may not
touch. They agree: **3406.668974249948 against 3406.6689742499498** (4e-13 relative) and
**0.3034230329964074 against 0.3034230329964074** (exactly), against a grid interval of 2.3e-02.

### 2. The consumer at the crossing (prompt §3 item 6, §4 last row)

Same command. `max |\varphi_{\rm spline} - \varphi_{\rm ref}|` at ten points per production
interval, QCD $k=10^5$:

| samples given to the production consumer | $G_k$ | $T_k$ |
|---|---|---|
| the production grid (shipped, and prompt 09's miss) | 8.1329e-06 rad, **34.11 ulp** | 1.3982e-05 rad, **1876.61 ulp** |
| **the straddling pair alone** (prompt §2 item 2's letter) | 4.1563e-06, **17.43** — 1.96× | 7.0724e-06, **949.24** — 1.98× |
| the ±5 × 2 neighbourhood alone (prompt 10's row, reproduced) | 3.9121e-07, **1.64** | 5.5584e-07, **74.60** |
| **what shipped: pair + ±5 × 2** | **3.8296e-07 rad, 1.61 ulp** | **4.9012e-07 rad, 65.78 ulp** |

Both shipped rows are inside prompt 10 §5's **1e-06 rad** target, $G_k$ also inside its **2-ulp**
one, and both are better than the campaign base (1.9073e-06 / 3.1859e-06 rad) as well as than
`HEAD` (8.1062e-06 / 1.3982e-05). The improvement is **21.2×** and **28.5×** on `HEAD`.

The scorer reproduces `resolution_ladder`'s shipped row exactly — 8.1329e-06 rad / 34.11 ulp /
1,016 samples and 1.3982e-05 / 1876.61 / 1,040 — asserted in code before anything is printed.

### 3. The standoff, scored (prompt §2 item 2's "say what you chose and why")

| standoff, as a fraction of a grid interval | $G_k$ $k=10^5$ | $T_k$ $k=10^5$ |
|---|---|---|
| 1/2 | 1.83 ulp | 74.35 ulp |
| **1/4 — shipped** | **1.61** | **65.78** |
| 1/8 | 1.86 | 88.64 |
| 1/16 | 1.61 | 76.16 |
| 1/32 | 1.67 | 81.77 |
| 1/64 | 1.98 | 96.38 |
| 1e-3 | 2.53 | 123.65 |
| 1e-4 | 2.52 | 123.34 |
| *(no pair at all)* | *1.64* | *74.60* |

**1/4 is the only value that beats the no-pair column on both rows.** The window is bounded above
by the grid (a pair wider than an interval straddles nothing in particular) and below by four
separate floors, of which the third is the binding one: the datastore's 1e-07 relative redshift
resolution; the $u\to z$ recovery granularity, ~ulp$(u)$ = 3.6e-15 relative, which the shipped
5.82e-03 clears by 1.6e12; the consumer's $\varphi$ storage granularity, which is what produces the
plateau at the bottom of the table; and the spline's own conditioning. All four are written into
the constant's comment block.

### 4. All three wavenumbers, both sectors — the trap prompts 02 and 10 both warn about

`--k 1e5 1e7 3e8 --no-standoff-scan`, load average 5.09. "near" is the worst within three grid
intervals of the crossing.

| model | $k$ | sector | shipped grid | cosmology-aware grid |
|---|---|---|---|---|
| QCD | 1e5 | $G_k$ | 8.1329e-06 (34.11; near 34.11) | **3.8296e-07 (1.61; near 1.61)** |
| QCD | 1e5 | $T_k$ | 1.3982e-05 (1876.61; near 1876.61) | **4.9012e-07 (65.78; near 65.78)** |
| QCD | 1e7 | $G_k$ | 1.4251e-05 (0.93; near 0.00) | 1.4251e-05 (0.93; near 0.00) — identical |
| QCD | 1e7 | $T_k$ | 1.2028e-06 (1.26; near 0.15) | 1.2028e-06 (1.26; **near 0.01**) |
| QCD | 3e8 | $G_k$ | 3.2200e-04 (0.66; near 0.00) | 3.2200e-04 (0.66; near 0.00) — identical |
| QCD | 3e8 | $T_k$ | 3.8674e-05 (1.27; near 0.00) | 3.8999e-05 (**1.28**; near 0.00) |
| LambdaCDM | all three | both | — | bit-identical grid, so bit-identical rows |

**One row moves the wrong way and it is recorded rather than argued away**: QCD $T_k$ at
$3\times10^8$ goes 3.8674e-05 → 3.8999e-05 rad, **+0.84 %**, 1.27 → 1.28 ulp of the span. It is not
at the crossing — `near break` is 0.00 both times — and the pair-alone column reproduces it
exactly, so it is the pair and not the neighbourhood; 1 ulp of the span is the floor at which ten
of the twelve production rows already sit. For scale, the $C^0$ knot schemes prompt 10 scored moved
this same row by **1.70×**.

### 5. `[03-derivative-pad-clamp-on-coarse-grids]` (prompt §2 item 5) — measured, does not bind, unchanged

`h_lo = min(x[1]-x[0], -\log(0.9)/12, 0.05(x[-1]-x[0])/12)` in $x = \log(1+z)$:

| grid | first $x$ interval | floor cap | fraction cap | `h_lo` | binds |
|---|---|---|---|---|---|
| base (1,732) | 2.115878e-03 | 8.780043e-03 | 1.561272e-01 | 2.115878e-03 | no |
| cosmology-aware (1,773) | 2.115878e-03 | 8.780043e-03 | 1.561272e-01 | 2.115878e-03 | no |

The same float. The lowest protected sample is matter–$\Lambda$ equality at $z=0.3034$, **48.19
base intervals above** $z_{\rm end}=0.1$; the bottom 40 samples are element-for-element the base
grid's, so neither endpoint nor either end's spacing moves. Nothing was changed.

### 6. The tree

| suite | before (`b580a78`) | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.55 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 361 OK | **380 OK** (151.3 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (14.5 s) | every module except `test_3bessel_analytic` |

The `LiouvilleGreen` figure is the **fast set**, as prompts 02–08 and 10 used: `test_3bessel_analytic`
is excluded and the full set is 148 (prompt 09 ran it). This commit touches no `LiouvilleGreen`
file and no file any of its tests import. No count falls; the +19 are `test_source_grid.py`.

`black` is clean on every file in the diff. `main.py` parses under `ast` (it cannot be imported).

### 7. What was *not* run

**No pipeline run.** `main.py` needs Ray and a datastore; the grid hunk's two new helpers are
executed by the test suite through `load_main_py_functions`, and the grid they produce is scored
by the consumer script, but **the assembled `run_pipeline` body is checked by reading and by `ast`,
not by running**. In particular the second `convert_to_redshifts` call that recovers the protected
points as `redshift` objects has not been exercised against a real datastore. That is the same
limitation `ComputeTargets/tests/test_main_plumbing.py`'s own docstring records, and the natural
place to discharge it is `docs/source-remediation-verification/scoped_pipeline_run.py`, which is
not among the files this prompt may touch.

---

## Observations not acted on

1. **`BackgroundModel` is not keyed on the source grid, and prompt 11 has just made that matter.**
   Its lookup filters on `(cosmology, atol, rtol)` plus `LargestSourceZTag`, `SmallestSourceZTag`
   and `SourceSamplesPerLog10ZTag` — **all three unchanged by this commit**, since the grid's
   endpoints and its samples-per-decade are unchanged — and its factory does not filter on
   `z_sample`. So a pre-prompt-11 datastore returns its **1,732-node** `BackgroundModel` for the new
   **1,773-node** grid, the one surviving row in an otherwise invalidated store. Opened below as
   `[11-background-model-not-keyed-on-the-source-grid]`. Not fixed here: adding a tag to
   `BackgroundModel`'s lookup is a separate key change with its own blast radius, and
   `BackgroundModel`'s `object_get` is outside the `:520-590` hunk this prompt may touch.

2. **`CosmologyConcepts.redshift.check_zsample` has no callers.** It exists to assert that two
   objects share a sample grid, which is exactly the check that would have caught observation 1,
   and nothing in the tree calls it. Noted rather than opened: it is a dormant helper, not a defect,
   and observation 1 carries the actionable half.

3. **`ComputeTargets/tests/wkb_reference.py::production_source_z_values` transcribes
   `populate_z_sample`'s arithmetic** and therefore still produces the *base* grid. Every reference
   measurement in the tree — prompt 10's scan included — is taken on it, which is correct and is why
   the comparisons in §2 above are like for like. But once a pipeline actually runs on the
   cosmology-aware grid, the QCD half of `wkb_reference_data.json` will be built on a different
   grid from the one that module hands out, and the two will have to be reconciled. Not this
   prompt's file, and no number moves until a run happens. Whoever regenerates the QCD fixture next
   should decide whether `production_source_z_values` gains the protected set.

4. **`docs/gktk-remedial/verify_production_path.py` still builds its own $G_k$ consumer**
   (`[02-verify-script-builds-its-own-Gk-consumer]`, open on the `phase-representation` board).
   Prompt 10's log flagged that this means six of §3.5's twelve rows would not exercise the
   production $G_k$ construction if prompt 11 changed the source grid. It has; the flag stands.

5. **The straddling pair is worth 2 % on $G_k$ and 12 % on $T_k$, and costs 6 samples in 1,773.**
   A later reader weighing simplicity against that could drop it and keep the neighbourhood alone
   (1.64 / 74.60 ulp, both still inside the target). It was kept because prompt §2 item 2 asks for
   it, because it is what makes "which side of the step is this sample on?" answerable without an
   equality comparison on a recovered redshift, and because it is the only part of the design that
   scales correctly when prompt 12 changes the density. Not an issue; a recorded trade.

---

## State handed to the next prompt

1. **The production grids, and the four tag labels.** LambdaCDM: **1,732** source samples,
   bit-identical to the campaign's, tagged `SourceRedshiftGrid_1732_0960e169`; **145** response
   samples, `ResponseRedshiftGrid_145_69050b4c`. QCD: **1,773** source samples (1,732 base + 39
   break-related + 2 features), tagged `SourceRedshiftGrid_1773_303f9ce7`; **156** response samples
   (148 under a blind stride plus the 8 protected), `ResponseRedshiftGrid_156_197b46de`.
   `T_Z_REPRESENTATION_VERSION` is **5**, untouched.

2. **The three constants prompt 12 will want to move, and the one it must not move blindly.**
   `SOURCE_GRID_BREAK_HALF_WIDTH = 5` and `SOURCE_GRID_BREAK_REFINEMENT = 2` are prompt 10's
   measurement and are local to a break. `SOURCE_GRID_BREAK_STANDOFF = 0.25` is **a fraction of a
   grid interval**, so it follows `samples_per_log10z` automatically — that is deliberate, and
   replacing it with an absolute relative number is measured in §3 above to cost a factor of 1.5 to
   1.7. `samples_per_log10z` itself is prompt 12's question and was not touched.

3. **The reproduction, one command, ~32 s, no Ray and no datastore:**
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py
   PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py \
       --k 1e5 1e7 3e8 --no-standoff-scan
   ```
   `--models`, `--k`, `--json` and `--no-standoff-scan` narrow it. Prompt 10's scan is unchanged and
   its command is unchanged. The QCD reference fixture is untouched by this commit and its
   regeneration command is still
   `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run`.

4. **What the consumer now reads at the crossing**, for anyone re-measuring §3.5 of
   `docs/gktk-remedial-verification.md`: QCD $k=10^5$ **3.8296e-07 rad (1.61 ulp)** in $G_k$ and
   **4.9012e-07 rad (65.78 ulp)** in $T_k$, against 8.1062e-06 / 1.3982e-05 at `HEAD` and
   1.9073e-06 / 3.1859e-06 at the campaign base. Those are grid-scored figures from the script
   above, **not** a re-run of `verify_production_path.py`, which needs a pipeline run on the new
   grid.

5. **The regeneration is owed and is quantified in §5 of "What shipped".** Eight stored object
   types; 15,020 objects and 6 m 35 s plus an unfinished `QuadSourceIntegral` stage for a measured
   5×5-wavenumber single-model run; production is ×10 in each wavenumber sample and two models. The
   LambdaCDM half is recoverable by relabelling two `store_tag` rows, **but only on a datastore that
   holds no pre-prompt-11 QCD objects**.

6. **Two open issues move.** `[10-consumer-phi-unresolved-at-the-eos-crossing]` is **closed** by
   this commit (board §4), and `[11-background-model-not-keyed-on-the-source-grid]` opens in §3.
   `[03-derivative-pad-clamp-on-coarse-grids]` was measured and is unchanged;
   `[02-consumer-phi-below-the-storage-granularity]` is untouched and is still 100 % of the two QCD
   $G_k$ `theta_deriv` rows that miss $10^{-6}$ away from $k=10^5$ — no grid change touches it.

7. **Prompt 12 inherits a grid whose density is still uniform.** Nothing here changes
   `samples_per_log10z`, and the prompt's "Do not touch" says so explicitly. What prompt 12 gains is
   that the grid construction now has a place to put a density criterion, and that the break
   neighbourhoods are already handled locally, so a density measurement taken away from a crossing
   is no longer contaminated by one.
