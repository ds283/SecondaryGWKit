# Prompt 16 — Retire the samples-per-decade tag and the arguments it fed

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Narrows:** `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]` on this
campaign's board §3 — its *tag* half, which this prompt closes; its *wavenumber-set* half is the
user's explicit no-action decision (§1 below) and stays recorded
**Implements:** the retirement prompt 14 deferred and prompt 15 did not carry out, because the
instruction lived only in prompt 14's text and prompt 15 was never given it
**Depends on:** 15 (which is what made the tag false). **Blocks:** nothing.
**Recommended model:** **Sonnet** — a mechanical removal across one file's tagging hunks, with one
real question (does removing a tag from a lookup broaden or narrow it?) that must be *measured*.

**Files you may touch:** `main.py` (**the argument-parsing and tagging hunks only**),
`Datastore/SQL/ObjectFactories/BackgroundModel.py` (**its module docstring only** — no filter, no
column, no `build()` logic), `ComputeTargets/tests/test_run_identity.py`,
`ComputeTargets/tests/test_source_grid.py`, `docs/source-remediation-verification/run_quadsource_integrals.py`
(**only to keep it consistent with `main.py`'s tagging; see §2 item 5**), a new test module if you
want one, plus this campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyConcepts/`; `CosmologyModels/`; `ComputeTargets/BackgroundModel.py`;
any `extract_*.py` — prompt 14's exception was prompt 14's; `QCD_EOS.py`; `Quadrature/`; any
compute target; any `Datastore/` factory's *code*; the grid, its construction, its density or
`SOURCE_GRID_CONSTRUCTION_VERSION`.

**Read first:** `prompts/qcd-background-audit/logs/15-equidistribute-the-source-grid.md` and
`logs/14-key-the-source-grid-construction.md` §4 (what `source_samples_per_log10z` was found to
feed); board entry `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]`;
`main.py`'s tag block at `:975-1040` and the 28 `SourceSamplesPerLog10ZTag` call sites.

---

## 1. What is wrong, and the two decisions already taken

Prompt 15 replaced the source grid's uniform density with a measured curvature criterion, capped so
that no interval is ever coarser than the uniform lattice would have put there. After that commit:

- **`SourceSamplesPerLog10ZTag` is false as a description.** It labels stored objects
  `SourceSamplesPerLog10Z_100` as though the grid had 100 samples per decade of $z$. It does not:
  QCD carries 1,996 samples and LambdaCDM 1,778, against the 1,773 and 1,732 of the lattice that
  number now merely *bounds*. **Impact on correctness: none** — since prompt 14 the grid's real
  identity is the content digest and `SOURCE_GRID_CONSTRUCTION_VERSION`, both of which are in the
  lookup key. The defect is that a reader of a store is told something untrue.
- **`delta_logz = 1.0 / float(source_samples_per_log10z)` at `main.py:1203` and `:1820` is
  vestigial.** Prompt 14 traced every consumer and established that nothing computes from it; both
  sites sit inside `object_get` calls that also carry the tag, so they are the same hunks.

**Two decisions the user has taken. Neither is open, and you do not re-litigate either.**

1. **`--source-samples-log10z` stays.** The command-line switch is *not* retired and must not be.
   Its job changed rather than ended: it no longer sets the grid's density, it sets the **base
   lattice the cap is measured against** — "no interval may ever be coarser than this". That makes
   it more load-bearing than before, not less. Its four surviving uses (`main.py:908`, `:919`,
   `:930`, `:3430`) all stay.
2. **Nothing is to be done about the grid's dependence on the wavenumber sample.** The user's
   words: *"we have to go further back for small `k`. I don't think there is anything we need to do
   about this; we just need to bear it in mind when we're doing a concrete calculation."* So **do
   not** add a `SourceKSampleTag`, do not fold a wavenumber digest into the construction version,
   and do not otherwise act on that half of the issue. Record the decision on the board in the
   user's framing so a later reader does not reopen it as an oversight.

## 2. The change

1. **Stop writing the tag.** Remove its declaration (`main.py:979`), its construction (`:1004`) and
   all 28 tag-list uses. Stored objects that already carry it keep it — it is harmless history and
   removing it from an existing store would destroy archival information for no gain.

2. **Establish, by measurement, whether removing a tag from a lookup broadens or narrows it**, and
   say which in the log. This is the one question in the prompt that is not mechanical. If
   `object_get` requires a row to carry *at least* the listed tags, removing one **broadens** the
   query and a pre-prompt-16 store still resolves; if it requires an exact set, it does not.
   **Do not assume either way** — read the implementation and demonstrate it with a test. Whether
   this commit carries a regeneration depends entirely on the answer, and the log must state the
   answer and the consequence plainly.

3. **Remove the two `delta_logz` arguments** at `:1203` and `:1820`. Confirm, do not assume, that
   the receiving call still has a sensible default and that nothing downstream reads the value; if
   either site turns out to need a step, say so and stop rather than inventing one.

4. **Correct the switch's help text.** `--source-samples-log10z` currently reads *"specify number of
   z-sample points per log10(z) for the source term"*, which is now the misleading sentence in a
   different place. It should say what the number actually does: it sets the base lattice, and the
   grid is refined above it by the curvature criterion and never coarsened below it.

5. **`docs/source-remediation-verification/run_quadsource_integrals.py:207` builds the same tag
   label** and writes with it. It is a scoped verification harness, not production. Decide whether
   it follows `main.py` or is left alone, **state the reason**, and note that
   `CLAUDE.md`'s "verification documents are additive" rule governs the *documents*, not
   necessarily the scripts beside them. Either answer is defensible; an unexamined one is not.

6. **Do not bump `SOURCE_GRID_CONSTRUCTION_VERSION`.** The construction does not change here — only
   what is said about it. State in the log why a bump would be wrong, so that a later reader does
   not read the absence as an oversight.

7. **The two docstrings that describe the old lookup key** —
   `Datastore/SQL/ObjectFactories/BackgroundModel.py:28` and
   `ComputeTargets/tests/test_run_identity.py:10` — narrate a tag list that no longer exists. Update
   them to describe what the key is now. Both are prose; change no logic.

## 3. Tests

1. **The tag is gone from every writer.** By `ast` over `main.py`: no tag list mentions it, nothing
   constructs the label. Follow prompt 14's `TestMainPyNamesItsRun` shape.
2. **The grid did not move.** QCD 1,996 / `a2c32f67` and LambdaCDM 1,778 / `60a3205a`, element for
   element — this prompt touches no construction and the digests must prove it.
3. **A pre-prompt-16 store still resolves**, or does not, per §2 item 2 — whichever you measured.
   This is the test that carries the answer and it must exercise a real lookup, not a docstring.
4. **`SOURCE_GRID_CONSTRUCTION_VERSION` is unchanged at 2**, and `T_Z_REPRESENTATION_VERSION` at 6.
5. **Prompt 14's run-identity tests still pass unmodified** except for the docstring and the tag
   list of §2 item 7 — the run label and construction version are untouched by this prompt.

## 4. Acceptance

| Quantity | Requirement |
|---|---|
| `SourceSamplesPerLog10ZTag` in any writer or filter | **none** |
| `delta_logz` at `:1203`, `:1820` | **removed** |
| `--source-samples-log10z` | **kept**, help text corrected |
| Source grid, both models | **element-for-element identical**, same digests |
| `SOURCE_GRID_CONSTRUCTION_VERSION` | **unchanged at 2** |
| `T_Z_REPRESENTATION_VERSION` | **unchanged at 6** |
| Pre-prompt-16 store | behaviour **measured and stated**, not assumed |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/16-retire-the-samples-per-decade-tag.md` per README §5.1, carrying: whether removing a
tag from a lookup broadens or narrows it and how you established it; whether this commit carries a
regeneration; the decision on `run_quadsource_integrals.py` and its reason; and why
`SOURCE_GRID_CONSTRUCTION_VERSION` is not bumped.

Narrow `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]` on the board §3 — the
tag half is closed here, the wavenumber-set half is the user's stated no-action decision and stays
recorded in the user's framing — and update `docs/OPEN_ISSUES.md` in the same commit with its count
and date corrected. Update the board's §1 row and header: the campaign runs to 16 prompts.

Commit subject, or something equally specific:
`Retire the samples-per-decade tag the grid no longer obeys`
