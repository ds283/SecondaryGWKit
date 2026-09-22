# Datastore read-back campaign — implementation state

**Last updated:** 2026-09-22 · **Status: STARTED — 1 of 1 written prompts landed (01).** Prompt 01
landed the one-line `QuadSourceIntegral` fix, the static guard in the new `Datastore/tests/` root,
and the audit of every object factory. **Stored `QuadSourceIntegral` rows read back**; the
interrupted `handover-A3-baseline-lambdacdm` run is resumable, which is the user's to do. Two issues
are open in §3, one of them the second defect of the same class that the audit found and this
campaign's rules forbade it to fix.

**Campaign:** [`README.md`](README.md) ·
**Datastore:** `var/datastores/handover-A3-baseline-lambdacdm.sqlite` and its manifest ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.10

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit** —
> the row added, moved or deleted, and the count and date in its header corrected. An issue owned
> by another board is moved to **that** board's §4 and its row deleted from the index. The index is
> an index: one line per issue, pointing here. Where the two disagree, this board is right.
> See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [The `QuadSourceIntegral` read-back](01-quadsourceintegral-readback.md) | **R1** | Opus 5 | ✍️ yes | ✅ 2026-09-22 | *"Add the missing numeric_quad column to the QuadSourceIntegral read"* | [`logs/01-…`](logs/01-quadsourceintegral-readback.md) |

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **REMEDY** | Add `table.c.numeric_quad` to the `SELECT` that `sqla_QuadSourceIntegral_factory.build()` executes. One line; the schema and the stored data are correct. | 01 | ✅ **Done, 2026-09-22.** `QuadSourceIntegral.py:246`. Row `serial=20806` of shard 0 reads back through the real `Datastore.object_get` with `numeric_quad = -1.2033078530407802e-11`, equal as a float to the stored value; the same read against the unfixed tree raises `Could not locate column in row for column 'numeric_quad'`. Store read-only throughout; shard mtimes unchanged at 2026-09-21 04:32; backup untouched; pipeline not started. |
| R2 | **FACILITY** | The guard: a test that catches the whole class — `build()` reading a column its own `SELECT` does not request — with no Ray and no datastore. | 01 | ✅ **Done, 2026-09-22.** `Datastore/tests/test_factory_select_columns.py`, 516 lines, **4 tests**, 0.3 s. Shape **(a)**, static `ast` consistency, as the prompt recommended, and it did **not** have to be weakened to pass. Two soundness corrections were forced by running it: a dynamically named column may clear a binding but may never convict one, and a variable rebound part-way through a `build()` is two bindings — merging them produced a false positive on `sqla_BackgroundModelFactory`. Beyond the subset assertion the guard pins its own reach (`EXPECTED_COVERAGE`, 20 bindings), its one declared blind spot (`KNOWN_BLIND`) and the currency of `KNOWN_UNFIXED`, so that a refactor which blinds it fails rather than passing vacuously. **Deliberate-breakage record:** unfixed tree → `FAILED (failures=1)` naming `numeric_quad`; fixed tree → `OK`. Six holes stated in the module docstring, three of them live in this tree. |
| R3 | **AUDIT** | Run the guard across every object factory; **open** what it finds, fix nothing. | 01 | ✅ **Done, 2026-09-22.** 21 modules, **38 classes defining `build()`**, of which **24 read nothing off a query result** (verified independently by scanning each for an attribute read, an `_mapping` subscript or a `getattr` on a row-named variable — none has one) and **14 read rows** across **21 bindings**, 20 decidable and 1 declared blind. **One further defect found, not fixed:** `sqla_BackgroundModelValue_factory.build` reads `row_data.Hubble` where the column is `Hubble_GeV` — `[01-backgroundmodelvalue-hubble]` in §3. It is **not** on a path reachable from a running pipeline or a resume, so prompt §7's third stop condition was not met. Everything else is clean. The campaign README's "22 factories, 12 reading `row_data`" counts registry entries; the figures here are class by class and supersede it. |

---

## 3. Active and unresolved issues

- **[01-backgroundmodelvalue-hubble]** *(opened 2026-09-22 by prompt 01)* —
  `Datastore/SQL/ObjectFactories/BackgroundModel.py:1003` and `:1005`, in
  `sqla_BackgroundModelValue_factory.build()`, read **`row_data.Hubble`** on the branch taken when
  the row already exists: the check that the stored Hubble rate matches the one being written, and
  the `ValueError` that reports a mismatch. The column is **`Hubble_GeV`** — the `SELECT` at `:926`
  requests `Hubble_GeV`, and the table declared by `register()` has no `Hubble` column at all. Any
  execution of that branch raises `NoSuchColumnError`. It is the **same defect class** as
  `[R1]`, found by the guard `[R2]` on the sweep `[R3]`, and deliberately **not fixed** (campaign
  README §5 rule 4 and prompt §4): it is carried in the guard's `KNOWN_UNFIXED` so the suite is
  green on this tree and fails the moment the entry is stale.
  **Reachability: not reachable from a running pipeline, and not from a resume** — which is why
  prompt §7's third stop condition was not met and the user was not interrupted.
  `BackgroundModel.build()` does not go through this factory: it reads its sample rows with its own
  `SELECT` over `tables["BackgroundModelValue"]` (`:454-486`) and constructs the
  `BackgroundModelValue` objects itself, and nothing in the tree calls
  `object_get("BackgroundModelValue", ...)`. Corroboration that the whole `build()` is dead: its
  `row_data is None` branch inserts with the key **`"wkb_serial"`** (`:964`), which is not a column
  of this table either, so that branch cannot ever have run successfully. Two defects, neither ever
  fired. **Impact:** prospective. The first caller of `object_get("BackgroundModelValue", ...)` — a
  diagnostic, an extract script, a per-sample lookup — hits it on the first stored row, and what it
  defeats is a data-integrity check. **Next step:** `row_data.Hubble` → `row_data.Hubble_GeV * GeV`
  in both places (the comparison is against `Hubble`, already converted to internal units above),
  and separately decide whether `"wkb_serial"` is a typo for `"model_serial"` or evidence the
  `build()` should be deleted. One prompt, here or in the next campaign that touches `Datastore/`;
  strike the `KNOWN_UNFIXED` entry in the same commit. Measurement:
  [`logs/01-quadsourceintegral-readback.md`](logs/01-quadsourceintegral-readback.md), "The audit".
  Indexed at `docs/OPEN_ISSUES.md` §1.10.

- **[01-read-batch-is-outside-the-guard]** *(opened 2026-09-22 by prompt 01)* — the guard covers
  `build()` only, because that is what prompt §3 scopes it to. **`read_batch()` reads rows too and
  is checked by nothing.** `sqla_QuadSourceIntegral_factory.read_batch` (`:530`) alone reads some
  sixty attributes off a `row` that is the parameter of a nested `make_object`, from a query
  extended by `add_columns` in two loops — neither shape is one the static analyser can follow, so
  extending the existing check to it is not a matter of widening a glob. The defect class is equally
  possible there, and `read_batch` is on the **live** path (`Datastore.object_read_batch`), not only
  on resume: it has been exercised, which is weak evidence that it is currently clean but is no
  guarantee for the columns a given call does not touch. **Impact:** the guard's coverage is
  narrower than "the factories are guarded" would suggest, and a later reader could take the
  green suite for more than it says. The module docstring and the log both state the limit; this
  issue is so that it is not only stated. **Next step:** either teach the analyser to link a nested
  helper's row parameter to the query it is called with, or write prompt §3 shape **(b)**, the
  in-memory round trip, for `read_batch`. Not attempted in prompt 01: widening the guard is a change
  with its own risk and would have obscured the one-line diff that commit exists to prove.
  Measurement: [`logs/01-quadsourceintegral-readback.md`](logs/01-quadsourceintegral-readback.md),
  "What it cannot see" item 3. Indexed at `docs/OPEN_ISSUES.md` §1.10.

---

## 4. Resolved issues

*None yet. `[R1]` was never indexed as an issue: the campaign was created to fix it and prompt 01
closed it in the same commit that opened the board. Its record is item R1 in §2 above and the
`run_history` entry of `var/datastores/handover-A3-baseline-lambdacdm.manifest.json`.*

---

## 5. Baselines

Campaign README §4's, plus the two this campaign measured.

| Suite | Count | Measured |
|---|---|---|
| `ComputeTargets` | 552 | `ab7079c`; `test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a known wall-clock flake — re-run that module alone before attributing a failure |
| `CosmologyModels` | 39 | `ab7079c` |
| `LiouvilleGreen` | 148 (skipped=1) | `ab7079c` |
| `AdaptiveLevin` | **32** | **first baselined 2026-09-22 by prompt 01**, on the pre-change tree |
| `Datastore` | **4** | **created 2026-09-22 by prompt 01** |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .
```
