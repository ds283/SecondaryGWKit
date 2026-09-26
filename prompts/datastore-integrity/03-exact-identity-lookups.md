# Prompt 03 — identity lookups match exactly, and a duplicate raises

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 02 has landed; `git status` is clean.
**Closes:** `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` (tolerance-convergence
board), `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` (store-fingerprint board) and
`[03-integrationsolver-stepping-minimum-lookup]` (GkTk-remedial board), each on its own board's §4,
with its index row deleted. **Narrows:** `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`
(tolerance-convergence). **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. Each change is a few lines; the care is in changing the match
and nothing else, and in naming every caller whose behaviour moves.

**Read first:**

1. [`README.md`](README.md): §0, §1, §5 and §6.1. Decision 6.1.3 is the user's: the exit-time
   tolerance lookup becomes an **exact match**. Do not reopen it.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §3.5–§3.9.
3. `docs/datastore-integrity-audit/lookup_keys/probe_04_exit_time_and_solver.py`,
   `probe_04b_exact_match_fix_shape.py` and `probe_06_oneloop_tags_and_redshift_zero.py`, with their
   `.out` files: the measurements this prompt turns into tests.
4. `Datastore/SQL/ObjectFactories/wavenumber.py` `sqla_wavenumber_exit_time_factory` (`register`,
   `build`); `integration_metadata.py`; `redshift.py`; `tolerance.py`; and the `.scalar()` lookups
   listed in L3.
5. The three issues' entries on their boards, and `[05a-…]`'s.

---

## 1. What is wanted

Four lookups on replicated and metadata classes can serve a row that is not the one asked for, or
hide a second row that should not exist (audit §3.5–§3.9):
- the exit time takes the **loosest stored row at least as tight** as the request, reports the
  stored pair as if it were the requested one, and raises `MultipleResultsFound` once a tighter row
  has been stored; its query runs six times inside a loop;
- `IntegrationSolver` takes the first row with `stepping >=` the request, so an exact row can be
  passed over for a higher-order one;
- eight identity lookups take `.scalar()`, which returns the first row and never notices a second;
- `redshift` at `z = 0` divides by zero, matches nothing, and inserts a new row on every call.

After this prompt each of them matches exactly one row or none, and raises on two.

---

## 2. What to change

**L1 — the exit time.** In `sqla_wavenumber_exit_time_factory.build`:
- match `atol_serial == target_atol.store_id`, `rtol_serial == target_rtol.store_id` and
  `stepping == target_stepping` — **serial equality**, which is how every other keyed type matches a
  tolerance, because `tolerance.build` has already mapped the value to one serial (audit §3.5);
- delete the `order_by`, which `one_or_none()` makes dead;
- move `row_data = conn.execute(query).one_or_none()` out of the sub-horizon loop, so it runs once;
- the joins to the tolerance table may stay if they feed the returned `log10_tol` values; say which
  you did.
- **Do not change** the registration's `"stepping": "minimum"`. It is recorded in the schema
  witness `Datastore/tests/data/schema_at_base.json`, which this prompt may not change (D5 is
  undecided, and prompt 06 is the first to change the witness). Nothing reads it. List it under
  "Observations not acted on" for prompt 06.

**L2 — `IntegrationSolver`.** In `integration_metadata.py`, `stepping == stepping` and
`.scalar_one_or_none()`. Same restriction on its registration's `"stepping": "minimum"`.

**L3 — duplicates raise.** `.scalar()` → `.scalar_one_or_none()` at the identity lookups of
`tolerance`, `store_tag`, `version`, `IntegrationSolver` (L2), `GkSourcePolicy`, `QuadSourcePolicy`,
`LambdaCDM` and `QCD_Cosmology` (audit §3.8 lists the lines). Change no filter. `.scalar()` calls
elsewhere — counts, `BackgroundModel`'s value checks — are not identity lookups and are left alone;
list any you judged either way in the log.

**L4 — `redshift` at `z = 0`.** An explicit branch: for `z == 0` match `z == 0`; for every other
`z` the relative match is unchanged. *Prompt's choice:* this rather than `max(|z|, 1)`, which would
loosen the match for every `z < 1`. If you choose otherwise, say why.

**L5 — who moves.** Establish, and record in the log, every caller of the exit-time and
`IntegrationSolver` lookups and whether its result changes:
- `main.py` and the `extract_*.py` scripts request the pair production stores, so they are
  expected to be unchanged — show it;
- `docs/source-remediation-verification/analyse_greens_and_source.py` and
  `run_quadsource_integrals.py` request `(1e-10, 1e-8)` against a stored `(1e-10, 1e-9)`. Today they
  are silently served the tighter row; after L1 they miss. **Do not edit them**: they are documents
  of the tree they ran on. Narrow `[05a-…]` on its board to say so.

---

## 3. Tests — a new module in `Datastore/tests/`, no Ray, nothing under `var/`

Build the schema in memory with `build_schema(MetaData(), Datastore.SQL.Datastore._factories)` and
call the real factory `build()`, as the probes do. At minimum:

1. **Exit time.** Stored `(1e-10, 1e-9)`: the exact request hits and reports its own pair;
   `(1e-10, 1e-8)`, `(1e-8, 1e-8)` and `(1e-12, 1e-9)` miss. With `(1e-12, 1e-9)` stored as well,
   each exact request is served its own row and nothing raises. Stepping 1 stored: request 0 misses,
   request 1 hits.
2. **One statement.** A `build()` of the exit time executes exactly one `SELECT` (count with an
   `event.listen(engine, "before_cursor_execute", …)`).
3. **`IntegrationSolver`.** Stored `{serial 1: 8, serial 2: 4}`: request 4 → serial 2, request 8 →
   serial 1, request 6 inserts a new row.
4. **Duplicates raise.** For each class in L3, two rows with the same identity make `build()` raise
   `MultipleResultsFound`.
5. **`redshift` at zero.** Three builds at `z = 0` give one row; a build at `z = 0.1` is unchanged.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) put `<=` back on `rtol` in the exit-time lookup;
- (ii) put the query back inside the loop;
- (iii) `>=` back in `IntegrationSolver`;
- (iv) `.scalar()` back on `tolerance`;
- (v) remove the zero branch.

Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. §3's tests in a new module; no Ray, nothing under `var/`; every existing test passes unmodified.
2. The deliberate-breakage record, (i)–(v).
3. The L5 table of callers, each "unchanged" or "now misses", with the evidence.
4. The three issues closed on their own boards' §4, each with a closing note naming this commit;
   their index rows deleted; `[05a-…]` narrowed on its board and in the index; the count and date
   corrected.
5. `black --check` clean. On the board: the §1 row, items L1–L5, §3.1, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- A production caller (`main.py`, an `extract_*.py` script) would miss after L1 or L2.
- Changing a `.scalar()` makes an existing test fail. That is a duplicate in a fixture, and whether
  the fixture or the lookup is wrong is not this prompt's to decide.
- An existing test, or the schema witness, would have to change.

---

## 6. What this prompt does not do

- It does not change any registration, column or schema; the `"stepping": "minimum"` flags are
  prompt 06's.
- It does not change any sharded factory's lookup, any read path (prompt 05) or the replicated
  write (01–02).
- It does not edit the two verification scripts, `main.py` or any `extract_*.py` script.

---

## 7. The log and the board

`logs/03-exact-identity-lookups.md`, using the template in README §5.1, with the L5 table and the
list of `.scalar()` calls judged not to be identity lookups.

`IMPLEMENTATION_STATE.md`: the §1 row for 03, items L1–L5, §3.1 and §5.
