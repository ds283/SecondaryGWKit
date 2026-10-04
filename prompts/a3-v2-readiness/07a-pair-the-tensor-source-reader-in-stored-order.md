# Prompt 07a — pair the tensor-source reader's k in stored order

**Campaign:** [`README.md`](README.md) · **Board items:** **E1**–**E3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-04, after 07 landed (`a6c8f39`) and was reviewed (`8dab816`), at the user's
request. **Gate:** 07 landed and reviewed; `git status` is clean; nothing is `running` in the
registry; **07's resumed store has not been retired** (E3 reads it).
**Closes:** `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`.
**Opens:** nothing expected (§6 names one thing to record).
**Recommended model:** **Sonnet**. Two small pure helpers, two call sites in one reader, one new
test module, and one short read-only run of the reader under Ray.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 and §5.1.
2. This board's §3.2 entry for `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`,
   and the *Orchestrator review of prompt 07*.
3. `docs/a3-v2-readiness-verification.md` §2.3, and §2.3.2 above all: the failure, measured.
4. `extract_tensor_source_data.py`, in full: `plot_tensor_source`, `run_pipeline` and
   `build_tensor_source_work`.
5. `extract_common.py`: `require_background` and `NoBackgroundModel`, the pattern for E2.
6. `CosmologyConcepts/wavenumber.py`: `wavenumber_exit_time` (`__eq__`, `__lt__`, `__hash__`) and
   `wavenumber_exit_time_array`.
7. `main.py`, the `--quad-source-queue` block (near line 1934), where the stored pairs come from.
8. The tests that already read this reader, all of which must pass unchanged:
   `ComputeTargets/tests/test_quadsource_parent_main.py` (`TestTheReaders`),
   `ComputeTargets/tests/test_reader_read_only.py`, `ComputeTargets/tests/test_run_identity.py` and
   `ComputeTargets/tests/test_main_plumbing.py`.
9. `CLAUDE.md`, "Long-running jobs — the run registry". E3 is short, but it runs under Ray beside
   the user's ChamPBH cluster.

---

## 1. What is wrong

Read by the orchestrator at `8dab816`, from the code and from 07's measurements. Pass-through facts;
you need not rediscover them.

- **The stored orientation is q ≤ r** [read]. `main.py` builds the `QuadSource` work list from
  `itertools.combinations_with_replacement(source_k_exit_times, 2)` (line 1938), and
  `source_k_exit_times` is a `wavenumber_exit_time_array` (line 1059). That class sorts its members
  ascending by `k` and drops duplicates (`CosmologyConcepts/wavenumber.py:882`). So every stored row
  has `q.k <= r.k`.
- **The reader does not keep that order** [read]. `run_pipeline` draws
  `k_subsample = sample(list(source_k_exit_times), k=round(0.9 n + 0.5))` (line 364), which returns
  the k in random order. `work_grid = list(itertools.combinations_with_replacement(k_subsample, 2))`
  (line 390) then pairs them in that order. Unless the draw happens to be ascending, some pairs have
  q > r.
- **A q > r pair matches nothing, silently** [read]. The lookup is labels-only (lines 378–386). On no
  match, `QuadSource.build` returns a labels-only miss: an unpopulated `QuadSource` whose `available`
  is false. `plot_tensor_source` reads `source.values` at once (line 135), which raises
  `RuntimeError: values has not yet been populated`. That message names neither q nor r.
- **Measured** (verification §2.3.2). On 07's resumed store, read-only, the reader exited 1 after
  45 s. With 3 k the sample is all three in random order, so it passes 1 time in 6; the 2026-10-01
  pass was such a draw. With A3 v2's 8 k it would pass about 1 time in 8! = 40 320. A wrapper that
  only sorts the draw (`docs/a3-v2-readiness/rehearsal/tensor_source_sorted_sample.py`, **not the
  shipped reader**) read all six rows and exited 0.
- **The other readers are not affected** [read]. `extract_QuadSourceIntegral_data.py` builds its
  pairs from `combinations_with_replacement(source_k_exit_times, 2)` over the sorted array (line
  1136) and then samples *pairs*, so every pair it asks for is in stored order. The other four
  readers look up one k at a time.

---

## 2. What to change

**E1 — pairs in stored order.** In `extract_common.py`, a pure function **`quad_source_pairs(k_exits)`**:
- It takes any iterable of `wavenumber_exit_time` and returns the list
  `list(itertools.combinations_with_replacement(wavenumber_exit_time_array(k_exits), 2))`. It orders
  through `wavenumber_exit_time_array` on purpose: that is the class `main.py`'s pairs come from, so
  the reader's order and the build's are one order by construction, not two that happen to agree.
- Its docstring says why: the store holds `QuadSource` for q ≤ r only (`main.py`'s
  `--quad-source-queue`), and a labels-only lookup of a q > r pair matches nothing.
- In `extract_tensor_source_data.py`, `run_pipeline` builds `work_grid = quad_source_pairs(k_subsample)`.
  The `sample` draw itself is unchanged: how many k are drawn is the reader's design, not this
  defect. `itertools` stays imported only if something else uses it.

**E2 — a miss names the pair it asked for.** In `extract_common.py`, a pure function
**`require_quad_source(source)`**:
- If `source.available` is true it returns nothing.
- Otherwise it raises `RuntimeError`, naming q and r by `k_inv_Mpc` and by `store_id` (the
  `wavenumber_exit_time`s the object was built with, `source._q_exit` and `source._r_exit`), and
  saying that no stored `QuadSource` matched them under the run's tags. *Wording is your choice*;
  it must contain both `k` values and the words "no stored QuadSource".
- In `extract_tensor_source_data.py`, `plot_tensor_source` calls `require_quad_source(source)`
  **before** its first read of `source.values` or `source.functions`.
- A miss is an error, not a skip. Every pair the reader asks for is built from k the store itself
  holds, so a miss means the store lacks a row it should have, and the reader must exit non-zero.

**Unchanged:** `build_tensor_source_work`, the one `object_get("QuadSource", …, labels_only=True, …)`
call, the `ShardedPool(… read_only=True …)` construction, `--models`, the run selection, every
other function of the reader, and every other `extract_*.py`. `ComputeTargets/`, `Datastore/` and
`main.py` are not touched: the factory's labels-only miss is correct, and the reader is what
misread it.

**E2's tests and E1's**, in a new module `ComputeTargets/tests/test_tensor_source_pairs.py`. No
Ray, no datastore, nothing under `var/`. Stand-in `wavenumber_exit_time`s that order and hash as
the real class does (by `k`, and by `store_id`), or real ones built without a store if the class
allows it; *your choice*, said in the log.
- **E1, run.** For every permutation of a 3-element and of a 4-element set of distinct k:
  - `quad_source_pairs` returns the same list;
  - that list equals `list(itertools.combinations_with_replacement(wavenumber_exit_time_array(items), 2))`;
  - every pair has `q.k <= r.k`, and the list has n(n+1)/2 pairs.
  - A duplicated member appears once, as `wavenumber_exit_time_array` makes it.
- **E1, read with `ast`** from `extract_tensor_source_data.py`, as `test_reader_read_only` reads the
  readers: `run_pipeline` assigns `work_grid` from a call of `quad_source_pairs`, and holds no call
  of `combinations_with_replacement`.
- **E2, run.** On an unpopulated labels-only `QuadSource` (built as
  `test_quadsource_parent_main.test_a_labels_only_miss_names_neither_and_cannot_be_computed` builds
  one, with stand-in q and r that carry `k.k_inv_Mpc` and `store_id`), `require_quad_source` raises
  `RuntimeError` whose message holds both `k` values and "no stored QuadSource". On an object whose
  `available` is true it raises nothing.
- **E2, read with `ast`**: in `plot_tensor_source`, a call of `require_quad_source` comes before the
  first attribute read of `values` or `functions` on `source`.

---

## 3. Verification

1. **Every existing test passes unchanged**, the four modules of "Read first" item 8 among them.
   Run the six suites serially on the staged tree, with README §7's command, output to files, the
   verdict grepped. Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
   `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
   can fail; re-run a failure alone, and on an export of the parent commit, before calling it real.
2. **Deliberate breakage.** Each must make a test of the new module fail. Restore it afterwards.
   - (i) `quad_source_pairs` pairs its input in the order given (no `wavenumber_exit_time_array`).
   - (ii) `quad_source_pairs` orders by descending `k`.
   - (iii) `run_pipeline` builds `work_grid` from `combinations_with_replacement(k_subsample, 2)`
     again.
   - (iv) `plot_tensor_source` reads `source.values` before calling `require_quad_source`.
   - (v) `require_quad_source` returns silently when `source.available` is false.

   Each goes in the log as a diff, taken with `git diff --` the one file it changes, so that
   `git apply` replays it at this commit. Name the tests that fail.
3. **E3 — the reader on 07's resumed store, read-only, three times.** This is the run 07's §2.3
   could not pass.
   - **Before:** `python -m RunRegistry list` shows nothing `running`. No Ray process from this
     checkout's `venv` is alive: search the whole process table for `SecondaryGWKit/venv` together
     with `ray`, not `pgrep -f 'gcs_server|raylet|ray::'` alone, which misses the runtime-env agent
     (`[07-a-killed-builds-ray-agents-outlive-the-driver]`). The user's ChamPBH cluster runs from
     `…/ChamPBH/venv/`; it is theirs, and you touch none of it. `RAY_ADDRESS` is unset.
     `store show var/datastores/a3-v2-readiness-07-resumed.sqlite` shows the sidecar, not a
     tombstone. If the store has been retired, **stop and ask**.
   - **Each run:**
     ```bash
     PYTHONPATH=. ./venv/bin/python extract_tensor_source_data.py \
         --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
         --ray-address local --models LambdaCDM \
         --output var/a3-v2-readiness-07a/extract/run-<n>/ \
         > var/a3-v2-readiness-07a/extract/run-<n>.log 2>&1
     ```
     It takes about a minute, so it runs in the foreground: it is not a long job, and is not
     registered.
   - **Around each run:** the SHA-256, mtime and size of the primary and all four shards, by
     `shasum -a 256` and **`/usr/bin/stat -f '%m %z'`** (a GNU `stat` earlier on `PATH` reads `-f` as
     "file system", which is how 07's first reader pass recorded a false change; log 07,
     deviation 4). And `ls` for any `-journal`, `-wal` or `-shm` beside them.
   - **Each run must:** exit 0; print the run selection and `RUNNING PIPELINE FOR MODEL LambdaCDM`;
     show `EXTRACT TENSOR SOURCE DATA` with 6 lookups; write a csv and plots for all six pairs; raise
     no `ReadOnlyMiss`, `ReadOnlyWrite` or `RuntimeError`; and leave every file byte-identical with
     its mtime unchanged, with no journal.
   - **Why three.** The unit tests are the proof that every order works. The runs show that the
     shipped reader works on a real store. Each run of the old reader passed with probability 1/6,
     so three passes of it would have had probability 1/216. Say so; do not overstate it.
   - **After the last run:** no Ray process from this checkout's `venv` left, waited for by pid if
     one lingers, never by a command-line pattern match. Paste the listing.
   - `store fingerprint var/datastores/a3-v2-readiness-07-resumed.sqlite` (without `--write`) still
     equals its sidecar's.

---

## 4. Acceptance

1. E1 and E2 in `extract_common.py` and `extract_tensor_source_data.py`, and nothing else in any
   reader.
2. The new test module, and the deliberate-breakage record (i)–(v).
3. E3: three runs on 07's resumed store, each exiting 0 with six pairs read, and the store
   byte-identical throughout.
4. Every existing test passes unmodified; the six suites at the baselines the brief gives.
   `black --check` clean on every Python file added or changed.
5. The issue closed on this board's §4 with its closing note, its index row deleted, and the index's
   count and date corrected by counting. The §1 row for 07a, items E1–E3, and §5. All in one commit.

---

## 5. Stop conditions — stop and ask

- An existing test would have to change.
- The fix seems to need a change to `ComputeTargets/`, `Datastore/`, `main.py` or another reader.
- 07's resumed store is absent, retired, or differs from its sidecar's fingerprint before E3.
- An E3 run exits non-zero, raises `ReadOnlyMiss`, `ReadOnlyWrite` or `RuntimeError`, reads fewer
  than six pairs, or changes any file of the store.
- A Ray process from this checkout's `venv` is alive before an E3 run, or does not exit after it.

---

## 6. What this prompt does not do

- It changes no factory and no `ComputeTargets` class. A labels-only miss stays an unpopulated
  object; `require_quad_source` is how a reader asks.
- It does not change `extract_QuadSourceIntegral_data.py`. Its pairs are in stored order already
  (§1). **Record in the log's observations, and open a §3.2 issue only if you confirm it by
  reading:** its `plot_QuadSourceIntegrand` checks `GkPolicy.available` and `Gk.available` but not
  the `QuadSource`'s, so a row genuinely missing from a store would surface there as an
  `AttributeError` on `source.z_sample` that names no pair.
- It does not change the `sample` draw, or how many k any reader draws.
- It writes no store, and nothing under `var/` but `var/a3-v2-readiness-07a/`. It opens no store
  but 07's resumed one, read-only. It retires nothing: D7-2 leaves that to the user.
- It does not edit `docs/a3-v2-readiness-verification.md` or the rehearsal's wrapper. The record is
  additive: E3's runs go in a new subsection, §2.3.4, beside §2.3.2, which stays as it is.

---

## 7. The log, the record and the board

**The files you may change, and nothing else:**
- `extract_common.py` (E1, E2) and `extract_tensor_source_data.py` (E1, E2);
- `ComputeTargets/tests/test_tensor_source_pairs.py`, new;
- `docs/a3-v2-readiness-verification.md`: a new §2.3.4 only, holding E3's commands verbatim, the
  hashes before and after each run, and each run's summary lines;
- `docs/a3-v2-readiness/e3/`, new, for E3's saved outputs small enough to commit;
- `logs/07a-pair-the-tensor-source-reader-in-stored-order.md`, in README §5.1's form;
- this board and `docs/OPEN_ISSUES.md`.

**The log** records, in addition: verification 2 and 3 verbatim; your choice of stand-ins; the six
suites beside the baselines.

**The board.**
- The §1 row for 07a, and items E1–E3.
- The issue moved from §3.2 to §4, with a closing note naming this commit's log and E3.
- `### 5.18 After prompt 07a (<date>)`.
- The header's status: **"10 of 10 prompts landed. The campaign's work is done; it is closed by the
  user after the orchestrator's review of 07a."** Do not write that it is closed.

**`docs/OPEN_ISSUES.md`.** Delete the issue's row; correct the count and date by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
