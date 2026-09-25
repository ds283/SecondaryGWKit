# Prompt 05 — fingerprint the real stores, and record each fingerprint in its sidecar

**Campaign:** [`README.md`](README.md) · **Board item:** **F13** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** nothing. **Narrows:** `[00-replicated-writes-can-diverge-across-shards]` (§3 of the
board), which says the A3 store and the backup "have not been read; prompt 05 reads them".
**Opens:** anything the fingerprints find in the stores (§6), **without acting on it**.
**Decisions:** D3 (README §6.2) is the user's request to write the `fingerprint` field of the three
existing sidecars. README §6.1 point 7 fixes what a fingerprint holds.
**Recommended model:** **Opus**. There is no code. The judgement is in three places:
- telling a finding about a store from a fault in the fingerprint;
- proving that each write changed one field and nothing else;
- showing that the digests localise two differences that are known only as row counts.

**Read first:**

1. [`README.md`](README.md) in full, especially §0, §1, §3, §6.1 and §6.2 (D3).
2. Prompt 04's log, [`logs/04-the-fingerprint.md`](logs/04-the-fingerprint.md): the format as
   shipped, "Observations not acted on" and "State handed to the next prompt". The orchestrator
   re-ran its demonstration on a fresh copy of the sweep store, and got the same digests.
3. `RunRegistry/stores.py`, from `# the fingerprint (store-fingerprint prompt 04)` to the end, and
   `RunRegistry/__main__.py` `_fingerprint`. You call these; you do not change them.
4. Prompt 02's log, [`logs/02-a-structured-inventory.md`](logs/02-a-structured-inventory.md), on
   the named problems (`replicated-divergence`, `orphan-tag`, …) and on `reference_digest`, by
   which a record names its parents.
5. The board's §3 entry `[00-replicated-writes-can-diverge-across-shards]`.
6. The three sidecars, which you read and do not yet write:
   - `var/datastores/handover-A3-baseline-lambdacdm.manifest.json`, the **live A3 store**;
   - `var/datastores/handover-atol-sweep.manifest.json`, the **sweep store**;
   - `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json`,
     the **backup**.
7. `CLAUDE.md`: "Long-running jobs — the run registry" and "Repository mechanics".

---

## 1. What is wanted

Three stores existed before the campaign. None has a fingerprint, because none was built after
prompt 04 landed. This prompt is the remedial step (README §2, D3):

- **Fingerprint each of the three, read-only**, with prompt 04's `store fingerprint`. This is the
  first time campaign code reads an original store. Prompts 01–04 read only copies, because whether
  the reader writes is what they were testing (README §3). So every read here is checked against a
  byte-level snapshot.
- **Record each fingerprint in its sidecar**, with `store fingerprint --write`. That writes the
  `fingerprint` field and nothing else.
- **Fingerprint a registry copy of one**, and show that it matches: the two-machine question
  answered for a real store.
- **Show that the digests localise the known differences** between the three stores. These are
  known today only as row counts (§2).

**Stores built before 04 landed.** README §2 names "any store built here before 04 landed, such as
the A3 v2 store". On 2026-09-25 there is none: `var/datastores/` holds the three stores above and
nothing else, and the A3 v2 store has not been built. Confirm this with
`find var -name '*.sqlite' -not -path '*/store-fingerprint-check-*'`, and record it. If a fourth
store has appeared, stop (§7).

**This prompt writes no code.** It has no tests and so no mutations. The log's deliberate-breakage
section says so, and points to §4 Phase A step 5, which plays the part of a discriminator.

---

## 2. What is known about the stores

Row counts, taken read-only by the prompt-04 orchestrator on 2026-09-25 (`sqlite3` `mode=ro`, all
four shards). **Re-measure them yourself** (§4 Phase A step 3); do not copy them.

| Class | Backup | Live A3 | Sweep | Kind |
|---|---:|---:|---:|---|
| `tolerance` | **6** | **12** | **13** | replicated |
| `QuadSourceIntegral` | **7 552** (1 890 / 1 888 / 1 889 / 1 885) | **7 654** (1 936 / 1 888 / 1 907 / 1 923) | **7 706** (1 955 / 1 892 / 1 920 / 1 939) | sharded |
| `version` | 1 | 1 | 1 | replicated |
| `store_tag` | 10 | 10 | 10 | replicated |
| `redshift` | 1 740 | 1 740 | 1 740 | replicated |
| `wavenumber` | 8 | 8 | 8 | replicated |
| `LambdaCDM` | 1 | 1 | 1 | replicated |
| `QCD_Cosmology` | 1 | 1 | 1 | replicated |
| `IntegrationSolver` | 7 | 7 | 7 | replicated |
| `GkSourcePolicy` | 2 | 2 | 2 | replicated |
| `QuadSourcePolicy` | 2 | 2 | 2 | replicated |
| `wavenumber_exit_time` | 8 | 8 | 8 | replicated |
| `BackgroundModel` | 1 | 1 | 1 | replicated |
| `TkNumericIntegration` | 8 | 8 | 8 | sharded |
| `TkWKBIntegration` | 8 | 8 | 8 | sharded |
| `GkNumericIntegration` | 4 549 | 4 549 | 4 549 | sharded |
| `GkWKBIntegration` | 13 920 | 13 920 | 13 920 | sharded |
| `GkSource` | 1 160 | 1 160 | 1 160 | sharded |
| `GkSourcePolicyData` | 1 160 | 1 160 | 1 160 | sharded |
| `QuadSource` | 36 | 36 | 36 | sharded |
| `OneLoopIntegral` | 0 | 0 | 0 | sharded |

Every `*Value` table has the same per-shard count in all three stores. `QuadSourceIntegral_tags`
rises by nine rows for every added `QuadSourceIntegral` row.

**What the counts say, and what they do not.**
- **Backup → live A3.** The backup was taken on 2026-09-21 before the first resume attempt. That
  attempt failed reading back an existing row, and the sidecar records that nothing changed. The
  registered resume `handover-03-a3-baseline-resume-20260923T024847` then ran on 2026-09-23 and
  failed at 92.31% of the quadratic-source-integral stage. By count, the live store has **102 more
  `QuadSourceIntegral` rows and 6 more `tolerance` rows**, and nothing else differs. Nothing
  records **which** rows, or why a resume added tolerance rows.
- **Live A3 → sweep.** The sweep store is a copy of the live A3 store (`copied_from`), made by
  `docs/handover/quadsource_atol_sweep.py --prepare`. That writes the sidecar's `created`, which is
  2026-09-23T12:41. `--prepare --force` replaces the whole store, and the first registered sweep
  run began at 11:06. So 12:41 is when the copy was **last** made, and that run's products may not
  be in it. Three registered sweep runs began after 12:41 (`lowz`, `seam`, `lowz-check`). By count,
  the sweep store has **52 more `QuadSourceIntegral` rows and 1 more `tolerance` row**. Whether
  every live-A3 record is still in it is not known.
- **The expectation**, which the fingerprints test: in each pair, **only** `tolerance` and
  `QuadSourceIntegral` differ, and only by additions. Every other class has the same digest.

**The backup's primary names the live store's shards.** Its `shards` table records the absolute
paths `…/var/datastores/handover-A3-baseline-lambdacdm-shard000N.sqlite`, which are the **live**
shards. `ShardedPool._resolve_shard_rows` reads a legacy absolute record as the sibling of that
name, so the backup should read its own four shards. The fingerprint depends on this. It is checked
directly: the backup's `QuadSourceIntegral` count must be 7 552, not the live store's 7 654
(§4 Phase A step 4).

**The sidecars are in the writer's byte form.** Each of the three, parsed and re-serialised as
`write_json_atomic` writes (`json.dump(indent=2, sort_keys=True)` plus a newline), gives back its
own bytes exactly. So after `--write`, deleting the `fingerprint` key and re-serialising must give
back the original bytes. That is both the proof that one field changed and the way to undo the
write (§4 Phase B step 2). Each sidecar is a problem-free registry sidecar, and none has a
`fingerprint` field.

| Sidecar | `store_id` | Bytes | SHA-256, 2026-09-25 |
|---|---|---:|---|
| live A3 | `5f58ac536362424499a4e94c6242928e` | 4 447 | `f722040836f4918439f2ade919d596222f862bec44788e714febd4c905ff4c28` |
| sweep | `04198f22f8704c52a252a72566af94ed` | 806 | `a595b7c942aa4a54cf71bb3eb360095b35d3c94d4735031209aaac351c7b9a36` |
| backup | `4c2ce77b0bf24671bcebd8dda3441041` | 3 007 | `ce7b476af75aa519e73cd14e0afd354912ce66a9626d86a1dcfb2bd7dc7c751a` |

**One digest is already known.** Prompt 04 fingerprinted a `cp -p` copy of the sweep store, and
the orchestrator repeated it on a fresh copy at `ef46154`. Both gave the overall digest
`2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18`. The original must give the
same.

---

## 3. The *prompt's choices*

Each may be overridden by a logged deviation that is at least as strong.

1. **Every read and every comparison comes before any write.** Phase A (§4) fingerprints all three
   stores read-only and compares them. Phase B writes only if Phase A met no stop condition. A
   fault found after a write would leave a wrong value in the one place a reader looks.
2. **A finding about a store is recorded, not stopped on.** For example: a named problem, a
   difference beyond the two expected classes, or a record present in the older store and absent
   from the newer. The fingerprint describes what the store holds, and recording that is what D3
   asks for. A finding is named in the log, and opened as a §3 issue if no known operation
   explains it. **A doubt about the fingerprint itself is a stop** (§7), because then the value
   written would be wrong.
3. **The writes are taken by a person, with the tree clean.** Run `--write` before you create any
   file in the repository, while `git status --porcelain` is empty and `HEAD` is this prompt's
   base. Then `taken` says `git_head` the base, `git_dirty` false, `run_id` null. Draft the log in
   your scratchpad until the writes are done.
4. **The registry copy is of the live A3 store.** It is the store whose product will travel
   between machines, and the one with a run's history behind it.
5. **The three fingerprints, as written, are committed** beside the log, as
   `logs/05-fingerprints.json`: an object from each sidecar's repo-relative path to its
   `fingerprint` value, byte-for-byte what `--write` put there. The sidecars live in the
   gitignored `var/`. Without this, what was recorded exists only there. The listings are **not**
   committed: they are generated on demand (README §6.1 point 7).
6. **The order of the writes is sweep, backup, live A3**, least valuable first. A failure stops
   before the store that matters.

---

## 4. The procedure

All scratch work goes in `var/store-fingerprint-check-05/`, which you delete at the end. Never
`/tmp`, and never a session scratchpad for anything that must survive the session. The listings and
the registry copy need about 0.5 GB.

### Phase A — read-only

1. **Discovery.** `python -m RunRegistry list`: nothing `running`. Record the output.
2. **Snapshot**, read-only, every file under `var/datastores/` and
   `var/datastores/backup-pre-resume-20260921T091011/`: the listing, and each file's `lstat` size,
   `st_mtime_ns` and SHA-256. Also `ls var/runs`. For each sidecar:
   - `store show` says it is a registry sidecar, with problems `none`;
   - it has no `fingerprint` field;
   - it re-serialises to its own bytes (§2).

   Copy the three sidecars' bytes into `var/store-fingerprint-check-05/sidecars-before/`.
3. **Independent counts.** Count every class table on every shard with
   `sqlite3 'file:<path>?mode=ro'`, and compare them with §2's table. Re-take the snapshot
   afterwards; it must be identical.
4. **Fingerprint each store twice, read-only**, with the default runs root, so that the
   running-run check reads the real `var/runs/`:
   - `python -m RunRegistry store fingerprint <primary>`. Record the output, the wall time and the
     peak resident memory (`/usr/bin/time -l`). Its exit code must be 0 with `none recorded`;
   - again, with `--listing var/store-fingerprint-check-05/listings/<name>.txt`. Apart from the
     `>> listing` line, the output must be byte-identical to the first run's;
   - to have each fingerprint as JSON, call `RunRegistry.stores.fingerprint_store(primary)` (the
     default `write=False`) from a scratch script, and save its `fingerprint` in the check
     directory. Its digests must equal the command line's.

   After **every** command, the store's directory must be unchanged, files and listing alike.
   Then check:
   - every class count equals step 3's count. A sharded class is the sum of its shards. A
     replicated class is one shard's count, and any disagreement among shards shows as a
     `replicated-divergence` problem;
   - **the backup reads its own shards.** Open the backup with `open_read_only`, and record each
     shard's path. All four must be inside the backup's directory. The legacy-path notice must name
     the live directory and say it reads siblings in the backup's. The backup's
     `QuadSourceIntegral` count is 7 552, and its `tolerance` count is 6;
   - **the sweep's overall digest is `2c2dde68…`** (§2);
   - the problem counts of each store. Where a store has any, read `main.py --inventory` on it
     (read-only since prompt 03) for their text.
5. **The known differences.** From the saved JSON, compute
   `compare_fingerprints(backup, live A3)` and `compare_fingerprints(live A3, sweep)` in a scratch
   script. Record every entry. The expectation (§2) is that each pair names only:
   - `tolerance`, its one tag set `[]`, 6 against 12, and 12 against 13;
   - `QuadSourceIntegral`, in whichever tag sets its records carry, with the tag sets' counts
     summing to 7 552 against 7 654, and 7 654 against 7 706.

   Then **name the records**, from the listings. For each pair, and for `tolerance` and
   `QuadSourceIntegral`, take the record lines only in the newer store and only in the older:
   - the added `tolerance` records, by `log10_tol` (decode the `float.hex`);
   - the added `QuadSourceIntegral` records, counted by `(atol, rtol)` and by tag set. A record
     names its parents by `reference_digest`. `tolerance` is untagged, so its reference is the
     digest of its key, and a map from each tolerance record's digest to its `log10_tol` decodes
     them;
   - how many records are only in the older store. The expectation is none.

   If any other class differs, name its differing records in the same way. That is a finding
   (§3 choice 2), not a stop, unless §7 says otherwise.

### Phase B — the writes

Only if Phase A met no stop condition.

1. `git status --porcelain` is empty, and `HEAD` is the base. `python -m RunRegistry list`:
   nothing `running`.
2. For the sweep store, then the backup, then the live A3 store:
   - `python -m RunRegistry store fingerprint <primary> --write`. It must exit 0 and say
     `it replaced none`;
   - **the sidecar changed in one field.** Parse it. Its keys are the old ones plus `fingerprint`.
     Delete `fingerprint`, and re-serialise as `write_json_atomic` does. The bytes must equal
     `sidecars-before/`'s exactly;
   - **the written fingerprint is Phase A's**, apart from `taken`. `taken` is
     `{"when", "git_head": <base>, "git_dirty": false, "run_id": null}`;
   - of the store's files, only the sidecar changed;
   - `store fingerprint <primary>`, read-only: `matches`, exit 0. `store show` prints it.

   Record each sidecar's new size and SHA-256.

### Phase C — a registry copy

1. `python -m RunRegistry store copy var/datastores/handover-A3-baseline-lambdacdm.sqlite
   var/store-fingerprint-check-05/copy/handover-A3-baseline-lambdacdm.sqlite --purpose "…"`, with
   the default runs root. The live A3 store's files and sidecar must be unchanged by it.
2. The copy's sidecar has a new `store_id`, `copied_from` the live A3 store, and the live
   sidecar's `fingerprint` **verbatim**, `taken` included.
3. `store fingerprint` on the copy: `matches`, exit 0.

### Phase D — the end

1. Re-take Phase A step 2's snapshot. Everything is identical except the three sidecars, whose
   new SHA-256 you recorded in Phase B. `ls var/runs` is identical, and `RunRegistry list` shows
   the same runs.
2. Write `logs/05-fingerprints.json` from the three sidecars' `fingerprint` values (§3 choice 5).
   Check that each value parses back equal to the sidecar's.
3. Delete `var/store-fingerprint-check-05/`.
4. Run every suite (README §7). No code changed, so each must match its baseline.

---

## 5. Acceptance

1. The three sidecars each hold a `fingerprint`, and each differs from its bytes before only in
   that field, proven by the exact re-serialisation of Phase B step 2.
2. Each written fingerprint equals the read-only one of Phase A apart from `taken`, and
   `store fingerprint` on each store says `matches`. The sweep's overall digest is `2c2dde68…`.
3. The registry copy of the live A3 store matches.
4. The two comparisons are recorded entry by entry. The differing records are named, by
   `log10_tol` and by `(atol, rtol)`, and the number of records only in the older store is
   stated.
5. Every store file except the three sidecars is byte-identical to before. `var/runs/` is
   unchanged, and `var/store-fingerprint-check-05/` is gone.
6. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the log, `logs/05-fingerprints.json`, this
   board, README §2's row for 05 and `docs/OPEN_ISSUES.md`. No code, and no test.
7. Every suite matches its baseline.
8. **This board:** the §1 row for 05; F13; §3, the measurement added to
   `[00-replicated-writes-can-diverge-across-shards]` for the A3 store and the backup, and any
   issue opened; §5's baselines.
9. **`docs/OPEN_ISSUES.md`**, in the same commit: any row added, and the replicated-divergence
   row's hook updated if the measurement narrows it; the count and the date corrected.

---

## 6. What this prompt does not do

- It changes no code, no test, no key, no schema and no inventory. If the fingerprint looks
  wrong, stop (§7). Do not repair it.
- It writes no store file, and no sidecar field but `fingerprint`. It never adopts, creates, moves
  or rewrites a sidecar.
- It does not act on a finding. It does not prune, repair a divergence, delete the backup, or
  finish the A3 resume. Each is recorded, and opened as an issue where no known operation explains
  it.
- It does not begin a run, under `var/runs/` or anywhere else. The writes are a person's (§3
  choice 3).
- It does not fingerprint a store elsewhere, such as on `macstudio-tunnel`.

---

## 7. Stop conditions — stop and ask the user

Stop before any further write, and report what you found:

- a `running` run, alive or stale, names any of the three stores;
- a fourth store exists under `var/` (§1);
- a sidecar already holds `fingerprint`, is not a problem-free registry sidecar, or does not
  re-serialise to its own bytes;
- **any store file changes under a read**, or a sidecar changes other than under `--write`. The
  reader was shown not to write on copies. If it writes an original, that is the finding that
  matters most;
- two read-only fingerprints of one store differ, or a class count differs from the independent
  count;
- the sweep's overall digest is not `2c2dde68…`;
- the backup reads any file outside its own directory, or its `QuadSourceIntegral` count is not
  7 552;
- a written fingerprint differs from Phase A's apart from `taken`, or a sidecar changes in any
  field but `fingerprint`;
- the registry copy does not match;
- anything seems to need a change to code, a test, a store, or a sidecar field other than
  `fingerprint`.

Differences between the stores beyond the two expected classes, removals, and named problems are
**findings, not stops** (§3 choice 2), unless one of the lines above also holds.

---

## 8. The log and the board

`logs/05-fingerprint-the-real-stores.md`, using README §5.1's template. In addition, record:
- the Phase A snapshot, and the independent counts;
- for each store: its overall digest, its per-class table (count, digest, tag sets), its problem
  counts, the wall time, the peak memory, and the fingerprint's size;
- the backup's four shard paths as the reader opened them;
- the two comparisons, entry by entry, and the records named in Phase A step 5;
- for each sidecar: its SHA-256 and size before and after, the `taken` written, and the
  re-serialisation proof;
- **how to undo a write**: delete the `fingerprint` key and re-serialise. Phase B step 2 proved
  that this restores the sidecar's exact bytes;
- each *prompt's choice*, and whether it was kept;
- that there are no tests and no mutations, and why.

On `IMPLEMENTATION_STATE.md`: the §1 row for 05; F13; §3 as §5 item 8 says; §5's baselines. Update
`docs/OPEN_ISSUES.md` in the same commit.
