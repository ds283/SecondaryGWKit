# Log 05 — Fingerprint the real stores, and record each fingerprint in its sidecar

**Prompt:** [`prompts/store-fingerprint/05-fingerprint-the-real-stores.md`](../05-fingerprint-the-real-stores.md)
**Commit:** *(this commit)*, "Record the three real stores' fingerprints in their sidecars"
**Base:** `50a24acb23e41a68f508be63db509c10b2ac6d7e` ("Write prompt 05 of the store fingerprint
campaign"), clean. Prompt 04 had landed.
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. F13 done. No §7 stop condition arose. The three original stores were each
fingerprinted read-only, three times, with identical digests every time, and no store file changed
under any read. Each fingerprint was then written into its sidecar, which changed in that one
field only, proven by exact re-serialisation. A registry copy of the live A3 store matches it.
- **Overall digests:**
  - sweep `2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18`, as prompt 04 found
    on its copies;
  - backup `eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222`;
  - live A3 `433b7fc3cad94a971020c0493e4062e4d3a7a590a6f084c6a8e8f451f2b648a3`.
- **No store has a problem of any kind.** In particular there is no `replicated-divergence` on any
  of the three.
- **The comparisons localise the known differences exactly.** Each pair differs in `tolerance` and
  `QuadSourceIntegral` only, each in its one tag set, and only by additions. Nothing is only in the
  older store.
- **Every added record is attributed to a recorded operation.** Backup → live A3 is
  +48 production-pair records from the registered resume, plus 54 records and 6 tolerance values
  from the first sweep run. That run wrote into the live store through a copied primary, as
  `2ebb7b6` and the `handover` board record. So the resume added no tolerance row. Live A3 →
  sweep is +52 records and 1 tolerance value from the sweep store's later writers.

No issue was opened. `[00-replicated-writes-can-diverge-across-shards]` gained the measurement on
all three original stores.

## The rules this prompt ran under

As the orchestrator stated them at dispatch, and kept:

1. **One commit**, holding this log, `logs/05-fingerprints.json`, the board, README §2's row for 05
   and `docs/OPEN_ISSUES.md`. Not pushed.
2. **The only files written outside `var/store-fingerprint-check-05/` are the three sidecars'
   `fingerprint` field**, through `python -m RunRegistry store fingerprint <primary> --write`, only
   in Phase B, in the order sweep, backup, live A3. No code, test or store file changed. No other
   sidecar field was written. The only sidecar created is the registry copy's, in Phase C, inside
   `var/store-fingerprint-check-05/`.
3. **No run was begun**, under `var/runs/` or anywhere else.
4. **`var/store-fingerprint-check-05/` was deleted at the end.**
5. **No file whose name starts with `orch_` was touched.** The session scratchpad holds several;
   this prompt's scripts there are named `impl05_*`.
6. **§7's stop conditions mean stop and ask**, before any further write. None arose.

The tree was kept clean (`git status --porcelain` empty, `HEAD` the base) from dispatch until the
third write, so each `taken` names a clean base (§3 choice 3). Drafts and scripts lived in the
session scratchpad, and the procedure's outputs in `var/store-fingerprint-check-05/`, which is
gitignored (`.gitignore:3`, `var/`). The first file created in the repository was
`logs/05-fingerprints.json`, after Phase D step 1.

## What shipped

- **Three sidecars each hold a `fingerprint`** (D3). They live in the gitignored `var/`, so they
  are not in this commit:

  | Sidecar | `store_id` | Before (bytes, SHA-256) | After (bytes, SHA-256) | `taken.when` |
  |---|---|---|---|---|
  | sweep, `var/datastores/handover-atol-sweep.manifest.json` | `04198f22…` | 806, `a595b7c942aa4a54cf71bb3eb360095b35d3c94d4735031209aaac351c7b9a36` | 10 564, `f7b705bd1335f006dfad67e91ef40a3b786c83e0d8f63703bb71c5a5dfb3f834` | `2026-09-25T07:31:21+01:00` |
  | backup, `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json` | `4c2ce77b…` | 3 007, `ce7b476af75aa519e73cd14e0afd354912ce66a9626d86a1dcfb2bd7dc7c751a` | 12 763, `ea1ff60d7f33fda16b7a0b961523f3e8f88f2d8497c704f7519a6674d7e95088` | `2026-09-25T07:32:10+01:00` |
  | live A3, `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` | `5f58ac53…` | 4 447, `f722040836f4918439f2ade919d596222f862bec44788e714febd4c905ff4c28` | 14 205, `394ff933c30c755f7beae03b9f9da4fad1651266d9730ab74dd412ceb851732a` | `2026-09-25T07:32:35+01:00` |

  Every `taken` is `{"git_dirty": false, "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
  "run_id": null, "when": …}`.
- **`logs/05-fingerprints.json`** (29 442 bytes, new; §3 choice 5). It is one object mapping each
  sidecar's repo-relative path to that sidecar's `fingerprint` value, serialised in the writer's
  form (`json.dump(indent=2, sort_keys=True)` plus a newline). Each value parses back equal to the
  sidecar's, `taken` included (Phase D step 2). The listings are not committed (README §6.1
  point 7).
- **This log, the board, README §2's row for 05 and `docs/OPEN_ISSUES.md`.**

No code, no test, no key, no schema and no inventory changed.

## Deviations from the prompt

1. **More read-only reads of the originals than the prompt lists.** *IMPLEMENTATION CHOICE.* §3
   choice 2 opens an issue for a finding only "if no known operation explains it". Deciding that
   needed row-level evidence the fingerprint does not carry, since timestamps and labels are
   outside every record by design. So, besides §4's reads, four more reads were made:
   - with stdlib `sqlite3` on `file:<path>?mode=ro`, each primary's `shards` table;
   - on every shard, each `QuadSourceIntegral` row's `label`, `timestamp`, `atol_serial` and
     `rtol_serial`, grouped;
   - shard 0's `tolerance` table, with its timestamps;
   - the four registered sweep runs' and the resume's `manifest.json`, `status.json` and
     `stdout.log` under `var/runs/`.

   Each read of a store was followed by a full snapshot, identical to the first (below). No store
   code was pointed at an original for these.
2. **The snapshot also records each directory's listing and `st_mtime_ns`.** *IMPLEMENTATION
   CHOICE.* This is stricter than §4 step 2 asks. The atomic write (`write_json_atomic`: a `.tmp`
   beside the sidecar, then `os.replace`) changes the containing directory's mtime. So after
   Phase B, `var/datastores/` and the backup's directory differ in mtime only. Their listings are
   identical, and no `.tmp` is left.
3. **The independent counts used Python's stdlib `sqlite3`**, with the URI `file:<path>?mode=ro`,
   rather than the `sqlite3` command-line shell. *IMPLEMENTATION CHOICE.* It is the same SQLite
   library and the same mode, and it gives the counts as data to compare.
4. **`logs/05-fingerprints.json` holds each value, not the sidecar's bytes of it.**
   *STRUCTURALLY REQUIRED.* §3 choice 5 says "byte-for-byte what `--write` put there". Inside a
   sidecar the value is nested one level deeper, and so indented differently, so its bytes cannot
   stand alone. The file holds each value serialised exactly as the writer serialises. Each parses
   back equal to the sidecar's value, and re-serialising either gives the same bytes.
5. **The scripts are in the session scratchpad** (`impl05_snapshot.py`, `impl05_counts.py`,
   `impl05_fpjson.py`, `impl05_compare.py`, `impl05_attrib.py`, `impl05_verify_write.py`).
   *IMPLEMENTATION CHOICE*, as the orchestrator directed. Everything the procedure produced, from
   snapshots and outputs to listings, JSON and the copy, was in `var/store-fingerprint-check-05/`
   until Phase D deleted it.

No UNINTENDED DRIFT. One scratch script (`impl05_fpjson.py`) failed on its first run, after the
sweep's `fingerprint_store` call, at a print of `inventory.shards`, which is a tuple of serials,
not of shard objects. That call was read-only, and the snapshot taken after it was identical. The
line was corrected, its one output file in the check directory removed, and the script re-run
whole.

## Verification performed

### Phase A — read-only

1. **Discovery.** `python -m RunRegistry list`: 7 runs, **none `running`**. 5 are finished:
   - `handover--quadsource-atol-sweep-lowz-check-20260924T001705`, `-seam-20260923T225928`,
     `-lowz-20260923T132309` and `-20260923T110622`, all `done`;
   - `handover-03-a3-baseline-resume-20260923T024847`, `failed`.

   2 are `unknown` pre-registry: `a3-pilot` and `a3-pilot-resume`. The volume had 81 GB free.
   **No fourth store.** `find var -name '*.sqlite' -not -path '*/store-fingerprint-check-*'` gives
   15 files, which are the three stores' primaries and four shards each, and nothing else. The A3
   v2 store has not been built.
2. **Snapshot**, read-only. It covers 18 files and 2 directories under `var/datastores/`:

   | File (under `var/datastores/`) | Size | mtime | SHA-256 (16) |
   |---|---:|---|---|
   | `backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite` | 32 768 | 2026-09-20 21:03:10 | `fdbe93a541083e8f` |
   | `backup-…/handover-A3-baseline-lambdacdm-shard0000.sqlite` | 87 044 096 | 2026-09-21 04:32:10 | `6667d5588146038a` |
   | `backup-…/handover-A3-baseline-lambdacdm-shard0001.sqlite` | 87 302 144 | 2026-09-21 04:32:24 | `d0ad78e6a9eabb7a` |
   | `backup-…/handover-A3-baseline-lambdacdm-shard0002.sqlite` | 87 588 864 | 2026-09-21 04:32:24 | `faeca5d7198f3411` |
   | `backup-…/handover-A3-baseline-lambdacdm-shard0003.sqlite` | 86 609 920 | 2026-09-21 04:32:24 | `83f6288df912c5e4` |
   | `backup-…/handover-A3-baseline-lambdacdm.manifest.json` | 3 007 | 2026-09-24 22:03:28 | `ce7b476af75aa519` |
   | `handover-A3-baseline-lambdacdm.sqlite` | 32 768 | 2026-09-20 21:03:10 | `fdbe93a541083e8f` |
   | `handover-A3-baseline-lambdacdm-shard0000.sqlite` | 87 367 680 | 2026-09-23 11:22:04 | `a10f3c3ce33ad056` |
   | `handover-A3-baseline-lambdacdm-shard0001.sqlite` | 87 302 144 | 2026-09-23 11:11:00 | `9a5a4827a996786f` |
   | `handover-A3-baseline-lambdacdm-shard0002.sqlite` | 87 711 744 | 2026-09-23 11:22:04 | `d196e8a34167aa3f` |
   | `handover-A3-baseline-lambdacdm-shard0003.sqlite` | 86 892 544 | 2026-09-23 11:22:04 | `13398fec0071e0d3` |
   | `handover-A3-baseline-lambdacdm.manifest.json` | 4 447 | 2026-09-24 22:03:27 | `f722040836f49184` |
   | `handover-atol-sweep.sqlite` | 32 768 | 2026-09-23 12:41:04 | `c3cda49d8e351806` |
   | `handover-atol-sweep-shard0000.sqlite` | 87 482 368 | 2026-09-24 02:58:24 | `4000a990c1ae6ec0` |
   | `handover-atol-sweep-shard0001.sqlite` | 87 334 912 | 2026-09-23 23:00:42 | `8cf35f4c9758e563` |
   | `handover-atol-sweep-shard0002.sqlite` | 87 814 144 | 2026-09-24 02:58:24 | `6c6995adc027a5bd` |
   | `handover-atol-sweep-shard0003.sqlite` | 86 994 944 | 2026-09-24 02:58:24 | `362ad59e5f62e28c` |
   | `handover-atol-sweep.manifest.json` | 806 | 2026-09-24 22:03:28 | `a595b7c942aa4a54` |

   The primaries and sidecars have the prefixes prompts 01–04 recorded:
   - primaries: A3 `fdbe93a5` (live and backup, byte-identical) and sweep `c3cda49d`;
   - sidecars: A3 `f7220408`, sweep `a595b7c9` and backup `ce7b476a`, each as §2's table gives it
     in full.

   `ls var/runs` gave the seven run directories, and `realistic_large_x_cells.jsonl`, `run.out`,
   `run.pid` and `run.progress`.

   **The sidecars.** For each of the three, `store show` says `kind: registry` and
   `problems: none`, and it has no `fingerprint` key. Parsed and re-serialised as
   `write_json_atomic` writes, each gives back its own bytes exactly. Their bytes were copied to
   `sidecars-before/` (`cp -p`), and the copies' SHA-256 are the three above.
3. **Independent counts** of every table on every shard of every store, with stdlib `sqlite3`
   `mode=ro`. Every shard is in `journal_mode` `delete`. Every class count equals §2's table,
   class for class and shard for shard:
   - **`tolerance`**: 6 / 12 / 13 on every shard (backup / live / sweep);
   - **`QuadSourceIntegral`**:
     - backup 7 552, as 1 890 / 1 888 / 1 889 / 1 885;
     - live A3 7 654, as 1 936 / 1 888 / 1 907 / 1 923;
     - sweep 7 706, as 1 955 / 1 892 / 1 920 / 1 939;
   - **`QuadSourceIntegral_tags`** is 9× those counts: 67 968, 68 886 and 69 354;
   - every other table has the same per-shard count in all three stores. Every replicated table
     has the same count on all four shards.

   The snapshot re-taken afterwards was **identical**, directory mtimes included.
4. **Three read-only fingerprints of each store.** After each command a full snapshot was taken
   and compared with the first. **All were identical.**
   - `store fingerprint <primary>`: exit 0, `problems: none`, `recorded: none recorded (the
     sidecar is registry)`.
   - `store fingerprint <primary> --listing var/store-fingerprint-check-05/listings/<name>.txt`:
     exit 0. Its output without the `>> listing` line is **byte-identical** to the first run's
     (`cmp`).
   - `RunRegistry.stores.fingerprint_store(primary)`, with `write=False`, from a scratch script:
     `wrote` false and `recorded` None. Its overall digest, and every class's count and 12-hex
     digest, equal the command line's.
   - **Each listing hashes to its fingerprint.** The header digest is the overall digest. Each
     class's record lines, sorted bytewise, hash to the class digest, and each tag set's lines to
     that set's digest. The listings have 30 222 lines (19.1 MB) for the backup, 30 330 (19.2 MB)
     for the live A3 store and 30 383 (19.3 MB) for the sweep.

   **Wall time and peak memory** (`/usr/bin/time -l`, whole command, imports included):

   | Store | Run 1: wall, max RSS | Run 2 (`--listing`): wall, max RSS | Peak memory footprint |
   |---|---|---|---|
   | sweep | 6.34 s, 256.4 MB | 5.94 s, 257.9 MB | 205 MB |
   | backup | 5.14 s, 253.5 MB | 5.58 s, 227.5 MB | 203–204 MB |
   | live A3 | 6.66 s, 235.5 MB | 6.97 s, 247.9 MB | 205–206 MB |

   **Size of each fingerprint:**

   | Store | Compact JSON | `indent=2` |
   |---|---:|---:|
   | sweep | 6 285 | 9 133 |
   | backup | 6 283 | 9 131 |
   | live A3 | 6 285 | 9 133 |

   Each sidecar grew by about 9 757 bytes. The sweep's is one byte longer than prompt 04's 6 284,
   because `taken.git_dirty` is `false` here and was `true` there.

   **The checks:**
   - **Every class count equals step 3's count.** A sharded class is the sum of its shards, and a
     replicated class is one shard's count. In every class of every store the tag-set counts sum to
     the class count.
   - **The backup reads its own shards.** Its primary's `shards` table names the **live** shards by
     absolute path: `…/var/datastores/handover-A3-baseline-lambdacdm-shard000{0,1,2,3}.sqlite`.
     `open_read_only` printed:
     ```
     !! Primary database ".../var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite" records 4 shard(s) by legacy absolute path in ".../var/datastores"; reading them as siblings in ".../var/datastores/backup-pre-resume-20260921T091011" instead (stored records not rewritten)
     ```
     It opened these four shards, every one inside the backup's directory:
     - `#0` `…/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite`;
     - `#1` `…/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite`;
     - `#2` `…/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite`;
     - `#3` `…/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite`.

     `store fingerprint` on the backup begins with the same line. **The backup's
     `QuadSourceIntegral` count is 7 552 and its `tolerance` count is 6**, so it did not read the
     live shards. The live and sweep primaries name their own shards, and print no notice.
   - **The sweep's overall digest is `2c2dde68…`**, as prompt 04 and the orchestrator found on
     copies.
   - **Problem counts: none, in every class of every store.** So `main.py --inventory` had no
     problem text to show, and was not run.
5. **The known differences:** below.

**The per-class table.** Every class has at most one tag set, whose count and digest equal the
class's. **B**, **T**, **G** and **Q** are prompt 02's tag sets. Nineteen of the 21 classes have the
same count and digest in all three stores.

| Class | Backup | Live A3 | Sweep | Digest | Tag set |
|---|---:|---:|---:|---|---|
| `version` | 1 | 1 | 1 | `482c17e477216a7821fbd15f13e17852f1f9c0a8993a00e12b092fca8d7f2d9a` (all three) | `[]` |
| `store_tag` | 10 | 10 | 10 | `8e36e81ea00925fbccd5611e1009fc01fe56bb4761c9fba39bd3b05ebd395d37` (all three) | `[]` |
| `redshift` | 1 740 | 1 740 | 1 740 | `f5851073507ae2f2989c79e49890d880f386c52e27908988584f8508e7e8c95a` (all three) | `[]` |
| `wavenumber` | 8 | 8 | 8 | `f82c942161647580916d3ccfbcf288f366145a047ccc71892b42fdcbfc084bbe` (all three) | `[]` |
| **`tolerance`** | **6** | **12** | **13** | `b582318d050185a1b398d3c3fef7d771f98e1e509918adef47b4ab22bb24ca05` / `4abc515d0a78a7427733db07267b7905d569db39bab05274421b9b518f3a6029` / `5ef7da79b88f47475ba5d8335bf226366dd02259c2d6d02ec47edc8db593aca8` | `[]` |
| `LambdaCDM` | 1 | 1 | 1 | `dcbaa9d413d45bf56b63a2a6ac1da865159e363f8b5ea2a479265113b82091b2` (all three) | `[]` |
| `QCD_Cosmology` | 1 | 1 | 1 | `178439a39e42aacff29617bf4fafae2d6b4f2faa27a5a8337781de080355fb4f` (all three) | `[]` |
| `IntegrationSolver` | 7 | 7 | 7 | `317b009bb7febb1b3e0980794a994f6d8bdeed3ca784464e2bc462acef1c18c1` (all three) | `[]` |
| `GkSourcePolicy` | 2 | 2 | 2 | `064c7949b3309290eda4bd27713f1b608e066bc0a73fcc815fb3c91720626ec7` (all three) | `[]` |
| `QuadSourcePolicy` | 2 | 2 | 2 | `064c7949b3309290eda4bd27713f1b608e066bc0a73fcc815fb3c91720626ec7` (all three) | `[]` |
| `wavenumber_exit_time` | 8 | 8 | 8 | `045639da7d40c62a233ac245525e0efb1e0d5740bed35c58310026ec1e6cb3fa` (all three) | `[]` |
| `BackgroundModel` | 1 | 1 | 1 | `e3fb6cb1cf7e179e1f4c94d9b27a1deaccf9d9e1c9cf9613ec2cf8fde51827d7` (all three) | **B** |
| `TkNumericIntegration` | 8 | 8 | 8 | `07061fe0dd0fd8890f3b210fa0ea48bc7845ca7bb6a6787be6f639202363b4f8` (all three) | **T** |
| `TkWKBIntegration` | 8 | 8 | 8 | `35a479b6320ff8e3d780be078657323ac18443cc1725e86a18673766266ecd34` (all three) | **T** |
| `GkNumericIntegration` | 4 549 | 4 549 | 4 549 | `c123f591d2b04a2d8358825c64340cfad5a60f8f57bfb9f0a6bb527f820e5bfe` (all three) | **G** |
| `GkWKBIntegration` | 13 920 | 13 920 | 13 920 | `7397175220005dab7233546c4a4ece0de5200ab76a6c2818b1a6408a1cd5acc5` (all three) | **G** |
| `GkSource` | 1 160 | 1 160 | 1 160 | `2245e52abd42fbbb89010a734fd510413b7c01625395f7c500f6569ecdcfa024` (all three) | **G** |
| `GkSourcePolicyData` | 1 160 | 1 160 | 1 160 | `df2424eb174d794a042edd3a69853e4ede786fa01107630384e5a17812716761` (all three) | `[]` |
| `QuadSource` | 36 | 36 | 36 | `2dd5a869a5f2017edda7d5a5e63552a9365c6362eed4b0f210116864b2b75526` (all three) | **T** |
| **`QuadSourceIntegral`** | **7 552** | **7 654** | **7 706** | `8545e21a4bf297c3b5505e1b1290344b26457d36f7fde5471f050cd4e94f915c` / `94c5883ba5c59fcc7617f9ed7b435525900109fef4bd6c53e4f16d89fce5d4f8` / `28c5c5e4d0393b0146c4ef88a88ef3ca5f288e9d26986e10ff113829d1cceb3b` | **Q** |
| `OneLoopIntegral` | 0 | 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (the empty digest) | none |

Records: backup 30 180, live A3 30 288, sweep 30 341 (prompt 02's count).

### Phase A step 5 — the two comparisons, and the records they name

`compare_fingerprints` on the saved JSON, and set differences of each class's record lines from
the listings. Parents were decoded through each class's reference digest, which is `digest(key)`
for an untagged class. This applies to `tolerance`, `wavenumber`, `redshift`,
`wavenumber_exit_time` and `GkSourcePolicy`, and the maps were taken over the union of the three
listings.

**`compare_fingerprints(backup, live A3)`: 2 entries.**
```
{"kind": "tag_set", "class": "QuadSourceIntegral", "tags": [Q], "recorded": 7552, "current": 7654,
 "text": "QuadSourceIntegral: tag set [GkOneLoopDensity, …, TkOneLoopDensity] differs: 7552 recorded, 7654 now"}
{"kind": "tag_set", "class": "tolerance", "tags": [], "recorded": 6, "current": 12,
 "text": "tolerance: tag set [] differs: 6 recorded, 12 now"}
```
There is no `problems` entry, because neither store has a problem.
- **`tolerance`: 6 records only in the live store, and none only in the backup.** The additions
  are `log10_tol`:
  - −22 (`-0x1.6000000000000p+4`);
  - −25 (`-0x1.9000000000000p+4`);
  - −28 (`-0x1.c000000000000p+4`);
  - −30 (`-0x1.e000000000000p+4`);
  - −6 (`-0x1.8000000000000p+2`);
  - −7 (`-0x1.c000000000000p+2`).

  The backup's six are −32, −13, −10.522878745280337 (that is, $3\times10^{-11}$), −10, −9 and −8.
- **`QuadSourceIntegral`: 102 records only in the live store, and none only in the backup.** All
  102 are in tag set **Q**, with policy `GkSourcePolicy` (1.5, `maximize-WKB`), model `ea6c118b…`
  and $z_{\rm source,max} = 1.63911\times10^{16}$. By `(log10 atol, log10 rtol)`:

  | Pair | Added | $z_{\rm response}$ |
  |---|---:|---|
  | (−32, −8) | **48** | 12 distinct, 8.327 to 174.13 |
  | (−32, −7) | 9 | 43.72, 75.99, 693.47, 914.24, 1 205.29, 1 589.00, 4 800.13 |
  | (−32, −6) | 9 | the same seven |
  | (−30, −8) | 9 | the same seven |
  | (−28, −8) | 9 | the same seven |
  | (−25, −8) | 9 | the same seven |
  | (−22, −8) | 9 | the same seven |

  The backup's 7 552 are all at (−32, −8). The live store's 7 654 are 7 600 at (−32, −8) and 9
  at each of the six other pairs.

**`compare_fingerprints(live A3, sweep)`: 2 entries.**
```
{"kind": "tag_set", "class": "QuadSourceIntegral", "tags": [Q], "recorded": 7654, "current": 7706, …}
{"kind": "tag_set", "class": "tolerance", "tags": [], "recorded": 12, "current": 13, …}
```
- **`tolerance`: 1 record only in the sweep, `log10_tol` −5 (`-0x1.4000000000000p+2`), and none
  only in the live store.**
- **`QuadSourceIntegral`: 52 records only in the sweep, and none only in the live store.** All are
  in tag set **Q**, with the same policy, model and $z_{\rm source,max}$ as above.

  | Pair | Added | $z_{\rm response}$ |
  |---|---:|---|
  | (−32, −8) | 4 | 0.1, 0.302, 6.317 |
  | (−32, −7) | 16 | 10 distinct, 0.1 to $1.456\times10^{11}$ |
  | (−32, −6) | 16 | the same 10 |
  | (−32, −5) | 16 | 12 distinct, 0.1 to 4 800.13 |

  The sweep's 7 706 are:
  - 7 604 at (−32, −8);
  - 25 at each of (−32, −7) and (−32, −6);
  - 16 at (−32, −5);
  - 9 at each of (−30, −8), (−28, −8), (−25, −8) and (−22, −8).

**The expectation holds in both pairs.** Only `tolerance` and `QuadSourceIntegral` differ, each in
its one tag set, and only by additions. The added line counts are exactly the count differences:
6 and 102, then 1 and 52. **No record is only in the older store, in any class.** The other 19
classes have identical record lines, line for line.

### Findings, and the operations that explain them

None of these is a stop (§3 choice 2), and each is explained by a recorded operation. They were
attributed from the rows' `label` (whose prefix is the job name, `main.py:3427`) and `timestamp`,
read with stdlib `sqlite3` `mode=ro` (deviation 1), and from the run records. The rows carry no run
id, so the attribution is by job name and time window.

1. **The 48 production-pair records added to the live store are the registered resume's.** Their
   label prefix is `handover-A3-baseline-lcdm`. They are stamped 2026-09-23, 4 in the 02h hour and
   44 in the 09h hour, inside `handover-03-a3-baseline-resume-20260923T024847`'s window (02:48 to
   10:52). This is the resume's last session, which the `handover` board records as "48 items in
   6 h 55 m" at $z \ge 8.3$ (`[a3-baseline-quadsource-integrals-are-1680-short]`).
2. **The 54 records at six non-production pairs in the live store are the first sweep run's**,
   `handover--quadsource-atol-sweep-20260923T110622` (11:06 to 11:23). Their label prefix is
   `handover-atol-sweep`, and they are stamped in the 11h hour. The first sweep run's
   `results` was the sweep store, but it wrote into the live A3 store, because the copied primary
   still named the live shards. That is recorded in `2ebb7b6`'s commit message, in
   `docs/handover/quadsource_atol_sweep.py`'s `assert_store_is_self_consistent`, on the `handover`
   board, and in `run-registry`'s `[04-sharded-store-paths-are-absolute-…]`. The live shards'
   mtimes agree: 11:11:00 and 11:22:04 on 2026-09-23.
3. **The resume added no tolerance row.** The six added `tolerance` rows are serials 17–22,
   stamped 2026-09-23 11:06:30, 11:07:05, 11:07:40, 11:08:16, 11:09:57 and 11:11:00. That is one
   at the start of each of the first sweep run's six non-production pairs, in the order its
   `stdout.log` runs them: 1e-22, 1e-25, 1e-28 and 1e-30 for atol, then 1e-6 and 1e-7 for rtol.
   This answers §2's "nothing records … why a resume added tolerance rows". It did not add them:
   the sweep did, into the live store.
4. **Every live-A3 record is in the sweep store**, the first sweep run's 54 included. §2 said
   that run's products "may not be in it". They are, because they were written into the live
   store, which `--prepare --force` copied at 12:41.
5. **The sweep store's 52 added records and 1 added tolerance value** are:
   - **9 at (−32, −5), stamped in the 12h hour**, by the unregistered verification child described
     in `2ebb7b6`'s commit message: "a child at a tolerance pair not previously present
     (atol=1e-32, rtol=1e-5) … computed and stored its nine rows". The `tolerance` row −5 is
     serial 23, stamped 12:41:25;
   - **21 by `…-lowz-20260923T132309`** (13:23 to 15:42), 7 at each of rtol −5, −6 and −7;
   - **18 by `…-seam-20260923T225928`** (22:59 to 23:01), 9 at each of rtol −6 and −7. The
     seam's (−32, −8) pair added nothing: those nine work items were already production rows;
   - **4 by `…-lowz-check-20260924T001705`** (00:17 to 02:58), at (−32, −8), stamped in the 02h
     hour on 2026-09-24.

   That is 9 + 21 + 18 + 4 = 52.
6. **No problem of any kind in any of the three stores**, so no replicated divergence. All 12
   replicated classes, including `BackgroundModel` with its tags and its value count, hold the
   same records on all four shards of each store. This is the measurement
   `[00-replicated-writes-can-diverge-across-shards]` waited for (board §3).
7. **The backup's primary names the live store's shards.** It is read correctly, as siblings in
   its own directory. This is `datastore-portability`'s legacy-path behaviour, working as designed.

No new issue: every difference is an addition that a recorded operation explains.

### Phase B — the writes

Before the first write, `git status --porcelain` was empty, `HEAD` was
`50a24acb23e41a68f508be63db509c10b2ac6d7e`, and `RunRegistry list` showed nothing `running`. The
same held before each of the other two writes.

For each store, in the order **sweep, backup, live A3**:
- `store fingerprint <primary> --write` exited 0, and printed `>> wrote the fingerprint into the
  sidecar; it replaced none`. Without that line, its output is byte-identical to Phase A's first
  read-only run.
- **The sidecar changed in one field.**
  - Its keys are the old ones plus `fingerprint`: 8 → 9 for the sweep, 16 → 17 for the backup and
    17 → 18 for the live A3 store.
  - The new file is in the writer's byte form.
  - **Deleting `fingerprint` and re-serialising (`json.dumps(indent=2, sort_keys=True) + "\n"`)
    gives the bytes of `sidecars-before/` exactly.**
  - No other field differs in value.
- **The written fingerprint equals Phase A's `fingerprint_store` result apart from `taken`**,
  whole dicts compared. Its `taken` is `{"when", "git_head": "50a24acb…", "git_dirty": false,
  "run_id": null}`.
- **Of the store's files only the sidecar changed.** The snapshot after the write differs from
  the one before it in two entries: the sidecar's size, mtime and SHA-256, and the mtime of the
  directory holding it (deviation 2). The listing is unchanged.
- `store fingerprint <primary>`, read-only: exit 0, with `recorded: matches the fingerprint taken
  <when> by a person, at 50a24acb23e41a68f508be63db509c10b2ac6d7e`. The digest lines are unchanged.
  `store show` gives `kind: registry`, `problems: none`, and prints the fingerprint with its
  overall digest. The snapshot after these two was identical to the one after the write.

**How to undo a write.** Parse the sidecar, delete its `fingerprint` key, and write it back in the
writer's form: `json.dump(fields, handle, indent=2, sort_keys=True)`, then `"\n"`, which is
`RunRegistry.write_json_atomic`. Phase B step 2 proved that this gives back each sidecar's exact
bytes. The SHA-256s to expect are those in "What shipped". The procedure is a person's, and no
code does it.

### Phase C — a registry copy of the live A3 store

1. `python -m RunRegistry store copy var/datastores/handover-A3-baseline-lambdacdm.sqlite
   var/store-fingerprint-check-05/copy/handover-A3-baseline-lambdacdm.sqlite --purpose "…"`, with
   the default runs root. It exited 0 in 3.5 s. The snapshot after it is **identical** to the one
   before, so the live store's files and sidecar are unchanged.
2. **The copy's sidecar:**
   - carries the live sidecar's `fingerprint` **verbatim**, `taken` included (whole dicts
     compared);
   - has a new `store_id`, `8870025f1220471697d7e6ca71d1674f`;
   - has `copied_from` `{"datastore": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
     "store_id": "5f58ac536362424499a4e94c6242928e"}`;
   - has history `adopt`, `copy`.

   The only other fields that differ are `created` and `purpose`.
3. **`store fingerprint` on the copy**: exit 0, `recorded: matches the fingerprint taken
   2026-09-25T07:32:35+01:00 by a person, at 50a24acb…`. Its overall digest is `433b7fc3…`, and its
   whole per-class table is identical to the live store's. This answers the two-machine question
   for a real store.

### Phase D — the end

1. **The final snapshot against the first** gives five differences:
   - the three sidecars, with the new SHA-256 above;
   - the mtimes of `var/datastores/` and of the backup's directory (deviation 2).

   Every other file, all 15 store files included, is byte-identical, with the same size and
   mtime. Both directory listings are identical, and no `.tmp` is left. `ls var/runs` is
   identical (`cmp`). `RunRegistry list` shows the same 7 runs, 5 finished and 2 `unknown`, and
   differs from the first listing only in the age column. **Nothing was created under
   `var/runs/`.**
2. `logs/05-fingerprints.json` was written from the three sidecars, and each value parses back
   equal to its sidecar's.
3. **`var/store-fingerprint-check-05/` was deleted.** `var/` holds `.DS_Store`,
   `bootstrap-a3-resume.log`, `datastores` and `runs`, as before.
4. **Suites.** No code changed, so each was expected to match its baseline, and each does:

   | Suite | Baseline at `50a24ac` (orchestrator) | After |
   |---|---|---|
   | `AdaptiveLevin` | 32 OK | 32 OK |
   | `ComputeTargets` | 552 OK (the wall-clock flake is known) | 552 OK |
   | `CosmologyModels` | 39 OK | 39 OK |
   | `Datastore` | 177 OK | 177 OK |
   | `LiouvilleGreen` | 148 OK (skipped=1) | 148 OK (skipped=1) |
   | `RunRegistry` | 117 OK | 117 OK |

   All six were run concurrently, each with
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`.

### Scope

`git diff HEAD~1 HEAD --stat` touches only:
- this log;
- `logs/05-fingerprints.json`;
- `prompts/store-fingerprint/IMPLEMENTATION_STATE.md`;
- `prompts/store-fingerprint/README.md` (§2's row for 05);
- `docs/OPEN_ISSUES.md`.

No code, no test. No Python file changed, so `black` has nothing new to check.

## Each *prompt's choice*, and whether it was kept

| Choice | Kept? |
|---|---|
| 1. Every read and every comparison comes before any write | **Kept.** Phase A, including the attribution reads of deviation 1, was complete before the first `--write`. |
| 2. A finding about a store is recorded, not stopped on; a doubt about the fingerprint is a stop | **Kept.** Seven findings are recorded above. None needed an issue, because each is explained by a recorded operation. No doubt about the fingerprint arose. Three read-only readings of each store agreed. The counts equal the independent counts, and the listings hash to the digests. The sweep's digest is prompt 04's, and the backup reads its own shards. |
| 3. The writes are taken by a person, with the tree clean | **Kept.** Every `taken` names `50a24acb…`, `git_dirty` false and `run_id` null. The first repository file was created after the third write. |
| 4. The registry copy is of the live A3 store | **Kept.** |
| 5. The three fingerprints, as written, are committed beside the log | **Kept**, as values in the writer's serialisation (deviation 4). |
| 6. The writes are in the order sweep, backup, live A3 | **Kept.** |

## The deliberate-breakage record

**This prompt writes no code, so it has no tests and no mutations.** README §5 rule 8 asks each
prompt to name mutations its tests must catch. There is nothing here to mutate that this prompt
added, and the fingerprint's own mutations are prompt 04's nine, all caught.

Phase A step 5 plays the part of a discriminator on real data. Two differences, known beforehand
only as row counts, were each localised by the digests to exactly two classes and one tag set
each. The records the listings named are exactly the count differences, 102 + 6 and 52 + 1, with
no removals. Each was then attributed to the operation that wrote it. A fingerprint that digested
the wrong thing would have named other classes, or records that no operation explains.

## Observations not acted on

- **The live A3 sidecar's `run_history` does not mention the 2026-09-23 resume or the 54 sweep
  records.** Its last entry is the 2026-09-21 attempt. The run registry records the resume
  (`var/runs/handover-03-a3-baseline-resume-20260923T024847`), and the `handover` board records
  the 54. That field is a human note, and this prompt may write only `fingerprint`. The note is
  incomplete, not wrong, so no issue is opened.
- **The live A3 fingerprint describes a store that holds 54 non-production records.** They are in
  the `QuadSourceIntegral` digest like any other. Anyone using the store as a comparator should
  select on the tolerance pair. The `handover` board already decided the store is to be
  regenerated.
- **The sweep store was written once outside any registered run**: the `2ebb7b6` verification
  child's nine rows and one `tolerance` row. Only that commit message records it. The registry
  records and does not enforce, and the fingerprint now covers those rows. No issue is opened.
- **`GkSourcePolicy` and `QuadSourcePolicy` have the same digest** on all three stores, as prompt
  04 saw on the sweep copy. This bears on `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`,
  and does not change it.
- **`store fingerprint` on the backup begins with the legacy-path `!!` line**, as prompt 04 saw on
  its copies. The live and sweep stores, whose primaries name their own shards, print none. Nor
  does the registry copy.

## State handed to the next prompt

- **This is the campaign's last prompt.** F1–F13 are done.
- **The three sidecars hold format-1 fingerprints**, taken by a person at `50a24ac`, clean.
  `python -m RunRegistry store fingerprint <primary>` on each should say `matches`. A difference
  now means the store changed after 2026-09-25 07:31–07:32.
- **`logs/05-fingerprints.json`** holds the three values, for a reader without `var/`.
- **The attribution of every `QuadSourceIntegral` record** by tolerance pair and writer is in
  Findings 1–5, for whoever regenerates the A3 store.
- **Baselines** are unchanged: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore 177, LiouvilleGreen 148 (1 skipped), RunRegistry 117.
