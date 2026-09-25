# Log 05 — Retire the two stores, and correct the one claim that retirement falsifies

**Prompt:** [`prompts/store-retirement/05-retire-the-two-stores.md`](../05-retire-the-two-stores.md)
**Commit:** *(this commit)* — "Retire the sweep store and the A3 backup"
**Model:** Claude Opus 5.5
**Date:** 2026-09-25 (Phases A and B, and the two retirements) and 2026-09-26 (the amendment, and
Phase C)
**Result:** COMPLETE. No §5 stop condition was met in any phase. The agent wrote nothing under
`var/`. The user retired the sweep store and the backup with `store retire` on 2026-09-25, and
amended the live A3 sidecar's `backup` field with `store amend` on 2026-09-26. Both retired
stores are completed tombstones, and their ten files are gone. **The live A3 store's five
`.sqlite` files are byte-identical, mtimes included,** from before the first dry run to after the
amendment. Its sidecar changed only by the amendment. No issue was opened.

This log was written as the prompt ran, in three phases. The irreversible steps between them were
the user's (README §5 rule 10).

## What shipped

**No code.** The user ran three commands, each prepared, shown and checked by the agent:
- **R10.** `store retire` on the sweep store, `var/datastores/handover-atol-sweep.sqlite`. Its four
  shards and its primary are gone, and its sidecar is a completed tombstone.
- **R11.** `store retire` on the backup,
  `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite`. Its
  four shards and its primary are gone. Every deleted file was inside its own directory, although
  its primary's records name the live store's shards. Its sidecar is a completed tombstone. Then
  `store amend` on the live A3 sidecar's `backup` field: `retained` is now `false`, and it
  names the tombstone.
- **R12.** `logs/05-tombstones.json` holds the two tombstones and the amended live sidecar,
  verbatim, because `var/` is gitignored. Dated, additive notes are on this campaign's board, the
  `run-registry` board and the `handover` board. `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` is now true,
  and was not edited.

## Deviations from the prompt

1. **The `store show` outputs in Phase A step 3 are recorded with each sidecar's
   `fingerprint.classes` block elided.** `IMPLEMENTATION CHOICE`. The three outputs are 1073 lines,
   almost all of them the per-class and per-tag-set digests. Everything else in each output is
   recorded verbatim below. The elided block is not lost: it is part of each sidecar, whose SHA-256
   is in the before-picture; it is committed already in `prompts/store-fingerprint/logs/05-fingerprints.json`;
   and Phase C's `logs/05-tombstones.json` holds all three sidecars verbatim, `fingerprint`
   included, which Phase B checks value-identical to the before-picture.
2. **The before-picture also records `var/runs/` (size and mtime of every file and directory) and
   the repository root's `physics-test-n20-*` store (size, mtime, SHA-256).** `IMPLEMENTATION
   CHOICE`. The prompt asks for `var/datastores/` only. Neither is to change, and both are cheap to
   include, so a change to either would be caught too. Neither is written by anything here.

3. **The exit codes of the user's three commands are inferred, not observed.** `IMPLEMENTATION
   CHOICE`. The prompt asks for the user's commands "with their exit codes". The outputs the user
   pasted do not show one. Each ended in its success line (`>> retired:` twice, `>> amended:`),
   which the command prints only on the path that exits 0, with empty stderr. So each is recorded
   as "exit 0, inferred", and the state each left was checked independently (B2–B3, C1–C2).

None is `UNINTENDED DRIFT`.

## Verification performed

### Phase A — prepare and show

#### A1. The tree

- Branch `handover-remedial`, `HEAD` `a8260b7825fe7084cfac3d57428e452076cf81f9` ("Write prompt 05
  of the store retirement campaign and its notes").
- `git status` is clean apart from untracked paths that are not this campaign's:
  `docs/datastore-integrity-audit.md`, `docs/datastore-integrity-audit/` and
  `prompts/datastore-integrity/`. They were left alone. From step 4 on, this log draft is untracked
  too.

#### A2. Nothing running

`./venv/bin/python -m RunRegistry list`, exit 0. Nothing is `running`:

```
   RUN                                                         STATE      PROGRESS      AGE  PURPOSE · STAGE
   handover--quadsource-atol-sweep-lowz-check-20260924T001705  done              -   1d 23h  Phase 3b: is rtol 1e-7 converged at low z, or is it another c5? Three cheapest phase-2 cases plus the control, at rtol 1e-8 only; their 1e-5/1e-6/1e-7 rungs are already stored, so this completes a four-rung ladder in the band that carries all the cost.  · low-z-check pair 1/1: atol=1e-32 rtol=1e-08 done in 9682 s
   handover--quadsource-atol-sweep-seam-20260923T225928        done              -   2d 00h  Phase 3a: the band where a numeric branch of G_k exists (z_response above the numeric solve's z_min, 6.3 e-folds inside horizon entry). Five mixed-policy and four numeric-policy work items, severe geometry, over rtol 1e-6/1e-7/1e-8 at atol 1e-32. Sub-5 s rows, so this is the cheap half of phase 3; it settles the tolerance in the only band where the consumer-side crossover and D1's numeric spline exist.  · seam pair 3/3: atol=1e-32 rtol=1e-08 done in 30 s
   handover--quadsource-atol-sweep-lowz-20260923T132309        done              -   2d 10h  Phase 2 of the QuadSourceIntegral tolerance sweep: seven work items in the z_response <= 6.32 block that no run has ever attempted, at rtol 1e-5/1e-6/1e-7 and atol 1e-32. No 1e-8 reference -- unaffordable at these eta_response -- so the test is self-convergence of the three. Decides the pair the store is regenerated at.  · low-z pair 3/3: atol=1e-32 rtol=1e-07 done in 7782 s
   handover--quadsource-atol-sweep-20260923T110622             done              -   2d 12h  Decide whether DEFAULT_QUADRATURE_ATOL=1e-32 buys any digit of QuadSourceIntegral's total that depth-20 Levin bisection was not already failing to deliver. Nine fixed production work items x seven tolerance pairs, on a copy of the A3 baseline store; the baseline store itself is untouched. Opened by [a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles].  · pair 7/7: atol=1e-32 rtol=1e-08 done in 54 s
   handover-03-a3-baseline-resume-20260923T024847              failed            -   2d 20h  Finish the interrupted A3 policy-geometry baseline: the quadratic-source-integral stage on the current-grid LambdaCDM store, resumed after the numeric_quad read-back fix (210f0a0). Policy rows are already complete; this stage is for QuadSourceIntegral outcomes useful later.  · CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaini
   a3-pilot-resume                                             unknown           -   4d 14h  (no manifest — predates the registry, or was not begun through it)
   a3-pilot                                                    unknown           -   4d 14h  (no manifest — predates the registry, or was not begun through it)

7 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 5 finished, 2 unknown.
```

#### A3. The before-picture

Taken 2026-09-25 by a read-only helper outside the tree (every file opened `rb`, `os.lstat` for
size and `st_mtime_ns`). `var/datastores/` holds 18 files in 2 directories, and nothing else:
no `.tmp`, no `.incomplete-move`, no journal, no `-wal` or `-shm` file. `var/runs/` holds 32 files
in 8 directories. The orchestrator's independent snapshot, taken before dispatch, agrees with this
one on every entry it holds (56 files and 10 directories: size, mtime and, where it has one, SHA-256).

Two things in it are expected and worth saying. The backup's primary and the live A3 primary are
byte-identical (`fdbe93a5…`, same size and mtime): the backup's primary is a copy whose four
`shards` rows are the live primary's absolute records (audit §2.7). And every sidecar's mtime is
from its fingerprint write of 2026-09-25 07:31–07:32.

#### The live A3 store — primary, four shards and sidecar

| File | Size | mtime_ns | SHA-256 |
|---|---|---|---|
| `var/datastores/handover-A3-baseline-lambdacdm-shard0000.sqlite` | 87367680 | 1790158924459056164 | `a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6` |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0001.sqlite` | 87302144 | 1790158260177696451 | `9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4` |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0002.sqlite` | 87711744 | 1790158924456294411 | `d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe` |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0003.sqlite` | 86892544 | 1790158924442869520 | `13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083` |
| `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` | 14205 | 1790317956713717437 | `394ff933c30c755f7beae03b9f9da4fad1651266d9730ab74dd412ceb851732a` |
| `var/datastores/handover-A3-baseline-lambdacdm.sqlite` | 32768 | 1789934590990297815 | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` |

#### The sweep store

| File | Size | mtime_ns | SHA-256 |
|---|---|---|---|
| `var/datastores/handover-atol-sweep-shard0000.sqlite` | 87482368 | 1790215104478790948 | `4000a990c1ae6ec0f24fd3bc67fbbe3b939336aae0751f1818438f82c9cbf22e` |
| `var/datastores/handover-atol-sweep-shard0001.sqlite` | 87334912 | 1790200842714567098 | `8cf35f4c9758e563cf5674c35f0464d9b453107e5d4a33db2511ad1547f6dd21` |
| `var/datastores/handover-atol-sweep-shard0002.sqlite` | 87814144 | 1790215104451901365 | `6c6995adc027a5bde97b8de8218166b3bc4dc45469a148b1b1c42d6aab165056` |
| `var/datastores/handover-atol-sweep-shard0003.sqlite` | 86994944 | 1790215104448929341 | `362ad59e5f62e28c720525f6895bd32953ab3b78042ec3fb20edf4ee581f8889` |
| `var/datastores/handover-atol-sweep.manifest.json` | 10564 | 1790317882346772856 | `f7b705bd1335f006dfad67e91ef40a3b786c83e0d8f63703bb71c5a5dfb3f834` |
| `var/datastores/handover-atol-sweep.sqlite` | 32768 | 1790163664518981413 | `c3cda49d8e3518064bb4b7112e9df7c98f35479ad8051d2a4c0a230a41db956e` |

#### The backup

| File | Size | mtime_ns | SHA-256 |
|---|---|---|---|
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite` | 87044096 | 1789961530881608041 | `6667d5588146038ab98f83f31db51facd11468e47f510e7bab284d1d0326c048` |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite` | 87302144 | 1789961544964306854 | `d0ad78e6a9eabb7acd42ad8cca387ef652f7edaa901fb4ec4ade3581286dd63f` |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite` | 87588864 | 1789961544984656759 | `faeca5d7198f3411a76a8fb9aec422ee5ae28a7b62271752ecc45fc89254bf14` |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite` | 86609920 | 1789961544992746604 | `83f6288df912c5e46493a85dddcf27c2fc20cd0ee533f2268becb63894420a59` |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json` | 12763 | 1790317931376053120 | `ea1ff60d7f33fda16b7a0b961523f3e8f88f2d8497c704f7519a6674d7e95088` |
| `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite` | 32768 | 1789934590990297815 | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` |

#### Directory listings under `var/datastores/`

| Directory | mtime_ns | Entries |
|---|---|---|
| `var/datastores/` | 1790317956713981296 | `backup-pre-resume-20260921T091011`, `handover-A3-baseline-lambdacdm-shard0000.sqlite`, `handover-A3-baseline-lambdacdm-shard0001.sqlite`, `handover-A3-baseline-lambdacdm-shard0002.sqlite`, `handover-A3-baseline-lambdacdm-shard0003.sqlite`, `handover-A3-baseline-lambdacdm.manifest.json`, `handover-A3-baseline-lambdacdm.sqlite`, `handover-atol-sweep-shard0000.sqlite`, `handover-atol-sweep-shard0001.sqlite`, `handover-atol-sweep-shard0002.sqlite`, `handover-atol-sweep-shard0003.sqlite`, `handover-atol-sweep.manifest.json`, `handover-atol-sweep.sqlite` |
| `var/datastores/backup-pre-resume-20260921T091011/` | 1790317931376335450 | `handover-A3-baseline-lambdacdm-shard0000.sqlite`, `handover-A3-baseline-lambdacdm-shard0001.sqlite`, `handover-A3-baseline-lambdacdm-shard0002.sqlite`, `handover-A3-baseline-lambdacdm-shard0003.sqlite`, `handover-A3-baseline-lambdacdm.manifest.json`, `handover-A3-baseline-lambdacdm.sqlite` |

#### Outside `var/`: the repository root's pre-registry `physics-test-n20-*` store (for reference only)

| File | Size | mtime_ns | SHA-256 |
|---|---|---|---|
| `physics-test-n20-lambdacdm-zend0p1-profile.sqlite` | 6971392 | 1789044464407111071 | `ca7388f9f2aa2a566a72818db30ca0f15bd474e07fd90f6b24478bbd7f807440` |
| `physics-test-n20-lambdacdm-zend0p1-shard0000.sqlite` | 19722240 | 1789044400526507924 | `b11dccff29fa3fe8def97dede9323992d5052762b7a6ccfb92e7012204168c2a` |
| `physics-test-n20-lambdacdm-zend0p1-shard0001.sqlite` | 19271680 | 1789044400542626780 | `76e5049a5a18218bb254bec1c57fd48ab2ff1b39b431f77f57a7b5bf1866e550` |
| `physics-test-n20-lambdacdm-zend0p1-shard0002.sqlite` | 18235392 | 1789044400535128728 | `603857abf7558fbdc4cfad54c2d6f497009757b7bee634f86ebf680ca35063b3` |
| `physics-test-n20-lambdacdm-zend0p1-shard0003.sqlite` | 21245952 | 1789044400461920958 | `f1f0aabd4206d2f11e5033ac82d7f3ef344f5e78fbd14c1eb93da5996a7e809d` |
| `physics-test-n20-lambdacdm-zend0p1.sqlite` | 32768 | 1789041316588073487 | `f8eb57ff0b253356ebac4a28fc15456509fb92055b7ab4b878204f3c9a3aae88` |

#### A3, continued. `store show` on each primary

Each exits 0, reads `kind: registry` and `problems: none`, and agrees with the prompt's §1 table:
`store_id`, recorded digest and one `adopt` entry each. The `fingerprint.classes` block is elided
(Deviation 1). Everything else is verbatim.

```
===== store show var/datastores/handover-atol-sweep.sqlite
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.manifest.json
kind:     registry
problems: none
datastore: 'handover-atol-sweep.sqlite'
fields:
  {
    "copied_from": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
    "created": "2026-09-23T12:41:04+0100",
    "datastore": "handover-atol-sweep.sqlite",
    "fingerprint": {
      "classes": { ... elided: the per-class digests, verbatim in the sidecar, whose SHA-256 is in the before-picture ... },
      "digest": "2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:31:21+01:00"
      }
    },
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/handover-atol-sweep.sqlite",
        "when": "2026-09-24T22:03:28+01:00"
      }
    ],
    "name": "handover-atol-sweep",
    "purpose": "Working copy of the A3 baseline store for docs/handover/quadsource_atol_sweep.py. Disposable: every row that is not at the production tolerance pair belongs to a sweep and nothing else reads it. The comparator is handover-A3-baseline-lambdacdm, not this.",
    "sidecar_format": 1,
    "store_id": "04198f22f8704c52a252a72566af94ed"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs:
  handover--quadsource-atol-sweep-lowz-check-20260924T001705  done      finished  by results
  handover--quadsource-atol-sweep-seam-20260923T225928  done      finished  by results
  handover--quadsource-atol-sweep-lowz-20260923T132309  done      finished  by results
  handover--quadsource-atol-sweep-20260923T110622  done      finished  by results
exit 0
===== store show var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json
kind:     registry
problems: none
datastore: 'handover-A3-baseline-lambdacdm.sqlite'
fields:
  {
    "created": "2026-09-20T21:02:30+01:00",
    "datastore": "handover-A3-baseline-lambdacdm.sqlite",
    "driver": "docs/gktk-remedial/scoped_pipeline_run.py (NOT the source-remediation copy, which is broken by [13-scoped-run-driver-k-grid-literal])",
    "fingerprint": {
      "classes": { ... elided: the per-class digests, verbatim in the sidecar, whose SHA-256 is in the before-picture ... },
      "digest": "eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:32:10+01:00"
      }
    },
    "git_dirty": true,
    "git_head": "0d7c05c0ddf5b6cd03763307c6a233d70c638fec",
    "grid_criterion": "post qcd-background-audit prompt 15 (measured curvature criterion)",
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite",
        "when": "2026-09-24T22:03:28+01:00"
      }
    ],
    "name": "handover-A3-baseline-lambdacdm",
    "note": "Pilot: purpose is to measure cost and mixed-row yield at full k span before sizing anything larger.",
    "purpose": "Baseline datastore for prompts/handover A3 (policy-geometry census, D5). The before-picture that D1 is scored against: current source grid, current hand-over. Keep.",
    "restart": {
      "command": "PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py --k-min 1e5 --k-max 3e8 --k-count 8 --cpus 10 --models LambdaCDM --allow-existing -- --database var/datastores/handover-A3-baseline-lambdacdm.sqlite --job-name handover-A3-baseline-lcdm --shards 4 --zend 0.1 --source-samples-log10z 100",
      "note": "--allow-existing is REQUIRED: the driver refuses an existing datastore path without it. The pipeline looks up completed work and should resume rather than recompute; verify that from the lookup-queue counts in the first minutes rather than assuming it."
    },
    "run_history": [
      {
        "elapsed": "10h 33m",
        "note": "The GkSourcePolicyData stage COMPLETED. A3's census reads those rows, so the census may already be runnable against this store without finishing the quadratic-source-integral stage.",
        "outcome": "INTERRUPTED, not failed. Exited cleanly on SIGTERM; all five sqlite files pass PRAGMA integrity_check.",
        "size_at_stop": "385 MB across 4 shards",
        "stage_reached": "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)",
        "stages_complete": [
          "background",
          "numerical tensor Green functions",
          "APPLY GKSOURCE POLICIES",
          "GkSourcePolicyData SUMMARY STATISTICS"
        ],
        "started": "2026-09-20T21:03:04+01:00",
        "stopped": "2026-09-21T07:37 (approx, SIGTERM by request so the laptop could sleep)"
      }
    ],
    "scope": "8 log-spaced k over 1e5-3e8 /Mpc, the FULL production span (the 2026-09 verification runs used 1e5-1e7 only), LambdaCDM, production redshift geometry: --zend 0.1, 100 source samples per log10 z, default response sparseness 12",
    "sidecar_format": 1,
    "status_files": {
      "pid": "var/runs/a3-pilot/run.pid",
      "stderr": "var/runs/a3-pilot/run.err",
      "stdout": "var/runs/a3-pilot/run.out"
    },
    "store_id": "4c2ce77b0bf24671bcebd8dda3441041"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: none
exit 0
===== store show var/datastores/handover-A3-baseline-lambdacdm.sqlite
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-A3-baseline-lambdacdm.manifest.json
kind:     registry
problems: none
datastore: 'handover-A3-baseline-lambdacdm.sqlite'
fields:
  {
    "backup": {
      "path": "var/datastores/backup-pre-resume-20260921T091011",
      "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly.",
      "retained": true
    },
    "created": "2026-09-20T21:02:30+01:00",
    "datastore": "handover-A3-baseline-lambdacdm.sqlite",
    "driver": "docs/gktk-remedial/scoped_pipeline_run.py (NOT the source-remediation copy, which is broken by [13-scoped-run-driver-k-grid-literal])",
    "fingerprint": {
      "classes": { ... elided: the per-class digests, verbatim in the sidecar, whose SHA-256 is in the before-picture ... },
      "digest": "433b7fc3cad94a971020c0493e4062e4d3a7a590a6f084c6a8e8f451f2b648a3",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:32:35+01:00"
      }
    },
    "git_dirty": true,
    "git_head": "0d7c05c0ddf5b6cd03763307c6a233d70c638fec",
    "grid_criterion": "post qcd-background-audit prompt 15 (measured curvature criterion)",
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
        "when": "2026-09-24T22:03:27+01:00"
      }
    ],
    "name": "handover-A3-baseline-lambdacdm",
    "note": "Pilot: purpose is to measure cost and mixed-row yield at full k span before sizing anything larger.",
    "purpose": "Baseline datastore for prompts/handover A3 (policy-geometry census, D5). The before-picture that D1 is scored against: current source grid, current hand-over. Keep.",
    "restart": {
      "command": "PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py --k-min 1e5 --k-max 3e8 --k-count 8 --cpus 10 --models LambdaCDM --allow-existing -- --database var/datastores/handover-A3-baseline-lambdacdm.sqlite --job-name handover-A3-baseline-lcdm --shards 4 --zend 0.1 --source-samples-log10z 100",
      "note": "--allow-existing is REQUIRED: the driver refuses an existing datastore path without it. The pipeline looks up completed work and should resume rather than recompute; verify that from the lookup-queue counts in the first minutes rather than assuming it."
    },
    "run_history": [
      {
        "elapsed": "10h 33m",
        "note": "The GkSourcePolicyData stage COMPLETED. A3's census reads those rows, so the census may already be runnable against this store without finishing the quadratic-source-integral stage.",
        "outcome": "INTERRUPTED, not failed. Exited cleanly on SIGTERM; all five sqlite files pass PRAGMA integrity_check.",
        "size_at_stop": "385 MB across 4 shards",
        "stage_reached": "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)",
        "stages_complete": [
          "background",
          "numerical tensor Green functions",
          "APPLY GKSOURCE POLICIES",
          "GkSourcePolicyData SUMMARY STATISTICS"
        ],
        "started": "2026-09-20T21:03:04+01:00",
        "stopped": "2026-09-21T07:37 (approx, SIGTERM by request so the laptop could sleep)"
      },
      {
        "consequence": "A3 is NOT blocked: its census reads GkSourcePolicyData (1160 rows, complete). The ~7552 QuadSourceIntegral rows are written but cannot be read back until the one-line fix lands, and the last work item cannot be completed.",
        "datastore_state": "UNDAMAGED. Live vs pre-resume backup: GkSourcePolicyData 290/290/290/290 and QuadSourceIntegral 1890/1888/1889/1885 on all four shards, identical. Pruning removed nothing.",
        "defect": "Datastore/SQL/ObjectFactories/QuadSourceIntegral.py: the SELECT at :234 omits table.c.numeric_quad, while build() at :324 reads row_data.numeric_quad. The sibling SELECT at :530 does include it. Latent: a fresh run computes rows and never reads them back, so only a RESUME (or any object_get of a stored row) can hit it.",
        "error": "sqlalchemy.exc.NoSuchColumnError: Could not locate column in row for column 'numeric_quad'",
        "outcome": "FAILED at the quadratic-source-integral stage. Resume replayed every earlier stage correctly (background, numeric and WKB Green functions, GkSource, APPLY GKSOURCE POLICIES, summary statistics) and then crashed reading back an EXISTING QuadSourceIntegral row.",
        "started": "2026-09-21T09:11 (resume attempt, --allow-existing --prune-unvalidated)"
      }
    ],
    "scope": "8 log-spaced k over 1e5-3e8 /Mpc, the FULL production span (the 2026-09 verification runs used 1e5-1e7 only), LambdaCDM, production redshift geometry: --zend 0.1, 100 source samples per log10 z, default response sparseness 12",
    "sidecar_format": 1,
    "status_files": {
      "pid": "var/runs/a3-pilot/run.pid",
      "stderr": "var/runs/a3-pilot/run.err",
      "stdout": "var/runs/a3-pilot/run.out"
    },
    "store_id": "5f58ac536362424499a4e94c6242928e"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs:
  handover-03-a3-baseline-resume-20260923T024847  failed    finished  by results
exit 0
```

#### A4. The reasons

Drafted from the prompt's §2 step 4, joined into one line with single spaces, and extracted from
the prompt text programmatically so that the join is exact. Neither contains a single quote or a
backtick.

- **Sweep** (312 characters):

  ```
  Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.
  ```
- **Backup** (348 characters):

  ```
  Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.
  ```

The reason recorded in a tombstone is final (log 03): the real run, and any completion, must use
the same text character for character. If the user rewords either, its dry run is run again with
the final text before the user runs the command.

#### A5. The dry runs

Both run from the repository root, with the default roots `var/runs/` and `var/datastores/`. Each
exits 0, with empty stderr.

**The sweep.**

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/handover-atol-sweep.sqlite --reason 'Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.' --dry-run
```

Exit 0. Stdout, verbatim:

```
store:    /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.sqlite
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.manifest.json
dry run:  nothing was written or deleted
references:
  runs naming it, under var/runs: 4
    handover--quadsource-atol-sweep-lowz-check-20260924T001705  done      by results
    handover--quadsource-atol-sweep-seam-20260923T225928  done      by results
    handover--quadsource-atol-sweep-lowz-20260923T132309  done      by results
    handover--quadsource-atol-sweep-20260923T110622  done      by results
  sidecars naming it, under var/datastores: 0
  not searched: sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files
fingerprint check:
  matched: a fresh read-only fingerprint matched the recorded one, digest 2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18
files to be deleted:
  var/datastores/handover-atol-sweep-shard0000.sqlite
  var/datastores/handover-atol-sweep-shard0001.sqlite
  var/datastores/handover-atol-sweep-shard0002.sqlite
  var/datastores/handover-atol-sweep-shard0003.sqlite
  var/datastores/handover-atol-sweep.sqlite
tombstone:
  {
    "completed": null,
    "files": [
      "var/datastores/handover-atol-sweep-shard0000.sqlite",
      "var/datastores/handover-atol-sweep-shard0001.sqlite",
      "var/datastores/handover-atol-sweep-shard0002.sqlite",
      "var/datastores/handover-atol-sweep-shard0003.sqlite",
      "var/datastores/handover-atol-sweep.sqlite"
    ],
    "files_present_only": false,
    "fingerprint": {
      "condition": "matched",
      "digest": "2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18"
    },
    "git_dirty": true,
    "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
    "reason": "Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.",
    "references": {
      "not_searched": "sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files",
      "runs": [
        {
          "id": "handover--quadsource-atol-sweep-lowz-check-20260924T001705",
          "matched_by": [
            "results"
          ],
          "state": "done"
        },
        {
          "id": "handover--quadsource-atol-sweep-seam-20260923T225928",
          "matched_by": [
            "results"
          ],
          "state": "done"
        },
        {
          "id": "handover--quadsource-atol-sweep-lowz-20260923T132309",
          "matched_by": [
            "results"
          ],
          "state": "done"
        },
        {
          "id": "handover--quadsource-atol-sweep-20260923T110622",
          "matched_by": [
            "results"
          ],
          "state": "done"
        }
      ],
      "runs_root": "var/runs",
      "sidecars": [],
      "stores_root": "var/datastores"
    },
    "state": "retiring",
    "when": "2026-09-25T23:29:50+01:00"
  }
```

**The backup.**

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite --reason 'Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.' --dry-run
```

Exit 0. Stdout, verbatim:

```
!! Primary database "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite" records 4 shard(s) by legacy absolute path in "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores"; reading them as siblings in "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011" instead (stored records not rewritten)
!! Primary database "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite" records 4 shard(s) by legacy absolute path in "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores"; reading them as siblings in "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011" instead (stored records not rewritten)
store:    /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json
dry run:  nothing was written or deleted
references:
  runs naming it, under var/runs: 0
  sidecars naming it, under var/datastores: 1
    var/datastores/handover-A3-baseline-lambdacdm.manifest.json: $.backup.path
  not searched: sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files
fingerprint check:
  matched: a fresh read-only fingerprint matched the recorded one, digest eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222
files to be deleted:
  var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite
  var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite
  var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite
  var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite
  var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite
tombstone:
  {
    "completed": null,
    "files": [
      "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite",
      "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite",
      "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite",
      "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite",
      "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite"
    ],
    "files_present_only": false,
    "fingerprint": {
      "condition": "matched",
      "digest": "eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222"
    },
    "git_dirty": true,
    "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
    "reason": "Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.",
    "references": {
      "not_searched": "sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files",
      "runs": [],
      "runs_root": "var/runs",
      "sidecars": [
        {
          "fields": [
            "$.backup.path"
          ],
          "sidecar": "var/datastores/handover-A3-baseline-lambdacdm.manifest.json"
        }
      ],
      "stores_root": "var/datastores"
    },
    "state": "retiring",
    "when": "2026-09-25T23:30:10+01:00"
  }
```

**The checks.**

| Check | Sweep | Backup |
|---|---|---|
| exit code, stderr | 0, empty | 0, empty |
| `files to be deleted:` | exactly the four shards, `shard0000`…`shard0003` in ascending serial, then the primary, all in `var/datastores/` and all named `handover-atol-sweep*`. No `handover-A3-baseline-lambdacdm*` path | exactly the four shards in ascending serial, then the primary. **Every path is inside `var/datastores/backup-pre-resume-20260921T091011/`**; none names `var/datastores/handover-A3-baseline-lambdacdm*` (checked programmatically too) |
| the resolver's notice | none, as expected: the sweep's records are its own siblings' | present, twice, on stdout. It is printed by `ShardedPool._resolve_shard_rows` (`ShardedPool.py:596`), and the primary's `shards` table is read through it twice: for the plan (`closed_store_files`) and for the fresh fingerprint (`read_inventory`). It reads `records 4 shard(s) by legacy absolute path in ".../var/datastores"; reading them as siblings in ".../var/datastores/backup-pre-resume-20260921T091011" instead (stored records not rewritten)`. This is audit §2.7's danger, handled: the stored records name the live store's shards, and the one resolver reads them as the backup's own |
| `fingerprint check:` | `matched`, digest `2c2dde68…4da5b18`, equal to §1's recorded digest | `matched`, digest `eedcdfb2…888db222`, equal to §1's recorded digest |
| `--without-fingerprint` suggested? | no | no |
| `references:`, runs | four, each `done` and matched `by results`: `…-atol-sweep-20260923T110622`, `…-lowz-20260923T132309`, `…-seam-20260923T225928`, `…-lowz-check-20260924T001705`. **Equal to §1's prediction** | none. **Equal to §1's prediction** |
| `references:`, sidecars | none. **Equal to §1's prediction.** Log 03's observation 4 warned that this report could be broad, because the sweep store sits directly in `var/datastores/`. No other sidecar's string resolves to that directory: the live A3 sidecar's `backup.path` resolves to the backup's directory, and its `restart.command` is a whole command line, not a path | one: `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` at `$.backup.path`. **Equal to §1's prediction** |
| roots named | `runs_root` `var/runs`, `stores_root` `var/datastores`, and `not_searched` | the same |
| `tombstone:` | state `retiring`, `completed` null, the reason exactly (compared with the text above), the five files equal to `files to be deleted:`, `files_present_only` false, the fingerprint condition, the references | the same |

**One reference no report can see, as expected.** `var/runs/a3-pilot/BACKUP_PATH` holds
`var/datastores/backup-pre-resume-20260921T091011`. It is a loose pre-registry file, not a run
manifest or a sidecar, so it is in the report's `not_searched` ("every reference that is not in a
run manifest or a sidecar"). It is a record of the past, it is out of scope (README §1), and it is
explained on the `run-registry` board in Phase C, not edited.

**`git_dirty` is `true` in both previews.** `git_provenance` counts untracked files
(`git status --porcelain`), and the tree has the three untracked paths of A1 and, from step 4,
this log draft. The real runs will record `true` for the same reason. Not a defect.

#### A6. Nothing changed

The before-picture was re-taken after both dry runs, and again after copying the three sidecars to
the scratchpad for Phase B's field-by-field comparison. Both re-takes are **byte-identical** to the
first (the snapshot files compare equal with `cmp`): every file's size, mtime and SHA-256 under
`var/datastores/`, both directories' mtimes and listings, every file and directory mtime under
`var/runs/`, and the `physics-test-n20-*` store.

#### A7. Handed back

The two dry-run outputs, the two reasons, and the two commands below, **sweep first**, were handed
to the orchestrator for the user.

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/handover-atol-sweep.sqlite --reason 'Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.'
```

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite --reason 'Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.'
```

- The first deletes `var/datastores/handover-atol-sweep-shard000{0,1,2,3}.sqlite` and then
  `var/datastores/handover-atol-sweep.sqlite`, and keeps `handover-atol-sweep.manifest.json` as its
  tombstone.
- The second deletes
  `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard000{0,1,2,3}.sqlite`
  and then that directory's `handover-A3-baseline-lambdacdm.sqlite`, and keeps that directory's
  `handover-A3-baseline-lambdacdm.manifest.json` as its tombstone.
- **Neither names any of the live A3 store's five files**,
  `var/datastores/handover-A3-baseline-lambdacdm{,-shard0000,-shard0001,-shard0002,-shard0003}.sqlite`,
  or its sidecar.

### The user's commands

Both were run by the user from the repository root, exactly as handed back in A7, sweep first. The
outputs were relayed by the orchestrator.

**User step 1, the sweep.**

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/handover-atol-sweep.sqlite --reason 'Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.'
```

- stderr empty. The output ended `>> retired: /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.manifest.json`.
  `files deleted:` listed the five sweep files. The tombstone read state `retired`, `completed`
  `2026-09-25T23:35:57+01:00`.
- **Exit 0, inferred from the `>> retired:` line; not shown in the user's output.**
- **The orchestrator's check 1**, against its pre-dispatch snapshot `orch_05_s0.json`: the only
  differences were the sweep's four shards and primary gone, its sidecar changed (SHA-256
  `f7b705bd…` → `1a498467…`, 10564 → 12976 bytes), and the mtime and listing of `var/datastores/`.
  The live A3 store's five files and sidecar, all six backup files, `var/runs/` and the
  `physics-test-n20-*` store were identical. `store show` on the sweep primary exited 0 and read as
  a completed tombstone with `problems: none`.

**User step 2, the backup.**

```bash
./venv/bin/python -m RunRegistry store retire var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite --reason 'Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.'
```

- stderr empty. The output ended `>> retired: …/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json`.
  The tombstone read state `retired`, `completed` `2026-09-25T23:36:55+01:00`, with the same five
  in-directory files deleted.
- **Exit 0, inferred from the `>> retired:` line; not shown in the user's output.**
- **One difference from the dry run:** the resolver's `!!` notice, "reading them as siblings in
  …/backup-pre-resume-20260921T091011 instead", printed **three** times, not twice. Explained
  below under B3.
- **The orchestrator's check 2**, against check 1's snapshot `orch_05_s1.json`: the only
  differences were the backup's four shards and primary gone, its sidecar changed (SHA-256
  `ea1ff60d…` → `d7625885…`, 12763 → 14927 bytes), and the mtime and listing of its directory,
  which now holds only `handover-A3-baseline-lambdacdm.manifest.json`. **The live A3 store's five
  files and its sidecar were byte-identical to `s0`, mtimes included** (SHA-256 prefixes: shard0000
  `a10f3c3c`, 0001 `9a5a4827`, 0002 `d196e8a3`, 0003 `13398fec`, primary `fdbe93a5`, sidecar
  `394ff933`). `store show` on the backup primary exited 0 and read as a completed tombstone with
  `problems: none`. Nothing was `running`.

**User step 3, the amendment.** Run after Phase B, exactly as handed back in B5, unchanged:

```bash
./venv/bin/python -m RunRegistry store amend var/datastores/handover-A3-baseline-lambdacdm.sqlite --field backup --json '{"path": "var/datastores/backup-pre-resume-20260921T091011", "retained": false, "retired": "2026-09-25T23:36:55+01:00", "tombstone": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json", "reason": "Retired with store retire (store-retirement prompt 05). The resume it guarded will never happen, because the A3 store is to be regenerated, not resumed, and store-fingerprint log 05 found nothing in it that is not also in this store."}' --reason 'The backup this field describes was retired with store retire, so retained true is no longer true. store-retirement prompt 05.'
```

- stderr empty. The output ended `>> amended: /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-A3-baseline-lambdacdm.manifest.json`.
  The history entry has `when` `2026-09-26T00:06:46+01:00`, `git_dirty` `true` and
  `git_head` `a8260b7`.
- **Exit 0, inferred from the `>> amended:` line; not shown in the user's output.**
- **The orchestrator's check 3**, against its post-Phase-B snapshot:
  - Only the live A3 sidecar changed (SHA-256 `394ff933…` → `3e43aee7a…`, 14205 → 15783
    bytes), and so did the mtime of `var/datastores/`, the directory entry that follows from the
    sidecar's atomic write.
  - Against `agent05_before_live.manifest.json`, whose SHA-256 is `s0`'s: the keys are the same,
    only `backup` and `history` differ, and `history[:-1]` equals the old history.
  - The new entry is `amend` on `backup`, with `before` `{"present": true, "value": <the old
    backup, verbatim>}` and the new value in `after`. `fingerprint` is identical.
  - `store show` on the live primary exits 0, with `kind: registry` and `problems: none`.
  - The live store's five `.sqlite` files are identical to `s0` in SHA-256, size and mtime.
  - Nothing is `running`.

### Phase B — check the retirements, and prepare the amendment

#### B1. Nothing running

`./venv/bin/python -m RunRegistry list`, exit 0. The listing is the same as A2: seven runs, five
finished and two `unknown` (no manifest), and nothing `running`.

#### B2. The tombstones

`store show` on each retired primary. Each exits 0 with empty stderr, and each begins with
`retirement:`. Verbatim, with the `fingerprint.classes` block elided (Deviation 1):

**The sweep.** `./venv/bin/python -m RunRegistry store show var/datastores/handover-atol-sweep.sqlite`, exit 0:

```
retirement:
  state:     retired
  when:      2026-09-25T23:35:57+01:00 at a8260b7825fe7084cfac3d57428e452076cf81f9 (dirty)
  reason:    Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.
  completed: 2026-09-25T23:35:57+01:00
  fingerprint:
  matched: a fresh read-only fingerprint matched the recorded one, digest 2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18
  files (every file of the store):
    var/datastores/handover-atol-sweep-shard0000.sqlite
    var/datastores/handover-atol-sweep-shard0001.sqlite
    var/datastores/handover-atol-sweep-shard0002.sqlite
    var/datastores/handover-atol-sweep-shard0003.sqlite
    var/datastores/handover-atol-sweep.sqlite
  references, when it was retired:
  runs naming it, under var/runs: 4
    handover--quadsource-atol-sweep-lowz-check-20260924T001705  done      by results
    handover--quadsource-atol-sweep-seam-20260923T225928  done      by results
    handover--quadsource-atol-sweep-lowz-20260923T132309  done      by results
    handover--quadsource-atol-sweep-20260923T110622  done      by results
  sidecars naming it, under var/datastores: 0
  not searched: sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep.manifest.json
kind:     registry
problems: none
datastore: 'handover-atol-sweep.sqlite'
fields:
  {
    "copied_from": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
    "created": "2026-09-23T12:41:04+0100",
    "datastore": "handover-atol-sweep.sqlite",
    "fingerprint": {
      "classes": { ... elided, value-identical to the before-picture (B2) ... },
      "digest": "2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:31:21+01:00"
      }
    },
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/handover-atol-sweep.sqlite",
        "when": "2026-09-24T22:03:28+01:00"
      },
      {
        "from": "var/datastores/handover-atol-sweep.sqlite",
        "git_dirty": true,
        "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
        "operation": "retire",
        "to": null,
        "when": "2026-09-25T23:35:57+01:00"
      }
    ],
    "name": "handover-atol-sweep",
    "purpose": "Working copy of the A3 baseline store for docs/handover/quadsource_atol_sweep.py. Disposable: every row that is not at the production tolerance pair belongs to a sweep and nothing else reads it. The comparator is handover-A3-baseline-lambdacdm, not this.",
    "retired": {
      "completed": "2026-09-25T23:35:57+01:00",
      "files": [
        "var/datastores/handover-atol-sweep-shard0000.sqlite",
        "var/datastores/handover-atol-sweep-shard0001.sqlite",
        "var/datastores/handover-atol-sweep-shard0002.sqlite",
        "var/datastores/handover-atol-sweep-shard0003.sqlite",
        "var/datastores/handover-atol-sweep.sqlite"
      ],
      "files_present_only": false,
      "fingerprint": {
        "condition": "matched",
        "digest": "2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18"
      },
      "git_dirty": true,
      "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
      "reason": "Disposable working copy of the A3 baseline for docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt 05.",
      "references": {
        "not_searched": "sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files",
        "runs": [
          {
            "id": "handover--quadsource-atol-sweep-lowz-check-20260924T001705",
            "matched_by": [
              "results"
            ],
            "state": "done"
          },
          {
            "id": "handover--quadsource-atol-sweep-seam-20260923T225928",
            "matched_by": [
              "results"
            ],
            "state": "done"
          },
          {
            "id": "handover--quadsource-atol-sweep-lowz-20260923T132309",
            "matched_by": [
              "results"
            ],
            "state": "done"
          },
          {
            "id": "handover--quadsource-atol-sweep-20260923T110622",
            "matched_by": [
              "results"
            ],
            "state": "done"
          }
        ],
        "runs_root": "var/runs",
        "sidecars": [],
        "stores_root": "var/datastores"
      },
      "state": "retired",
      "when": "2026-09-25T23:35:57+01:00"
    },
    "sidecar_format": 1,
    "store_id": "04198f22f8704c52a252a72566af94ed"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs:
  handover--quadsource-atol-sweep-lowz-check-20260924T001705  done      finished  by results
  handover--quadsource-atol-sweep-seam-20260923T225928  done      finished  by results
  handover--quadsource-atol-sweep-lowz-20260923T132309  done      finished  by results
  handover--quadsource-atol-sweep-20260923T110622  done      finished  by results
```

**The backup.** `./venv/bin/python -m RunRegistry store show var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite`, exit 0:

```
retirement:
  state:     retired
  when:      2026-09-25T23:36:55+01:00 at a8260b7825fe7084cfac3d57428e452076cf81f9 (dirty)
  reason:    Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.
  completed: 2026-09-25T23:36:55+01:00
  fingerprint:
  matched: a fresh read-only fingerprint matched the recorded one, digest eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222
  files (every file of the store):
    var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite
    var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite
    var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite
    var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite
    var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite
  references, when it was retired:
  runs naming it, under var/runs: 0
  sidecars naming it, under var/datastores: 1
    var/datastores/handover-A3-baseline-lambdacdm.manifest.json: $.backup.path
  not searched: sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json
kind:     registry
problems: none
datastore: 'handover-A3-baseline-lambdacdm.sqlite'
fields:
  {
    "created": "2026-09-20T21:02:30+01:00",
    "datastore": "handover-A3-baseline-lambdacdm.sqlite",
    "driver": "docs/gktk-remedial/scoped_pipeline_run.py (NOT the source-remediation copy, which is broken by [13-scoped-run-driver-k-grid-literal])",
    "fingerprint": {
      "classes": { ... elided, value-identical to the before-picture (B2) ... },
      "digest": "eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:32:10+01:00"
      }
    },
    "git_dirty": true,
    "git_head": "0d7c05c0ddf5b6cd03763307c6a233d70c638fec",
    "grid_criterion": "post qcd-background-audit prompt 15 (measured curvature criterion)",
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite",
        "when": "2026-09-24T22:03:28+01:00"
      },
      {
        "from": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite",
        "git_dirty": true,
        "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
        "operation": "retire",
        "to": null,
        "when": "2026-09-25T23:36:55+01:00"
      }
    ],
    "name": "handover-A3-baseline-lambdacdm",
    "note": "Pilot: purpose is to measure cost and mixed-row yield at full k span before sizing anything larger.",
    "purpose": "Baseline datastore for prompts/handover A3 (policy-geometry census, D5). The before-picture that D1 is scored against: current source grid, current hand-over. Keep.",
    "restart": {
      "command": "PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py --k-min 1e5 --k-max 3e8 --k-count 8 --cpus 10 --models LambdaCDM --allow-existing -- --database var/datastores/handover-A3-baseline-lambdacdm.sqlite --job-name handover-A3-baseline-lcdm --shards 4 --zend 0.1 --source-samples-log10z 100",
      "note": "--allow-existing is REQUIRED: the driver refuses an existing datastore path without it. The pipeline looks up completed work and should resume rather than recompute; verify that from the lookup-queue counts in the first minutes rather than assuming it."
    },
    "retired": {
      "completed": "2026-09-25T23:36:55+01:00",
      "files": [
        "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0000.sqlite",
        "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0001.sqlite",
        "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0002.sqlite",
        "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm-shard0003.sqlite",
        "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite"
      ],
      "files_present_only": false,
      "fingerprint": {
        "condition": "matched",
        "digest": "eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222"
      },
      "git_dirty": true,
      "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
      "reason": "Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in it that is not also in the live A3 store, and its retention condition, a clean resume, can never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which says Keep, was inherited from the live store it copies. store-retirement prompt 05.",
      "references": {
        "not_searched": "sidecars outside var/datastores, and the `.tmp` and `.incomplete-move` files beside sidecars; run manifests outside var/runs; and every reference that is not in a run manifest or a sidecar, such as docs, boards, logs and other committed files",
        "runs": [],
        "runs_root": "var/runs",
        "sidecars": [
          {
            "fields": [
              "$.backup.path"
            ],
            "sidecar": "var/datastores/handover-A3-baseline-lambdacdm.manifest.json"
          }
        ],
        "stores_root": "var/datastores"
      },
      "state": "retired",
      "when": "2026-09-25T23:36:55+01:00"
    },
    "run_history": [
      {
        "elapsed": "10h 33m",
        "note": "The GkSourcePolicyData stage COMPLETED. A3's census reads those rows, so the census may already be runnable against this store without finishing the quadratic-source-integral stage.",
        "outcome": "INTERRUPTED, not failed. Exited cleanly on SIGTERM; all five sqlite files pass PRAGMA integrity_check.",
        "size_at_stop": "385 MB across 4 shards",
        "stage_reached": "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)",
        "stages_complete": [
          "background",
          "numerical tensor Green functions",
          "APPLY GKSOURCE POLICIES",
          "GkSourcePolicyData SUMMARY STATISTICS"
        ],
        "started": "2026-09-20T21:03:04+01:00",
        "stopped": "2026-09-21T07:37 (approx, SIGTERM by request so the laptop could sleep)"
      }
    ],
    "scope": "8 log-spaced k over 1e5-3e8 /Mpc, the FULL production span (the 2026-09 verification runs used 1e5-1e7 only), LambdaCDM, production redshift geometry: --zend 0.1, 100 source samples per log10 z, default response sparseness 12",
    "sidecar_format": 1,
    "status_files": {
      "pid": "var/runs/a3-pilot/run.pid",
      "stderr": "var/runs/a3-pilot/run.err",
      "stdout": "var/runs/a3-pilot/run.out"
    },
    "store_id": "4c2ce77b0bf24671bcebd8dda3441041"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: none
```

**Each reads as a completed tombstone:**

| Condition (prompt §3 step 2) | Sweep | Backup |
|---|---|---|
| `store show` exits 0, begins `retirement:`, `problems: none` | yes | yes |
| state `retired`, with `completed` set | `retired`, `2026-09-25T23:35:57+01:00` | `retired`, `2026-09-25T23:36:55+01:00` |
| the reason, exactly as given | equal to A4's text, compared by program | equal to A4's text, compared by program |
| fingerprint condition `matched`, with §1's digest | `matched`, `2c2dde68…4da5b18`; equal to the sidecar's own `fingerprint.digest` | `matched`, `eedcdfb2…888db222`; equal to the sidecar's own `fingerprint.digest` |
| `files` equal to the dry run's `files to be deleted:` | equal (list comparison) | equal (list comparison) |
| `references` equal to the dry run's | equal (dict comparison) | equal (dict comparison) |
| a history that ends in one `retire` entry after the `adopt` | `[adopt, retire]`. The `retire` entry's `from` is the primary's path and its `to` is `null`; its `when`, `git_head` and `git_dirty` equal `retired`'s | the same |

The only keys of `retired` that differ from the dry run's preview are `state` (`retiring` →
`retired`), `completed` (`null` → the time) and `when` (the dry run's time → the real run's).
Nothing else in `retired` differs.

**The sidecars as JSON, field by field, against the pre-images copied in A6.**

| Field | Sweep | Backup |
|---|---|---|
| keys added | `retired` only | `retired` only |
| keys removed | none | none |
| `fingerprint` | value-identical | value-identical |
| `history` | the old one-entry history plus one `retire` entry, `history[:-1]` value-identical | the same |
| `copied_from` | value-identical | *(absent before and after)* |
| `created`, `datastore`, `name`, `purpose`, `sidecar_format`, `store_id` | each value-identical | each value-identical |
| `driver`, `git_dirty`, `git_head`, `grid_criterion`, `note`, `restart`, `run_history`, `scope`, `status_files` | *(absent before and after)* | each value-identical |

So every field except `history` and `retired` is value-identical to the before-picture, the
`fingerprint` included. The backup's unknown fields, including `restart.command`, which names the
live store, are unchanged records of the past (prompt §6).

#### B3. The files

The after-picture was taken with the A3 helper and compared with the first before-picture
(`agent05_s1.json`):

| Section | Before → after | Differences |
|---|---|---|
| files under `var/datastores/` | 18 → 8 | **10 gone**: the sweep's four shards and primary, and the backup's four shards and primary. **2 changed**: the sweep sidecar (10564 → 12976 bytes, SHA-256 `f7b705bd…` → `1a4984677a673b54bdfd46ec5169b5968b25503fcc60ceb6969c9a688f1f678a`, mtime `1790317882346772856` → `1790375757460021032`) and the backup sidecar (12763 → 14927 bytes, SHA-256 `ea1ff60d…` → `d76258854ae57d3f08fa8295ac12f100b5d002f1900cdb8e03d08725189a872a`, mtime `1790317931376053120` → `1790375815169136727`). **0 new** |
| directories under `var/datastores/` | 2 → 2 | `var/datastores/`: mtime `1790317956713981296` → `1790375757460337696`, listing loses the sweep's five files. `backup-pre-resume-20260921T091011/`: mtime `1790317931376335450` → `1790375815169392100`, listing is now only `handover-A3-baseline-lambdacdm.manifest.json` |
| `var/runs/`, files and directories | 32 and 8 → 32 and 8 | none |
| `physics-test-n20-*` | 6 → 6 | none |

These are exactly the differences the prompt allows: the ten store files gone, the two retired
sidecars changed, and the directory entries that follow from those. The orchestrator's SHA-256
prefixes for the two new sidecars agree with these.

**The live A3 store's five files and its sidecar are byte-identical to the before-picture, mtimes
included:**

| File | SHA-256, Phase A | SHA-256, Phase B | mtime_ns, Phase A = Phase B |
|---|---|---|---|
| `var/datastores/handover-A3-baseline-lambdacdm-shard0000.sqlite` | `a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6` | `a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6` | 1790158924459056164 = 1790158924459056164 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0001.sqlite` | `9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4` | `9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4` | 1790158260177696451 = 1790158260177696451 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0002.sqlite` | `d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe` | `d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe` | 1790158924456294411 = 1790158924456294411 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0003.sqlite` | `13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083` | `13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083` | 1790158924442869520 = 1790158924442869520 |
| `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` | `394ff933c30c755f7beae03b9f9da4fad1651266d9730ab74dd412ceb851732a` | `394ff933c30c755f7beae03b9f9da4fad1651266d9730ab74dd412ceb851732a` | 1790317956713717437 = 1790317956713717437 |
| `var/datastores/handover-A3-baseline-lambdacdm.sqlite` | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` | 1789934590990297815 = 1789934590990297815 |

**The third `!!` notice on the backup's real run**, explained from the code. The notice is printed
by `ShardedPool._resolve_shard_rows` (`Datastore/SQL/ShardedPool.py:596`) each time the backup
primary's `shards` table is read through the one resolver. The dry run reads it twice, and then
returns (`RunRegistry/stores.py:1925-1927`, "a dry run stops here"):
1. the plan, `ShardedPool.closed_store_files` → `_plan_deletion` → `_read_closed_store`;
2. the fresh read-only fingerprint, `read_inventory`.

The real run goes on past that return, and makes one more read:

3. the deletion, `ShardedPool.delete_store` (`stores.py:1953`), which builds its own plan with
   `_plan_deletion` (`ShardedPool.py:734`) and so reads the table through `_read_closed_store`
   (`:1091`) a third time.

The tombstone writes do not read it: `_update_sidecar` → `_check_before_writing` →
`_registry_problems` does not open the store. That third read cannot resolve outside the backup's
directory without a refusal. `_plan_deletion` asserts, before any unlink, that every resolved shard's
parent is the primary's own directory (`ShardedPool.py:1102-1107`). `delete_store` unlinks only what
that plan names, and re-checks each file just before it unlinks it. Then `retire_store` fails the
retirement if `delete_store` returned any file not in the tombstone's `files` (`stores.py:1959`,
`unlisted`). The notice itself names the backup's directory as where the records were read. The
after-picture confirms it: the five files deleted are the backup's own, and the live A3 store is
byte-identical. **Not a stop.**

#### B4. An interrupted retirement

Neither tombstone reads `retiring`. There is nothing to complete, and no remedy is needed.

#### B5. The amendment, drafted

The live A3 sidecar's `backup` field, read as JSON, still equals the value the prompt quotes:

```json
{"path": "var/datastores/backup-pre-resume-20260921T091011", "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly.", "retained": true}
```

The draft is the prompt's §3 step 5 text. The one change is that `<retired.completed>` is filled
with the backup tombstone's `retired.completed`, `2026-09-25T23:36:55+01:00`, read from the
tombstone sidecar. The command, compared with the prompt's text by program, is identical apart from
that substitution:

```bash
./venv/bin/python -m RunRegistry store amend var/datastores/handover-A3-baseline-lambdacdm.sqlite --field backup --json '{"path": "var/datastores/backup-pre-resume-20260921T091011", "retained": false, "retired": "2026-09-25T23:36:55+01:00", "tombstone": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json", "reason": "Retired with store retire (store-retirement prompt 05). The resume it guarded will never happen, because the A3 store is to be regenerated, not resumed, and store-fingerprint log 05 found nothing in it that is not also in this store."}' --reason 'The backup this field describes was retired with store retire, so retained true is no longer true. store-retirement prompt 05.'
```

Checks made, none of which touches `var/`:
- **The `--json` argument parses with `json.loads`.** I checked both the literal text and the
  argument as a shell would split it (`shlex.split`). It gives an object with keys `path`,
  `retained`, `retired`, `tombstone` and `reason`.
- **It differs from the current value.** `retained` is `false`, a JSON boolean, where it was
  `true`. `reason` is new. `retired` and `tombstone` are added. `path` is kept, so a reader who
  follows it lands on the tombstone. The change from `true` to `false` is boolean to boolean, so
  `[04-amend-calls-true-1-and-1-0-identical]` does not reach it.
- **`retired`** equals the backup tombstone's `retired.completed`. **`tombstone`** names the backup
  sidecar, which exists.
- **No single quote** appears in the JSON or in `--reason`. `--reason` survives the shell split
  character for character.
- **`backup` is an unknown field**, so `store amend` may change it. The live A3 sidecar is a
  problem-free registry sidecar with history `[adopt]`, and nothing is `running`.

There is no dry run for `store amend`, so the command was not run. The after-picture was re-taken
after these checks and is identical to B3's.

### Phase C — check the amendment, and record

#### C1. The amendment

- **`list`** exits 0, with the same seven runs as in A2 and B1. Nothing is `running`.
- **The live A3 sidecar as JSON**, compared with its pre-image copied in A6. The pre-image's
  SHA-256 is `394ff933…`, the before-picture's.

| Condition (prompt §4 step 1) | Result |
|---|---|
| `backup` is the value the user gave | equal, by `json.loads`, to the `--json` argument of B5's command as a shell splits it |
| keys | the same set. Only `backup` and `history` differ |
| `history` | `history[:-1]` value-identical to the old history (`[adopt]`); one new entry, the last |
| the new entry | `operation` `amend`, `field` `backup`, `from` and `to` `null`, `reason` equal to B5's `--reason`, `when` `2026-09-26T00:06:46+01:00`, `git_head` `a8260b7`, `git_dirty` `true` |
| its `before` | `{"present": true, "value": <the old backup>}`, the old value verbatim: `{"path": "var/datastores/backup-pre-resume-20260921T091011", "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly.", "retained": true}` |
| its `after` | `{"present": true, "value": <the new backup>}` |
| every other field | value-identical, `fingerprint` included |
| the live store's five `.sqlite` files | byte-identical to the before-picture, mtimes included (C2) |

**`store show`** on the live primary exits 0, with empty stderr, and prints `kind:     registry`
and `problems: none`. Verbatim, with `fingerprint.classes` elided (Deviation 1):

```
sidecar:  /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-A3-baseline-lambdacdm.manifest.json
kind:     registry
problems: none
datastore: 'handover-A3-baseline-lambdacdm.sqlite'
fields:
  {
    "backup": {
      "path": "var/datastores/backup-pre-resume-20260921T091011",
      "reason": "Retired with store retire (store-retirement prompt 05). The resume it guarded will never happen, because the A3 store is to be regenerated, not resumed, and store-fingerprint log 05 found nothing in it that is not also in this store.",
      "retained": false,
      "retired": "2026-09-25T23:36:55+01:00",
      "tombstone": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json"
    },
    "created": "2026-09-20T21:02:30+01:00",
    "datastore": "handover-A3-baseline-lambdacdm.sqlite",
    "driver": "docs/gktk-remedial/scoped_pipeline_run.py (NOT the source-remediation copy, which is broken by [13-scoped-run-driver-k-grid-literal])",
    "fingerprint": {
      "classes": { ... elided, value-identical to the before-picture (C1) ... },
      "digest": "433b7fc3cad94a971020c0493e4062e4d3a7a590a6f084c6a8e8f451f2b648a3",
      "fingerprint_format": 1,
      "problems": {},
      "taken": {
        "git_dirty": false,
        "git_head": "50a24acb23e41a68f508be63db509c10b2ac6d7e",
        "run_id": null,
        "when": "2026-09-25T07:32:35+01:00"
      }
    },
    "git_dirty": true,
    "git_head": "0d7c05c0ddf5b6cd03763307c6a233d70c638fec",
    "grid_criterion": "post qcd-background-audit prompt 15 (measured curvature criterion)",
    "history": [
      {
        "from": null,
        "git_dirty": false,
        "git_head": "f53598f4fb97cbb1269079b05760157605bc49fd",
        "operation": "adopt",
        "to": "var/datastores/handover-A3-baseline-lambdacdm.sqlite",
        "when": "2026-09-24T22:03:27+01:00"
      },
      {
        "after": {
          "present": true,
          "value": {
            "path": "var/datastores/backup-pre-resume-20260921T091011",
            "reason": "Retired with store retire (store-retirement prompt 05). The resume it guarded will never happen, because the A3 store is to be regenerated, not resumed, and store-fingerprint log 05 found nothing in it that is not also in this store.",
            "retained": false,
            "retired": "2026-09-25T23:36:55+01:00",
            "tombstone": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json"
          }
        },
        "before": {
          "present": true,
          "value": {
            "path": "var/datastores/backup-pre-resume-20260921T091011",
            "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly.",
            "retained": true
          }
        },
        "field": "backup",
        "from": null,
        "git_dirty": true,
        "git_head": "a8260b7825fe7084cfac3d57428e452076cf81f9",
        "operation": "amend",
        "reason": "The backup this field describes was retired with store retire, so retained true is no longer true. store-retirement prompt 05.",
        "to": null,
        "when": "2026-09-26T00:06:46+01:00"
      }
    ],
    "name": "handover-A3-baseline-lambdacdm",
    "note": "Pilot: purpose is to measure cost and mixed-row yield at full k span before sizing anything larger.",
    "purpose": "Baseline datastore for prompts/handover A3 (policy-geometry census, D5). The before-picture that D1 is scored against: current source grid, current hand-over. Keep.",
    "restart": {
      "command": "PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py --k-min 1e5 --k-max 3e8 --k-count 8 --cpus 10 --models LambdaCDM --allow-existing -- --database var/datastores/handover-A3-baseline-lambdacdm.sqlite --job-name handover-A3-baseline-lcdm --shards 4 --zend 0.1 --source-samples-log10z 100",
      "note": "--allow-existing is REQUIRED: the driver refuses an existing datastore path without it. The pipeline looks up completed work and should resume rather than recompute; verify that from the lookup-queue counts in the first minutes rather than assuming it."
    },
    "run_history": [
      {
        "elapsed": "10h 33m",
        "note": "The GkSourcePolicyData stage COMPLETED. A3's census reads those rows, so the census may already be runnable against this store without finishing the quadratic-source-integral stage.",
        "outcome": "INTERRUPTED, not failed. Exited cleanly on SIGTERM; all five sqlite files pass PRAGMA integrity_check.",
        "size_at_stop": "385 MB across 4 shards",
        "stage_reached": "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)",
        "stages_complete": [
          "background",
          "numerical tensor Green functions",
          "APPLY GKSOURCE POLICIES",
          "GkSourcePolicyData SUMMARY STATISTICS"
        ],
        "started": "2026-09-20T21:03:04+01:00",
        "stopped": "2026-09-21T07:37 (approx, SIGTERM by request so the laptop could sleep)"
      },
      {
        "consequence": "A3 is NOT blocked: its census reads GkSourcePolicyData (1160 rows, complete). The ~7552 QuadSourceIntegral rows are written but cannot be read back until the one-line fix lands, and the last work item cannot be completed.",
        "datastore_state": "UNDAMAGED. Live vs pre-resume backup: GkSourcePolicyData 290/290/290/290 and QuadSourceIntegral 1890/1888/1889/1885 on all four shards, identical. Pruning removed nothing.",
        "defect": "Datastore/SQL/ObjectFactories/QuadSourceIntegral.py: the SELECT at :234 omits table.c.numeric_quad, while build() at :324 reads row_data.numeric_quad. The sibling SELECT at :530 does include it. Latent: a fresh run computes rows and never reads them back, so only a RESUME (or any object_get of a stored row) can hit it.",
        "error": "sqlalchemy.exc.NoSuchColumnError: Could not locate column in row for column 'numeric_quad'",
        "outcome": "FAILED at the quadratic-source-integral stage. Resume replayed every earlier stage correctly (background, numeric and WKB Green functions, GkSource, APPLY GKSOURCE POLICIES, summary statistics) and then crashed reading back an EXISTING QuadSourceIntegral row.",
        "started": "2026-09-21T09:11 (resume attempt, --allow-existing --prune-unvalidated)"
      }
    ],
    "scope": "8 log-spaced k over 1e5-3e8 /Mpc, the FULL production span (the 2026-09 verification runs used 1e5-1e7 only), LambdaCDM, production redshift geometry: --zend 0.1, 100 source samples per log10 z, default response sparseness 12",
    "sidecar_format": 1,
    "status_files": {
      "pid": "var/runs/a3-pilot/run.pid",
      "stderr": "var/runs/a3-pilot/run.err",
      "stdout": "var/runs/a3-pilot/run.out"
    },
    "store_id": "5f58ac536362424499a4e94c6242928e"
  }
runs naming this store, under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs:
  handover-03-a3-baseline-resume-20260923T024847  failed    finished  by results
```

#### C2. The final picture

Re-taken with the A3 helper, and compared first with B3's after-picture:

| Section | B → C | Differences |
|---|---|---|
| files under `var/datastores/` | 8 → 8 | **1 changed**: the live A3 sidecar, 14205 → 15783 bytes, SHA-256 `394ff933…` → `3e43aee7aa17b02f2d2f7041a382a520a9ef5693553b0df40aa222ae8a3a7eb3`, mtime `1790317956713717437` → `1790377606326005386`. None gone, none new |
| directories | 2 → 2 | `var/datastores/`: mtime `1790375757460337696` → `1790377606326249800`, and its listing unchanged. This is the directory entry that follows from the sidecar's atomic write, a `.tmp` created and renamed onto the sidecar, and it is within prompt §3.3's allowance. `backup-pre-resume-20260921T091011/` is unchanged |
| `var/runs/`, `physics-test-n20-*` | unchanged | none |

So the differences from Phase A are exactly Phase B's, plus the live sidecar and the directory
entry that follows from it. The orchestrator's check 3 found the same.

`var/datastores/` now holds eight files: the live A3 store's five and its sidecar, the sweep's
tombstone `handover-atol-sweep.manifest.json`, and the backup's tombstone, alone in
`backup-pre-resume-20260921T091011/`.

**The live A3 store's final hashes, beside Phase A's:**

| File | Size | SHA-256, Phase A (before the first dry run) | SHA-256, Phase C (after the amendment) | mtime_ns, A = C? |
|---|---|---|---|---|
| `var/datastores/handover-A3-baseline-lambdacdm-shard0000.sqlite` | 87367680 | `a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6` | `a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6` | yes, 1790158924459056164 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0001.sqlite` | 87302144 | `9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4` | `9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4` | yes, 1790158260177696451 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0002.sqlite` | 87711744 | `d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe` | `d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe` | yes, 1790158924456294411 |
| `var/datastores/handover-A3-baseline-lambdacdm-shard0003.sqlite` | 86892544 | `13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083` | `13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083` | yes, 1790158924442869520 |
| `var/datastores/handover-A3-baseline-lambdacdm.sqlite` | 32768 | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` | `fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b` | yes, 1789934590990297815 |
| `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` (the sidecar, amended) | 14205 → 15783 | `394ff933c30c755f7beae03b9f9da4fad1651266d9730ab74dd412ceb851732a` | `3e43aee7aa17b02f2d2f7041a382a520a9ef5693553b0df40aa222ae8a3a7eb3` | no: 1790317956713717437 → 1790377606326005386, the amendment |

#### C3. The evidence file

[`05-tombstones.json`](05-tombstones.json) holds three sidecars, keyed by repository path in the
shape of `store-fingerprint`'s `logs/05-fingerprints.json`:
- the sweep's tombstone;
- the backup's tombstone;
- the live A3 sidecar after the amendment.

Each was read from disk as bytes, and its SHA-256 was checked equal to C2's before it was parsed.
Written with `json.dump(indent=2)`. Read back, each entry equals `json.load` of the file on disk.

#### C4. The records (R12)

Each is an additive, dated note. Nothing a board already says was rewritten.
- **This board**, `IMPLEMENTATION_STATE.md`:
  - the header's status says the campaign is complete;
  - the §1 row for 05 is filled in, and a dated paragraph, "Prompt 05 landed (2026-09-26)", is
    added after the existing 05 paragraph;
  - R10–R12 are marked done;
  - §5 gains an "After prompt 05" column ("not re-run (no code changed)") and a dated sentence
    saying why.
- **The `run-registry` board**: a dated note after the sentence that ends the paragraph beginning
  "A second instance, found when the campaign was planned". It says the backup was retired and its
  directory holds only its tombstone. It explains `var/runs/a3-pilot/BACKUP_PATH`: a pre-registry
  record of the past, out of scope, not edited, and it now lands on the tombstone. Additions only
  (`git diff --numstat`: 13 added, 0 removed).
- **The `handover` board**: a dated note after the "Superseded" paragraph of
  `[a3-baseline-quadsource-integrals-are-1680-short]`. It says the sweep store, whose first run
  wrote the 54 rows, is retired, and that `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` is now true and was
  not edited. Additions only (12 added, 0 removed).
- **`docs/OPEN_ISSUES.md`**: unchanged, because no issue was opened.

`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` was not edited. No run manifest or file under
`var/runs/` was touched.

#### C5. The commit

One commit, staged by name: this log, `logs/05-tombstones.json` and the three boards. The
untracked `docs/datastore-integrity-audit*` and `prompts/datastore-integrity/` were not staged. No
`.py` file changed, so `black` had nothing to format. The suites were not re-run, because no code
changed.

## The deliberate-breakage record

**None, by design.** This prompt writes no code, so there is nothing to mutate, and no test was
added or run. Its checks are of real state under `var/`, each made against a before-picture taken
by a read-only helper: the dry runs (A5), the tombstones and files (B2–B3), and the amendment
(C1–C2). They are also compared with the orchestrator's independent snapshots. The code those
checks exercised, `store retire` and `store amend`, was mutation-tested where it was written, in
prompts 03 and 04, whose deliberate-breakage records stand.

## Observations not acted on

1. **The resolver's `!!` notice is printed once per read of a legacy primary's `shards` table.**
   It appeared twice on the backup's dry run and three times on its real run. The deletion's own
   plan is the third read, which the dry run never reaches (B3). It goes to stdout, not stderr.
   Cosmetic, and in `ShardedPool`, which this prompt does not change. Not an issue.
2. **`git_dirty` is `true` in both tombstones and in the `amend` entry.** `git_provenance`
   counts untracked files. The tree held the three untracked `datastore-integrity` paths, which
   are not this campaign's, and this log draft. It is correct, not a defect. Not an issue.
3. **`[03-the-package-docstring-still-says-the-registry-deletes-nothing]` stays open.** Its entry
   offers it to prompt 05 "if its orchestrator admits it". This prompt writes no code and was not
   given `RunRegistry/__init__.py`, so it was left.
4. **The backup's tombstone keeps `purpose` ("… Keep.") and `restart.command`, which names the
   live store.** Both are records of the past, as prompt §6 says, and amend refuses a tombstone.
   Its `retired.reason` says why "Keep" no longer holds. Not an issue.

## State handed to the next prompt

- **The campaign is complete.** R1–R12 are done.
- **`var/datastores/`** now holds the live A3 store (five files and its amended sidecar), the
  sweep's tombstone, and the backup's tombstone, alone in `backup-pre-resume-20260921T091011/`.
- **Evidence that survives in git:** `logs/05-tombstones.json`.
- **Both retired names are never reused (D6).** `begin`, create, copy, move, fingerprint, adopt
  and amend refuse them. A store written at either name reads as "exists again".
- **Open issues are unchanged:** the seven in §3 of this board, and none new.
