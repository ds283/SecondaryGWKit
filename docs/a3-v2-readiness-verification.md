# A3 v2 readiness campaign — the rehearsal of A3 v2's launch (prompt 07)

**Campaign:** [`prompts/a3-v2-readiness/README.md`](../prompts/a3-v2-readiness/README.md) ·
**Board:** [`prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md`](../prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md) ·
**Prompt:** [`prompts/a3-v2-readiness/07-rehearse-the-a3-v2-launch.md`](../prompts/a3-v2-readiness/07-rehearse-the-a3-v2-launch.md) ·
**Log:** [`prompts/a3-v2-readiness/logs/07-rehearse-the-a3-v2-launch.md`](../prompts/a3-v2-readiness/logs/07-rehearse-the-a3-v2-launch.md)

Every launch ran on branch `handover-remedial` at **`dbd8a51`** ("Record the user's decisions D7-1
and D7-2 for prompt 07"). Its code is `7bd202f`'s (06a). `git status --short` was empty at every
launch, and every manifest stamps `git_dirty: false` (§1.3). A3 v2's own launcher,
`docs/handover/quadsource_atol_sweep.py --build --rehearsal --register`, built two throwaway stores
under `var/datastores/`:
- **`a3-v2-readiness-07-reference.sqlite`** (sidecar `store_id` `b7ad6350c920495d9c8618deb5b39ec8`),
  built straight through by the run **`reference`**;
- **`a3-v2-readiness-07-resumed.sqlite`** (`store_id` `b880a3a3eec64e8cac7521b9fc9756b9`), built by
  **`resumed-a`** (killed by `SIGTERM` in the `GkSource` stage), **`resumed-b`** (`--resume`, killed
  in the `QuadSourceIntegral` stage) and **`resumed-c`** (`--resume`, to `done`).

The runs are under `var/runs/a3-v2-readiness-07-<slug>-<stamp>/`. Both stores stay, with their
sidecars, for the orchestrator's review, and the user then retires them, or not (D7-2). Neither is
retired here. The small outputs quoted below are saved in
[`a3-v2-readiness/rehearsal/`](a3-v2-readiness/rehearsal/); the large ones stay under `var/` (§6).

**Headline.**
1. **The resumed store's fingerprint equals the reference's.** `compare_fingerprints` is empty, both
   at format 6, digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21
   classes equal, each equal to its sidecar's (§2.2.4).
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0. `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, each driver exiting 143. Each handler's
   line came within the second of the interrupter's signal (§2.1).
3. **The resume recomputed nothing it had stored.** At its open, `resumed-b` pruned the 35 `GkSource`
   rows `resumed-a` left unvalidated, and recomputed exactly those plus the 189 missing. `resumed-c`
   computed the 471 `QuadSourceIntegral` rows that were in flight at the second signal, and nothing
   else. Nothing is duplicated (§2.2).
4. **Acceptance 4 is not met: `extract_tensor_source_data.py` exits 1.** It draws its k in random
   order and asks for `QuadSource` pairs with q > r, which the store does not hold. The defect
   predates this campaign: the reader passes 1 time in 6 with three k, and with A3 v2's eight
   essentially never. Opened as
   `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`; it needs a fix
   before A3 v2's read-back. A wrapper that only sorts its draw, **not the shipped reader**, reads
   all six `QuadSource` rows read-only and exits 0. The other five readers exit 0, and with
   `--models` omitted the skip line prints and the exit is 0. No reader refuses anything or changes
   any file of the store. The `QuadSourceIntegral` extractor reads every stored row of the triples
   it draws (288 of 288), but by design it draws 6 triples at random, 3 of them among the store's 9,
   so it does not read all 864
   (§2.3).
5. **A finding: a killed build's Ray agents outlive its driver.** After `resumed-b` was killed, its
   dashboard agent and runtime-env agent were orphaned to pid 1. They were still alive 40 s after
   the driver exited, and the dashboard agent about 90 s after. Both exited on their own, by about
   2 minutes after. So "no Ray process of this repository left" is not true when the driver exits.
   A3 v2's check after a kill must wait for these by pid (§2.1.3). Opened as
   `[07-a-killed-builds-ray-agents-outlive-the-driver]`.

---

## 1. The runs

### 1.1 How each was launched

Each launch was §1.3's command, verbatim but for the store, the slug, the purpose and `--resume`,
run from the repository root, detached:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/a3-v2-readiness-07-<store>.sqlite \
    --job-name a3-v2-readiness-07 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign a3-v2-readiness --prompt 07; \
  echo "** driver exit status $?"' \
  > var/runs/a3-v2-readiness-07-<slug>.launch.log 2>&1 &
disown
```

After each launch the agent made one check, that the manifest existed, that `status.json` said
`running` with a pid answering `kill -0`, and that `RunRegistry list` showed it, and ended its turn.
The orchestrator ran one wait per launch, on the run's liveness (on `resumed-c`, on the pid), and
resumed the agent when the run was no longer alive. Neither polled.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `a3-v2-readiness-07-reference-20261004T012909` | reference | no | 27325 (27322) | `done` | 0 | 0 | 12 m 58 s (01:29:09–01:42:07) | 13.10 13.27 15.59 / 20.57 43.94 39.60 (at 01:44:30) |
| `resumed-a` | `a3-v2-readiness-07-resumed-a-20261004T014440` | resumed | no | 41825 (41820) | `killed` | 143 | 143 | 6 m 29 s (01:44:40–01:51:09) | 20.57 43.94 39.60 / 22.18 44.48 43.95 |
| `resumed-b` | `a3-v2-readiness-07-resumed-b-20261004T015203` | resumed | yes | 49456 (49451) | `killed` | 143 | 143 | 8 m 18 s (01:52:03–02:00:21; pid gone by ~02:00:30) | 19.04 42.28 43.17 / 123.24 110.98 76.89 |
| `resumed-c` | `a3-v2-readiness-07-resumed-c-20261004T020421` | resumed | yes | 64172 (64166) | `done` | 0 | 0 | 4 m 54 s (02:04:21–02:09:15; pid gone 02:09:26) | 16.72 68.25 65.91 / 62.95 72.16 67.76 |

Wall time is from the manifest's `created` to the final heartbeat. The load is `uptime`'s, at the
pre-launch check and at the first check after the run (10 CPUs). The machine was shared with
another of the user's workspaces throughout; the one-minute load reached 188 during §2.3's first
pass and 365 before the sorted-sample run.

`store create` ran on each closed primary after its first close: the reference's at 01:44:18,
after `reference`, followed by `store fingerprint --write`; the resumed store's at 01:51:55, after
`resumed-a` ([`store_sidecars.txt`](a3-v2-readiness/rehearsal/store_sidecars.txt)). Each
manifest, final `status.json` (less the fingerprint's per-class detail), `stderr.log` and last
launch-log line is in [`run_records.txt`](a3-v2-readiness/rehearsal/run_records.txt).

### 1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch** (prompt §3), each into a new directory under
the agent's scratchpad, outside `var/`:

```bash
bash docs/datastore-integrity-verification/interrupter_dummy_test.sh <scratch>/work/dummy-scoped
bash docs/a3-v2-readiness/interrupter_build_dummy_test.sh <scratch>/work/dummy-build
```

Both exited 0, and both pass
([`interrupter_dummy_test.txt`](a3-v2-readiness/rehearsal/interrupter_dummy_test.txt),
[`interrupter_build_dummy_test.txt`](a3-v2-readiness/rehearsal/interrupter_build_dummy_test.txt)):
- **five cases, 01:27:46–01:28:11.** Case A signalled at 45 % ≥ 40 % (child 143). B exited
  silently when the child died first. C refused `'sleep 30'` with exit 2, naming
  "`scoped_pipeline_run.py` (the run's own driver, its manifest's argv[0])", and the child lived.
  D held 4 s and signalled (child 143). E did not signal when the stage completed during the hold.
- **the `--build` cases, 01:28:17.** A signalled the `quadsource_atol_sweep.py --build --register
  dummy-build-a` child (interrupter 0, child 143). B refused the `scoped_pipeline_run.py` child
  (exit 2, child alive). C: "the run's manifest names no argv[0]", exit 2, child alive.
- No `sleep` child was left alive.

**The points, from the reference's `stdout.log`.**
- `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **1 m 2.16 s** and printed no intermediate
  progress line, so `resumed-a` was interrupted from the banner with `--hold`. The agent chose
  `--interval 5 --hold 15`, to put the signal 15–20 s into a stage of about a minute (log 07,
  deviation 2).
- `CALCULATE QUADRATIC SOURCE INTEGRALS` printed its first, and only, progress line at **5 m
  10.2 s**: `0/2 work items remaining = 100.00% complete`, with 603 computed and 533 stored. It
  completed at 7 m 1.74 s. `resumed-b` was interrupted at that line's 100 %.

**The real firings** ([`interrupter_firings.txt`](a3-v2-readiness/rehearsal/interrupter_firings.txt)):

```
resumed-a (interrupter pid 41858; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 15)
2026-10-04T01:50:53+01:00 liveness alive; pid 41825; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-04T01:50:53+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 15 s, then reading again
2026-10-04T01:51:08+01:00 sent SIGTERM to pid 41825 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/a3-v2-readiness-07-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 52753; '^CALCULATE QUADRATIC SOURCE INTEGRALS, ' 100 --interval 5)
2026-10-04T01:54:59+01:00 liveness alive; pid 49456; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-04T02:00:06+01:00 liveness alive; pid 49456; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS, 100.00% (0 of 2 work items remaining)'
2026-10-04T02:00:06+01:00 sent SIGTERM to pid 49456 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS, 100.00% (0 of 2 work items remaining)' (100% >= 100%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume … --register resumed-b …'; exiting
```

Each confirmed, before it signalled, that the pid's command line named `quadsource_atol_sweep.py`,
its manifest's `argv[0]`, and `--register <slug>`. The interrupter refused no real run's pid.

**An interrupter launched at the wrong directory.** For `resumed-b`, the agent's wait for the run
directory used a zsh glob qualifier that never matched, and at 01:53:43 it launched an
interrupter (pid 52113) with an empty run directory. It read the repository root, logged "the
run's manifest names no --register slug; exiting without signalling", and exited. It signalled
nothing. Its log, written to `<repository>/interrupter.log`, was moved to
`var/a3-v2-readiness-07/stray-interrupter-resumed-b.log` and is reproduced in
`interrupter_firings.txt`. It was created after `resumed-b`'s manifest (01:52:03), whose
`git_dirty: false` stands. The correct interrupter started at 01:54:12, with the run in `APPLY
GKSOURCE POLICIES`, before the stage it watched (log 07, deviation 1).

### 1.3 The pre-launch checks

Before each launch ([`prelaunch_checks.txt`](a3-v2-readiness/rehearsal/prelaunch_checks.txt)):
- `git rev-parse HEAD` was `dbd8a511fd3c22f1c8aed8374c5c452b7b25f453` and `git status --short`
  was empty. No tracked file was modified, and no untracked file was present, so each manifest
  says `git_dirty: false`. The record's files were written after the last launch.
- `RAY_ADDRESS` was unset.
- `python -m RunRegistry list` showed nothing `running`: 12 runs before `reference` (10 finished,
  2 `unknown`, `a3-pilot` and `a3-pilot-resume`), then 13, 14 and 15.
- **No Ray process of this repository.** The listing was `pgrep -f 'gcs_server|raylet|ray::'`,
  each pid's parent and command line, then a whole-table search for a command line containing
  both `SecondaryGWKit/venv` and `ray`. Each time it listed the user's ChamPBH cluster only, all
  started from `…/ChamPBH/venv/` (Python 3.13): `gcs_server` 62525, `raylet` 62574, its dashboard
  agent 62594, and spill/restore workers 96937, 96961, 97272, 97276, children of 62574. None was
  touched. This repository's builds run under `/opt/local/…/Python.framework/Versions/3.12`.
- Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
  naming either. No file under `var/datastores/` matched `a3-v2`, and nothing under `var/runs/`
  began `a3-v2`. Before `resumed-a`, no `a3-v2-readiness-07-resumed*` file existed.
- 105–106 GB free on `var/`'s volume.

---

## 2. What was verified

### 2.1 L1 — every launch ends as recorded

#### 2.1.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: no sidecar yet | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `16b29948…`, problems `{}` | absent | not written: "…a3-v2-readiness-07-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `02e97d50…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint's `taken` names its run and `git_head` `dbd8a511…`, `git_dirty` false. The full
digests: `reference` and `resumed-c`
`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`; `resumed-a`
`16b29948c171c38f7e98fe457db19ca192d97d9843511c838dcb584241234999`; `resumed-b`
`02e97d50edb63732ad9295c5105bd99c81f810e1146dfd849b50e8fe17fc6720`.

#### 2.1.2 Each killed launch, against prompt 04's account

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 01:51:08 | 02:00:06 |
| Handler's line | `** SIGTERM at 2026-10-04 01:51:08 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-04 02:00:06 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s | under 1 s |
| `stderr.log`'s second line | `** SIGTERM: the pipeline ended with SystemExit(143); recording killed, exit code 143` | `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, 15 s in (102 of 291 rows stored) | the `QuadSourceIntegral` 100 % line, 5 m 8.74 s in (393 of 864 stored) |

- **The delay.** Both timestamps are whole seconds: the handler's from `time.strftime`, the
  interrupter's from `RunRegistry.now_iso()`. Equal stamps bound the delay below 1 s; they do not
  measure it. Prompt 04 measured 0.001–1.006 s.
- **`SystemExit(1)` in `resumed-b`.** The signal arrived during `ray.wait`, where Ray's signal
  check turns the handler's `SystemExit(143)` into `SystemExit(1)` (audit S1). The latch is what
  `finish_after_exec` reads, so the record is `killed`/143 and the exit is 143 either way. In
  `resumed-a` the exception arrived unconverted. Not a defect: it is the case prompt 04's latch
  exists for.
- **The record is written before the fingerprint.** The orchestrator's first read of `resumed-b`,
  at 02:00:06, saw `state: killed`, `exit_code: 143` and no `fingerprint` yet, with pid 49456
  alive. That is the design: the handler writes `killed` at once, and `finish_after_exec` adds
  the fingerprint after the pool closes. A wait that ends on the recorded state ends before the
  fingerprint is written. The orchestrator then waited on the pid, which was gone by about 02:00:30
  (log 07, observation 3).

#### 2.1.3 No Ray process of this repository left — a finding

| After | When | Processes of this repository |
|---|---|---|
| `reference` | ~01:43 | none |
| `resumed-a` | ~01:51:40 | none |
| `resumed-b` | 02:01:09 (~40 s after the driver exited) | **49729** `…/SecondaryGWKit/venv/…/ray/dashboard/agent.py`, **49730** `…/SecondaryGWKit/venv/…/ray/_private/runtime_env/agent/main.py`; both ppid 1, process group 49456, started 01:52:07, session `session_2026-10-04_01-52-05_453941_49456` |
| `resumed-b` | 02:01:30 | 49729 still alive (etime 09:23, state `SN`); 49730 gone |
| `resumed-b` | 02:02:05 (the orchestrator, read-only) | 49729 alive (etime 09:58); load 65.70 97.32 74.32 |
| `resumed-b` | by 02:02:36 (the orchestrator's bounded wait on the pid, 10 s granularity) | none; load 48.15 90.06 72.49 |
| `resumed-b` | 02:04:12 (before `resumed-c`) | none |
| `resumed-c` | 02:09:46 (20 s after its pid was gone) | none |

- **What it is.** When the killed driver ran `ray.shutdown()` and exited, Ray's per-node dashboard
  agent and runtime-env agent of its local instance did not exit with it. They were re-parented
  to pid 1, kept the driver's process group, and exited on their own about 1 and 2 minutes later.
  No `gcs_server`, `raylet` or `ray::` worker of this repository was left at any check.
- **It is a §5 stop condition**, "leaves a Ray process of this repository alive", and the agent
  stopped. The user decided (2026-10-04): wait for the orphan to exit on its own; if it does,
  record it and continue. It did, and nothing was killed.
- **Why `reference` and `resumed-a` showed none** is not established. Their checks came about 60 s
  and 30–40 s after exit, at loads of 20–44. `resumed-b`'s came about 40 s after exit at a load of
  123. If the agents exit when a periodic check finds their raylet gone (inferred, not read in
  Ray's source), the load is the likely difference. Not measured.
- **The listing.** `pgrep -f 'gcs_server|raylet|ray::'` listed 49729 only because its command line
  holds `--raylet-name=…`. It would not have listed 49730, whose command line names neither. The
  agent's listing added a whole-table search for `SecondaryGWKit/venv` together with `ray`, which
  found both. A check that relies on the `pgrep` pattern alone can miss a runtime-env agent.
- **What it means for A3 v2.** "No Ray process left" is not true when the driver exits. The check
  after a kill must list processes from the build's venv (or the build's process group), and wait
  for each by pid until it exits, rather than assume they are gone. Opened as
  `[07-a-killed-builds-ray-agents-outlive-the-driver]` (§5).
- **A second process, not this checkout's.** During §2.3, a whole-table search for `SecondaryGWKit`
  also matched another of the user's sessions, working in the worktree
  `SecondaryGWKit-wt-replicated-key`. Worktrees share this checkout's venv by a symlink, so a Ray
  process started from one may show either path. Not touched.

#### 2.1.4 The manifests

From [`run_records.txt`](a3-v2-readiness/rehearsal/run_records.txt), for all four:
- `git_head` `dbd8a511fd3c22f1c8aed8374c5c452b7b25f453`, the dispatch `HEAD`; `git_dirty` false;
- `script` `docs/handover/quadsource_atol_sweep.py`; `script_sha256`
  `3d0e6ecdba33315a550761797e108e85ba46b0b660b4481cfa1ddbd535b12073`, the same in all four;
- `argv[0]` `docs/handover/quadsource_atol_sweep.py`; `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only;
- `results` names the reference store for `reference`, the resumed store for the other three;
- `scope`, in each, the rehearsal text, which names the pair, the grid and `zend 250`, and not
  `9280`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into
  a3-v2-readiness-07-<store>.sqlite`.

#### 2.1.5 Every launch's `stdout.log`

Counted in each of the four:

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution: the two `np.logspace(… NUMBER_SOURCE_K_VALUES / NUMBER_RESPONSE_K_VALUES)` grids, `tol=DEFAULT_QUADRATURE_ATOL`, `tol=DEFAULT_QUADRATURE_RTOL` |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--shards', '4', '--zend', '250', '--source-samples-log10z', '100']` | 1 |
| `--no-prune-unvalidated` anywhere | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |

The two resumed launches' launch logs begin `** quadsource_atol_sweep --build: resuming into
a3-v2-readiness-07-resumed.sqlite`, printed before the run registers, and so not in `stdout.log`.

#### 2.1.6 A3 v2's guard

After `resumed-c`, in the foreground, with no `--register`
([`guard.txt`](a3-v2-readiness/rehearsal/guard.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
    --job-name a3-v2-readiness-07 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (02:10:00–02:10:01), with:

```
RuntimeError: quadsource_atol_sweep --build: a3-v2-readiness-07-resumed.sqlite, a3-v2-readiness-07-resumed-shard0000.sqlite, a3-v2-readiness-07-resumed-shard0001.sqlite, a3-v2-readiness-07-resumed-shard0002.sqlite, a3-v2-readiness-07-resumed-shard0003.sqlite already exist(s). Refusing to build into an existing store: a comparator that silently inherited earlier rows is one nobody can interpret. Pass --resume to continue an interrupted build at the SAME tolerance pair, or choose a path that does not exist.
```

It names the five files that exist. The SHA-256 of every file of the store, and of its sidecar, was
the same before and after; no run directory was created; no Ray process of this repository
appeared. The refusal is raised before `RunRegistry.begin` and `ray.init`.

### 2.2 L2 — the resume is the reference

#### 2.2.1 Nothing stored is recomputed

`docs/datastore-integrity-verification/stage_counts.py` takes each column's label from the run id
by splitting on `-10-`, prompt 10's number, and raised `IndexError` on these run ids. It was not
changed. [`stage_counts_07.py`](a3-v2-readiness/rehearsal/stage_counts_07.py) prints the same
table through its `read`, splitting on `-07-` (log 07, observation 1;
`[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]`):

```bash
PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/stage_counts_07.py \
    var/runs/a3-v2-readiness-07-reference-20261004T012909 \
    var/runs/a3-v2-readiness-07-resumed-a-20261004T014440 \
    var/runs/a3-v2-readiness-07-resumed-b-20261004T015203 \
    var/runs/a3-v2-readiness-07-resumed-c-20261004T020421
```

Lookup / compute / store ([`stage_counts.txt`](a3-v2-readiness/rehearsal/stage_counts.txt)):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **224/224/224** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | **864/393/343 (last progress)** | **471/471/471** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.** `resumed-a` stored 102 rows before the signal, 35 of them unvalidated (§2.2.2).
  `resumed-b` pruned the 35 at open and computed 224 = 291 − 102 + 35: what was missing, plus the
  pruned rows. No validated row was computed again. Summed, less the pruned: 102 + 224 − 35 = 291,
  the reference's.
- **`QuadSourceIntegral`.** `resumed-b`'s last progress line, at 02:00:05, reads
  `inflight: 0 lookup, 471 compute, 50 store | completed: 864 lookup, 393 compute, 343 store`.
  The closed store held **393** rows (134 + 121 + 0 + 138 by shard), so the 50 stores in flight
  completed before the store closed. The 471 computes in flight at the signal were never stored,
  and `resumed-c` computed exactly 471. Summed: 393 + 471 = 864, the reference's. No stored row
  was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b` because `resumed-a` never reached it.

#### 2.2.2 The prune at startup (prompt 05)

The closed store after each launch, read `mode=ro` by
[`closed_store_state.py`](a3-v2-readiness/rehearsal/closed_store_state.py), the store's SHA-256s
unchanged by the read ([`closed_store_state.txt`](a3-v2-readiness/rehearsal/closed_store_state.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **35 `GkSource`**: shard 0 serials 11, 12, 13, 14, 15, 22, 28, 29, 30, 31, 33, 34; shard 1 214, 215, 216, 217, 218, 219, 228; shard 3 522, 523, 528, 529, 530, 531, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549. Every other class with a `validated` column: none |
| `resumed-b` | none | 0 rows | none |
| `resumed-c` | none | 0 rows | none |

After `resumed-a`, `GkSource` held 34 + 30 + 0 + 38 = 102 rows. After `resumed-b`, 97 + 97 + 0 + 97.

**By serial** ([`prune_serials.py`](a3-v2-readiness/rehearsal/prune_serials.py),
[`prune_serials.txt`](a3-v2-readiness/rehearsal/prune_serials.txt), after `resumed-c`): 32 of the 35
serials are absent. 547, 548 and 549, the three highest on shard 3, are held by rows stored and
validated by `resumed-b` at 01:52:33–01:52:40, whose labels carry those times (SQLite reuses a
deleted row's id when it was the table's largest). None of the 35 unvalidated rows remains.

**What `resumed-b` printed.** Each actor's integrity warning, at open (01:52:09):
- shard 1 in full: `!! INTEGRITY WARNING … datastore "shard0001-store" …`, `The following
  unvalidated GkSource objects were detected in the datastore:`, seven entries, store_ids 214–219
  and 228, and `** Note: these values have been pruned from the datastore.`;
- shards 0 and 3 folded by Ray: `!! INTEGRITY WARNING … "shard0000-store" … [repeated 2x across
  cluster]`, one entry shown with `[repeated 28x across cluster]`, and the note `[repeated 2x across
  cluster]`. 7 + 28 = 35, the count from the store.

`resumed-c` printed no integrity warning; nothing was unvalidated. `QuadSourceIntegral` has no
`validated` column, so the rows `resumed-b` stored need no prune.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class, so no
`duplicate` (§2.2.5). Each finish's fingerprint is built from `read_inventory`, and its `problems`
tallies every problem kind by class: it is `{}` for `resumed-a`, `resumed-b` and `resumed-c`, so no
`duplicate` after any resumed launch. Per shard, `GkSource`'s 97 rows hold 97 distinct keys
(`prune_serials.txt`).

**The pool's prune of a replicated class.** None. No resumed launch printed a prune of a replicated
class, and after every launch each shard's `BackgroundModel` row is validated and the census finds
the replicated tables equal (§2.2.5).

#### 2.2.3 The version row (prompt 02)

`SELECT serial, label FROM version`, `mode=ro`, on every shard, after every launch: exactly one row,
`(1, '2025.1.1')`, on each of the four shards of both stores. The primary has no `version` table.

#### 2.2.4 The comparison

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/a3-v2-readiness-07-reference.sqlite \
    var/datastores/a3-v2-readiness-07-resumed.sqlite
```

Exit 0 ([`fingerprint_comparison.txt`](a3-v2-readiness/rehearsal/fingerprint_comparison.txt)):

```
reference: format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none; against its sidecar: 0 difference(s), it matches
resumed: format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none; against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
verdict: equal
```

| Class | Reference | Resumed | Digests | `datastore-integrity` 10 (§2.1.5) |
|---|---|---|---|---|
| `version` | 1 | 1 | equal | 1 |
| `store_tag` | 10 | 10 | equal | 10 |
| `redshift` | 1159 | 1159 | equal | 1159 |
| `wavenumber` | 3 | 3 | equal | 3 |
| `tolerance` | **7** | **7** | equal | **6** |
| `LambdaCDM` | 1 | 1 | equal | 1 |
| `QCD_Cosmology` | 1 | 1 | equal | 1 |
| `IntegrationSolver` | 7 | 7 | equal | 7 |
| `GkSourcePolicy` | 2 | 2 | equal | 2 |
| `QuadSourcePolicy` | 2 | 2 | equal | 2 |
| `wavenumber_exit_time` | 3 | 3 | equal | 3 |
| `BackgroundModel` | 1 | 1 | equal | 1 |
| `TkNumericIntegration` | 3 | 3 | equal | 3 |
| `TkWKBIntegration` | 3 | 3 | equal | 3 |
| `GkNumericIntegration` | 1368 | 1368 | equal | 1368 |
| `GkWKBIntegration` | 3477 | 3477 | equal | 3477 |
| `GkSource` | 291 | 291 | equal | 291 |
| `GkSourcePolicyData` | 291 | 291 | equal | 291 |
| `QuadSource` | 6 | 6 | equal | 6 |
| `QuadSourceIntegral` | 864 | 864 | equal | 864 |
| `OneLoopIntegral` | 0 | 0 | equal | 0 |

**Against `datastore-integrity` 10.** Every class's count equals that record's but `tolerance`:
**7 here, 6 then.** Not a defect. `main.py` stores one `tolerance` row per distinct value among
eight defaults (`main.py` L4249–4256). The reference's rows are $\log_{10}$ tol = −10, −9, −32, −8,
−13, −10.52 ($3\times10^{-11}$) and −7 (read `mode=ro` from shard 0). On 2026-10-01 the quadrature
`rtol` was `1e-8`, equal to `DEFAULT_GK_NUMERIC_REL_TOLERANCE`, so the two shared a row. Now it is
`1e-7` (prompt 01, D1), a value of its own. As expected, the digests are not compared.

#### 2.2.5 Integrity, both stores, read-only

**The census** (`docs/datastore-integrity-audit/replicated_serial_census.py`, both primaries;
[`census.txt`](a3-v2-readiness/rehearsal/census.txt)): every table `OK` on both, verdict `no
divergence` on both. Its table list is `config/sharding.py`'s 13 `replicated_tables`, in the same
order, plus `BackgroundModel_tags`, which it adds; it omits none. Each table has equal rows on the
four shards: `version` 1, `store_tag` 10, `redshift` 1159, `wavenumber` 3, `wavenumber_exit_time` 3,
`tolerance` 7, `LambdaCDM` 1, `QCD_Cosmology` 1, `IntegrationSolver` 7, `BackgroundModel` 1,
`BackgroundModelValue` 1159, `GkSourcePolicy` 2, `QuadSourcePolicy` 2, `BackgroundModel_tags` 4.

**`integrity_check`, `foreign_key_check` and the inventory**
([`store_soundness.py`](a3-v2-readiness/rehearsal/store_soundness.py),
[`store_soundness.txt`](a3-v2-readiness/rehearsal/store_soundness.txt)): on the primary and every
shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is `['ok']` and `PRAGMA
foreign_key_check` returns 0 rows. `read_inventory(primary)` reports no problem in any class of
either store; each class's record count is the table of §2.2.4, identical in the two. Every file of
both stores had the same SHA-256 before and after these reads, and no journal appeared.

### 2.3 L3 — every reader, read-only, on the resumed store

**Acceptance 4 is not met.** `extract_tensor_source_data.py` exits 1 on the resumed store. The cause
is a defect of the reader that predates this campaign (§2.3.2). The other five readers exit 0, no
reader refuses anything, and no reader changed any file of the store. The `QuadSourceIntegral`
extractor reads every stored row of every `(k, q, r)` it asks for, but by design it asks for a
random subsample, so it does not read all 864 (§2.3.3).

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance,
through [`run_readers.sh`](a3-v2-readiness/rehearsal/run_readers.sh)
([`readers.txt`](a3-v2-readiness/rehearsal/readers.txt)):

```bash
PYTHONPATH=. ./venv/bin/python <script>.py --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
    --ray-address local --models LambdaCDM --output var/a3-v2-readiness-07/extract/<script>/ \
    > var/a3-v2-readiness-07/extract/<script>.log 2>&1
```

#### 2.3.1 What each read, and the store afterwards

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 33 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 414 | none | identical |
| `extract_GkWKB_data.py` | 0 | 94 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 82 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 156 | none | identical |
| `extract_TkWKB_data.py` | 0 | 111 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | **1** | 45 s | `QUERY K_EXIT VALUES`: 3; then **failed** in `EXTRACT TENSOR SOURCE DATA` (§2.3.2) | 22 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 50 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 (6 `(k, q, r)`, 4 tasks each) | 73 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 84 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 351 | none | identical |
| `tensor_source_sorted_sample.py`, **not the shipped reader** (§2.3.2) | 0 | 69 s | `EXTRACT TENSOR SOURCE DATA`: 6, all six `QuadSource` rows | 27 | none | identical |

- **The model and the run.** Every run printed `** reading run "default", the only run in this
  datastore (no --run-label given)`, from `resolve_run_selection`: the store records one run label,
  so it is read. With `--models LambdaCDM` each printed `>> RUNNING PIPELINE FOR MODEL LambdaCDM`
  and nothing about `QCD_Cosmology`. Without `--models`, `extract_Gk_data.py` (the quickest) read
  LambdaCDM, then printed the skip line log 03 describes, and exited 0.
- **No file of the store changed.** Before and after each reader, the SHA-256 of the primary and
  of every shard was identical: 6401569d… (primary), 4a17ef11…, 4854bd93…, fd6c21f1…, 53c7abca…
  (shards 0–3), the values taken after `resumed-c`. No `-journal`, `-wal` or `-shm` file appeared.
  The mtimes (02:04:28 for the primary, 02:09:10 for shards 0, 1 and 3, 01:44:50 for shard 2) are
  all earlier than the first reader (02:12:54) and were the same after the last.
- **The first pass's mtime column was broken, by the agent.** `run_readers.sh` as first written
  called `stat -f '%Sm %z %N'`. A GNU `stat` is first on this machine's `PATH`, and read `-f` as
  "file system status", so the column held the file system's free-block counts, which change, and
  the script printed `STORE FILES CHANGED` after every reader of the six. That line is false: the
  SHA-256 column of the same output is identical before and after each, and every mtime predates
  the pass. The script now calls `/usr/bin/stat`, and the last two runs used it (log 07,
  deviation 4). The first pass's output is kept, with a header saying this, in `readers.txt`.
- **Ray.** After each of the last two runs no Ray process of this checkout's venv was left.
- **`store fingerprint`** (no `--write`) on the resumed store afterwards: digest `39809dca…`, format
  6, problems none, "matches the fingerprint taken 2026-10-04T02:09:15+01:00 by run
  a3-v2-readiness-07-resumed-c-20261004T020421".

#### 2.3.2 `extract_tensor_source_data.py` exits 1 — acceptance 4 not met for this reader

**What happened** ([`reader_rows.txt`](a3-v2-readiness/rehearsal/reader_rows.txt)):

```
** EXTRACT TENSOR SOURCE DATA
Traceback (most recent call last):
  File ".../extract_tensor_source_data.py", line 439, in <module>
    read_models(model_list, run_pipeline)
  ...
  File ".../RayTools/RayWorkPool.py", line 331, in run
    obj = ray.get(ref)
  ...
ray.exceptions.RayTaskError(RuntimeError): ray::plot_tensor_source() (pid=78922, ip=127.0.0.1)
  File ".../extract_tensor_source_data.py", line 135, in plot_tensor_source
    values: List[QuadSourceValue] = source.values
  File ".../ComputeTargets/QuadSource.py", line 428, in values
    raise RuntimeError("values has not yet been populated")
RuntimeError: values has not yet been populated
```

**Why.**
- The reader draws its k with `random.sample` (line 364), which returns them in random order. It
  pairs them with `itertools.combinations_with_replacement` (line 390), so unless the draw happens
  to be ascending, some pairs have q > r.
- The store holds `QuadSource` rows for q ≤ r only, the six pairs of the three k (read `mode=ro`).
  A pair with q > r matches nothing.
- `QuadSource.build`'s labels-only lookup, on a miss, returns an unpopulated object
  (`Datastore/SQL/ObjectFactories/QuadSource.py`, the `row_data is None` branch) rather than
  reporting absence. The remote plot then fails on `.values`.
- A sharded miss is not a write and was never meant to raise `ReadOnlyMiss`. The store was
  byte-identical across the failing run. It is not prompt 03's read-only pool.

**It predates this campaign.** With three k, `sample` returns a permutation of all three, and the
reader passes only when it is ascending, 1 time in 6. On 2026-10-01, at `d144d04` with a
read-write pool, the same script read LambdaCDM in full (6 lookups, plots and csv written) and
failed only on `QCD_Cosmology` (`var/datastore-integrity-10/extract/extract_tensor_source_data.log`):
a 1-in-6 draw. With A3 v2's eight k the reader would pass about 1 time in 8! (40 320): essentially
never. **A fix is needed before A3 v2's read-back.** Opened as
`[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`. Not fixed here (§6).

**The rest of the read path, through a wrapper that is not the shipped reader.** By the user's
decision (2026-10-04),
[`tensor_source_sorted_sample.py`](a3-v2-readiness/rehearsal/tensor_source_sorted_sample.py) runs
the unmodified `extract_tensor_source_data.py` through `runpy.run_path(..., run_name="__main__")`,
after replacing `random.sample` with a function that returns the real draw in ascending k when
every item is a `wavenumber_exit_time`, and the real draw unchanged otherwise. Nothing else is
changed.

```bash
PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/tensor_source_sorted_sample.py \
    --database var/datastores/a3-v2-readiness-07-resumed.sqlite --ray-address local \
    --models LambdaCDM --output var/a3-v2-readiness-07/extract/extract_tensor_source_data-sorted-sample/
```

It printed `random.sample drew k = [316227.7660168379, 1000000.0, 100000.0] /Mpc; returned in
ascending order [100000.0, 316227.7660168379, 1000000.0] /Mpc`. The shipped reader would have
failed on that draw. It exited 0 and read all six `QuadSource` rows (6 lookups; a csv and plots
for each of the six (q, r)). It raised no `ReadOnlyMiss` or `ReadOnlyWrite`, and left every file
of the store byte-identical, by SHA-256, mtime and size, with no journal. **It shows only that the
rest of the reader's read path works, read-only, on every `QuadSource` row. It does not show that
the shipped reader works: it does not.**

#### 2.3.3 `extract_QuadSourceIntegral_data.py` and the 864 rows

The prompt asked that the extractor's count of rows read equal the store's 864. It does not, and
cannot as written. The script draws a random subsample: 2 of the 3 response k
(`round(0.4 × 3 + 0.5)`) and 3 of the 6 (q, r) pairs (`round(0.4 × 6 + 0.5)`). It reads, with
`object_read_batch`, the rows of each of the 2 × 3 = 6 `(k, q, r)` drawn. What it did read, against
the store (read `mode=ro`; `reader_rows.txt`):

| `(k, q, r)` drawn, /Mpc | Stored rows | Rows in `integral_data.csv` |
|---|---|---|
| (1e5, 1e6, 1e6) | 96 | 96 |
| (3.16e5, 1e5, 3.16e5) | 96 | 96 |
| (3.16e5, 1e6, 1e6) | 96 | 96 |
| (1e5, 1e5, 3.16e5) | 0 (outside the triangle filter) | none written |
| (1e5, 3.16e5, 1e6) | 0 | none written |
| (3.16e5, 3.16e5, 1e6) | 0 | none written |

- **Prompt 01's point holds.** At its default pair, now the build's `(1e-32, 1e-7)`, the extractor
  found every stored row of every triple it asked for: 288 of 288, each triple's 96 z_response
  values. On 2026-10-01 a store built at `1e-7` would have yielded none.
- **"Every row of the store" was not exercised.** The store holds 864 rows in 9 triples, 96 each,
  and this run drew 3 of them. The prompt's acceptance assumed the script reads every row; it
  never did (on 2026-10-01 it made 24 lookups too). Not a defect of the script, which subsamples on
  purpose. Recorded, not opened (log 07, deviation 5).

#### 2.3.4 `extract_tensor_source_data.py` after prompt 07a — three runs, each exit 0

*Added 2026-10-04 by prompt 07a. §2.3.2 above stays as it is: it was right for the reader it ran.*

Prompt 07a changed the reader in two places (log 07a): `run_pipeline` pairs its `sample` draw with
`extract_common.quad_source_pairs`, which orders it through `wavenumber_exit_time_array`, the
class `main.py`'s stored pairs come from, so every pair asked for has q ≤ r; and
`plot_tensor_source` calls `extract_common.require_quad_source(source)` before it reads
`source.values`, so a labels-only miss raises a `RuntimeError` naming q and r and the words "no
stored QuadSource". The `sample` draw itself is unchanged.

The shipped reader, on the staged tree of prompt 07a (parent `e923b3c`), ran three times, serially,
read-only, on the same closed resumed store, each in the foreground with its own local Ray
instance, through [`e3_run.sh`](a3-v2-readiness/e3/e3_run.sh), which hashes the store before and
after with [`snapshot.sh`](a3-v2-readiness/e3/snapshot.sh). Each run's command:

```bash
PYTHONPATH=. ./venv/bin/python extract_tensor_source_data.py \
    --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
    --ray-address local --models LambdaCDM \
    --output var/a3-v2-readiness-07a/extract/run-<n>/ \
    > var/a3-v2-readiness-07a/extract/run-<n>.log 2>&1
```

**Before the first run** (10:23–10:31): `python -m RunRegistry list` showed 16 runs, none
`running` (14 finished, 2 `unknown`). No process whose command line names `SecondaryGWKit/venv`
was alive; every Ray process on the machine was the user's ChamPBH cluster (`gcs_server` 62525,
`raylet` 62574, their agents and four idle workers, the pids and start times prompt 07 recorded).
`RAY_ADDRESS` was unset. `store show` showed the store's registry sidecar, not a tombstone. `store
fingerprint` (no `--write`): digest `39809dca…`, format 6, problems none, "matches the fingerprint
taken 2026-10-04T02:09:15+01:00 by run a3-v2-readiness-07-resumed-c-20261004T020421".

| Run | Start | Exit | Wall | `QUERY K_EXIT VALUES` | `EXTRACT TENSOR SOURCE DATA` | csv | plots (full, parts, q-zoom, r-zoom) | `ReadOnlyMiss` / `ReadOnlyWrite` / `RuntimeError` | Store files |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10:31:37 | 0 | 25 s | 3 lookup | 6 lookup | 6 | 6, 6, 6, 3 | none | identical |
| 2 | 10:32:18 | 0 | 27 s | 3 lookup | 6 lookup | 6 | 6, 6, 6, 3 | none | identical |
| 3 | 10:32:47 | 0 | 26 s | 3 lookup | 6 lookup | 6 | 6, 6, 6, 3 | none | identical |

- **Each run's summary lines** ([`run-1.log`](a3-v2-readiness/e3/run-1.log),
  [`run-2.log`](a3-v2-readiness/e3/run-2.log), [`run-3.log`](a3-v2-readiness/e3/run-3.log), ANSI
  colour codes stripped):
  ```
  >> Opened existing sharded datastore ".../a3-v2-readiness-07-resumed.sqlite" read-only with 4 shards (version label "2025.1.1", serial 1)
  ** reading run "default", the only run in this datastore (no --run-label given)
  >> RUNNING PIPELINE FOR MODEL LambdaCDM
     @@ LambdaCDM background model: source grid construction version 2, digest 7139cb9c
  ** QUERY K_EXIT VALUES
        Queue summary: 3 lookup
  ** EXTRACT TENSOR SOURCE DATA
     -- ALL WORK ITEMS COMPLETE
        Queue summary: 6 lookup
        (no compute items were evaluated - all assets read from datastore)
  ```
  No `Traceback`, `ReadOnlyMiss`, `ReadOnlyWrite` or `RuntimeError` in any log.
- **All six pairs, each run.** The six csv files are the q ≤ r pairs of the store's three k:
  (1e5, 1e5), (1e5, 3.1623e5), (1e5, 1e6), (3.1623e5, 3.1623e5), (3.1623e5, 1e6), (1e6, 1e6)
  /Mpc ([`files-written.txt`](a3-v2-readiness/e3/files-written.txt)). The `r-zoom` plot is drawn
  only when q ≠ r, so three. The three runs' csv directories are identical (`diff -r`), and equal to
  the one 07's sorted-draw wrapper wrote (`var/a3-v2-readiness-07/extract/extract_tensor_source_data-sorted-sample/`).
- **The store, around each run**, by `shasum -a 256` and `/usr/bin/stat -f '%m %z'`
  ([`run-<n>.state.txt`](a3-v2-readiness/e3/run-1.state.txt), [`final.txt`](a3-v2-readiness/e3/final.txt)).
  The same before and after every run, and no `-journal`, `-wal` or `-shm` beside any file:

  | File | SHA-256 | mtime (epoch) | size |
  |---|---|---|---|
  | `a3-v2-readiness-07-resumed.sqlite` | `6401569db93fbff8dafdb0a8093596ccea850d2b830ac38d5aff5db59bb16963` | 1791075868 | 36864 |
  | `-shard0000.sqlite` | `4a17ef11df1fc0c3532e740e9cbf9aecb3e11f7e11e1a60ecd820946e5b1b684` | 1791076150 | 23777280 |
  | `-shard0001.sqlite` | `4854bd9305bde6063efc3c791b09776fa0ea443684131f104af8057c1a29667e` | 1791076150 | 23904256 |
  | `-shard0002.sqlite` | `fd6c21f119d841d958ec33e74bd415835f2ded8bc8815be285156375cdcbba06` | 1791074690 | 1077248 |
  | `-shard0003.sqlite` | `53c7abca1a661bfae6dae17f648482bdf12b292be026db30da0cd7ab038853f6` | 1791076150 | 23531520 |

  These are the values §2.3.1 gives after `resumed-c`.
- **Ray, after each run and after the last** ([`ray-after.txt`](a3-v2-readiness/e3/ray-after.txt),
  10:33:32): no process whose command line names `SecondaryGWKit/venv`. Every `gcs_server`,
  `raylet`, `ray::` or `/ray/` process listed is ChamPBH's, with the pids and start times above.
  Nothing lingered, so nothing was waited for.
- **`store fingerprint`** (no `--write`) afterwards: the same output as before, line for line,
  digest `39809dca…`, matching the sidecar ([`fingerprint-after.txt`](a3-v2-readiness/e3/fingerprint-after.txt)).

**What three passes show, and what they do not.** The unit tests are the proof that every order of
the draw is paired correctly: `ComputeTargets/tests/test_tensor_source_pairs.py` runs
`quad_source_pairs` on every permutation of three and of four k. These runs show that the shipped
reader works on a real store. Each run of the old reader passed with probability 1/6 (§2.3.2), so
three passes of it would have had probability 1/216. That bounds how likely these three passes
were by luck alone; it is not a proof. The reader does not print the order `sample` drew, so which
draws these were is not recorded.

### 2.4 What A3 v2's launch must now pass

This checklist supersedes `docs/datastore-integrity-verification.md` §2.6, which stays as it is.
Each item points to where this prompt exercised it.

- [ ] **The command**, on the macstudio, from a clean checkout of the commit to be built:
  ```bash
  nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
      --build \
      --database var/datastores/<A3 v2 store>.sqlite \
      --job-name <A3 v2 job name> --atol 1e-32 --rtol 1e-7 --cpus 6 \
      --register <slug> --purpose "<one line>" --campaign handover --prompt <prompt>; \
    echo "** driver exit status $?"' \
    > var/runs/<campaign>-<prompt>-<slug>.launch.log 2>&1 &
  disown
  ```
  That is §1.1's command without `--rehearsal`. The store and job names are the `handover`
  campaign's to choose. The macstudio keeps its own registry, so a `list` here sees nothing there
  (README §3). Check the manifest's `scope` begins `full build:` and names the pair (§2.1.4 is its
  rehearsal counterpart), and that `stdout.log` shows four `replaced 1 occurrence` lines, `main.py
  argv` with `--zend 0.1` and no `--no-prune-unvalidated`, and the local Ray bootstrap (§2.1.5).
- [ ] **The pre-launch checks**, on that machine (§1.3): `RunRegistry list` shows nothing `running`
  or stale; `store show` on the store's name says `absent`, no runs; no Ray process from that
  checkout's venv, by a listing that searches the command lines for the venv's path and `ray`, not
  `ray status` and not `pgrep`'s pattern alone (§2.1.3); `RAY_ADDRESS` unset; `git status --short`
  empty; the load average recorded.
- [ ] **One check after the launch, then no polling** (§1.1): the manifest, `status.json` `running`
  with a pid answering `kill -0`, and `RunRegistry list`.
- [ ] **`store create` after the first close, and the fingerprint at finish** (§1.1; §2.1.1:
  `fingerprint_sidecar` is "not written … (there is none)" until the sidecar exists, then
  `written`).
- [ ] **An interruption**, if one comes, is recorded `killed`/143 within about a second, then gains
  its fingerprint, and the launch log ends `** driver exit status 143` (§2.1.2). **A wait must end on
  the pid, not on the recorded state**, since `killed` is written before the fingerprint (§2.1.2).
  **Then wait, by pid, for every Ray process from the build's venv to exit**: the dashboard agent
  and runtime-env agent can outlive the driver by minutes (§2.1.3,
  `[07-a-killed-builds-ray-agents-outlive-the-driver]`).
- [ ] **A resume with `--build --resume`**, the same pair, the same scope flags (none for A3 v2), and
  no `--no-prune-unvalidated`. It prunes the unvalidated rows at open, recomputes only what is
  missing, and ends equal to an uninterrupted build (§2.2). `--resume` does not check the scope a
  store was built with (log 06a, observation 1): resume with exactly the original command plus
  `--resume`.
- [ ] **A3 v2's guard**: `--build` without `--resume` on an existing store refuses before Ray and
  before registering, naming the files (§2.1.6).
- [ ] **After the build:** the census, with verdict `no divergence` (§2.2.5); `integrity_check` `ok`
  and `foreign_key_check` empty on every file (§2.2.5); `read_inventory` with no problem (§2.2.5);
  one `version` row per shard, `2025.1.1` (§2.2.3); no journal and no `replication_in_flight` row
  (§2.2.2).
- [ ] **Every reader with `--ray-address local`**, and `--models LambdaCDM` or not, each leaving every
  file of the store as it was, by SHA-256 and mtime, and no journal (§2.3). The default `auto`
  attaches to any `ray start` cluster on the machine. The `QuadSourceIntegral` extractor finds every
  stored row of each triple it draws (§2.3.3). `store fingerprint` afterwards equals the sidecar's.
  **`extract_tensor_source_data.py` must be fixed first**: as shipped it fails on almost every draw
  of A3 v2's eight k (§2.3.2).

**What this prompt could not exercise, and why.**
- **The production grid and `--zend 0.1`.** The rehearsal scope is 3 k over $10^5$–$10^6$/Mpc at
  `--zend 250` (06a, D5). Production's work list, its batch count and its low-$z$ cost are not
  measured here.
- **The macstudio itself**: its registry, its Ray, its load and its disk.
- **`QCD_Cosmology`**: the build is LambdaCDM-only, as A3 v2 is. The readers' skip path is
  exercised (§2.3), not a `QCD_Cosmology` read.
- **A crash mid-replicated-write**: both interruptions were in sharded stages, which make no
  replicated write, so no in-flight record was left (§2.5).
- **A signal during `ray.init`**: `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]`.
  Both signals came minutes after the handler was installed.
- **The pool's prune of an unvalidated `BackgroundModel`**: none existed (§2.5).
- **A `SIGKILL` or a power loss**: not tried; D7-1 is a `SIGTERM`.

**Every issue still open on this board, and what it means for A3 v2** (board §3.2):
- `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]` — **blocks A3 v2's
  read-back of `QuadSource`**: the shipped reader would fail on all but 1 in 40 320 draws of eight k
  (§2.3.2). Fix it before A3 v2's read-back.
- `[07-a-killed-builds-ray-agents-outlive-the-driver]` — after an interruption, wait by pid for the
  build's Ray agents before calling the machine clear, or before relaunching (§2.1.3).
- `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]` — none on the build; a verification
  tool must be wrapped to tabulate A3 v2's runs, as here.
- `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` — an interrupted A3 v2's
  `status.json` may lose `pid` and `stage`, and a `SIGKILL` right after could leave it unparseable.
  Not seen here.
- `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]` — a signal in A3 v2's first
  seconds may record `failed` and exit 1 or 15, not 143. Not exercised.
- `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]` — none: A3 v2 is
  `--build`.
- `[01-reader-tolerance-agreement-does-not-scan-object-read-batch]` — none today; a test gap.
- `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` — an A3 v2 interrupted in its
  first open leaves an empty primary a person deletes; `--build` and `--build --resume` both refuse
  it.
- `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` — none: A3 v2 passes no
  `--drop`.

**Superseded in part, 2026-10-04, after prompt 07a (`3b96ba1`).** Two lines above were true for
the tree at `a6c8f39` and are no longer:
- The readers' item's last sentence, "`extract_tensor_source_data.py` must be fixed first". It is
  fixed: the reader pairs its k in stored order and exited 0 three times on the resumed store, all
  six pairs read and the store byte-identical (§2.3.4). For A3 v2 the item now reads: every reader,
  `extract_tensor_source_data.py` included, as the rest of that item says.
- `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]` in the list of open
  issues. It is closed (board §4), and no longer blocks A3 v2's read-back.

07a opened one issue, which the list should be read as including:
- `[07a-the-quadsourceintegral-extractor-reads-a-quadsource-miss-unchecked]` — none on a complete
  store: the `QuadSourceIntegral` extractor asks for pairs in stored order. A row genuinely missing
  from A3 v2 would surface there as an error that names no pair.

### 2.5 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen. After every launch, each shard's
  `BackgroundModel` row is validated (§2.2.2), and no resumed launch printed a prune of it.
  Prompt 02's pool-level prune ran on nothing.
- **An in-flight record left by a killed launch.** Did not happen. `replication_in_flight` had 0 rows
  after every launch, the two killed ones included (§2.2.2).
- **A repair by the check at open.** Did not happen. No launch printed `_print_reconciliation`'s
  output, or any repair, reconciliation or in-flight line (a search of every `stdout.log` and launch
  log for `reconcil`, `repair`, `in.flight`, `replicat`).

---

## 3. Commands, verbatim

The launches are §1.1's, with these substitutions:
- `reference`: `--database var/datastores/a3-v2-readiness-07-reference.sqlite`, `--register reference`,
  no `--resume`;
- `resumed-a`: `--database var/datastores/a3-v2-readiness-07-resumed.sqlite`, `--register resumed-a`,
  no `--resume`;
- `resumed-b`, `resumed-c`: the resumed store, `--register resumed-b` / `resumed-c`, and `--resume`.

Each `--purpose` is in its manifest (`run_records.txt`). The interrupters, launched detached right
after their builds:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/a3-v2-readiness-07-resumed-a-20261004T014440 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 15 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/a3-v2-readiness-07-resumed-b-20261004T015203 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS, ' 100 --interval 5 > /dev/null 2>&1 &
disown
```

The stores' sidecars:

```bash
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/a3-v2-readiness-07-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint --write \
    var/datastores/a3-v2-readiness-07-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/a3-v2-readiness-07-resumed.sqlite
```

The checks, each read-only:

```bash
./venv/bin/python docs/a3-v2-readiness/rehearsal/closed_store_state.py var/datastores/<store>.sqlite
./venv/bin/python docs/a3-v2-readiness/rehearsal/prune_serials.py var/datastores/a3-v2-readiness-07-resumed.sqlite
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py \
    var/datastores/a3-v2-readiness-07-reference.sqlite var/datastores/a3-v2-readiness-07-resumed.sqlite
PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/store_soundness.py \
    var/datastores/a3-v2-readiness-07-reference.sqlite var/datastores/a3-v2-readiness-07-resumed.sqlite
```

with `stage_counts_07.py`, `compare_stores.py`, the guard and the readers as quoted in §2.

---

## 4. The two stores

Both stay in `var/datastores/` with their sidecars (D7-2): `a3-v2-readiness-07-reference.sqlite`,
its four shards and `a3-v2-readiness-07-reference.manifest.json`; `a3-v2-readiness-07-resumed.sqlite`,
its four shards and `a3-v2-readiness-07-resumed.manifest.json`. Shard 2 of each is 1 077 248 bytes
and holds only replicated rows: with three k values, the sharded classes fall on shards 0, 1 and 3.
`store retire` is the user's to run.

---

## 5. Issues opened

- `[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]` (§2.3.2), with the
  silent unpopulated object a labels-only `QuadSource` miss returns.
- `[07-a-killed-builds-ray-agents-outlive-the-driver]` (§2.1.3).
- `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]` (§2.2.1).

All three are on the board's §3.2 and in `docs/OPEN_ISSUES.md` §1.17.

---

## 6. What stays under `var/`

- `var/runs/a3-v2-readiness-07-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`: each run's
  manifest, `status.json` with its full fingerprint, `stdout.log`, `stderr.log`, and, for
  `resumed-a` and `resumed-b`, `interrupter.log`.
- `var/runs/a3-v2-readiness-07-<slug>.launch.log`, beside each run directory.
- `var/a3-v2-readiness-07/`: the pre-launch captures (`prelaunch/`), the closed-store states and
  hashes (`state/`), the check outputs (`checks/`), the readers' outputs and logs (`extract/`,
  `extract-summary*.txt`), the stray interrupter's log, and the agent's working notes (`notes.txt`).
