# Review — handover prompt 10, re-read A3 v2

**Orchestrator:** Claude Opus 5.5, in a Claude Code session on the Mac Studio (`mps026298`), in
`/Users/ds283/Documents/Code/SecondaryGWKit`, following `prompts/handover/orchestrator/prompt-10.md`
read at the dispatch SHA with `git show`. **Date:** 2026-10-25 (UTC).
**Dispatch SHA:** `57f67a9a5b44eeca55d5ecd17f2fbd34480298a6`. **Subagent:** one, fresh context, Opus.
**Nothing committed or pushed on this machine.**

**Verdict: checks 1–8 pass; check 9 fails on one point**: `verification.md`'s links are not
relative to `docs/handover/`. That is reported, not repaired (note §5). Every other part of check 9
passes. The science is unaffected: acceptance 1–3 hold, as checks 5–8 show.

## Before dispatch (note §1)

The Gate, re-read here at 10:46Z before dispatch:

```
git fetch origin                    -> 087d420..57f67a9  handover-remedial -> origin/handover-remedial
git rev-parse origin/handover-remedial -> 57f67a9a5b44eeca55d5ecd17f2fbd34480298a6
git rev-parse HEAD                  -> b90752dda46e57d588f8107cbd4d4557134fadb2
git status --short                  -> (empty)
./venv/bin/python -m runregistrykit list -> 5 run(s): 5 finished; none running or stale; exit 0
ls -ld ~/.raypoolkit/rehearsal-lock ~/ScienceData/.rehearsal-lock -> both: No such file or directory
ls -d var/handover-10               -> No such file or directory
ls var/                             -> datastores handover-06 handover-07 handover-08 runs
git diff --stat b90752d 57f67a9 -- ObjectFactories config main.py
                                    -> config/ray_local.py | 7 +++; main.py | 10 ++--; 2 files
df -h .                             -> 120G available
```

The marker was touched in the session scratchpad at 2026-10-25 10:46:19.32Z, immediately before
dispatch. No suite baselines were taken, as the note says: prompt 10 changes no tracked file.

## 1. Scope — PASS

```
git status --short                 -> (empty)
git rev-parse HEAD                 -> 57f67a9a5b44eeca55d5ecd17f2fbd34480298a6   (the dispatch SHA)
find var -newer <marker> -not -path var/handover-10 -not -path 'var/handover-10/*'
                                   -> var
```

`HEAD` is the dispatch SHA, so the agent committed nothing. The only path outside
`var/handover-10/` that is newer than the marker is the directory `var` itself, whose mtime moved
when `var/handover-10/` was created in it. The store's files and its sidecar are not listed.

## 2. The step 1 diff — PASS

The agent's saved `out/step1_checkout.txt` (line 160), `git diff --stat b90752d HEAD -- ObjectFactories
config main.py` after the move, lists `config/ray_local.py` (+7) and `main.py` (+8 −2), 2 files. I
re-took it as `git diff b90752d 57f67a9 -- ObjectFactories config main.py` and read it in full:

- `config/ray_local.py` is new: a comment and `RAY_TEMP_DIR = "/tmp/ray-sgk"`.
- `main.py` gains the import, and `ray.init(address=args.ray_address)` becomes
  `ray.init(address=…, _temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None)`.

This is §0's two files and no more, and neither touches a schema.

## 3. The venv — PASS

- `out/pip_10.txt` line 213, the `pip install --dry-run -r requirements.txt` output: the only
  "Would install" line is `Would install raypoolkit-0.3.0`.
- I diffed `./venv/bin/pip freeze`, taken now, against lines 7–107 of
  `docs/handover/a3-v2/rehearsal/out/pip_06.txt`, which are its 101-line freeze. One line differs
  (line 76):
  ```
  < raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976
  > raypoolkit @ git+https://github.com/ds283/RayPoolKit@67a75b97a656accb98195a4bda93f64e6a5f3fb2
  ```
- `pip list --editable` prints nothing, so none of the three packages is editable. Each is a
  `git+https` pin:
  - `datastorekit@f47b9a25…`
  - `raypoolkit@67a75b97…`
  - `runregistrykit@0bae3a3c…`

## 4. The store before — PASS

- `out/store_10.txt`, `store show`: sidecar present, `kind: registry`, `problems: none`.
- The pre-reader `store fingerprint` (no `--write`): digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none,
  `recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 … at b90752d…`.
- `state/store_files_before.txt` has the primary and four shards, each with SHA-256, size and mtime
  (epoch and ISO). It also has a `SIDECAR` line for the manifest, and no `JOURNAL` line.
- The agent's `storefiles.sh` takes the shard list from the primary's own `shards` table
  (`sqlite3 … mode=ro`). It looks for `-journal`, `-wal` and `-shm` beside each file.
- The agent also found the file states identical to 08's `store_files_before.txt` (`diff` rc 0).

## 5. The readers — PASS

`out/readers_10.txt`: `run_readers.sh` returned rc 0 both times. Every one of the seven readers
printed `exit 0` and `store files: identical before and after (SHA-256, mtime, size); no journal`.
Each was followed by `Ray processes from this checkout's venv after: 0`.

I scanned each reader's own log under `var/handover-10/extract/*.log` with `grep -c`. All seven
show `Traceback=0 ReadOnlyMiss=0 ReadOnlyWrite=0 FATAL=0`.

| Reader | Exit | Wall | Files | Size (KiB) | 08: files |
|---|---|---|---|---|---|
| `extract_Gk_data` | 0 | 54 s | 1251 | 78,616 | 1593 |
| `extract_GkWKB_data` | **0** (08: 1) | 171 s (08: 24 s) | **9744** | 551,064 | **344** |
| `extract_GkSource_data` | 0 | 34 s | 847 | 69,684 | 873 |
| `extract_TkWKB_data` | 0 | 19 s | 72 | 6,936 | 72 |
| `extract_tensor_source_data` | 0 | 22 s | 172 | 7,672 | 172 |
| `extract_QuadSourceIntegral_data` | 0 | 31 s | 687 | 58,512 | 681 |
| `extract_Gk_data`, no models | 0 | 58 s | 1980 | 127,032 | 1593 |

- **Total:** 879 MB (08: 360 MB).
- **Why the counts differ:** the agent attributes the differences to the readers' unseeded
  `random.sample` and to 09's filtering. A difference in count is recorded, not a stop (note §0).

## 6. The GkWKB read-back, counted by me — PASS

```
find …/LambdaCDM/plots/full-range -type f -name '*.pdf' | wc -l   -> 1392
find …/LambdaCDM/plots/full-range -type f -name '*.png' | wc -l   -> 1392
find …/LambdaCDM/csv -type f -name '*.csv' | wc -l                -> 1392
(zoom-matching and theta: 1392 PDFs and 1392 PNGs each; 9744 files in all)
```

- **The three sets are equal.** I sorted the `k=…/zsource=…` stems of the full-range PDFs, PNGs
  and CSVs, and `cmp` reports them identical.
- **The grid:** 8 distinct k and 174 distinct z_source, and 8 × 174 = 1392.
- **No damaged files:** there are no empty files, and no full-range PDF is under 2,000 B.

**The work list.** I read the reader's code at the dispatch SHA and checked it against its log:

- `extract_GkWKB_data.py:509` sets `work_grid = product(k_subsample, z_subsample)`.
- `k_subsample` (`:457–464`) is every response k, since there are no more than 20. There are 8.
- `z_subsample` (`:471–473`) is `sample(list(z_source_sample), k=int(round(0.10·N + 0.5)))`, which
  gives 174 here.
- The log `extract/extract_GkWKB_data.log` agrees. Line 32 reads `792/1392 work items remaining`,
  line 34 `252/1392`, and line 37 `Queue summary: 1392 lookup`.

So the work list is 1392 pairs, and every one was written.

**The failing pair** `k=1e+05-k-serial=1/zsource=1.3633e+16-z-serial=19` (`stat -f %z`):

| File | Size |
|---|---|
| full-range PDF | 30,366 B (08 left a 1,039 B fragment) |
| full-range PNG | 128,806 B |
| CSV | 34,150 B |

Its zoom-matching and theta PDF and PNG are present too.

**Dropped points.** The reader reports no count, so the agent took it from the CSVs, through
`extract_common.sensible_log_points` itself (`scripts/gkwkb_readback.py`, which I read). The scope
is stated:

- **The pair:** 35 of 154 points dropped. All 35 are above the ceiling, 4.01e24 = 10³ × max|G|.
  The largest is 7.57e+243.
- **This run's 1392 sampled pairs (not the store):** 5162 of 109,594 dropped. 5034 are above the
  ceiling and 128 are exact zeros. 518 pairs lost at least one point.

**For the record** (the agent's `issues.md` 1): `z_subsample` is an unseeded random draw of about
10 %. So the failing pair was in this run's sample by chance, with odds of about 1 in 10. Acceptance
2 holds as written, but the prompt's procedure did not guarantee it.

## 7. The store after — PASS

```
diff state/store_files_before.txt state/store_files_after.txt            -> exit 0 (identical)
bash var/handover-10/scripts/storefiles.sh > <scratchpad>/store_files_now.txt
diff state/store_files_before.txt <scratchpad>/store_files_now.txt       -> exit 0 (identical, sidecar line included)
ls -a var/datastores | grep -E -- '-(journal|wal|shm)$'                  -> nothing (exit 1)
./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite
   digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56
   problems: none
   recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752dda46e57d588f8107cbd4d4557134fadb2
   exit 0
```

I re-ran the fingerprint myself, without `--write`. It equals the sidecar's `"digest"`.

## 8. The lock — PASS

`out/lock_10.txt`:

| Step | Command | Exit | Output |
|---|---|---|---|
| step 4 | `status` | 0 | free |
| step 4 | `acquire` | 0 | at 10:49:36Z |
| step 4 | `status` | **0** | HELD, `pid: -`, `owner pid: - (no pid recorded yet)`, no Ray process |
| step 8 | `ray-processes` | 0 | empty |
| step 8 | `release` | 0 | at 10:57:50Z |
| step 8 | `status` | 0 | free |
| step 8 | `ray-processes` | 0 | empty |

- **No `ray-processes` during a reader.** The agent took none while a reader ran, which the check
  makes optional.
- **The first `status`.** It was run without `--ray-temp-dir` and without its exit recorded. It was
  re-run at once, and that is classified UNINTENDED DRIFT in the log (deviation 2).
- **`status` while held exited 0, not 3.** Note §0 expected exit 3 while the lock is held with no
  pid. That was stated not to be a stop either way. The difference is recorded in `issues.md` 2, for
  the laptop to check against RayPoolKit `docs/rehearsal-lock.md` at `v0.3.0`.

I re-ran both commands myself, each as its own shell command, appending to a scratchpad file, with
`; echo $?` in the same command:

```
./venv/bin/python -m raypoolkit.rehearsal_lock status --ray-temp-dir /tmp/ray-sgk   -> exit 0
   lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)
   Ray processes of /tmp/ray-sgk: none
./venv/bin/python -m raypoolkit.rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk -> exit 0, no output
ls -ld ~/.raypoolkit/rehearsal-lock ~/ScienceData/.rehearsal-lock -> both: No such file or directory
```

## 9. The records — FAIL on one point (links); the rest PASS

**`log.md` — pass.**

- **The headings.** It has every heading of 08's log. That log is at
  `prompts/handover/logs/08-after-the-a3-v2-build.md`; the note's `logs/…` is relative to
  `prompts/handover/`. The headings are What shipped, Deviations from the prompt, Verification
  performed, Observations not acted on, and State handed to the next step.
- **The deviations.** It classifies all eight: one STRUCTURALLY REQUIRED, one UNINTENDED DRIFT and
  six IMPLEMENTATION CHOICE. Step 2's `raypoolkit` move is listed under "Outside
  `var/handover-10/`" as the prompt's own amendment, not as a deviation.

**`issues.md` — pass.** It exists and holds three observations, none acted on:

1. The readers' work lists are unseeded random draws.
2. `status` exits 0 while the lock is held with no pid.
3. `extract_GkWKB_data.py` leaks its theta figure: matplotlib's "More than 20 figures" warning,
   which predates 09.

**`verification.md` — fails on its links.**

- **What passes:** it is headed `## 4. Re-read with the fixed readers`, with subsections 4.1–4.8,
  and it says where its records will be committed (`a3-v2/reread/`).
- **What fails:** every link has the form `reread/out/<file>`. For example,
  `[gate_10.txt](reread/out/gate_10.txt)`, and the same for all ten files linked.
- **Where it breaks:** relative to `docs/handover/`, where the section will live, these resolve to
  `docs/handover/reread/out/…`, which will not exist. The records are to be committed at
  `docs/handover/a3-v2/reread/out/…`, so the links need the form `a3-v2/reread/out/<file>`. §1 of
  `A3-V2-VERIFICATION.md` uses that form (`a3-v2/rehearsal/out/host_06.txt`).
- **Why it may have happened:** §3 of the same document, already committed, links
  `after/readers_08.txt`. That is broken in the same way, and the agent's form follows it.
- **Not repaired:** the note says to report a failed check and not repair it.

The links are in a record that the laptop rewrites when it commits. Whether to correct them there,
and whether §3's are corrected too, is the user's decision.
