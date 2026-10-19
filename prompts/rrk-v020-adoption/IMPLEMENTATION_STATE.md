# rrk-v020-adoption campaign — implementation state

**Last updated:** 2026-10-19 · **Status: IN PROGRESS — 3 of 4 landed.** 01 landed and reviewed
(the pin moved, the venv on `v0.2.0`); 02 landed and reviewed (the hard-kill route in `CLAUDE.md`,
the interrupter's `--signal`, the driver test); 03 landed and reviewed (R3 holds; the hard-kill route holds;
Ray's GCS server outlived the `SIGKILL`, and U3's stop was ruled by the user); 04 written
2026-10-19, after 03's review (README §2.1), with U4 added to its charter.
U1–U4 ruled by the user (README §6.1). The issue assigned here from `handover` is closed (§4); one
is open, opened by 03 (§3).

The plan is [`README.md`](README.md), written on 2026-10-18 at `79434bf` from RunRegistryKit's
`v0.2.0` release notes and its adoption note for SGK.

The campaign moves SGK's `runregistrykit` pin from `v0.1.0` to `v0.2.0`, puts the route after a
hard kill (`end`, then the resume) into `CLAUDE.md`, and shows it on the A3 v2 launcher in a
rehearsal, R3, before the A3 v2 compute run depends on it.

**Campaign:** [`README.md`](README.md) ·
**Code:** `requirements.txt`, `CLAUDE.md`, `docs/datastore-integrity-verification/interrupt_when.py`,
`docs/a3-v2-readiness/interrupter_build_dummy_test.sh`, a new test module in `RunDrivers/tests/` ·
**Verification:** `docs/rrk-v020-adoption-verification.md` (created by 03) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.22

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-18, and U4 2026-10-19 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the shape | four prompts: pin, adopt, rehearse, close |
| **U2** — the hard kill | a pair of launches on a third store, after the unchanged four-launch protocol |
| **U3** — Ray after a `SIGKILL` | stop and ask if a Ray process of this repository outlives the killed driver |
| **U4** — the GCS step (2026-10-19) | `CLAUDE.md` rule 5 gains it, in 04: after `end`, check `ray-processes`; a GCS server still listed is ended by a person with `SIGKILL` by its pid |

The orchestrator's choices of method, C1–C8, are in README §6.2. **C3 departs from the plan as
accepted** (the drivers keep raising `begin`'s refusal with its traceback); it is the user's to
reverse (README §6.3).

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Pin `runregistrykit` at `v0.2.0`](01-pin-runregistrykit-at-v0.2.0.md) | the pin; the baselines; `CLAUDE.md`'s pin mentions | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | `0a27b0c` | [`logs/01-pin-runregistrykit-at-v0.2.0.md`](logs/01-pin-runregistrykit-at-v0.2.0.md) |
| 02 | [Adopt the hard-kill route](02-adopt-the-hard-kill-route.md) | `CLAUDE.md`'s rules; `interrupt_when.py --signal`; the driver test | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | `2818780` | [`logs/02-adopt-the-hard-kill-route.md`](logs/02-adopt-the-hard-kill-route.md) |
| 03 | [Rehearse through `v0.2.0` (R3)](03-rehearse-through-v0.2.0.md) | **R3**; the hard-kill pair; closes `[a3-v2-a-hard-killed-run-blocks-its-store]` | Opus | ✍️ 2026-10-18 | ✅ 2026-10-19 | `17916cb` | [`logs/03-rehearse-through-v0.2.0.md`](logs/03-rehearse-through-v0.2.0.md) |
| 04 | [Close the campaign](04-close-the-campaign.md) | the final measurements; G1 drafted; the index's notes; what A3 v2 inherits; rule 5's GCS step (U4) | Sonnet | ✍️ 2026-10-19 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–03 now; 04 after 03's review. The hold is for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, under
`orchestrator/`: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-18, used for
01), [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-18, used for 02),
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-18, used for 03) and
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-19, for 04).

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **K1** | 01 | `requirements.txt:81` moved to `v0.2.0`; the venv installed from it | ✅ `0a27b0c`: venv `runregistrykit` 0.2.0 at `0bae3a3c…`, not editable; `pip freeze` differs by that line alone |
| **B1** | 01 | the baselines: `pip freeze`, provenance, `list`, `store show` on the 21 sidecars, the seven roots | ✅ `0a27b0c`: taken before and after; 45 rows and 21 blocks equal; the seven roots at §0.2's counts both times |
| **A1** | 02 | `CLAUDE.md`: liveness by host and boot; the route after a hard kill; no `Run.finish` by hand | ✅ `2818780`: rule 2 by host and boot; new rule 5; the pointer; `end` in the command list |
| **A2** | 02 | `interrupt_when.py --signal {TERM,KILL}`; the build dummy test's `KILL` case | ✅ `2818780`: default output unchanged; case D, child exit 137 |
| **A3** | 02 | the driver test: a `SIGKILL`ed run refuses `register`; `end`; `register` accepted | ✅ `2818780`: `test_hard_killed_predecessor.py`, 4 tests; `RunDrivers` 99 → 103 |
| **L1**–**L4** (R3) | 03 | README §4.1's four launches through `v0.2.0`; L4 against R2 | ✅ `17916cb`: both stores `39809dca…`, `encoding_format` 1, equal to each other, their sidecars and R2's two recorded fingerprints; prune 48 = 48; every L4 count R2's, every statistics block byte-identical to R2's same slug |
| **H1** | 03 | the hard-kill pair: `hardkill-a` killed, the refused resume, `end`, `hardkill-b` to `done`, `39809dca…` | ✅ `17916cb`: 137 with status `running`; `!!`; `begin` refused, nothing created or changed; `end` verified, second refused; `hardkill-b` `done`, sidecar `written`, `39809dca…` |
| **H2** | 03 | no Ray process outlives the `SIGKILL`ed driver (U3) | ❌ not as stated: the GCS server outlived it 20 m 50 s, until a `SIGKILL` on the user's ruling; opened as `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (§3) |
| **Z1** | 04 | the final measurements | ⬜ |
| **Z2** | 04 | G1 drafted for RunRegistryKit's `interrupted-runs` board | ⬜ |
| **Z3** | 04 | `docs/OPEN_ISSUES.md`'s notes on the transferred issues `v0.2.0` closed, and one line on each of their SGK boards | ⬜ |
| **Z4** | 04 | what A3 v2 inherits; `CLAUDE.md` rule 5's GCS step (U4); the GCS issue narrowed | ⬜ |
| **Z5** | 04 | the records: verification §3, this board, README §2, `prompts/INDEX.md`, `docs/OPEN_ISSUES.md` | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RunRegistryKit has tagged `v0.2.0` with release notes and an SGK adoption note | — | ✅ holds (2026-10-18): tag object `895e3c70…` on `0bae3a3c…`; `docs/releases/v0.2.0.md` and `docs/adoption/v0.2.0/secondarygwkit.md` in the tag; nothing under `runregistrykit/` or in `pyproject.toml` changed after it (RunRegistryKit `f6d4b80`) |
| R3 | the four launches through `v0.2.0` reproduce `39809dca…`, equal to R2's recorded fingerprints | 03 | ✅ holds (2026-10-18, `17916cb`): both stores `39809dca…`, 0 differences against R2's `reference` and `resumed-c` |
| HK | a `SIGKILL`ed build's resume is refused at `begin`, `end` records the run `lost`, and the resume then reproduces `39809dca…` with its fingerprint recorded | 03 | ✅ the route holds (2026-10-19, `17916cb`), every step as written. **H2 did not hold as written**: Ray's GCS server outlived the killed driver past ten minutes; the prompt's stop fired, and the user ruled a `SIGTERM` (no effect) and then a `SIGKILL` (U3); it is §3's issue |
| RunRegistryKit G1 | SGK has adopted `v0.2.0` in a campaign of its own, its pin moved in a commit of its own | 01 | ✅ pin moved at `0a27b0c`; the record on RunRegistryKit's board is the user's (README §6.3); 04 drafts it |

## 3. Active and unresolved issues

- **[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]** *(opened 2026-10-19 by prompt 03,
  at R3's hard-kill pair)* — after a `SIGKILL` of a build's driver, Ray's GCS server does not exit.
  `hardkill-a`'s driver (pid 99903) was killed at 23:49:15. By 23:49:35 every Ray process of its run
  was gone but one: the GCS server, pid 99918, reparented to pid 1. Its `gcs_server.out` logs
  `Stopping GCS server.` and "Not sending reply because executor stopped" at 23:49:15, a metrics
  export failure at 23:49:21, and nothing after. It was alive at 23:59:15, the prompt's ten-minute
  deadline (correction 10), with no children and no listening socket, and the prompt stopped. On the
  user's rulings the orchestrator sent it a `SIGTERM` at 00:06:13, which had no effect, then took a
  read-only stack sample: the main thread, in all 1386 samples, sits in
  `boost::asio … signal_handler … → ray::gcs::GcsServer::Stop() → ray::rpc::GrpcServer::Shutdown() →
  grpc::Server::ShutdownInternal(gpr_timespec) → gpr_cv_wait`, a shutdown waiting for in-flight
  calls that will not complete. A `SIGTERM` goes through the same handler. A `SIGKILL` at 00:10:05,
  on the user's second ruling, ended it at once. `ray stop` was never run. Record:
  [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §2.3,
  [`hardkill_03.txt`](../../docs/rrk-v020-adoption/rehearsal/hardkill_03.txt); the sample is
  `var/rrk-v020-adoption-03/gcs_server_99918_sample.txt` (not committed).

  **Impact on A3 v2.** The route after a hard kill (`CLAUDE.md` rule 5: `end`, then the resume)
  holds, and the resume does not need the GCS server gone: `hardkill-b` started its own fresh
  instance. But the dead build's GCS server stays, so `tools.rehearsal_lock ray-processes` is not empty:
  `release` refuses (`tools/rehearsal_lock.py:376-380`) and a next `acquire` exits 4. After `end`, a person checks `ray-processes`, and may need to `SIGKILL` a listed GCS server by
  its pid, never by pattern and never with `ray stop`. It is one observation: whether every
  `SIGKILL` of a driver leaves the GCS server like this, and what signalled it into `Stop()`, are
  not established. **Next step:** 04 states this among what A3 v2 inherits. Any fix is Ray's or the
  launcher's (for example a parent-death watch on the GCS server), and is not taken here. Indexed at
  `docs/OPEN_ISSUES.md` §1.22.

## 4. Resolved issues

Closed by prompt 03, 2026-10-19:

- **[a3-v2-a-hard-killed-run-blocks-its-store]** — *opened 2026-10-16 on the
  [`handover`](../handover/IMPLEMENTATION_STATE.md) board, which holds it, and assigned here on
  2026-10-18 (README §6.2 C6).* After a hard kill, `runregistrykit` 0.1.0 blocked the store, and no
  registry command ended the dead run.

  **Resolution.** Shown on A3 v2's own launcher with `runregistrykit` `v0.2.0`: `hardkill-a`,
  `SIGKILL`ed, stayed `running` and `list` marked it `!!`; the resume was refused at `begin` with
  nothing created and nothing changed; `end` recorded it `lost`, verified; `hardkill-b` then ended
  `done` with its fingerprint `39809dca…` recorded and the sidecar `written`. **Closed on the
  `handover` board's §4**, which holds the resolution in full; the row leaves `docs/OPEN_ISSUES.md`
  §1.22. Record: verification §2, [`logs/03-rehearse-through-v0.2.0.md`](logs/03-rehearse-through-v0.2.0.md).

---

## 5. Dispatch record

### 5.0 Planning (2026-10-18)

The plan was proposed in conversation on 2026-10-18 and accepted as proposed (U1–U3). It was
measured at `79434bf` (README §0.2):
- 45 runs, none `running` or stale;
- 21 sidecars, all tombstones;
- the seven roots at AdaptiveLevin 32, ComputeTargets 700, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped), ObjectFactories 279, RunDrivers 99 and tools 91, all OK;
- a dry run of the moved pin installs `runregistrykit-0.2.0` and nothing else.

Writing 02 against the drivers led to **C3**: the plan's clean-error change to the two drivers is
not taken, and the reason is recorded in README §6.2. The planning commit assigns
`[a3-v2-a-hard-killed-run-blocks-its-store]` here: the `handover` board gains the **Assigned**
line, and the index row moves from `docs/OPEN_ISSUES.md` §1.1 to §1.22.

### 5.1 After prompt 01 (2026-10-18)

Landed at `0a27b0c`, on `f0d280a`. K1 and B1 pass; no stop condition fired. Log:
[`logs/01-pin-runregistrykit-at-v0.2.0.md`](logs/01-pin-runregistrykit-at-v0.2.0.md); measurements:
`measure/out/01-{pip,list,store-show,roots}.txt`.
- **The pin.** `requirements.txt:81` is `@v0.2.0`. The dry run said `Would install
  runregistrykit-0.2.0` and nothing else. The venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`,
  `requested_revision` `v0.2.0`, not editable; `datastorekit` and `raypoolkit` unchanged;
  `pip check` clean; importing the core loads none of `ray`, `sqlalchemy`, `datastorekit`.
- **The registry read as before.** `list`: the same 45 rows, state and marker, exit 0, nothing on
  stderr. The summary gains "45 with no host recorded", not the 43 the prompt expected: the two
  `a3-pilot` directories have no status, so they count (log, deviation 1). The 21 sidecars read
  block for block as before, `problems: none`.
- **The roots**, before and after, serially: AdaptiveLevin 32, CosmologyModels 39, ObjectFactories
  279, tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers 99, ComputeTargets 700, all OK; no known
  flake fired.
- **`this_host()`:** `id` `8d2b247824882486`, `name` `mps026088.phys.susx.ac.uk`, `boot`
  `376461AF-4985-4F42-8E41-F4E914F5E3E6`. 03 checks its launches against that boot.
- **No issue opened or closed;** `docs/OPEN_ISSUES.md` stays at 100.
- **The venv is ahead of `git revert`:** reverting 01 leaves `v0.2.0` installed (README §4.2).

**Orchestrator review of prompt 01** (2026-10-18, against `0a27b0c`, by
[`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) §3). Dispatched from `f0d280a` to one
Sonnet subagent, after the seven roots were measured at `8a8cd10`. Nothing landed after the agent's
commit. **All seven checks pass**, and no stop condition fired.
1. **Scope.** Eight files: `requirements.txt`, `CLAUDE.md`, the four `measure/out/01-*` files, the
   log and this board. `requirements.txt`'s diff is line 81 alone. Nothing under `README.md`,
   `prompts/INDEX.md`, `orchestrator/`, code, tests or `docs/`; the index untouched, at 100.
2. **`CLAUDE.md`.** `:56`'s pin and `:88`'s say `v0.2.0`; the clause "and moved to `v0.2.0` on
   2026-10-18 in the `rrk-v020-adoption` campaign" follows the adopting campaign, the sentence
   re-wrapped over three lines. No other line moved; `v0.1.0` no longer appears.
3. **The venv, measured here.** `direct_url.json`: `commit_id` `0bae3a3c…`, `requested_revision`
   `v0.2.0`, no `dir_info`; `pip show -f` lists no `runregistrykit/tests/`. `pip freeze` here
   equals the agent's after-freeze, and differs from its before-freeze by line 82 alone; `pip check`
   clean; `pip install --dry-run -r requirements.txt` now installs nothing. `01-pip.txt` has the
   dry run before the install, and the install's only uninstall is `runregistrykit-0.1.0`.
4. **The imports and the host.** The `sys.modules` check prints `[]`. `package_provenance(...)`
   equals the agent's after-reading, and its `datastorekit` and `raypoolkit` entries the
   before-reading. `TERMINAL_STATES` ends `lost`; `this_host()` gives the note's `id`, `name` and
   `boot`, as the agent's did.
5. **The readings.** `01-list.txt`, parsed by a scratch script into `(id, marker, state)`: 45 rows
   before, 45 after and 45 in a fresh `list` here, all equal; no marker; exit 0 and an empty stderr
   at each. The after-summary is the note's line exactly, "45 with no host recorded".
   `01-store-show.txt`: 21 blocks before and 21 after, each ending `exit=0` with `problems: none`,
   equal block for block; a fresh `store show` on the backup's, `handover-stale-run-probe`'s and
   `runregistry-extraction-08-resumed`'s equals the after-blocks. The roots, re-run here at
   `0a27b0c`, serially, load 10–27 from other work: all seven at §0.2's counts, no failure, as in
   both of the agent's runs.
6. **Nothing else moved.** `find var -newer <marker>`, the marker set just before dispatch, is
   empty. `git status --short` was clean after the agent's commit. RunRegistryKit's checkout is
   unchanged, at `f6d4b80`, clean.
7. **The records.** The log has every §5.1 heading and classifies its five deviations, none
   `UNINTENDED DRIFT`; deviation 1 is the summary clause, as the note asked; deliberate breakage
   "none"; §2.6's check of the adoption note recorded. The commit message is in `CLAUDE.md`'s form.
- **The prompt's expectation of 43 was the plan's error, not the package's.** `list` counts every
  run whose status has no `host` object (`__main__.py:659`, `__init__.py:1243` at the tag), and a
  directory with no status has none. README §2.2 01's acceptance 3 carries a correction.
- **Residue fixed in the follow-up commit:** "this commit" → `0a27b0c` on this board and in the
  log; the header re-wrapped (its first two lines ran past 100 columns); this note listed as "used
  for 01"; README §2's row for 01; README §2.2's correction above; `prompts/INDEX.md`'s row.
- **What 02 inherits:** the venv on `v0.2.0`; `CLAUDE.md`'s registry rules as they were, now
  describing an older release (02's task); this host's `boot`, `376461AF-…`, for 03.

### 5.2 After prompt 02 (2026-10-18)

Landed at `2818780`, on `c435666`. A1–A3 pass; no stop condition fired. Log:
[`logs/02-adopt-the-hard-kill-route.md`](logs/02-adopt-the-hard-kill-route.md); measurements:
`measure/out/02-{roots,dummy-build,dummy-scoped}.txt`.
- **`CLAUDE.md`.** Rule 2 gains host and boot; a new rule 5 is the route after a hard kill
  (Durability and Provenance are 6 and 7); the pointer names "Ending a run whose end was never
  recorded" as rule 5's source; `end` joins the command list (log, deviation 1, sanctioned by the
  orchestrator under README §6.2 C4).
- **The interrupter.** `--signal {TERM,KILL}`, default `TERM`; the default's log is as before. The
  build dummy test's case D: `interrupter exit=0`, `child exit=137`, "sent SIGKILL". Cases A–C of
  it and A–E of the scoped test give 08's outcomes.
- **The test.** Four tests: the refusal at `register` (nothing taken over, observed before the
  helper restores), `end` then the second registration, `end` refused on a `lost` run, `end`
  refused on a live one (`--unverified` included).
- **The roots.** Every root at its count; `RunDrivers` 103; no known flake fired.
- **The mutations,** all five caught (log). No child process left; `var/` untouched.
- **No issue opened or closed;** `docs/OPEN_ISSUES.md` stays at 100.

**Orchestrator review of prompt 02** (2026-10-18, against `2818780`, by
[`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) §3). Dispatched from `c435666` to one
Sonnet subagent, after the probe of the route and the seven roots at `c288eb8`. Nothing landed
after the agent's commit. **All seven checks pass**, and no stop condition fired.
1. **Scope.** Nine files: `CLAUDE.md`, `interrupt_when.py`, `interrupter_build_dummy_test.sh`, the
   new module, the three `measure/out/02-*` files, the log and this board. No driver, existing test,
   `README.md`, `prompts/INDEX.md` or `orchestrator/`; the index untouched, at 100.
2. **`CLAUDE.md`.** The registry section alone, and `end` in the command list (`:75`). Read against
   RunRegistryKit's README at the tag: rule 2's host and boot, `foreign` untested, an earlier boot
   `stale` whatever its pid, no-host records as before; rule 5's steps, "on the host it ran on",
   no override for an answering pid, `--unverified` for another host or an unsaying status,
   `lost`; `Run.finish` never by hand; the unreaped-process sentence and the `nohup sh -c` reaping.
   Rules 6 and 7 are the old 5 and 6 word for word. The pointer names the section as rule 5's
   source and does not attribute it to "The rules".
3. **The interrupter.** `--signal` is a keyword with default `signal.SIGTERM`; every check before
   the send is unchanged; the send's line takes the signal's name. Both dummy tests re-run here at
   `2818780`: normalised for time, pid and path, cases A–C of the build test and A–E of the scoped
   test equal R2's recorded outputs line for line; case D `interrupter exit=0`, `child exit=137`,
   "sent SIGKILL". `black --check` clean.
4. **The test.** Read whole against prompt §2.3 and the note: `load_driver` imported; no Ray;
   nothing under `var/`; each child reaped by a cleanup registered before its first read; the
   restoration observed inside the helper, before its `finally`; the four written keys excluded
   from "every other key is kept". Run here three times: 4 tests, OK, 1.6 s each; no stray child.
5. **The mutations and nothing else.** Breakages 1, 3 and 4 re-applied here from the log's diffs:
   1 fails tests 1 and 2 at `assertIsInstance(outcome, RuntimeError)`; 3 fails test 1 with
   `{'stdout': True, …}`; 4 gives case D `child exit=143` under a log that says `SIGKILL`. Each
   reverted; `git status --short` clean. The roots, re-run here at `2818780`, serially, load 8–17:
   AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91, LiouvilleGreen 148
   (1 skipped), RunDrivers 103, ComputeTargets 700, all OK. `find var -newer <marker>` is empty;
   `list` reads 45 runs as before. RunRegistryKit's checkout unchanged at `f6d4b80`, clean.
6. **The measurements.** `02-roots.txt` has the seven roots with their counts and loads; the two
   dummy outputs are whole, each under a header naming the command and `c435666` with 02's changes
   uncommitted (log, deviation 3), and equal to the runs here but for one pid.
7. **The records.** Every §5.1 heading; each changed `CLAUDE.md` paragraph quoted; four deviations,
   none `UNINTENDED DRIFT`; the five breakages as diffs. The commit message is in `CLAUDE.md`'s
   form.
- **An observation, not acted on.** `CLAUDE.md`'s preamble still says `prompts/run-registry`'s five
  failures are "the reason for each rule below"; rule 5's reason is the hard kill, which the rule
  states itself. Left as is: the sentence was a summary before, and rule 5 carries its own reason.
- **Residue fixed in the follow-up commit:** "this commit" → `2818780` on this board and in the
  log; the log's five diff blocks indented into their list items, so that they render (their text
  unchanged); this note listed as "used for 02"; the header; README §2's row for 02;
  `prompts/INDEX.md`'s row.
- **What 03 inherits:** `CLAUDE.md`'s rule 5 as the route it runs; `interrupt_when.py --signal
  KILL` for `hardkill-a` alone; the refusal's words as the probe and the test pin them (no
  trailing full stop); and the agent's note that a `KILL`ed build's `interrupter.log` is the one
  record of why it died, to be saved before `end`.

### 5.3 After prompt 03 (2026-10-19)

Landed at `17916cb`, on `aab4075`. L1–L4 and H1 pass; H2 does not hold as written, and the prompt's
§2.3 stop fired. Log: [`logs/03-rehearse-through-v0.2.0.md`](logs/03-rehearse-through-v0.2.0.md);
record: [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §1–§2;
outputs: `docs/rrk-v020-adoption/rehearsal/*_03*`.
- **R3.** Six launches under one lock, each launched detached, checked once, waited on by the
  orchestrator by pid. `reference` and `resumed-c` `done`/0, `resumed-a` and `resumed-b`
  `killed`/143. Both stores `39809dca…`, format 6, `encoding_format` 1, equal to each other, their
  sidecars and R2's two recorded fingerprints. Every manifest: `packages` the note's record,
  `git_dirty` false, twenty keys with `host`; every status `host` this machine's. No `.tmp` left by
  either handler's `finish`. `resumed-a` left 168 `GkSource` rows, 48 unvalidated; `resumed-b`'s open
  pruned exactly those 48; `resumed-c` computed 683 = 864 − 181 QSI rows. Readers, guard, census and
  soundness clean. L4 is R2's; R2's disk-space warning is gone (93 % used).
- **The hard-kill pair.** `hardkill-a` killed by `SIGKILL` 30–35 s into QSI (238 rows stored,
  inferred); exit 137, status `running`, no journal, no `.tmp`. `list` `!!`, stderr naming `end`.
  The resume refused at `begin` (`RuntimeError … No run directory was created`, exit 1, no Ray
  started, nothing changed). `end` exit 0, `lost`, verified; the second refused. Sidecar made.
  `hardkill-b` `done`, `39809dca…`, `fingerprint_sidecar` `written`; it pruned nothing (none was
  unvalidated) and computed 626 QSI rows.
- **Ray after the `SIGKILL`.** The GCS server outlived the driver. The prompt stopped at 23:59:15 with
  the lock held; the user ruled `SIGTERM` (00:06:13, no effect) then `SIGKILL` (00:10:05), carried out
  by the orchestrator. Opened as `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (§3).
- **The issues.** `[a3-v2-a-hard-killed-run-blocks-its-store]` closed on the `handover` board's §4;
  one opened. `docs/OPEN_ISSUES.md` stays at 100.
- **The registry:** 51 runs, 49 finished, 2 `unknown`, none `running` or stale. **The lock** released;
  free. **The roots** at 02's counts.

**Orchestrator review of prompt 03** (2026-10-19, against `17916cb`, by
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) §3). Dispatched from `aab4075` to one Opus
subagent, after the seven roots, the venv, the host, the store names and a hot-journal probe were
checked at `b84c53d`. The agent was resumed with `SendMessage` after each of the six builds, with
one wait per build ending on the driver pid's death: `ENDED done`, `killed`, `killed`, `done`,
`running` (`hardkill-a`, as expected) and `done`, heartbeat never stale. **The prompt's §2.3 stop
fired** at `hardkill-a` (below), and the user ruled twice. Nothing landed after the agent's commit.
**Every check passes but H2**, which fails as the record says, and the record is right.
1. **Scope.** 29 files: 26 added (the verification document, the log and 24 `*_03*` outputs) and
   three modified (this board, the `handover` board and `docs/OPEN_ISSUES.md`). README,
   `prompts/INDEX.md`, `orchestrator/`, and every code file, test and script are untouched. The
   `handover` diff moves the one entry from §3 to §4 and corrects its header's count, "Eight" to
   "Seven".
2. **The runs**, read from each run directory. `reference`, `resumed-c` and `hardkill-b` are
   `done`/0 at `39809dca…`; `resumed-a` and `resumed-b` are `killed`/143 at `c60e0d19…` and
   `7e41cbe2…`; all three of those fingerprints and the two killed ones are format 6,
   `encoding_format` 1, 21 classes, problems `{}`, with no `fingerprint_error`. `hardkill-a` is
   `lost`, `exit_code` `null`, with no fingerprint. No `.tmp` in any run directory. Each killed
   `stderr.log` holds the handler's two lines (`SystemExit(1)` both times); `hardkill-a`'s is empty.
   The launch logs end 0, 143, 143, 0, 137, 1 (the refused launch) and 0.
3. **The manifests and statuses.** All six: 20 keys, `packages` equal as JSON to
   `package_provenance` of the three, `git_head` `aab4075…`, `git_dirty` false, `script_sha256`
   `2a985303…`, `--resume` on `resumed-b`, `-c` and `hardkill-b` only, the rehearsal `scope`, the
   right `results`, the manifest `host`'s `id` and `name` this machine's, and every status's `host`
   `{"id": "8d2b247824882486", "boot": "376461AF-…"}`.
4. **Nothing stored was recomputed.** `stage_counts.py` on the six runs here equals the committed
   table. `GkSource` 291 − 168 + 48 = 171; QSI 864 − 181 = 683 on the resumed chain; on the
   hard-kill chain, `hardkill-b` computed 626 QSI rows and nothing else. The four shards of the
   `hardkill` store now hold 288 + 288 + 0 + 288 = 864 QSI rows. `hardkill-a`'s 238 is inferred
   (864 − 626), as the log says: the killed run recorded no fingerprint.
5. **The prune.** `prune_check_03.txt` gives 48 of 48 absent, 0 reused, twice. The hard-kill pair
   pruned nothing, and that is correct: `GkSource` had completed, and `QuadSourceIntegral` has no
   `validated` column: the shard's `pragma_table_info` lists none, and its factory says "No validated
   flag" (`ObjectFactories/QuadSourceIntegral.py:970`).
   `hardkill-a` left no journal, so DatastoreKit's rollback at open was not exercised either
   (log, observation 2).
6. **The fingerprints, here.** `compare_stores.py`: reference = resumed and reference = hardkill,
   21 classes equal. `compare_with_recorded.py` on all three primaries against R3's `reference` and
   `resumed-c` and R2's two: 0 differences in all twelve comparisons, and against each sidecar.
   `runregistrykit store fingerprint` on each: `39809dca…`, problems none, `recorded: matches`.
7. **Integrity, here.** `store_soundness.py` on all three: `=> clean`, 0 problems, 0 duplicates.
   The census: no divergence, three times. No journal under `var/datastores/`. All 18 files of
   R3's stores are byte-identical across this review's reads.
8. **The readers.** Recorded in `readers_03.txt`: all seven exit 0, every file identical around
   each. None was run here.
9. **L4.** `policy_statistics.py` here on `reference`, `resumed-b`, `resumed-c` and `hardkill-b`: 14
   lines each, each equal to its committed output, and the first three byte-identical to R2's same
   slug.
10. **The printed lines.** As the log records; the only line naming `runregistrykit` is the banner.
11. **H1, the route.** `hardkill-a`'s `ended`: `verified` `true`, `not_verified_because` `null`;
    evidence `host` `this`, `boot` `this`, `pid` 99903, `pid_answers` `false`, `state` `running`,
    `status_readable` `true`, `heartbeat_age` 1348.9 s; `git_head` `aab4075…`; `at` 00:11:12, after
    the kill. The refused launch's log is the `resuming into` line, the traceback from `:556` into
    `begin`, the refusal word for word and ending `No run directory was created`, and `** driver
    exit status 1`, with no `bootstrapping` line. `hardkill-b` is `fingerprint_sidecar` `written`.
12. **H2, Ray after the `SIGKILL`.** Not met, and recorded truly. The orchestrator's own reading at
    23:49:35 listed only the GCS server, pid 99918, reparented to pid 1. It was alive at 00:00:03
    and ignored a `SIGTERM` at 00:06:13. A read-only stack sample
    (`var/rrk-v020-adoption-03/gcs_server_99918_sample.txt`) shows the main thread, from `main`'s
    asio `signal_handler`, in `GcsServer::Stop()` → `GrpcServer::Shutdown()` →
    `grpc::Server::ShutdownInternal` → `gpr_cv_wait`. So the GCS server *was* signalled at the kill:
    its log says `Stopping GCS server.` at 23:49:15.152. Its shutdown then waited for ever on
    in-flight calls, and a second signal goes through the same handler. A `SIGKILL` at 00:10:05,
    on the user's ruling, ended it. `ray stop` was never run.
13. **The lock.** `lock_03.txt`: `acquire` at 23:27:51, 47 s before `reference`'s manifest;
    `set-pid` at the six builds with their driver pids (90795, 93122, 94708, 95847, 99903, 10646),
    none at the refused launch; `release` at 00:14:55, exit 0; one `status` record cut short by the
    agent's display pipe (log, deviation 17). Now `status`: free; `ray-processes`: none;
    `~/ScienceData/` holds `ChamPBH-rehearsals/` alone.
14. **`var/`.** The 21 tombstones hash to `7ad566f4…`, as before dispatch. `find var -newer <marker>`
    lists only R3's run directories, launch logs, stores, sidecars and working directory. `list`:
    51 runs, 49 finished, 2 `unknown`, "45 with no host recorded"; no R3 process remains.
15. **The records.** This board, the `handover` board, the index (100, the closed row gone, the new
    row in §1.22), the verification document (§1 in R2's §3 shape, §2 with §2.1–§2.10) and the log
    (every §5.1 heading; 21 deviations classified, three `UNINTENDED DRIFT`, each a record slip
    annotated where it occurs). The commit message is in `CLAUDE.md`'s form.
16. **The suites**, here at `17916cb`, serially, load 4–6: AdaptiveLevin 32, CosmologyModels 39,
    ObjectFactories 279, tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers 103, ComputeTargets
    700, all OK.
- **The orchestrator's own actions outside the note.** On the user's two rulings under U3: one
  `kill -TERM 99918` (00:06:13), one read-only `sample 99918 2`, and one `kill -KILL 99918`
  (00:10:05). Nothing else was signalled.
- **An observation, not acted on.** The index header's "across twenty-two campaigns" and "Two of
  these hold no open issue" cannot be reproduced by counting rows: the `Board` column has 21
  distinct values after this commit, with two of them `*none yet*`. The agent moved the count by
  this commit's change alone and said so (log, deviation 14). It is left for a later reconciliation
  of the index, which is not this campaign's.
- **Residue fixed in the follow-up commit:** "this commit" → `17916cb` on this board, in the log
  and on the `handover` board; this board's header; this note listed as "used for 03"; README §2's
  row for 03; `prompts/INDEX.md`'s row.
- **What 04 inherits:** R3 and HK hold; the route after a hard kill as run here (`list` `!!`, `end`
  on the host it ran on, then the resume); and
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`: after a hard kill a person checks
  `ray-processes`, and a GCS server still listed after ten minutes is ended by `SIGKILL` by its
  pid, never with `ray stop`. 04 states this among what A3 v2 inherits. R3's three stores stay
  live until the user decides, after 04's review.
