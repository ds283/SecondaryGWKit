# rrk-v020-adoption campaign — implementation state

**Last updated:** 2026-10-19 · **Status: CLOSED — 5 of 5 landed and reviewed.** Closed at 04 on
2026-10-19 (`34bf4f7`, reviewed), then reopened the same day for one prompt, 05, when the user
reversed C3 (README §6.1 U5), and closed again at 05 (`96a168e`, reviewed): the drivers print their
refusals as one `!!` line. U1–U5 ruled by the user. 01 landed and reviewed (the pin moved, the venv
on `v0.2.0`); 02 landed and reviewed (the hard-kill route in `CLAUDE.md`, the interrupter's
`--signal`, the driver test); 03 landed and reviewed (R3 holds; the hard-kill route holds; Ray's GCS
server outlived the `SIGKILL`, and U3's stop was ruled by the user); 04 landed and reviewed (the
final measurements, G1 drafted for RunRegistryKit, the index's notes on the two issues
RunRegistryKit closed, what A3 v2 inherits, and `CLAUDE.md` rule 5's GCS step under U4); 05 landed
and reviewed (`Refusal` in both drivers, `begin`'s refusals included, `ConfigError` excepted). The
issue assigned here from `handover` is closed (§4); one is open, opened by 03, narrowed by 04 and
assigned to `handover` by the user after the close (§3). R3's three stores are retired (§5.5).

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

**Made by the user, 2026-10-18, and U4 and U5 2026-10-19 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the shape | four prompts: pin, adopt, rehearse, close |
| **U2** — the hard kill | a pair of launches on a third store, after the unchanged four-launch protocol |
| **U3** — Ray after a `SIGKILL` | stop and ask if a Ray process of this repository outlives the killed driver |
| **U4** — the GCS step (2026-10-19) | `CLAUDE.md` rule 5 gains it, in 04: after `end`, check `ray-processes`; a GCS server still listed is ended by a person with `SIGKILL` by its pid |
| **U5** — C3 reversed (2026-10-19, after the close) | the drivers print their refusals cleanly, `begin`'s included: prompt 05 |

The orchestrator's choices of method, C1–C8, are in README §6.2. **C3 departed from the plan as
accepted** (the drivers kept raising `begin`'s refusal with its traceback); the user reversed it on
2026-10-19 (U5), and 05 carries the reversal.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Pin `runregistrykit` at `v0.2.0`](01-pin-runregistrykit-at-v0.2.0.md) | the pin; the baselines; `CLAUDE.md`'s pin mentions | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | `0a27b0c` | [`logs/01-pin-runregistrykit-at-v0.2.0.md`](logs/01-pin-runregistrykit-at-v0.2.0.md) |
| 02 | [Adopt the hard-kill route](02-adopt-the-hard-kill-route.md) | `CLAUDE.md`'s rules; `interrupt_when.py --signal`; the driver test | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | `2818780` | [`logs/02-adopt-the-hard-kill-route.md`](logs/02-adopt-the-hard-kill-route.md) |
| 03 | [Rehearse through `v0.2.0` (R3)](03-rehearse-through-v0.2.0.md) | **R3**; the hard-kill pair; closes `[a3-v2-a-hard-killed-run-blocks-its-store]` | Opus | ✍️ 2026-10-18 | ✅ 2026-10-19 | `17916cb` | [`logs/03-rehearse-through-v0.2.0.md`](logs/03-rehearse-through-v0.2.0.md) |
| 04 | [Close the campaign](04-close-the-campaign.md) | the final measurements; G1 drafted; the index's notes; what A3 v2 inherits; rule 5's GCS step (U4) | Sonnet | ✍️ 2026-10-19 | ✅ 2026-10-19 | `34bf4f7` | [`logs/04-close-the-campaign.md`](logs/04-close-the-campaign.md) |
| 05 | [Print the drivers' refusals cleanly](05-print-refusals-cleanly.md) | C3 reversed (U5): `Refusal`, one `!!` line, exit 1; `begin`'s refusals included | Sonnet | ✍️ 2026-10-19 | ✅ 2026-10-19 | `96a168e` | [`logs/05-print-refusals-cleanly.md`](logs/05-print-refusals-cleanly.md) |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–03 now; 04 after 03's review. The hold is for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, under
`orchestrator/`: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-18, used for
01), [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-18, used for 02),
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-18, used for 03),
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-19, used for 04) and
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) (2026-10-19, used for 05).

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
| **Z1** | 04 | the final measurements | ✅ `34bf4f7`: the seven roots at 02's counts (ComputeTargets' one wall-clock failure, ruling 9); `pip freeze` equal to 01's; provenance equal to R3's six manifests; `list` 51 runs; 24 `store show` blocks, `problems: none`; disk 94 % |
| **Z2** | 04 | G1 drafted for RunRegistryKit's `interrupted-runs` board | ✅ `34bf4f7`: the G1 row, `**holds** 2026-10-18`, in verification §3.3 |
| **Z3** | 04 | `docs/OPEN_ISSUES.md`'s notes on the transferred issues `v0.2.0` closed, and one line on each of their SGK boards | ✅ `34bf4f7`: the index's §1.11, §1.17 and §1.21 corrected; one dated line on each of the two boards' §4 entries; the other three confirmed still open in RunRegistryKit |
| **Z4** | 04 | what A3 v2 inherits; `CLAUDE.md` rule 5's GCS step (U4); the GCS issue narrowed | ✅ `34bf4f7`: verification §3.5; rule 5's two sentences; the GCS issue narrowed |
| **Z5** | 04 | the records: verification §3, this board, README §2, `prompts/INDEX.md`, `docs/OPEN_ISSUES.md` | ✅ `34bf4f7`: verification §3, this board, README §2, `prompts/INDEX.md`, `docs/OPEN_ISSUES.md` |
| **P1** | 05 | `Refusal(RuntimeError)` in each driver; the refusal sites and `begin`'s three calls raise it, messages unchanged | ✅ `96a168e`: 7 sites in the launcher, 4 in the scoped driver; `ConfigError` re-raised first (the orchestrator note for 05, ruling 1) |
| **P2** | 05 | each `__main__` block prints a `Refusal` as one `!!` line on stderr and exits 1; failures keep their tracebacks | ✅ `96a168e`: both drivers, from a subprocess, print one `!!` line, no traceback, exit 1 |
| **P3** | 05 | `test_clean_refusals.py`; the six breakages; verification §4 with the launcher's new hash | ✅ `96a168e`: 12 tests, `RunDrivers` 103 → 115; seven breakages caught; verification §4; launcher `9b40e457…` |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RunRegistryKit has tagged `v0.2.0` with release notes and an SGK adoption note | — | ✅ holds (2026-10-18): tag object `895e3c70…` on `0bae3a3c…`; `docs/releases/v0.2.0.md` and `docs/adoption/v0.2.0/secondarygwkit.md` in the tag; nothing under `runregistrykit/` or in `pyproject.toml` changed after it (RunRegistryKit `f6d4b80`) |
| R3 | the four launches through `v0.2.0` reproduce `39809dca…`, equal to R2's recorded fingerprints | 03 | ✅ holds (2026-10-18, `17916cb`): both stores `39809dca…`, 0 differences against R2's `reference` and `resumed-c` |
| HK | a `SIGKILL`ed build's resume is refused at `begin`, `end` records the run `lost`, and the resume then reproduces `39809dca…` with its fingerprint recorded | 03 | ✅ the route holds (2026-10-19, `17916cb`), every step as written. **H2 did not hold as written**: Ray's GCS server outlived the killed driver past ten minutes; the prompt's stop fired, and the user ruled a `SIGTERM` (no effect) and then a `SIGKILL` (U3); it is §3's issue |
| RunRegistryKit G1 | SGK has adopted `v0.2.0` in a campaign of its own, its pin moved in a commit of its own | 01 | ✅ pin moved at `0a27b0c`; the record on RunRegistryKit's board is the user's (README §6.3); 04 drafts it |

*2026-10-19, at the close (04, `34bf4f7`):* R3 and HK hold, and G1's draft is in the verification
document's §3.3, as `**holds** 2026-10-18`. The commit that records it in RunRegistryKit is the
user's.

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

  **Narrowed (2026-10-19, prompt 04):** what A3 v2 inherits is stated, in
  [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §3.5 and in
  `CLAUDE.md` rule 5, which now says that after `end` a person checks `ray-processes`, and ends a GCS
  server still listed ten minutes after the kill with `kill -KILL <pid>`, by its pid, never by
  pattern and never with `ray stop` (the user's ruling U4). What remains is the cause and any fix,
  which are Ray's or the launcher's. **The issue has no owner.** Whether it is assigned to a campaign,
  `handover`'s launcher being the place a parent-death watch would go, is the user's (README §6.3).

  **Assigned (2026-10-19):** to `handover`, by the user, after 04's review. That campaign owns the
  A3 v2 launch and its launcher, `docs/handover/quadsource_atol_sweep.py`, where a parent-death
  watch on the GCS server would go. The user may not take it up at once; until then `CLAUDE.md`
  rule 5's manual step stands. The entry stays here, its text unchanged; the `handover` board's §3
  points at it, and its index row moves to `docs/OPEN_ISSUES.md` §1.1.

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

### 5.4 After prompt 04 (2026-10-19)

Landed at `34bf4f7`, on `aa0b57c`. Z1–Z5 pass; no stop condition fired. Log:
[`logs/04-close-the-campaign.md`](logs/04-close-the-campaign.md); record:
[`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §3;
measurements: `measure/out/04-{roots,pip,list,store-show,disk}.txt`, taken at `aa0b57c` before any edit.
- **Z1.** The seven roots at 02's counts: AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279,
  tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers 103 OK; ComputeTargets 700 with one failure,
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the first of README §8's known wall-clock
  tests, which failed again alone under a Time Machine backup (orchestrator's ruling 9). `pip freeze`
  equals 01's after-install freeze; `pip check` clean; the provenance equals R3's six manifests'.
  `list`: 51 runs, 49 finished and 2 `unknown`, "45 with no host recorded", nothing `running` or stale,
  exit 0, stderr empty. 24 sidecars, all `problems: none`: the 21 tombstones as 01's after-blocks, R3's
  three live at `39809dca…`. Disk 94 % used, 64 GB free.
- **Z2.** The G1 row for RunRegistryKit's `interrupted-runs` board, `**holds** 2026-10-18`, drafted in
  verification §3.3. **Not committed in RunRegistryKit.**
- **Z3.** `[04-runs-do-not-say-which-machine-produced-them]` and
  `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` are resolved on
  RunRegistryKit's board (01 `387c58f`; 02 `2bf9c72`) and released in `v0.2.0`. The index's §1.11, §1.17
  and §1.21 say so, and the two boards' §4 entries gain one dated line each. The other three
  transferred issues are still open in RunRegistryKit.
- **Z4.** What A3 v2 inherits is verification §3.5. `CLAUDE.md` rule 5 gains U4's two sentences. The GCS
  issue is narrowed (§3).
- **Z5.** The verification document's §3, this board, README §2, `prompts/INDEX.md`,
  `docs/OPEN_ISSUES.md`. The index stays at 100.
- **The campaign leaves one issue open:** `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`
  (§3), narrowed, with no owner. It closed one (`[a3-v2-a-hard-killed-run-blocks-its-store]`, by 03).
- **Left to the user** (README §6.3): retiring R3's three stores
  (`rrk-v020-adoption-03-{reference,resumed,hardkill}`, live, `39809dca…`), or not; the G1 commit in
  RunRegistryKit; C3, if the clean error is wanted after all; and whether
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` is assigned to a campaign.

**Orchestrator review of prompt 04** (2026-10-19, against `34bf4f7`, by
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) §3). Dispatched from `aa0b57c` to one
Sonnet subagent, after the roots, the venv, the registry, the 24 `store show`s and RunRegistryKit's
board were checked at `ebb70a3`. The prompt was written after the user's ruling **U4** (README
§6.1), which added rule 5's GCS step to its charter. Nothing landed after the agent's commit. **All
ten checks pass**, and no stop condition fired.
1. **Scope.** 14 files: six `A` (the log and `measure/out/04-{disk,list,pip,roots,store-show}.txt`)
   and eight `M`, exactly the note's list. The verification diff is one hunk appending §3, 299 lines,
   none removed; README's is row 04 alone. No code, test, script, `requirements.txt` or `config/`.
2. **(V1) The measurements, here at `34bf4f7`.** `pip freeze` equals 01's after-install block;
   `package_provenance` equals the `packages` of all six R3 manifests; `list` gives 51 runs, "45 with
   no host recorded", exit 0, stderr empty; `04-store-show.txt` holds the `find` and 24 blocks, each
   `problems: none`. The 21 tombstones had equalled 01's after-blocks in the orchestrator's own check.
3. **(V2) The G1 draft,** read whole: the whole row, in the `extraction` board's G2 form (log,
   deviation 10, which is the better choice: the user pastes one line over one line). Dated
   2026-10-18 (ruling 1). Each commit and run id it names exists. Its §1–§8 mapping was read against
   the adoption note at `v0.2.0`: `requirements.txt:17`, `:75`, `:76`, `:88` are `datastorekit`, `ray`,
   `raypoolkit` and `SQLAlchemy`, as the note's §1 lists them; §3's `interrupt_when.py`, §4's
   `:556`, §6's `:249-253` and §7's prepare test are the note's own citations. The GCS finding is
   said to be Ray's, and no defect of the package is claimed.
4. **(V3) Z3.** RunRegistryKit at `f6d4b80`: `:1597` **Resolved (2026-10-16, by 01, `387c58f`)**,
   `:1763` **Resolved (2026-10-17, by 02, `2bf9c72`)**; the release notes' §2.1, §2.3 and §2.4 are the
   sections cited. The index's three paragraphs change their present-tense clauses only; the two
   boards gain one paragraph each, in place, and nothing else. The three left alone are in its
   `docs/OPEN_ISSUES.md` §1.2.
5. **(V4) Z4.** Verification §3.5 against the tree: the launcher's hash `2a985303…`, last commit
   `7af17bd`, `address="local"` at `:610`, `begin` at `:556`, the defaults at `:1000` and `:1014`; the
   pins by line; `list`'s markers (`!!` stale, `??` foreign: `__main__.py:643` at the tag); the
   lock's exits 1 and 4 at the lines given; the route's six steps against §2; the issues closed where
   it says. The headroom to 95 % is 17 GB by its own figures.
6. **(V4) `CLAUDE.md`.** One hunk, in rule 5, after the unreaped-process sentence (log, deviation
   12, a better place than the prompt's). It says what U4 says: check `ray-processes` after `end`;
   a GCS server may be left, stuck, beyond a `SIGTERM`; a **person** ends one still listed after ten
   minutes with `kill -KILL <pid>`, by its pid, never by pattern or `ray stop`; the lock waits. It
   contradicts neither RunRegistryKit's README nor the lock section.
7. **(V5) The records.** The board, the index (100 by counting; the hook; §1.22's sentence) and
   `prompts/INDEX.md` (the row closed; 27 = 2 live + 1 seeded + 24, counted from its rows) agree.
8. **(V5) Verification §3.1** agrees with the run directories, as 03's review read them.
9. **Nothing else moved.** `find var -newer <marker>` is empty; RunRegistryKit unchanged at
   `f6d4b80`, clean; the tree clean after the commit; the lock free, `ray-processes` none.
10. **(V6) The suites**, here at `34bf4f7`, serially, at load 21–45 (Spotlight, `duetexpertd`):
    AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91, LiouvilleGreen 148
    (1 skipped), RunDrivers 103 OK; ComputeTargets 700 with the one failure of
    `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, as at dispatch and in both of the agent's
    runs (ruling 9). It passed at 03's review at load 4–6, and 04 changed no code.
- **An observation, not acted on.** `store show` given a sidecar's path answers `kind: absent`,
  `problems: none` (log, deviation 13): it reads the path as a store whose sidecar would be
  `….manifest.manifest.json`. That is the package's documented argument, not a defect; not opened.
- **Residue fixed in the follow-up commit:** "this commit" → `34bf4f7` on this board, in README §2,
  the log's header and the verification document (the G1 draft included); this board's header
  re-wrapped and marked reviewed; a stray "each" in `docs/OPEN_ISSUES.md` §1.21's corrected lead-in;
  this note listed as "used for 04", and one of its lines re-wrapped.
- **The campaign is closed.** It leaves `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`
  open, narrowed, with no owner. **The user's:** R3's three stores; the G1 row, ready to paste into
  RunRegistryKit's `interrupted-runs` board §2 from verification §3.3; C3; the GCS issue's owner.

### 5.5 After the close: R3's stores retired, the GCS issue assigned (2026-10-19)

Two of the user's decisions left at the close (README §6.3), taken after 04's review.

**R3's three rehearsal stores retired, by the user's instruction.** Each was retired with
`python -m runregistrykit store retire` at `454c7d8`, clean, after a dry run of each:
`var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`.
- **The dry runs** exited 0. In each, a fresh read-only fingerprint matched the recorded `39809dca…`,
  and the only references were the store's own runs: `reference` its one; `resumed` `resumed-a`, `-b`
  and `-c`; `hardkill` `hardkill-a` (`lost`) and `hardkill-b`. No sidecar named any.
- **The retirements** exited 0, at 11:26:29, 11:26:42 and 11:26:55. Five files went per store (the
  primary and four shards), 211 MB in all. Each sidecar stays as its tombstone, with `completed` set,
  and the six run directories stay as the record. The outputs are under
  `var/rrk-v020-adoption-04/retire/` (`dry-run-03-*.log`, `retire-03-*.log`).
- **Afterwards** all 24 sidecars under `var/datastores/` are tombstones, `store show` reads each of
  R3's three with `problems: none`, and no `.sqlite` file remains there. `list` gives 51 runs, none
  `running`. Verification §3.2 and §3.5, which call R3's three live, were true at `aa0b57c` and stay
  as written.

**`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` assigned to `handover`**, by the user,
who may not take it up at once (§3, its **Assigned** line). The `handover` board's §3 gains a
pointer to the entry, which stays here; the index row moves from §1.22 to §1.1, and the count stays
100.

**Still the user's:** the G1 commit in RunRegistryKit, and C3.

### 5.6 After prompt 05 (2026-10-19)

Landed at `96a168e`, on `affaf84`. P1–P3 pass; no stop condition fired. Log:
[`logs/05-print-refusals-cleanly.md`](logs/05-print-refusals-cleanly.md); record:
[`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §4;
measurements: `measure/out/05-roots.txt`.
- **P1.** `Refusal(RuntimeError)` in each driver. `raise Refusal(`: the launcher's `run_build` 3,
  `assert_store_is_self_consistent` 1, `prepare` 2, `sweep` 1; the scoped driver's `main` 2, `register` 2.
  The other `raise RuntimeError(` sites (5 and 2) stay. At each of the three `begin` calls a
  `runregistrykit.ConfigError` is re-raised first, as it is a `RuntimeError` at `v0.2.0` (the orchestrator
  note for 05, ruling 1, `STRUCTURALLY REQUIRED`); then a `RuntimeError` becomes `Refusal(str(e))`.
- **P2.** Each `__main__` prints a `Refusal` as `!! <message>` on stderr and exits 1. No refusal is
  raised after `register` or `begin` has taken over the streams.
- **P3.** `test_clean_refusals.py`, 12 tests; `RunDrivers` 103 → 115, every other root at 02's counts
  (ComputeTargets' one wall-clock failure, re-run alone twice, recorded). Seven deliberate breakages,
  each caught (the prompt's six, and the removal of `run_build`'s `ConfigError` clause). The launcher's
  SHA-256 is `9b40e457f444db10d2b406b749e0b4391d53753e93c928fa431856b4c645e649` (R3's was `2a985303…`);
  the scoped driver's is `d2e38510acf5cff09dc63219e5c59cead2eb28042ba58c770ab8abe90d972c83`.
- **Opens nothing; closes nothing.** `docs/OPEN_ISSUES.md` is unchanged (100).
- **Left to the user:** nothing of this campaign's. *(Corrected by the orchestrator's review: the
  agent's text named the G1 commit and the GCS issue's owner as still the user's. The G1 record is
  in RunRegistryKit at `33a2cd2`, the user's commit; the GCS issue was assigned to `handover` in
  §5.5; and C3 was taken by U5, which this prompt carries.)*

**Orchestrator review of prompt 05** (2026-10-19, against `96a168e`, by
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) §3). Dispatched from `affaf84` to one
Sonnet subagent, after the prompt's citations were checked against both drivers and against `begin`
at `v0.2.0`. In writing the note the orchestrator found that `runregistrykit.ConfigError`
subclasses `RuntimeError` at the tag, which prompt §2.3's bare `except RuntimeError` would have
made a `Refusal`; ruling 1 settled it before dispatch (`ConfigError` re-raised first, a test case,
a seventh breakage). Nothing landed after the agent's commit. **All eight checks pass**, and no stop
condition fired.
1. **Scope.** Seven files: three `A` (the test module, the log, `05-roots.txt`) and four `M` (the two
   drivers, the verification document, this board), exactly the note's list. The verification diff
   is one hunk appending §4, none removed. `docs/OPEN_ISSUES.md` unchanged (100).
2. **The diffs.** Each driver's diff is the class, the `raise` sites, the three `try` blocks round
   `begin`, and the `__main__` block. Every string constant of every `raise` at `13d993c` is present
   at `96a168e`, by an `ast` extraction of the orchestrator's own; the three new raises carry none
   (`Refusal(str(e))`).
3. **The counts**, re-counted with that script: the launcher's `raise Refusal(` `run_build` 3,
   `assert_store_is_self_consistent` 1, `prepare` 2, `sweep` 1; `raise RuntimeError(` 5 in the
   functions ruling 3 names; the scoped driver's `main` 2 and `register` 2, and 2 `RuntimeError`.
   Each of the three `begin` calls sits in a `try` whose handlers are `runregistrykit.ConfigError`
   (bare `raise`) then `RuntimeError` (`raise Refusal(str(e)) from e`).
4. **The clean print, by hand.** The launcher's `--build` on an existing file, the scoped driver on an
   existing `--database`, and the launcher's `--prepare --force`: each exit 1, empty stdout, stderr
   one `!!` line holding the message, no `Traceback`.
5. **The test module, read whole.** Temporary directories and runs root only; no `ray.init`;
   `HardKilledPredecessorTestCase` imported alone; `args` from the launcher's own `main()` with the
   modes stubbed; no seam in either driver. One residue: its docstring described test 3 in ruling 2's
   relaxed form while the test asserts the strict one; corrected here.
6. **The breakages**, re-applied in a scratch copy of `96a168e`: 3 (`run_build` lets `begin`'s
   `RuntimeError` through) fails `test_run_build_makes_begins_refusal_a_refusal` and both
   `TestTheSites` tests; 5 (the model filter at `run_build._filtered_build` raises `Refusal`) fails
   `test_the_raise_sites_are_the_expected_ones`; 7 (no `ConfigError` clause in `run_build`) fails
   `test_a_config_error_from_begin_is_not_a_refusal` and the handler-order test. Each as the log says;
   the copy green again after each revert.
7. **Nothing else moved.** `find var -newer <marker>` empty; RunRegistryKit at `33a2cd2`, clean; lock
   free; `ray-processes` none; `black --check` clean on the three `.py` files.
8. **The suites** at `96a168e`, serially, at load 8–13: AdaptiveLevin 32, CosmologyModels 39,
   ObjectFactories 279, tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers **115**, ComputeTargets
   700, all OK. The known wall-clock test, which failed at the baselines and in the agent's run under
   an I/O load of 27–430 (a backup, Box, Munki), passed here.

**Residue fixed in the review's commit:** `96a168e` for "this commit" here, in the log's header and
in README §2's row; this board's header, closed again; §5.6's "Left to the user", corrected above;
verification §4's hash table, whose "Before 05" column named both hashes R3's (only the launcher's
is); the test module's docstring; `prompts/INDEX.md`'s row and header; this note listed as used.

**For the user, not opened as an issue** (the log's observation 1): `prepare` makes any
`RuntimeError` of `copy_store` a `Refusal`, as prompt §2.2 asks, and that includes a `ConfigError`.
Before 05 it was already re-raised as a plain `RuntimeError` with its cause chained; now it prints as
one `!!` line, the `ConfigError`'s own text inside it, with no traceback. By ruling 1's reasoning it
is not a refusal; `prepare` is run by a person at a terminal, and the message names the cause. A
one-clause change if the user wants it.
