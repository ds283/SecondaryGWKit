# rrk-v020-adoption campaign — implementation state

**Last updated:** 2026-10-18 · **Status: IN PROGRESS.** 01 landed and reviewed (the pin moved,
the venv on `v0.2.0`); 02 landed, awaiting review; 03 written, not dispatched; 04 held until 03's
review (README §2.1).
U1–U3 ruled by the user (README §6.1). One issue assigned here from `handover` (§3).

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

**Made by the user, 2026-10-18 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the shape | four prompts: pin, adopt, rehearse, close |
| **U2** — the hard kill | a pair of launches on a third store, after the unchanged four-launch protocol |
| **U3** — Ray after a `SIGKILL` | stop and ask if a Ray process of this repository outlives the killed driver |

The orchestrator's choices of method, C1–C8, are in README §6.2. **C3 departs from the plan as
accepted** (the drivers keep raising `begin`'s refusal with its traceback); it is the user's to
reverse (README §6.3).

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Pin `runregistrykit` at `v0.2.0`](01-pin-runregistrykit-at-v0.2.0.md) | the pin; the baselines; `CLAUDE.md`'s pin mentions | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | `0a27b0c` | [`logs/01-pin-runregistrykit-at-v0.2.0.md`](logs/01-pin-runregistrykit-at-v0.2.0.md) |
| 02 | [Adopt the hard-kill route](02-adopt-the-hard-kill-route.md) | `CLAUDE.md`'s rules; `interrupt_when.py --signal`; the driver test | Sonnet | ✍️ 2026-10-18 | ✅ 2026-10-18 | this commit | [`logs/02-adopt-the-hard-kill-route.md`](logs/02-adopt-the-hard-kill-route.md) |
| 03 | [Rehearse through `v0.2.0` (R3)](03-rehearse-through-v0.2.0.md) | **R3**; the hard-kill pair; closes `[a3-v2-a-hard-killed-run-blocks-its-store]` | Opus | ✍️ 2026-10-18 | ⬜ | — | — |
| 04 | Close the campaign | the final measurements; G1 drafted; the index's notes; what A3 v2 inherits | Sonnet | ⏸ after 03's review | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–03 now; 04 after 03's review. The hold is for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, under
`orchestrator/`: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-18, used for
01).

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **K1** | 01 | `requirements.txt:81` moved to `v0.2.0`; the venv installed from it | ✅ `0a27b0c`: venv `runregistrykit` 0.2.0 at `0bae3a3c…`, not editable; `pip freeze` differs by that line alone |
| **B1** | 01 | the baselines: `pip freeze`, provenance, `list`, `store show` on the 21 sidecars, the seven roots | ✅ `0a27b0c`: taken before and after; 45 rows and 21 blocks equal; the seven roots at §0.2's counts both times |
| **A1** | 02 | `CLAUDE.md`: liveness by host and boot; the route after a hard kill; no `Run.finish` by hand | ✅ this commit: rule 2 by host and boot; new rule 5; the pointer; `end` in the command list |
| **A2** | 02 | `interrupt_when.py --signal {TERM,KILL}`; the build dummy test's `KILL` case | ✅ this commit: default output unchanged; case D, child exit 137 |
| **A3** | 02 | the driver test: a `SIGKILL`ed run refuses `register`; `end`; `register` accepted | ✅ this commit: `test_hard_killed_predecessor.py`, 4 tests; `RunDrivers` 99 → 103 |
| **L1**–**L4** (R3) | 03 | README §4.1's four launches through `v0.2.0`; L4 against R2 | ⬜ |
| **H1** | 03 | the hard-kill pair: `hardkill-a` killed, the refused resume, `end`, `hardkill-b` to `done`, `39809dca…` | ⬜ |
| **H2** | 03 | no Ray process outlives the `SIGKILL`ed driver (U3) | ⬜ |
| **Z1** | 04 | the final measurements | ⬜ |
| **Z2** | 04 | G1 drafted for RunRegistryKit's `interrupted-runs` board | ⬜ |
| **Z3** | 04 | `docs/OPEN_ISSUES.md`'s notes on the transferred issues `v0.2.0` closed | ⬜ |
| **Z4** | 04 | what A3 v2 inherits | ⬜ |
| **Z5** | 04 | the records: verification §3, this board, README §2, `prompts/INDEX.md`, `docs/OPEN_ISSUES.md` | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RunRegistryKit has tagged `v0.2.0` with release notes and an SGK adoption note | — | ✅ holds (2026-10-18): tag object `895e3c70…` on `0bae3a3c…`; `docs/releases/v0.2.0.md` and `docs/adoption/v0.2.0/secondarygwkit.md` in the tag; nothing under `runregistrykit/` or in `pyproject.toml` changed after it (RunRegistryKit `f6d4b80`) |
| R3 | the four launches through `v0.2.0` reproduce `39809dca…`, equal to R2's recorded fingerprints | 03 | ⬜ |
| HK | a `SIGKILL`ed build's resume is refused at `begin`, `end` records the run `lost`, and the resume then reproduces `39809dca…` with its fingerprint recorded | 03 | ⬜ |
| RunRegistryKit G1 | SGK has adopted `v0.2.0` in a campaign of its own, its pin moved in a commit of its own | 01 | ✅ pin moved at `0a27b0c`; the record on RunRegistryKit's board is the user's (README §6.3); 04 drafts it |

## 3. Active and unresolved issues

- **[a3-v2-a-hard-killed-run-blocks-its-store]** — *opened 2026-10-16 on the
  [`handover`](../handover/IMPLEMENTATION_STATE.md) board, §3, which holds it. Assigned to this
  campaign on 2026-10-18 (README §6.2 C6).* RunRegistryKit closed the defect in `v0.2.0`: `end`
  (its `interrupted-runs` 04) and `begin`'s second-writer refusal (05). It reaches SGK by 01's pin.
  **Next step:** 03 shows the route on the real launcher (README §2.2, 03, acceptance 3), then
  closes the entry on the `handover` board's §4, with `docs/OPEN_ISSUES.md` §1.22, in its commit.

## 4. Resolved issues

None yet.

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

Landed at this commit, on `c435666`. A1–A3 pass; no stop condition fired. Log:
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
