# py313-venv campaign — implementation state

**Last updated:** 2026-10-25 · **Status: 01 landed, awaiting review — 1 of 2 landed.** The laptop's
venv is Python 3.13.16 with `ray` 2.59.0 and `SQLAlchemy` 2.0.55, and R6's three stores are
`39809dca…` (FP6 holds). 02 is written and **held** until 01's review (README §2). U1–U4 were ruled
by the user.

The plan is [`README.md`](README.md), written on 2026-10-25 at `ebfae11`.

The campaign moves the venv on the laptop and on the Mac Studio from Python 3.12.15 to 3.13.16,
`ray` from 2.43.0 to 2.59.0, and `SQLAlchemy` from 2.0.39 to 2.0.55. Every other pin is unchanged.
The rehearsal fingerprint must stay `39809dca…`, and A3 v2's must read `0e32f24a…` under the new
venv.

**Campaign:** [`README.md`](README.md) ·
**Code:** `requirements.txt`, `runregistrykit.toml`, `CLAUDE.md`. No `.py` file changes ·
**Verification:** [`docs/py313-venv-verification.md`](../../docs/py313-venv-verification.md)
(written by 01), with the saved outputs in `measure/out/` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.23

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-25 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — Ray | `2.59.0`, the latest |
| **U2** — SQLAlchemy | `2.0.55`; 2.1 later, after a DatastoreKit release that allows it |
| **U3** — the other pins | unchanged |
| **U4** — when | before the queued analysis jobs, on both hosts |

The orchestrator's choices of method, C1–C7, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Where | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|---|
| 01 | [Move the laptop's venv to Python 3.13](01-move-the-laptop-venv-to-python-3.13.md) | the two pins; the venv; `distributions`; `CLAUDE.md`; the roots; R6 with the hard-kill pair | laptop | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25, awaiting review | the commit that adds its log | [log](logs/01-move-the-laptop-venv-to-python-3.13.md) |
| 02 | [Move the Mac Studio's venv to Python 3.13](02-move-the-mac-studio-venv-to-python-3.13.md) | the checkout and venv there; the roots; R7; A3 v2 read under 3.13 | Mac Studio | Opus | ✍️ 2026-10-25 · ⏸ held on 01's review | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

The review of 02 closes the campaign (README §2).

**Orchestrator notes** are written just before dispatch, under `orchestrator/`:
[`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-25, for 01). The planning commit
is `1fd1c36`. The commit that adds the note also corrects prompt 01's §2.10 and §4. The plan had
R6's manifests at `git_dirty: false`, which cannot hold while the prompt's edits are uncommitted.
The stop is now a dirty `.py` file.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **B1** | 01 | the baselines under 3.12: freeze, provenance, the seven roots, `black` | ✅ taken before the move: 101-line freeze; roots 32/39/148/280/120/49/737; `black` 369 files clean |
| **P1** | 01 | `requirements.txt:75` `ray==2.59.0`, `:88` `SQLAlchemy==2.0.55`; no other line | ✅ the two lines only |
| **V1** | 01 | the laptop's venv on 3.13.16; its freeze differs from 3.12's in the two lines only; the old venv moved aside | ✅ freeze `diff` is exactly `ray` and `SQLAlchemy`; git packages at their commits, not editable; `pip check` clean; SQLite 3.53.4 both; old venv at `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025` |
| **G1** | 01 | `runregistrykit.toml`'s `distributions` gains `ray`, `SQLAlchemy`, `numpy`, `scipy` | ✅ with its comment; every R6 manifest names the seven distributions |
| **C1** | 01 | `CLAUDE.md` "Repository mechanics" names the 3.13 venv | ✅ one bullet; the packages section's `ray`/`SQLAlchemy` sentence unchanged |
| **S1** | 01 | the seven roots under 3.13 at 3.12's counts; `test_ray_address_default` passes; `black` clean; warnings recorded | ✅ every count 3.12's; the one failure, the known wall-clock test, fails under both (3.13 faster); `test_ray_address_default` 7 OK; `black` 369 clean; one new `ResourceWarning` (§3) |
| **R6** | 01 | R3's four launches on the laptop under 3.13: both stores `39809dca…`; the lock lists a 2.59.0 instance by its temp dir; the readers exit 0 | ✅ `done`/`killed`/`killed`/`done`; both stores `39809dca…`, 0 differences against R3's recorded; `ray-processes` exit 1 listing `gcs_server` and `raylet`; the readers exit 0, store unchanged; one new log line from Ray (§3) |
| **K1** | 01 | R3's hard-kill pair under Ray 2.59.0: the route holds; how long the GCS server outlives the `SIGKILL` | ✅ the route holds, `hardkill` store `39809dca…`; ❌ H2: the GCS server deadlocked as at R3 (same stack), alive ~1 h 56 m until the user's `kill -KILL`; narrows `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` |
| **M1** | 02 | the Mac Studio's checkout fast-forwarded; its venv on 3.13.16, with a freeze identical to the laptop's | ⬜ |
| **M2** | 02 | the seven roots there at 01's counts | ⬜ |
| **R7** | 02 | R6's four launches on the Mac Studio: both stores `39809dca…` | ⬜ |
| **A1** | 02 | A3 v2 under 3.13: every reader exits 0, the files are byte-identical, the fingerprint is `0e32f24a…` | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **FP6** | R6's three stores are `39809dca…` under Python 3.13.16, `ray` 2.59.0 and `SQLAlchemy` 2.0.55 | 01 | ✅ holds (2026-10-25): reference, resumed and hardkill all `39809dca…`, 0 differences against R3's `reference` and `resumed-c` |
| **FP7** | R7's two stores are `39809dca…` on the Mac Studio | 02 | ⬜ |
| **A3** | A3 v2's fingerprint, computed under the new venv, is `0e32f24a…` | 02 | ⬜ |

If FP6 or FP7 fails, the bisection of README §4.3 is the user's call.

## 3. Active and unresolved issues

- **[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]** *(opened 2026-10-25 by prompt
  01, from R6's logs; not acted on, by `CLAUDE.md` rule 4)* — Ray 2.59.0 turns on token
  authentication for a local cluster. Every R6 launch log, and three reader logs, carry
  `WARNING authentication_token_setup.py:… -- Token authentication is enabled for this Ray cluster.
  Set RAY_AUTH_MODE=disabled to opt out.` The first Ray start under 2.59.0 (`reference`, 12:51)
  also logged `Generated new authentication token and saved to /Users/ds283/.ray/auth_token`. That
  file is 64 bytes, mode 644, **outside the checkout**, and is read by every Ray ≥ 2.59 instance of
  this user on the machine (ChamPBH's and StochasticInstantons' too, once they move). R3's 2.43.0
  logs have no such line. **Impact:** none on what a build stores or a fingerprint reads (FP6
  holds). It is a new line in every build and reader log (README §5 rule 5), and a credential file
  in `~` that a rehearsal writes and nothing records. Whether the lock or an `auto` attach is
  affected was not measured. **Next step:** decide whether this repository's `local` instances
  should run with `RAY_AUTH_MODE=disabled` (a launcher and `config/` change, a campaign's) or keep
  the default and record the token file as expected. Measured in
  [`docs/py313-venv-verification.md`](../../docs/py313-venv-verification.md) §1.2.5. Not assigned.
  Indexed at `docs/OPEN_ISSUES.md` §1.23.

- **[01-a-test-leaves-an-in-memory-sqlite-engine-undisposed]** *(opened 2026-10-25 by prompt 01,
  from the warnings of §2.7; not acted on, by README §5 rule 1)* — under Python 3.13 the
  ObjectFactories root prints one `ResourceWarning: unclosed database in <sqlite3.Connection …>`
  (reported at `sqlalchemy/sql/selectable.py:4765`, where the collector found it). 3.13 newly warns
  when a `sqlite3.Connection` is collected unclosed. Traced with `PYTHONTRACEMALLOC=40` to
  `ObjectFactories/tests/test_quadsource_parent_key.py:443-444`, where `_read` does
  `sqla.create_engine("sqlite://")` and `metadata.create_all(engine)` and never disposes it. 3.12
  prints nothing there. **Impact:** none on any store or result: it is a test's in-memory engine,
  and the root passes at its count. It is noise in the suite under 3.13. **Next step:**
  `engine.dispose()` at the end of `_read` (or a `try`/`finally`), and a check that no other test
  does the same. Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.23.

**Measured here, held on another board:** `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`
(the `rrk-v020-adoption` board, §3) was narrowed on 2026-10-25 by 01's hard-kill pair under Ray
2.59.0 (verification §1.10.3).

## 4. Resolved issues

**Closed on another board:**

- **[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]** *(the `handover` board's; assigned
  here 2026-10-25)* — **Closed 2026-10-25 by py313-venv 01**, on the `handover` board's §4: this
  branch is on Python 3.13.16 with `ray` 2.59.0, later than `main`'s 2.45.0. A merge resolves
  `requirements.txt:75` to this branch's line. Its row left `docs/OPEN_ISSUES.md` §1.23.

## 5. Dispatch record

### 5.0 Planning (2026-10-25)

Proposed in conversation on 2026-10-25, after `handover` 10 landed. The user ruled U1–U4 the same
day. Measured at `ebfae11` (README §0.2):
- both hosts idle, with no run `running` or stale and the lock free;
- MacPorts 3.13.16 on both hosts;
- a dry run of the two moved pins in a fresh 3.13.16 venv resolves 101 packages, with only the two
  pins changing;
- no live module imports a module that 3.13 removed.

The planning commit also assigns `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`
here: the `handover` board gains an `**Assigned (2026-10-25):**` line, and its row moves to
`docs/OPEN_ISSUES.md` §1.23.

### 5.1 After prompt 01 (2026-10-25)

01 ran on the laptop, in the main checkout, at the dispatch `HEAD` `55c6a54`, from 12:12 to 15:37
GMT. It landed in one commit, awaiting review. The record is
[`docs/py313-venv-verification.md`](../../docs/py313-venv-verification.md) §1, and the log is
[`logs/01-move-the-laptop-venv-to-python-3.13.md`](logs/01-move-the-laptop-venv-to-python-3.13.md).
- **The venv** is Python 3.13.16 (`/opt/local/bin/python3.13 -m venv venv`, then `pip install -r
  requirements.txt`, pip not upgraded). Its freeze differs from 3.12's in exactly `ray` 2.59.0 and
  `SQLAlchemy` 2.0.55. SQLite is 3.53.4, unchanged. The 3.12 venv is at
  `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`; retiring it is the user's.
- **The roots, which 02 must equal:** AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
  (skipped=1), ObjectFactories 280, RunDrivers 120, tools 49, ComputeTargets 737. These are 3.12's
  counts. The known wall-clock test failed under both interpreters under the load of a ChamPBH
  suite, and passed in the `-W default` run.
- **The freeze 02 must equal** is the last freeze in `measure/out/01-pip.txt` (101 lines).
- **R6:** `reference` and `resumed-c` `done`; `resumed-a` and `resumed-b` `killed` at the points
  chosen. **All three stores are `39809dca…`** (FP6), with 0 differences against R3's recorded
  `reference` and `resumed-c`. The lock listed the 2.59.0 instance by its temp dir. The readers
  exited 0 and left the store unchanged.
- **The hard-kill pair** ran R3's route to `hardkill-b` `done` at `39809dca…`.
  - Ray 2.59.0's GCS server deadlocked in its `SIGTERM` handler as 2.43.0's did, frame for frame.
  - Prompt §2.9's ten-minute stop fired, and the user ruled: sample it, then the user kills it.
  - It was gone by 15:31:14, about 1 h 56 m after the `SIGKILL`.
- **Two issues opened** (§3), the GCS issue narrowed on its own board, and
  `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` closed (§4).
- **Not done:** `set-pid` for `hardkill-b`, by the agent's slip. It is the log's deviation 6: the
  lock was held but named a dead driver for `hardkill-b`'s 105 s.
- **Under `var/`:** R6's six run directories, seven launch logs, three stores with sidecars, and
  `var/py313-venv-01/`. None is retired.
- **02 is released by the review of this prompt**, not by this commit.
