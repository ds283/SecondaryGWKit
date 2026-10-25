# py313-venv campaign — implementation state

**Last updated:** 2026-10-25 · **Status: PLANNED — 0 of 2 landed.** 01 is written and ready to
dispatch. 02 is written and **held** until 01 has landed and been reviewed (README §2). U1–U4 were
ruled by the user.

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
| 01 | [Move the laptop's venv to Python 3.13](01-move-the-laptop-venv-to-python-3.13.md) | the two pins; the venv; `distributions`; `CLAUDE.md`; the roots; R6 with the hard-kill pair | laptop | Opus | ✍️ 2026-10-25 | ⬜ | — | — |
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
| **B1** | 01 | the baselines under 3.12: freeze, provenance, the seven roots, `black` | ⬜ |
| **P1** | 01 | `requirements.txt:75` `ray==2.59.0`, `:88` `SQLAlchemy==2.0.55`; no other line | ⬜ |
| **V1** | 01 | the laptop's venv on 3.13.16; its freeze differs from 3.12's in the two lines only; the old venv moved aside | ⬜ |
| **G1** | 01 | `runregistrykit.toml`'s `distributions` gains `ray`, `SQLAlchemy`, `numpy`, `scipy` | ⬜ |
| **C1** | 01 | `CLAUDE.md` "Repository mechanics" names the 3.13 venv | ⬜ |
| **S1** | 01 | the seven roots under 3.13 at 3.12's counts; `test_ray_address_default` passes; `black` clean; warnings recorded | ⬜ |
| **R6** | 01 | R3's four launches on the laptop under 3.13: both stores `39809dca…`; the lock lists a 2.59.0 instance by its temp dir; the readers exit 0 | ⬜ |
| **K1** | 01 | R3's hard-kill pair under Ray 2.59.0: the route holds; how long the GCS server outlives the `SIGKILL` | ⬜ |
| **M1** | 02 | the Mac Studio's checkout fast-forwarded; its venv on 3.13.16, with a freeze identical to the laptop's | ⬜ |
| **M2** | 02 | the seven roots there at 01's counts | ⬜ |
| **R7** | 02 | R6's four launches on the Mac Studio: both stores `39809dca…` | ⬜ |
| **A1** | 02 | A3 v2 under 3.13: every reader exits 0, the files are byte-identical, the fingerprint is `0e32f24a…` | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **FP6** | R6's three stores are `39809dca…` under Python 3.13.16, `ray` 2.59.0 and `SQLAlchemy` 2.0.55 | 01 | ⬜ |
| **FP7** | R7's two stores are `39809dca…` on the Mac Studio | 02 | ⬜ |
| **A3** | A3 v2's fingerprint, computed under the new venv, is `0e32f24a…` | 02 | ⬜ |

If FP6 or FP7 fails, the bisection of README §4.3 is the user's call.

## 3. Active and unresolved issues

None opened by this campaign.

**Assigned here (2026-10-25), held on another board:**
`[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`, on the `handover` board's §3.
01 closes it.

## 4. Resolved issues

None.

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
