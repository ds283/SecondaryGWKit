# dk-and-rpk-migration campaign — implementation state

**Last updated:** 2026-10-11 · **Status: PLANNED — 0 of 9 prompts written, none landed.** The plan
is [`README.md`](README.md), written on 2026-10-11 at `b510bc9` with the user's decisions U1–U4.
Prompts 01–04 are written once the user has reviewed the plan (README §2.1).

The campaign adopts `datastorekit` `v0.2.2` and `raypoolkit` `v0.2.0`, in that order. A
fingerprint rehearsal comes before the first swap (R0) and after each swap (R1, R2). Each
package's gate G2 is recorded here when it holds.

**Campaign:** [`README.md`](README.md) ·
**Briefs:** [`sgk-datastorekit-planning-brief.md`](sgk-datastorekit-planning-brief.md),
[`sgk-raypoolkit-planning-brief.md`](sgk-raypoolkit-planning-brief.md) ·
**Code:** `RunRegistry/__init__.py`, `requirements.txt`, `Datastore/`, `RayTools/`, and every layer
import outside `prompts/` ·
**Verification:** `docs/dk-and-rpk-migration-verification.md` (created by 02) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md), a section opened with this board's
first issue

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-11 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the order | DatastoreKit first, then RayPoolKit, each followed by its own rehearsal |
| **U2** — the rehearsal | prompt 07's full protocol (README §4.1) at R0, R1 and R2 |
| **U3** — the factories | stay in `Datastore/SQL/ObjectFactories/`; `Datastore/__init__.py` and `Datastore/SQL/__init__.py` become empty files |
| **U4** — provenance | each run's manifest records the installed packages, from prompt 01 |

The orchestrator's choices of method, C1–C14, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | Record the installed packages in each manifest | **U4** | Sonnet | ⬜ | ⬜ | — | — |
| 02 | Rehearse the start tree (R0) | §0.1's gap `53d4e18` → `b510bc9` | Opus | ⬜ | ⬜ | — | — |
| 03 | Pin `datastorekit` `v0.2.2` | the DatastoreKit pin | Sonnet | ⬜ | ⬜ | — | — |
| 04 | Adopt `datastorekit` | the DatastoreKit swap; **U3** | Opus | ⬜ | ⬜ | — | — |
| 05 | Rehearse through `datastorekit` (R1) | **DatastoreKit G2** | Opus | ⬜ | ⬜ | — | — |
| 06 | Pin `raypoolkit` `v0.2.0` | the RayPoolKit pin | Sonnet | ⬜ | ⬜ | — | — |
| 07 | Adopt `raypoolkit` | the RayPoolKit swap | Sonnet | ⬜ | ⬜ | — | — |
| 08 | Rehearse through both packages (R2) | **RayPoolKit G2**; `main.py:3247` | Opus | ⬜ | ⬜ | — | — |
| 09 | Close the migration | the inherited issues; the close | Sonnet | ⬜ | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Held, by design (README §2.1).**
- **01–04** are written after the user's review of the plan.
- **05–08** are written after 04's review, whose log fixes the counts and names they cite.
- **09** is written after 08's review.

No prompt waits on another's *scope*. Each charter is fixed in README §2.2.

The measurement the plan was written against is [`measure/`](measure/): both packages' scripts at
`b510bc9`, each equal to its checklist's appendix byte for byte; the six roots' counts; the
Datastore root by module; and the venv's `pip freeze` (README §0.2).

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ⬜ |
| DatastoreKit G2 | the 17 files are gone (two emptied, U3), the pin is in place and installed, the suites pass, and R1 reproduces `39809dca…` | 04, 05 | ⬜ |
| RayPoolKit G2 | `RayTools/` is gone, 0 imports, the pin is installed, the six roots pass, all 54 sites construct, and R2 reproduces `39809dca…` | 07, 08 | ⬜ |

Recording each gate on its package's own extraction board is a commit in that repository, not
here (README §6.2 C12).

## 3. Active and unresolved issues

None yet.

## 4. Resolved issues

None yet.
