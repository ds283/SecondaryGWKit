# Orchestrator — prompt 03, pin `datastorekit` at `v0.4.0`

Read [`../README.md`](../README.md) first: §0.1, §2.2 (03's charter), §4.2, §5 (rules 7, 9, 11–14)
and §8. Then the board's §2 (E1), §3 (`[00-recorded-fingerprints-lack-the-encoding-format]`) and
§5.5. The conventions are [`prompt-01.md`](prompt-01.md)'s §0 last list, carried over; what differs
is said here.

**You do not write code.** You may:
- run the suites, `pip` read-only commands, `python -m RunRegistry list`, `store show` and
  `store fingerprint` without `--write` or `--listing`;
- run a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`03-pin-datastorekit-at-v0.4.0.md`](../03-pin-datastorekit-at-v0.4.0.md)
**Board items:** K1 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing expected ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- E1 holds and 03–04 are written (`fd65055`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (outside `prompts/`
  and `docs/`).
- Nothing is `running` in the registry; the rehearsal lock is free; `ray-processes` lists nothing.
- One prompt at a time in this checkout (README §4.2).

## 0. What makes this prompt unusual

**It changes the venv, which no commit records.** The commit is one line of `requirements.txt` and
the records. The install happens before the commit, from the working tree, and is not undone by
`git revert` (README §4.2). So the review checks the venv directly, not only the diff.

**It is the last prompt before SGK's fingerprint has a second implementation.** 03 does not touch
the package's new module beyond one import to read `ENCODING_FORMAT`. Its §2.4 reads R0's two
stores through SGK's own code at the new pin: the cheap half of R1's fingerprint acceptance, taken
before the rehearsal so that a miss would point at the pin and not at a build.

**What moves on purpose:**
- `requirements.txt:17`, `v0.2.2` → `v0.4.0`. No other line.
- The venv: `datastorekit` 0.2.2 (`6d8ee9ae…`) → 0.4.0 (`c8570e84…`). Nothing else.
- `measure/out/03-pip.txt`, `03-fingerprint.txt`, `03-roots.txt`, `03-addenda.txt`.
- `logs/03-pin-datastorekit-at-v0.4.0.md`.
- The board: header, §1 row 03, K1, `### 5.6 After prompt 03`.
- No issue opens or closes; the index stays at 107.

**The facts, checked by the orchestrator on 2026-10-14** at `a5a6f6d` (E1, board §5.5) and at
`fd65055` (the rest). Pass them on.
- **The release.** Tag `v0.4.0`, object `8e7c1d77b3792b57b98983e84b99fc68ab88cb89`, commit
  `c8570e849334d09085bbd84876095fc9e22bd005`; CI run 38538726292 green on that commit. Under
  `datastorekit/`, `v0.2.2` → `v0.4.0` changes `SQL/factory_base.py`, `object.py` and adds
  `store_fingerprint.py`; `pyproject.toml` changes only its version.
- **The dry run.** With line 17 edited in a scratch copy of `requirements.txt`, `pip install
  --dry-run -r` printed `Would install datastorekit-0.4.0` and nothing else.
- **The venv now.** `datastorekit` 0.2.2, `package_provenance()` `{"commit":
  "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false, "requested_revision": "v0.2.2",
  …}`; `raypoolkit` 0.2.0 at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, not editable; no
  `runregistrykit`.
- **The registry.** Nothing `running` or stale. Sixteen sidecars: fourteen tombstones and R0's two
  live stores, `var/datastores/runregistry-extraction-02-{reference,resumed}.sqlite`.
- **The roots**, at `fd65055`, serially, load 6–14 on 10 cores:

  | Root | Ran | Result | Wall |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | OK | 1 s |
  | `CosmologyModels/tests` | 39 | OK | 3 s |
  | `ObjectFactories/tests` | 271 | OK | 14 s |
  | `tools/tests` | 91 | OK | 11 s |
  | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 25 s |
  | `RunRegistry/tests` | 286 | OK | 73 s |
  | `ComputeTargets/tests` | 693 | 1 failure | 218 s |

  The failure is README §8's `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.106 s
  against 0.06 s). Alone it failed twice more (0.103 s, 0.114 s). It fails alone at this machine's
  load, as at 02a's review; no code it runs has changed. The agent's runs will most likely show it
  too; that is not a stop.
- **The addendum's SGK claims checked:** `fingerprint_store` passes `taken` positionally
  (`RunRegistry/stores.py:1499-1507`), as does `test_store_fingerprint.py:378`. These are 06's.
- **Not touched, and staying:** every file under `RunRegistry/`, every test, every `docs/` script,
  the other pins, and every store and run directory under `var/`.

**What the review exists to establish.**
- **(P1) The pin.** One line moved, to the tag; the venv at the tag's commit, not editable; nothing
  else installed, upgraded or removed.
- **(P2) Nothing SGK runs changed.** The roots at 02a's counts; SGK's fingerprint of R0's stores
  `39809dca…`, the stores untouched.
- **(P3) The notes read.** Every item of the three addenda tabled, and the `v0.4.0` table's claims
  about SGK's code checked against `HEAD`.
- **(P4) The records** in the campaign's form.

**Conventions,** in addition to `prompt-01.md`'s:
- **The venv.** The agent runs exactly the two `pip install` commands of the prompt (`--dry-run`,
  then the real one) and no other `pip install` or `pip uninstall`. If the dry run shows anything
  but `datastorekit`, it stops before the real install.
- **`var/` is read, never written.** `store fingerprint` without `--write` and without
  `--listing` only reads; the agent hashes every store file around it.
- **Ray.** None. The lock is not taken.
- **The SHA.** "this commit" where 03's hash belongs; the orchestrator substitutes it.

## 1. Before you dispatch

Done at `fd65055` on 2026-10-14; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; clean; `git worktree list` shows this checkout alone.
2. **The registry.** Nothing `running` or stale.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free, no Ray process of this repository.
4. **The venv.** As §0.
5. **Baselines.** The seven roots, as §0's table.
6. **The `var/` marker**, for check 5: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README and the board, `HEAD`,
the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in
the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §6 list.
- **The two `pip install` commands, and no other venv change.**
- **The wall-clock flake** is expected, and is re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §4 conditions; if it stops after the install, say so.

## 3. The review — six checks

1. **Scope.** `git show --stat <03's commit>`: `requirements.txt`, the four `measure/out/03-*`
   files, the log and the board; `docs/OPEN_ISSUES.md` only with a new §3 entry.
   `git diff <03>~1 <03> -- requirements.txt` is the one line. Nothing under `RunRegistry/`,
   `docs/`, `README.md`, `prompts/INDEX.md` or `orchestrator/`.
2. **P1, the venv, measured here.** `pip show datastorekit`; `package_provenance()` gives 0.4.0,
   `c8570e84…`, `v0.4.0`, not editable, and `raypoolkit` unchanged; `pip check` clean;
   `pip install --dry-run -r requirements.txt` now installs nothing. The agent's `03-pip.txt` has the
   dry run before the install, and the freeze diff is the one line.
3. **P2, the fingerprint.** `03-fingerprint.txt`: both stores format 6, `39809dca…`, 21 classes,
   `recorded: matches`, exit 0, and the file hashes equal before and after. Re-run the two
   `store fingerprint` commands here, and hash the stores again: equal to the agent's.
4. **P2, the roots.** `03-roots.txt`: seven roots at §0's counts; any known flake re-run alone and
   both runs recorded. Re-run at the agent's commit, as §0's table.
5. **Nothing else moved.** `find var -newer <marker>` is empty (a read-only `store fingerprint`
   opens with `mode=ro`; if it shows a store file, compare hashes before calling it a change).
   `git status --short` clean.
6. **P3 and P4, the records.** `03-addenda.txt` covers every item of the three addenda, each with
   a disposition; the `v0.4.0` table's SGK claims are checked against `HEAD`. The board's header,
   §1 row, K1 and §5.6; the log's §5.1 headings and classified deviations; the commit message in
   `CLAUDE.md`'s form, ending `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 03* paragraph in §5.6, with
  residue fixed in a follow-up commit: "this commit" → the SHA; this note listed as "used for 03";
  `prompts/INDEX.md`'s row (4 of 10 landed).
- **Report to the user:** the pin and the venv; the roots; SGK's fingerprint at the pin; the
  addenda's items handed to 06.
- **Then 04**, R1. Its note is written after this review; it is the first rehearsal under the lock,
  and the one after which R0's and R1's stores are retired (D6).
