# Review — `handover` prompt 12, Part B, rehearsal R8 (Mac Studio)

Orchestrator review against `prompts/handover/orchestrator/prompt-12.md` §B.3, on mps026298,
2026-10-25, at the dispatch SHA `8a02f0c` (B1's code `049c7c8`; the commits between touch only
markdown, four files).

**Verdict: PASS, 7 of 7, check 3 on the user's ruling** (see "User's rulings (2026-10-25)" at the
end). As first written, the verdict was "6 of 7 checks hold; check 3 does not hold as written",
with this explanation. The fingerprint differs from
R3's in `QuadSource` only. `QuadSourceIntegral` has R3's digest, so the check's expected pair is
not what R8 shows. The half of the check that guards against a changed producer holds: no class
outside the two differs, so prompt §8's stop condition did not fire. The cause is the fingerprint's
design, read in `ObjectFactories/QuadSourceIntegral.py:965`: the class's `inventory_spec()` records
identity and tags only ("No validated flag and no value table"), so B1's change to the integrals'
values and `partition` cannot reach the digest. **Whether a `QuadSource`-only difference meets
§B.3 check 3 and prompt §6.4 / §7.3 is the user's ruling.** It is not repaired here.

## Gate

Re-read before dispatch, matching §B.5:
- `HEAD` `8a02f0c` and `git status --short` empty; Python 3.13.16.
- 11 runs, all finished; the lock was free, with `~/.raypoolkit/` empty and no
  `~/ScienceData/.rehearsal-lock`.
- No `handover-12` store and no `var/handover-12/`.
- R3's two recorded runs present at `var/handover-06/recorded/`.
- `~/.ray/auth_token` mode `-rw-------`; 117 GiB free.

The one difference is load: 6.2 at the Gate against "about 4.5" in §B.5. That is a reading, not
state, and the agent read 4.6 four minutes later. `d5f5a2b` and `049c7c8` are ancestors of `HEAD`.

## The checks

1. **Scope — holds.**
   - `HEAD` is `8a02f0c` and `git status --short` is empty.
   - `find var -newer <marker>`, excluding `var/handover-12/`, lists only:
     - the five R8 store files;
     - the run directory `var/runs/handover-12-r8-reference-20261025T210551/` (`manifest.json`,
       `status.json`, `stdout.log`, `stderr.log`);
     - `var/runs/handover-12-reference.launch.log`.
   - No sidecar was written; `--build` writes none.
   - A3 v2's six files (`handover-A3-v2-lambdacdm*`, the sidecar included) have identical SHA-256
     before dispatch and after the review.
   - The agent's one write outside the allowed paths, `/tmp/x` (its deviation 6), no longer exists.
2. **The run — holds.**
   - `status.json`: `done`, `exit_code` 0, pid 21718. The launch log ends
     `** driver exit status 0`.
   - The manifest:
     - `campaign` `handover`, `prompt` `12`, `--register r8-reference`;
     - `run_id` `handover-12-r8-reference-20261025T210551`;
     - `git_head` `8a02f0c421b1…`, `git_dirty` false, host Python 3.13.16;
     - launcher `RunDrivers/quadsource_atol_sweep.py` at SHA-256 `5b984c75…`.
   - The argv is §B.1's launch verbatim. The `--purpose` names R8, the tree and B1's SHA, and the
     expected difference, and names no temp dir.
   - The manifest names the dispatch SHA `8a02f0c` rather than the landed `049c7c8`, as the
     dispatch set out.
3. **The fingerprint — does not hold as written.**
   - `runregistrykit store fingerprint` without `--write`, run by me: digest
     **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`**, format 6, no
     problems. It equals `status.json`'s. This is R8's digest.
   - `compare_with_recorded.py` against R3's
     `rrk-v020-adoption-03-reference-20261018T232838`, run by me: exit 1, `verdict: DIFFERENT`,
     **one difference**:
     `QuadSource: tag set […] differs: 6 recorded, 6 now`.
   - All 20 other classes have R3's digest, **`QuadSourceIntegral` among them** (`8913eda5ce7b`
     in both). See the verdict above.
4. **The rows — holds.** By my own scratch reader, every file opened `mode=ro`:
   - All **864 of 864** `QuadSourceIntegral` rows have `partition.handover` with `Tq` and `Tr`,
     each with three `handover_continuity` entries.
   - No row's metadata contains `clamp_gaps_log1pz` or `max_clamp_gap_log1pz`.
   - **`QuadSource` value counts:**

     | | node counts of the 6 rows |
     |---|---|
     | R8, from the store | 371, 371, 371, 428, 428, 478 |
     | R4's extract, R3-equal | 368, 368, 368, 425, 425, 475 |

     Each row gained exactly 3 nodes.
   - On the three triples whose extract file names match between R4 and R8, the comparison is the
     same each time:
     - 3 nodes added and none removed;
     - the top (z = 6.8788e13) unchanged;
     - the added nodes are the switch node and the two overlap nodes below it (for example
       1.7126e9, 1.6736e9, 1.6355e9 for k = 1e5).
   - The other three triples' extract files are named by r-serial, which differs between the
     stores, so they were compared by count only.
   - That R3's values are unchanged at R3's nodes is the agent's measurement (verification.md),
     not re-derived here.
5. **The readers — holds.**
   - `out/readers.txt`: all six readers exit 0, and so does the script.
   - The store's five SHA-256 values are identical in the agent's before-reads and after-reads
     records.
   - My own hash, taken before my fingerprint and reader, equals both, and is unchanged after them.
6. **The lock — holds.** The transcript `state/lock.log`:
   - `acquire`: exit 0
   - `set-pid 21718`: exit 0. 21718 is the driver's pid, by `status.json` and by the agent's
     `pgrep -P 21715`.
   - `ray-processes`: empty, exit 0
   - `release`: exit 0
   - A mistyped `status --project …` exited 2. It is deviation 5, harmless.

   Run by me: `ray-processes --ray-temp-dir /tmp/ray-sgk` shows none, exit 0, and `status` says
   free, exit 0. The registry has 12 runs, all finished.
7. **The records — hold.**
   - Present: `out/`, `scripts/`, `state/`, `log.md`, `verification.md`, `issues.md` and
     `extract/`; §B.4 says not to copy `extract/` back.
   - `log.md` classifies all seven deviations: two `STRUCTURALLY REQUIRED`, two
     `IMPLEMENTATION CHOICE`, three `UNINTENDED DRIFT`.
   - `issues.md` proposes one new issue,
     `[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]`, opened on no board.

## The roots (§6.1, the agent's)

Both equal the laptop's at Part A's review, with no FAIL or ERROR, and the flake did not occur:
- `ComputeTargets`: 742, OK.
- `ObjectFactories`: 280, OK.

## The continuity residuals (recorded, not judged)

From the agent's `r8_rows.py`, over the three k:

| node | \|dT_rel\| | \|dTprime_rel\| |
|---|---|---|
| switch | 1.7e-7 to 3.2e-7 | 4.2e-6 to 5.7e-6 |
| overlap 1 | 4.9e-6 to 5.5e-6 | 2.1e-5 to 2.2e-5 |
| overlap 2 | 1.5e-5 to 1.6e-5 | 3.0e-5 to 3.1e-5 |

Every `dT_rel` is negative and every `dTprime_rel` positive. One row's metadata, read by me, agrees
(Tq switch node `dT_rel` −3.15e-7).

## For the user

1. **Check 3.** Does a fingerprint difference in `QuadSource` alone meet §B.3 check 3 and prompt
   §6.4 / §7.3, given that `QuadSourceIntegral`'s inventory cannot see values or `partition`?
2. **The agent's question, verbatim:** "do you accept that a difference in `QuadSource` alone
   meets §6.4/§7.3? If not, everything after §6.4 was read-only, and only the lock release would
   need redoing (a re-run would acquire it again)."
3. **The agent carried on past the unexpected comparison** rather than stopping with the lock held
   (its deviation 7). No §8 condition fired, and every step after §6.4 was read-only, so nothing
   needs undoing either way.
4. **The proposed issue** `[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]`: whether
   to open it, and on which board.
5. **Whether R8's store is kept or retired** (§B.4).

## User's rulings (2026-10-25)

Relayed by the laptop session that dispatched this review, quoting the user's answers verbatim:

1. **Check 3: "Accept as met (Recommended)".** A fingerprint difference in `QuadSource` alone
   meets §B.3 check 3 and prompt §6.4 / §7.3, because `QuadSourceIntegral`'s inventory cannot see
   values or `partition`. Check 4's row read shows B1's change there. This also answers the agent's
   question: nothing is redone.
2. **The proposed issue: "Open on handover (Recommended)".** The laptop opens
   `[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]` on the `handover` board in the
   records commit.
3. **R8's store: "Retire after commit (Recommended)".** The laptop runs `store retire`, with a dry
   run first, after the records commit.
