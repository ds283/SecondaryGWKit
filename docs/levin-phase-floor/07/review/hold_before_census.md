# 07 Part B — the rehearsal agent done; Part B held before the census (Mac Studio, 2026-11-08 ~10:45–11:05 UTC)

The user's word at stop 1 (recorded in `stop1.md`) approved the replay, added references for 2–3 of
the `q-r` calls with $q = r$, and held Part B before the census. The census agent is not dispatched.

The rehearsal agent reported done: S0, S1, S2 hold; no §8 stop; lock released; records in
`rehearsal/` (`log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`), 15 deviations
(3 unintended drift: 2, 5, 11). The orchestrator's checks, each by its own read-only reads:

1. **Lock and runs.** `status`: free; `ray-processes`: none; `~/.raypoolkit` empty. Registry: no run
   `running` or stale; `levin-phase-floor-07-p1-references-20261108T103132` `done`, exit 0, 2/2.
2. **Hashes.** `checks/protected_before.sha256`: 181/181 OK. `stop1/stores_before_fp.sha256` (R27,
   R28, 10 files): 10/10 OK. No file under `var/`, `docs/` or `prompts/` newer than
   `dispatch_marker` outside `var/levin-phase-floor-07/{rehearsal,review}/`, the R27/R28 stores and
   07's run directories and launch logs. All three trees `status --short` empty.
3. **Replay against R28, by `qsi_full_compare.read` on R28.** 224 units, all `done`; every unit's
   `total`, `total_abserr`, `analytic_rad`, `total_converged` and `num_regions` equal one R28 row's
   bit for bit, 224 distinct rows. `out/replay_calls.csv`'s 470 records are exactly R28's records with
   a nonzero count (key sets equal), and all four counts equal on all 470 (444 × 1 point,
   26 × 4,097).
4. **S1 re-scored** from R28's stored group record and 06g's `scores.json` `V1`: each of the 12
   `store_value28`/`store_abserr28` equals R28's `G+q+r` record bit for bit with 1 point; the
   call-form/stored-form factor is consistent between value and abserr (≤1.1e-16) and with 06g's
   R24 factor (≤1.4e-16); `ref_hex` and key equal 06g's; every ratio agrees with the agent's (largest
   0.2176, w11); all converged. **S1 holds.**
5. **S2 re-scored:** the reference is the FCC r2 reading; scatter 6.11e-37 and 1.19e-36 against
   tolerances 6.59e-33 and 9.18e-33; ratios R28 0.02025 and 0.2632, R27 5.611e5 and 4.075e4,
   equal to the agent's; each store's call-form value consistent with its `G+q+r` record (factor
   agreement ≤1.6e-16 and between stores ≤1.2e-16). **S2 holds.**
6. **`q-r` references:** ratios recomputed from the `float.hex` values agree with the agent's. On the
   worst call R28's abserr (≈7.9e-37) is 0.173 × the reference's stated error (4.56e-36, set by
   |QAGS − GL14|; QAGS's own abserr 9.29e-37), so R28's honest verdict there rests on the GL
   sequence (m = 8–14 within 5.9e-39, GL12 = GL14), not on the stated error; the agent states this
   in `verification.md`. On the other two the stated error is below R28's abserr (6.5×, 45.5×).
7. **Readers:** 6 exit 0, 6 store-identical, 0 Tracebacks, 6 shard-count warnings (06m's). Lock log:
   10 commands, each exit 0, ending free.

What is not yet checked: §B.3's full review (it follows Part B's end); the agent's `log.md` and
`verification.md` read in full.

## The user's decision on the hold (relayed by the laptop session, 2026-11-08)

Verbatim: "Yes, rule A alone, please. Open a separate issue for the `6pi` gate. We'll deal with this
afterwards, as you say. Record the resonance issues as belonging to the one-loop build and point at
the planning document. Please make the fix as an amendment of 07."

The laptop's account: the zero-rate case is fixed before the census as an amendment of 07 (rule A:
location, and the nodal guard, skip a phase whose total variation over the call's span, on the
scan, is below `SIX_PI`), made and reviewed on the laptop. The Mac Studio keeps holding: no census
dispatch, the stores, worktrees and rehearsal records untouched, until a new dispatch SHA comes.
Then Part B resumes from a third worktree at that SHA: R27 stands; R28 is rebuilt under a new slug
with the current R28 kept; the rebuild is compared with R28 and R27; S1 and S2 re-scored; the
readers; then the census. Details come with the SHA.
