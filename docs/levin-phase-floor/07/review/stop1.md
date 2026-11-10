# 07 Part B — planned stop 1, the orchestrator's checks (Mac Studio, 2026-11-08, ~00:16–00:50 UTC)

The rehearsal agent stopped at planned stop 1 with the lock held; S1, S2's capture and the readers
not run. It asked one question (how to obtain located points' kind and z), relayed verbatim to the
user. Checks, each by the orchestrator's own scripts in `review/scripts/`, outputs in
`review/stop1/`:

1. **Fingerprints.** R27 `levin-phase-floor-07-r27-parent-20261108T000634` and R28
   `levin-phase-floor-07-r28-guard-20261108T001916`: both `status.json` `done`, exit 0, digest
   `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, `problems {}`, taken at
   `91472dbe` and `74600350` with `git_dirty` false. Re-taken by `runregistrykit store fingerprint`
   without `--write`, from the code worktree: both `1b332ffa…`, problems none
   (`fingerprint_r27.txt`, `fingerprint_r28.txt`); store files unchanged by it. **Pass.**
2. **S0, R27 against R26.** `qsi_full_compare.py` (`34eded4d…`, 06m's orchestrator script, copied):
   864 rows, 143,831 fields, **0 differing** (`qsi_full_r26_r27.txt`). `vdiff_values.py`
   (`abdd5501…`) on the agent's JSON (`458bbce2…`): every value table identical, no row in one store
   only; 6,012 label differences, all job-name prefix and stamp only (`vdiff_r26_r27.txt`). 05's
   `classify_non_qsi.py` (`26690e61…`, committed 05 copy) on the agent's normalised copy: 0
   unclassified, exit 0 (`classify_r26_r27_normalised.txt`). **Pass: S0 holds.**
3. **R28 against R27 outside `QuadSourceIntegral`.** `vdiff_values.py` on the agent's JSON
   (`927d4e70…`): every value table identical, no row in one store only (`vdiff_r27_r28.txt`). 05's
   classifier on it: 0 unclassified, exit 0 (`classify_r27_r28.txt`). **Pass.**
4. **Inside `QuadSourceIntegral`.** `qsi_guard_compare.py` (`907fafae…`, new; imports
   `qsi_full_compare.py`'s reader): count layout correct in every Levin record of R28 (3,105 group,
   10,072 analytic), 0 problems. **640 untouched rows, 640 bit-identical** in every field but the
   counts. 224 other rows: 70 group records (44 `G+q+r` × 1 point, 26 `q-r` × 4,097) and 400
   analytic blocks with a nonzero count; `guarded_regions` and `guard_at_depth_max` 0 everywhere;
   `total_converged` 17 gained, 0 lost; largest $|\Delta{\rm total}|$/`total_abserr` 7.38e4 (R27),
   6.42e4 (R28) at $(10^6)^3$, $z$ 2284.25 (`qsi_guard_r27_r28.txt`, CSV of the 224). Every figure
   compared agrees with the agent's tabulation. **Pass.**
5. **The agent's finding 1 re-checked** (`qr_degenerate.txt`): 26 `q-r` records with $q = r$ at
   4,097 points; $|v_{28} - v_{27}| > {\rm abserr}_{28}$ in 23 of 26, largest 3,604× at
   $(10^5, 10^6, 10^6)$, $z$ 6.985e9; all 26 `phase_limited` in R28; $|v_{28}-v_{27}|$ at most
   0.0149 of ${\rm abserr}_{27}$. **Confirmed as the agent states it.**

No §8 stop condition is met. Relayed to the user: the agent's question, its two findings, and these
checks.

## The user's word at stop 1 (relayed by the laptop session, 2026-11-08)

Verbatim: "I approve the replay. Also, add a reference to check 2 or 3 of the `q-r` calls, to settle
whether R28 is honest. Hold Part B before the census."

The laptop's account of its terms, passed to the rehearsal agent verbatim with the word:
(1) the replay as the agent proposed (06g's `references/v1/scripts/` harness copied into
`rehearsal/scripts/`, code worktree, Ray-free, stores read-only, only the 470 calls with a nonzero
count, each checked bit for bit against R28 before use), under the held lock alongside S2's
capture, not a registered run, command and output under `rehearsal/out/`; for each of the 26 `q-r`
calls with $q = r$ report the count, that every scan sample was an exact zero, and the $z$ range;
(2) references for 2 or 3 of those 26 including the worst, (1e5, 1e6, 1e6), $z$ 6.985e9, each giving
R27's and R28's value and abserr against the reference, method chosen by the agent, recorded as a
deviation, its own error stated; (3) then S1, S2, the readers and the lock's release as written;
(4) hold before the census: the census agent is not dispatched; report and wait.
