# The seven group records — my re-score of the rehearsal agent's "which store understates" (2026-11-16 ~20:50Z)

`scripts/understates_rescore.py` → `understates_rescore.txt`. For each of the agent's seven sids: the reference is
the readings file's top-level value (asserted equal to FCC r2), its stated error the four readings' scatter. v and e are the
call-level outputs of the agent's bit-checked replays (`refcap_replay_a07.json` for R30, `refcap_replay_h.json` for
R31). Each replay's `stored_value`/`stored_abserr`/regions/converged was asserted equal to the record read from the store
by my stop-1 reader. Exact arithmetic in Fractions, the agent's rule (understates if |v−ref| − e > s; honest if
e − |v−ref| > s).

A first run of the script compared store-unit values against the call-unit reference (ratios ~1e7, meaningless). It was
corrected to the call-level outputs as above and re-run; the output file is the corrected run's.

Result, equal to the agent's `understates.txt` in every ratio to the figures printed: R30 understates in all seven
(2.912, 59.29, 35.6, 9.202, 5.619, 3.867, 4.36), R31 honest in all seven (0.0045, 0.0055, 0.0070, 0.298, 0.0150,
0.0014, 0.0050); s/min(e) at most 5.88e-4. All 14 calls converged.
