# 07 Part B §B.7 — the orchestrator's own read of R29 (2026-11-08 ~12:45 UTC)

R29 `levin-phase-floor-07-r29-guard-rule-a-20261108T123429`: done, exit 0, `1b332ffa…`, problems {},
taken at e182bc7a, git_dirty false; 12:34:28Z–12:44:03Z (about 9.6 min; R27 about 9, R28 about 16.5).
Store hashes before my reads: `r29_store_before.sha256`.

`qsi_rulea_compare.py` (9aeaeef4…, self-tested with R28 as R29: 0 changed of 13,177), exit 1
(`qsi_rulea_r27_r28_r29.txt`, rows in `qsi_rulea_r29_rows.csv`):
- 864 rows, keys equal; 13,177 Levin records, 13,087 bit-identical to R28 (counts included); 90
  changed: the 26 `q-r` (q = r, 4,097 at R28) and 64 analytic `2pt5` J4/Y4 blocks (1 point at R28).
  The 44 `G+q+r` records keep their located point, unchanged from R28.
- 86 of 90 changed records equal R27 on every leaf R27 has, with R29's four counts all 0.
- **4 do not** (the §B.7 rule fails): the `2pt5` J4 and Y4 blocks at $(3.162\times10^5)^3$,
  $z$ 1.329e9, and at $(10^6)^3$, $z$ 4.018e9. In each, R28 located 1 point; R29 locates none
  (Rule A skipped the phase) but `guarded_regions` is 1: the nodal guard caught the sign change, so
  the block differs from R27 in abserr, regions and (one of each pair) phase_limited
  (`four_records.txt`). abserr29/abserr27: J4 19.4 and 10.6; Y4 0.071 and 0.035. The blocks store
  no value leaf; the rows' `analytic_rad` differs from R27's by 7 and 3 ulp (≈1.5e-6 and ≈7e-8 of
  `metadata.analytic.abserr`); the rows' `total` equals R27's and R28's.
- No row differs outside its Levin records without a changed record. 54 rows changed; all 54 have
  `total` equal to R27's; 52 equal R27 in every field but the counts, the other 2 differ only in
  the four blocks above, `analytic_rad` and `metadata.analytic.abserr`
  (`changed_rows_outside_fields.txt`). `total_converged` 862 of 864, as R28.
- Summed `WKB_Levin_num_regions`: R27 114,826; R28 227,883; R29 114,460.
- Records located in R29: 44 group + 336 analytic = 380 (not about 444: Rule A also skipped 64
  analytic blocks); records with any nonzero count 384 (the 4 guarded).

This is O9's case (§B.7, `[07-c2-gate-threshold-was-never-measured]`): a phase Rule A skips with a
root, caught by the guard. §B.7 item 2's expectation ("every such record equals R27's") does not
hold for these 4; a failed check, reported, not adjudicated.

## The agent's stop (relayed to the laptop, ~12:55 UTC)

The rehearsal agent stopped on the same 4 records, lock held, replay/S1/S2/readers not run. Its
figures agree with mine in every number compared: 13,087 identical, 90 changed (26 `q-r` + 64
analytic, of which 60 equal R27 and 4 do not), 54 rows changed, all with `total` equal to R27's;
R29 against R27: 692 untouched rows all bit-identical, 172 tabulated, 17 gained / 0 lost (R28's
17); regions 114,460; outside QSI identical, 0 unclassified. Its question, relayed verbatim:
"Should I go on to the replay, S1/S2 and the readers with these 4 records reported as findings, or
hold?"

## The user's word on the agent's stop (relayed by the laptop, 2026-11-08)

Verbatim: "Yes, please proceed as you recommend." The laptop's recommendation, as relayed: go on,
with the 4 blocks reported as findings, not a stop (the "equals R27" rule was the laptop
orchestrator's error, to be corrected at the records commit; the 64 skipped analytic blocks are
the addendum's oversight); a new check on the 4 blocks, Ray-free under the held lock before the
readers: replay at R27's, R28's and R29's trees each checked against its own store, a reference
per block with its stated error, and per block each store's value, abserr and |value − ref|/abserr,
with V and the gate's span estimate on the root leaf; then the replay, S1, S2, the readers and the
release; stop 1′ continues. Passed to the agent verbatim.

## Planned stop 1′ — the orchestrator's checks after the agent's report (~13:30 UTC)

1. **Lock and runs.** `status` free, `ray-processes` none, `~/.raypoolkit` empty; lock log 8
   commands, all exit 0, ending free. Registry: nothing running or stale.
2. **Hashes.** R29's 5 files equal `r29_store_before.sha256` (taken before any comparison); R27 and
   R28 10/10; protected 181/181. Nothing under `var/`, `docs/`, `prompts/` newer than
   `dispatch_marker_b7` outside `rehearsal/r29/`, `review/`, R29's store and run; the three
   worktrees' `status --short --ignored` empty; the main checkout's `status --short` empty and no
   ignored file newer than the marker.
3. **R29 against R28 and R27** by `qsi_rulea_compare.py`: as recorded above; the agent's figures
   agree. The 4 guarded blocks are findings by the user's word, not a stop.
4. **Four blocks** (`block_ratios_recomputed.txt`): ratios re-derived from the agent's `float.hex`
   values agree; every store honest on every block (largest 6.2e-4, R28 (1e6)³ Y4); the stated
   error is 1.99e3–5.2e5 times below every abserr. V 15.113 and 15.896 < SIX_PI 18.850 < the gate's
   root-leaf estimates 19.160 and 20.196.
5. **Replay** (`replay_s1_s2_recheck.txt`): 224 units done, each equal to one R29 row (total,
   total_abserr, analytic_rad, converged, regions), 224 distinct; the 470 records' counts equal
   R29's; R29's 384 records with a nonzero count and the 90 changed records all among them.
6. **S1 on R29**: each R29 record matches the agent's stored value and abserr with 1 point; refs
   06g's; ratios agree (largest 0.2176, w11); every call equals R28's. **S1 holds.**
7. **S2 on R29**: ratios 0.02025 and 0.2632, as the agent's; store-consistent. **S2 holds.**
8. **Readers**: 6 exit 0, 6 store-identical, 0 Tracebacks, 6 shard-count warnings (06m's).
