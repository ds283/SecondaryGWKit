### §7c — Part C §5: rehearsal R32 against R31 (Mac Studio, 2026-11-19)

Records: [`07c/rehearsal/`](07c/rehearsal/). Log: [`07c/rehearsal/log.md`](07c/rehearsal/log.md).

**The tree.** `SecondaryGWKit-lpf07c-cc3368c`, detached at Part C's dispatch SHA `cc3368ce`. Its code is the code commit
`67b5af74`, variant (S), whose parent is `cff4237f`. The main checkout stayed at `28ebaf07`, and nothing was written in
either tree.

**R32.** Run `levin-phase-floor-07c-r32-07c-20261119T140417`, on host mps026298.
- Launched 14:04:16Z: driver pid 19463, `sh` pid 19451, Ray address `local`, `/tmp/ray-sgk`, 6 cpus.
- Finished `done`, exit 0: about 13.8 min, against R31's 13.6. `QuadSourceIntegral`'s stage took 11 m 19 s, against
  11 m 28.6 s.
- Fingerprint `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, taken at `cc3368ce` with `git_dirty`
  false.
  - Re-derived read-only by `runregistrykit store fingerprint`.
  - Its `classes` equal R31's.
- Store primary `4dd39837…`; see [`KEPT_ON_MAC_STUDIO.txt`](07c/rehearsal/KEPT_ON_MAC_STUDIO.txt).
- Records: [`out/r32_end_check.txt`](07c/rehearsal/out/r32_end_check.txt),
  [`out/r32_status.json`](07c/rehearsal/out/r32_status.json) and
  [`out/r32_fingerprint_own.txt`](07c/rehearsal/out/r32_fingerprint_own.txt).

| Check (prompt 07c §5) | Measured | Verdict |
|---|---|---|
| S0′: `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists nothing the build imports ([`out/s0prime.json`](07c/rehearsal/out/s0prime.json), [`out/00-gate.txt`](07c/rehearsal/out/00-gate.txt)) | `CLAUDE.md` only | **holds** |
| R32 gives `1b332ffa…` | `1b332ffa…` twice; `classes` equal R31's | **holds** |
| Outside `QuadSourceIntegral`: every value table identical; every other difference one of the four volatile kinds; no row in one store only ([`out/classify_non_qsi_r31_r32_labels_normalised.txt`](07c/rehearsal/out/classify_non_qsi_r31_r32_labels_normalised.txt)) | 360,462 value-table rows identical; 0 unclassified; the differences are timing, `label` (job-name prefix normalised in a copy), `parents_digest` 291 and `rho_reuse` 26 | **holds** |
| Inside it, every moved row tabulated, with the `total_converged` changes each way ([`out/qsi_tabulation_r31_r32.txt`](07c/rehearsal/out/qsi_tabulation_r31_r32.txt), [`out/qsi_rows_r31_r32.csv`](07c/rehearsal/out/qsi_rows_r31_r32.csv), [`out/qsi_records_r31_r32.csv`](07c/rehearsal/out/qsi_records_r31_r32.csv), [`out/qsi_summary_r31_r32.txt`](07c/rehearsal/out/qsi_summary_r31_r32.txt)) | 305 of 864 rows moved: `total` in 251, `total_abserr` in 252, `analytic_rad` in 106. Largest \|Δtotal\| is 1.05e-5 of either abserr. Regions 152,117 → 144,531; none rose. `total_converged`: 1 up, 0 down | tabulated |
| S3: \|t32 − t31\| ≤ e31 + e32 on every row with finite abserr in both, evaluated exactly ([`out/s3_scores.txt`](07c/rehearsal/out/s3_scores.txt)) | 864 scored; 0 violations; largest 5.25e-6; §9.5's named row 7.0e-8 | **holds** |
| §9.5's S3 item: R31's record G+q+r at (1e5, 1e6, 1e6), z 144627.7, converges at 86 regions, honest at 0.053 ([`out/s3_record.txt`](07c/rehearsal/out/s3_record.txt)) | converged, 86 regions, depth 16, equal to Part A's (S) record bit for bit; the integrand bit-identical at 1,001 samples to the capture Part A's `ref-R31-3-20363-g1` was read on; \|v − ref\|/abserr 0.05331484572900594 | **holds** |
| S1: V1's twelve locate z ≈ 10211 and are honest against 06g's references ([`out/s1_scores.txt`](07c/rehearsal/out/s1_scores.txt)) | 12 of 12 locate (within 8.52e-6; spacing ≥ 31.0); 12 of 12 honest, largest 0.0670 (w06); 8 equal R31; 4 moved (w08–w11, one chase stop each): ratios 0.000474137, 0.020798, 0.0132531, 0.00194838 against R31's 0.000473577, 0.0207975, 0.0132535, 0.00194974 | **holds** |
| S2: P1's two rows locate within one spacing and are honest against `p1_references` ([`out/s2_scores.txt`](07c/rehearsal/out/s2_scores.txt)) | \|x − x_s\| 7.1e-9 and 5.6e-9 against 4.12e-3; ratios 0.0202 (= R31) and 0.111 (R31 0.111; one chase stop) | **holds** |
| S4: R29's four guarded 2pt5 J4/Y4 blocks each locate their root, guard no leaf, and are honest against R29's block references ([`out/block_scores.txt`](07c/rehearsal/out/block_scores.txt)) | 4 of 4 locate one root at the reference root, bit for bit; 0 guarded; ratios 4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4; each equal to R31 and R28 bit for bit | **holds** |
| The readers on R32: exit 0, store byte-identical ([`out/readers_check.txt`](07c/rehearsal/out/readers_check.txt)) | 6 of 6 exit 0; R32 identical before and after | **holds** |
| The lock released; `ray-processes` lists nothing ([`state/lock.log`](07c/rehearsal/state/lock.log)) | released 14:49:20Z; free; nothing listed | **holds** |
| R31 and R32 kept; R29, R30 and R31 unchanged ([`state/stores_sha256_final.txt`](07c/rehearsal/state/stores_sha256_final.txt)) | R29, R30 and R31 equal to their before-hashes, and R31 to 07b's KEPT; R32 equal to its post-build hashes | **holds** |

**The replay behind S1, S2, S4, the S3 record and the chase counts** ([`out/cmd/r32-replay/`](07c/rehearsal/out/cmd/r32-replay/),
[`out/replay_calls.checks.json`](07c/rehearsal/out/replay_calls.checks.json),
[`out/blocks.json`](07c/rehearsal/out/blocks.json), [`out/refcap.json`](07c/rehearsal/out/refcap.json)).
- 07b's rehearsal capture harness was run at the tree, Ray-free, under the lock, as the registered run
  `levin-phase-floor-07c-r32-replay-20261119T143558` (8.5 min).
- 203 rows: the 202 located rows, P1, the w rows, the block rows, and the row of the one record of interest.
- **445 of 445 records reproduce R32 bit for bit.**
- **Over the 4,180 replayed calls:**
  - range-rule refusals: 567 in 359 of 932 group calls, and 6,517 in 2,587 of 3,248 analytic calls; the edge rule
    refuses nothing;
  - **chase stops:** 180 in 164 group calls, and 225 in 225 analytic calls;
  - no call has a chase stop without a rate-rule refusal.

**The two `phase_limited` flips** (descriptive, at the orchestrator's note;
[`out/phase_limited_flips.txt`](07c/rehearsal/out/phase_limited_flips.txt)).
- **The records:** (1e5, 3.162e5, 3.162e5), z 755.687, `subintervals[1].groups[1]` G+q−r and `groups[2]` G−q+r.
- **The change:** each went False → True, both converged throughout, regions 38 → 36.
- **How far:** \|Δv\| is 1.09e-7 of each abserr, and 5.47e-8 of their sum.
