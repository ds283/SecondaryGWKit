### §7a — Part C §5: rehearsal R30 against R29 (Mac Studio, 2026-11-12)

Records: [`07a/rehearsal/`](07a/rehearsal/). Log: [`07a/rehearsal/log.md`](07a/rehearsal/log.md).

**The tree.** `SecondaryGWKit-lpf07a-4bf50d3`, detached at Part C's dispatch SHA `4bf50d34`. Its code is the code commit
`1ec734d7`, whose parent is `dadd63a0`. The main checkout stayed at `28ebaf07`, and nothing was written in either tree.

**R30.** Run `levin-phase-floor-07a-r30-07a-20261112T113806`, on host mps026298.
- Launched 11:38:06Z: driver pid 75161, `sh` pid 75153, Ray address `local`, `/tmp/ray-sgk`, 6 cpus.
- Finished `done` with exit 0 at 11:47:48Z, about 9.7 min.
- Fingerprint `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, taken at 4bf50d34 with `git_dirty`
  false. Its `classes` equal R29's.
- Store primary `e5f8b443…`; see [`KEPT_ON_MAC_STUDIO.txt`](07a/rehearsal/KEPT_ON_MAC_STUDIO.txt).
- Records: [`out/r30_end_check.txt`](07a/rehearsal/out/r30_end_check.txt) and
  [`out/r30_status.json`](07a/rehearsal/out/r30_status.json).

| Check (prompt 07a §5) | Measured | Verdict |
|---|---|---|
| S0′: `git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'` lists nothing the build imports ([`out/00-gate.txt`](07a/rehearsal/out/00-gate.txt)) | `CLAUDE.md` only | **holds** |
| R30 gives `1b332ffa…` | `1b332ffa…`; `classes` equal R29's | **holds** |
| Outside `QuadSourceIntegral`: every value table identical; every other difference one of the four volatile kinds; no row in one store only ([`out/classify_non_qsi_r29_r30_labels_normalised.txt`](07a/rehearsal/out/classify_non_qsi_r29_r30_labels_normalised.txt)) | 360,462 value-table rows identical; 0 unclassified; differences are timing, `label` (job-name prefix normalised in a copy), `parents_digest` 291 and `rho_reuse` 19 | **holds** |
| Inside it, every moved row tabulated ([`out/qsi_tabulation_r29_r30.txt`](07a/rehearsal/out/qsi_tabulation_r29_r30.txt), [`out/qsi_rows_r29_r30.csv`](07a/rehearsal/out/qsi_rows_r29_r30.csv), [`out/qsi_records_r29_r30.csv`](07a/rehearsal/out/qsi_records_r29_r30.csv), [`out/qsi_summary_r29_r30.txt`](07a/rehearsal/out/qsi_summary_r29_r30.txt)) | 380 of 864 rows moved: `total` in 25, `total_abserr` in 60, `analytic_rad` in 168. Largest \|Δtotal\| is 0.0470 of R29's abserr and 0.0745 of R30's. `total_converged` changes: 0 each way | tabulated |
| S3: \|t30 − t29\| ≤ e29 + e30 on every row with finite abserr in both, evaluated exactly ([`out/s3_scores.txt`](07a/rehearsal/out/s3_scores.txt)) | 864 scored; 0 violations; largest 0.0288 | **holds** |
| S1: V1's twelve locate z ≈ 10211 and are honest against 06g's references ([`out/s1_scores.txt`](07a/rehearsal/out/s1_scores.txt)) | 12 of 12 locate a root at z 10211.31789, within 8.5e-6 (spacing ≥ 31); 12 of 12 honest, largest 0.2176 (w11). Equal to R29 bit for bit | **holds** |
| S2: P1's two rows locate within one spacing and are honest against `p1_references` ([`out/s2_scores.txt`](07a/rehearsal/out/s2_scores.txt)) | \|x − x_s\| 7.1e-9 and 5.6e-9 against 4.1e-3; ratios 0.0202 and 0.263. Equal to R28 and R29 | **holds** |
| S4: R29's four guarded 2pt5 J4/Y4 blocks each locate their root, guard no leaf, and are honest against R29's block references ([`out/block_scores.txt`](07a/rehearsal/out/block_scores.txt)) | 4 of 4 locate one root at the reference root, bit for bit; 0 guarded; ratios 4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4; reference errors ≤ 5.0e-4 of abserr30. Equal to R28's values bit for bit | **holds** |
| The readers on R30: exit 0, store byte-identical ([`out/readers_check.txt`](07a/rehearsal/out/readers_check.txt)) | 6 of 6 exit 0; R30 identical before and after | **holds** |
| The lock released; `ray-processes` lists nothing ([`state/lock.log`](07a/rehearsal/state/lock.log)) | released; free; nothing listed | **holds** |
| R29 and R30 kept; R29 unchanged ([`state/stores_sha256_final.txt`](07a/rehearsal/state/stores_sha256_final.txt)) | R29 equal to 07's `r29/KEPT_ON_MAC_STUDIO.txt`; R30 equal to its post-build hashes | **holds** |

**The replay behind S1, S2 and S4** ([`out/replay_check.txt`](07a/rehearsal/out/replay_check.txt),
[`out/blockcheck.txt`](07a/rehearsal/out/blockcheck.txt)).
- R29's capture harness was run at h, Ray-free and under the lock.
- The rows: 223 in all, covering 202 located rows, P1, the w rows, the block rows, and the rows of the 49 records whose
  abserr moved with no change to value, regions or location.
- **493 of 493 records reproduce R30 bit for bit.** Every one of the 223 workers had Ray uninitialised.
- Each located record holds exactly one root: 44 group (G+q+r), 200 J4 and 200 Y4 at 2pt5. None is guarded.
- The 4 S4 block calls reproduce R30 in pass 1 and pass 2.
- Their captures are not byte-identical to R29's. They differ only in the unit id, the tree tag and the call's own
  result ([`out/capture_compare.txt`](07a/rehearsal/out/capture_compare.txt)). Their integrands equal R29's bit for bit
  at 1001 points, with equal `x_span` and `kw`.
- The trust reading of `issues.md` item 1 is in [`out/trust_reading.txt`](07a/rehearsal/out/trust_reading.txt).

**Float values in hex** (from [`out/block_scores.txt`](07a/rehearsal/out/block_scores.txt)):

| Block | value30 | abserr30 | reference | regions |
|---|---|---|---|---|
| (3.162e5)³, z 1.329e9, J4 | `-0x1.9017c8e4681f6p-30` | `0x1.65dacd1543ce8p-66` | `-0x1.9017c8e4681f2p-30` | 14 |
| (3.162e5)³, z 1.329e9, Y4 | `0x1.2018ed31b447ep-27` | `0x1.78af331f037f8p-66` | `0x1.2018ed31b4477p-27` | 4 |
| (1e6)³, z 4.018e9, J4 | `-0x1.e6f859d2f7c72p-33` | `0x1.ff3cae87aca98p-69` | `-0x1.e6f859d2f7c57p-33` | 6 |
| (1e6)³, z 4.018e9, Y4 | `0x1.9714e016a7982p-30` | `0x1.faa21d2682f6bp-69` | `0x1.9714e016a7978p-30` | 13 |
