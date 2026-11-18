### §7b — Part C §5: rehearsal R31 against R30 (Mac Studio, 2026-11-16)

Records: [`07b/rehearsal/`](07b/rehearsal/). Log: [`07b/rehearsal/log.md`](07b/rehearsal/log.md).

**The tree.** `SecondaryGWKit-lpf07b-0a103b5`, detached at Part C's dispatch SHA `0a103b5a`. Its code is the code commit
`5e19c4b6`, variant (h), whose parent is `7b82a0cd`. The main checkout stayed at `28ebaf07`, and nothing was written in
either tree.

**R31.** Run `levin-phase-floor-07b-r31-07b-20261116T191849`, on host mps026298.
- Launched 19:18:48Z: driver pid 10624, `sh` pid 10620, Ray address `local`, `/tmp/ray-sgk`, 6 cpus.
- Finished `done`, exit 0, at 19:32:27Z: about 13.6 min, against R30's 9.7. `QuadSourceIntegral`'s stage took 1.52×.
- Fingerprint `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, taken at `0a103b5a` with `git_dirty`
  false.
  - Re-derived read-only by `runregistrykit store fingerprint`.
  - Its `classes` equal R30's.
- Store primary `5981563b…`; see [`KEPT_ON_MAC_STUDIO.txt`](07b/rehearsal/KEPT_ON_MAC_STUDIO.txt).
- Records: [`out/r31_end_check.txt`](07b/rehearsal/out/r31_end_check.txt),
  [`out/r31_status.json`](07b/rehearsal/out/r31_status.json) and
  [`out/r31_fingerprint_own.txt`](07b/rehearsal/out/r31_fingerprint_own.txt).

| Check (prompt 07b §5) | Measured | Verdict |
|---|---|---|
| S0′: `git diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'` lists only `CLAUDE.md` and 89c368f2's four files, each AST-equal with docstrings removed ([`out/s0prime.json`](07b/rehearsal/out/s0prime.json), [`out/00-gate.txt`](07b/rehearsal/out/00-gate.txt)) | those five; the four AST-equal | **holds** |
| R31 gives `1b332ffa…` | `1b332ffa…` twice; `classes` equal R30's | **holds** |
| Outside `QuadSourceIntegral`: every value table identical; every other difference one of the four volatile kinds; no row in one store only ([`out/classify_non_qsi_r30_r31_labels_normalised.txt`](07b/rehearsal/out/classify_non_qsi_r30_r31_labels_normalised.txt)) | 360,462 value-table rows identical; 0 unclassified; differences are timing, `label` (job-name prefix normalised in a copy), `parents_digest` 291 and `rho_reuse` 23 | **holds** |
| Inside it, every moved row tabulated, with the `total_converged` changes each way ([`out/qsi_tabulation_r30_r31.txt`](07b/rehearsal/out/qsi_tabulation_r30_r31.txt), [`out/qsi_rows_r30_r31.csv`](07b/rehearsal/out/qsi_rows_r30_r31.csv), [`out/qsi_records_r30_r31.csv`](07b/rehearsal/out/qsi_records_r30_r31.csv), [`out/qsi_summary_r30_r31.txt`](07b/rehearsal/out/qsi_summary_r30_r31.txt)) | 569 of 864 rows moved: `total` in 379, `total_abserr` in 386, `analytic_rad` in 548. Largest \|Δtotal\| is 0.278 of R30's abserr. Regions 114,245 → 152,117. `total_converged`: 0 up, 1 down | tabulated |
| S3: \|t31 − t30\| ≤ e30 + e31 on every row with finite abserr in both, evaluated exactly ([`out/s3_scores.txt`](07b/rehearsal/out/s3_scores.txt)) | 864 scored; 0 violations; largest 0.128 | **holds** |
| S1: V1's twelve locate z ≈ 10211 and are honest against 06g's references ([`out/s1_scores.txt`](07b/rehearsal/out/s1_scores.txt)) | 12 of 12 locate (within 8.5e-6; spacing ≥ 31); 12 of 12 honest, largest 0.0670 (w06); 6 equal R30, 6 moved (range refusals) and more accurate | **holds** |
| S2: P1's two rows locate within one spacing and are honest against `p1_references` ([`out/s2_scores.txt`](07b/rehearsal/out/s2_scores.txt)) | \|x − x_s\| 7.1e-9 and 5.6e-9 against 4.1e-3; ratios 0.0202 (= R30) and 0.111 (R30 0.263) | **holds** |
| S4: R29's four guarded 2pt5 J4/Y4 blocks each locate their root, guard no leaf, and are honest against R29's block references ([`out/block_scores.txt`](07b/rehearsal/out/block_scores.txt)) | 4 of 4 locate one root at the reference root, bit for bit; 0 guarded; ratios 4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4; each equal to R30 and R28 bit for bit | **holds** |
| The readers on R31: exit 0, store byte-identical ([`out/readers_check.txt`](07b/rehearsal/out/readers_check.txt)) | 6 of 6 exit 0; R31 identical before and after | **holds** |
| The lock released; `ray-processes` lists nothing ([`state/lock.log`](07b/rehearsal/state/lock.log)) | released 20:04:53Z; free; nothing listed | **holds** |
| R30 and R31 kept; R29 and R30 unchanged ([`state/stores_sha256_final.txt`](07b/rehearsal/state/stores_sha256_final.txt)) | R29 and R30 equal to their KEPT hashes; R31 equal to its post-build hashes | **holds** |

**The replay behind S1, S2 and S4, and the refusals** ([`out/cmd/r31-replay/`](07b/rehearsal/out/cmd/r31-replay/),
[`out/replay_calls.checks.json`](07b/rehearsal/out/replay_calls.checks.json),
[`out/blocks.json`](07b/rehearsal/out/blocks.json)).
- R30's capture harness was run at the tree, Ray-free, under the lock, as the registered run
  `levin-phase-floor-07b-r31-replay-20261116T195034`.
- 208 rows: the 202 located rows, P1, the w rows, the block rows, and the 8 records of interest.
- **452 of 452 records reproduce R31 bit for bit.**
- **The refusals over the 4,295 replayed calls:**
  - range rule: 591 in 373 of 967 group calls, and 6,733 in 2,667 of 3,328 analytic calls;
  - edge rule: none.
- No row is beyond S3's bound, so §9.5's addition names none.

**Which store understates** (the user's word at stop 1; descriptive, not scored against any rule;
[`out/refs/understates.txt`](07b/rehearsal/out/refs/understates.txt)).
- **The records:** the 7 group records whose \|v31 − v30\| exceeds e30 + e31.
- **The captures** were made from R31 ([`out/refcap.json`](07b/rehearsal/out/refcap.json)). Each reproduces R31 at the tree
  and R30 at 07a's module bit for bit ([`out/refs/refcap_replay_h.json`](07b/rehearsal/out/refs/refcap_replay_h.json),
  [`out/refs/refcap_replay_a07.json`](07b/rehearsal/out/refs/refcap_replay_a07.json)).
- **The readings:** read by `tools/filon_reference.py` (`d37e633b`), four readings each (run
  `levin-phase-floor-07b-r31-refs-20261116T201120`, 15 min).
- **The rule** was fixed in [`state/refs_plan_r31.md`](07b/rehearsal/state/refs_plan_r31.md) before any reading.

| Record (k, q, r; z) | label | ref (FCC r2) | R30 v / e → \|v−ref\|/e | R31 v / e → \|v−ref\|/e | s / min(e) | answer |
|---|---|---|---|---|---|---|
| (1e6, 1e5, 1e6; 329.64) | G−q+r | `0x1.5632c898e5fb4p-82` | `0x1.5632cafb0fb4cp-82` / `0x1.494e3de357011p-111` → **59.29** | `0x1.5632c89648947p-82` / `0x1.d79fe909f16f2p-106` → 0.0055 | 5.9e-4 | R30 understates |
| (3.162e5, 1e5, 3.162e5; 996.42) | G−q−r | `0x1.23787817e63bap-78` | `0x1.23787936ea1b6p-78` / `0x1.01f48d18f2a15p-107` → **35.60** | `0x1.2378781644174p-78` / `0x1.d1ff6e16c79edp-103` → 0.0070 | 7.7e-5 | R30 understates |
| (3.162e5, 1e5, 3.162e5; 6.945e6) | G−q−r | `0x1.713988d6e71e3p-78` | `0x1.71398a46846f0p-78` / `0x1.3f94350ebc856p-105` → **9.20** | `0x1.7139887db5cfep-78` / `0x1.2ae23c1b2f3d1p-102` → 0.298 | 4.4e-5 | R30 understates |
| (1e6, 3.162e5, 1e6; 1.918e8) | G−q−r | `0x1.1fa14d5bf27f8p-83` | `0x1.1fa14e8a6665dp-83` / `0x1.ae9ef5b803440p-110` → **5.62** | `0x1.1fa14d5ef3244p-83` / `0x1.8fe6aae4f450bp-108` → 0.0150 | 3.5e-5 | R30 understates |
| (1e6, 1e6, 1e6; 36286.8) | G−q−r | `-0x1.2842349c30c56p-88` | `-0x1.28423f2458f0ep-88` / `0x1.35355dc8dcb99p-111` → **4.36** | `-0x1.284234a1dbe00p-88` / `0x1.1c1352bde5bcfp-110` → 0.0050 | 2.2e-5 | R30 understates |
| (1e6, 1e6, 1e6; 331553) | G−q−r | `-0x1.96e8eccadfe18p-87` | `-0x1.96e8e80d54444p-87` / `0x1.39d49df69c224p-111` → **3.87** | `-0x1.96e8ecc9c3340p-87` / `0x1.853df17bc4f6bp-110` → 0.0014 | 2.5e-5 | R30 understates |
| (1e6, 1e5, 1e6; 36286.8) | G−q+r | `-0x1.8f8a08c933cd0p-82` | `-0x1.8f8a0ad2a0c70p-82` / `0x1.66266a174d12dp-107` → **2.91** | `-0x1.8f8a08ca38b19p-82` / `0x1.c5533eaecb7e0p-107` → 0.0045 | 3.4e-5 | R30 understates |

Values and abserrs are the calls' own, unscaled by (1 + z). Every R30 and R31 record here is converged.
