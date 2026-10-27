### 6.x The re-measurement on A3 v2, with the producers re-run in-process (prompt 06 §6)

Run `levin-phase-floor-06-wider-20261026T223634` on the Mac Studio (mps026298), Ray-free, no lock,
22:36:34Z–23:23:16Z: `done`, 109/109, exit 0, no unit over budget, no error. Two trees, both
worktrees: **t5** = `9f66622` (05a's t; its consumers and A3 v2's stored producer samples) and
**s** = `f5cee4c` (code `dc9d28c`; its producers re-run on every `GkWKBIntegration` and
`TkWKBIntegration` each row uses, the `GkSource` re-assembled and the `GkSourcePolicyData`
recomputed from them, and its consumers). Records: [`06/wider/`](06/wider/).

**Route (B), the user's decision.** `s` cannot open A3 v2 (its factories declare `theta_residual`,
which A3 v2 lacks: [`06/wider/out/cmd/06_dev_smoke_s.txt`](06/wider/out/cmd/06_dev_smoke_s.txt)). It
read, read-only, an unregistered copy of A3 v2 with the three nullable columns added, all NULL:
every one of 3,411,578 rows identical by bit pattern, schema differing only by those columns
([`06/wider/out/cmd/12_certify_copy_rerun.txt`](06/wider/out/cmd/12_certify_copy_rerun.txt)),
fingerprint `0e32f24a…`, problems none
([`06/wider/out/cmd/15_fingerprint_copy_at_s_rerun.txt`](06/wider/out/cmd/15_fingerprint_copy_at_s_rerun.txt)).
t5 read A3 v2 itself.

**Gate U0** ([`06/wider/out/U0_a.txt`](06/wider/out/U0_a.txt), [`06/wider/out/U0.txt`](06/wider/out/U0.txt)): holds.

| Clause | Result | |
|---|---|---|
| (a) | `git diff --stat 1fc349c 9f66622 -- . ':!docs' ':!prompts'`: `CLAUDE.md`, `primitive_phase.py`, `test_g_phase_reanchor.py`; both ASTs without docstrings equal `1fc349c`'s | holds |
| (b) | t5@1e-7 on `low-G+r-1`, `ctrl-low-1` against 05a's `t-1e-7-*`: total, abserr, every call, the whole result summary and region arrays bit-identical | holds |
| (c) | producers re-run at the parent on A3 v2 (3,156 G objects, 241,146 values; 4 T objects, 5,172 values): `theta_div_2pi`, `theta_mod_2pi`, `G_WKB`, `T_WKB` and every coefficient bit-identical; beyond the clause the re-assembled GkSource, the recomputed GkSourcePolicyData and each unit's result and regions reproduce the stored-input unit bit for bit | holds |
| (d) | two in-process `s` runs per row: result, calls, regions, the re-run's record, the policy, the probe and every `deltaTheta` bit-identical | holds |

**Predictions** ([`06/wider/state/predictions.md`](06/wider/state/predictions.md); §6 as amended,
quoted unchanged, written after U0 and before the launch; the pure-WKB tag read two ways in a
section added before the launch, the user ruling for (I) with no S1 number in existence):

| | Number | Verdict |
|---|---|---|
| **S1**, reading (I) of record (`z_init == z_source`) | 6,700 pure-WKB G windows on R1–R7, all matched, true in 6,700. `sigma_t5` is exactly 0 in all 6,700 (production's `phi_G` quantised to zero); `sigma_s` ≤ 8.8e-14 F (median 5.7e-16 F); largest `sigma_s`/bound 8.8e-8 | **holds** |
| (S1, reading (L), unscored) | no window has a null `z_init` (A3 v2 stores none) | (undecided) |
| **S2** | 12,862 T windows, all matched; true in 10,747, **false in 2,115**, all at z ≥ 1.75e7 with `\|theta_T\|` 0.05–1.5e3 rad, where `sigma_s/sigma_t5` = 0.952–1.04 and `sigma` = 5–1.5e7 `eps\|theta\|`: `phi_T`'s own structure, not storage noise. Elsewhere `sigma_s/F` median 7e-10 | **fails** |
| **S3** | 39 of 39; largest depth under s@1e-7 is 11 (11 on the rows capped under B1) | **holds** |
| **S4** | 34 of 39; false on `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 (× `total_abserr_t5`); median ratio 0.29, median relative change 1.3e-6; bit-identical on the two NULL controls | **fails** |
| **S5** | 28 of 31, none excluded; false on `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31; true on all 11 capped under B1 (largest 0.823). 05a's T4 under t5: 29 of 31 | **fails** |

**Numerically seeded G windows (not scored):** 3,594; `sigma_s` 2.1e-4 F at the median (max 1.2e-3
F) against `sigma_t5` 0.27 F (max 0.78 F); the re-run `deltaTheta`'s window RMS is 2.8e-5 F at the
median, about 1/8 of `sigma_s`.

**Descriptive:**
- No unit over budget; largest s wall/budget 0.715 (`s-1e-8-R2`). No `!! WARNING
  (GkSourcePolicyData` line in any unit.
- Regions 901 (t5) → 742 (s@1e-7), 1,053 at 1e-8; R1–R7 250 → 200; the 11 capped 426 → 336. 48 of
  219 group calls change regions, depth or convergence. Unconverged: 21 → 20 at 1e-7, 19 at 1e-8.
- `theta_deriv` (log) at the window centres: T relative change ≤ 4.6e-14; G median 3.7e-14, up to
  4.9 in seeded windows.
- `Levin_z` changes on one row (`low-G+r-1`); type, quality and crossover on none.
- `convergence_kinds.py` under s ([`06/wider/out/convergence_kinds_s.txt`](06/wider/out/convergence_kinds_s.txt)):
  1e-7 resolution 14, bound only 32, phase-limited 23; 1e-8 20, 25, 32 (05a's t: 18, 32, 17 and 21, 24, 27).
- The re-run values against A3 v2: 17 G cycle changes, each one cycle at a remainder within
  `4 eps|theta|` of a boundary; none for T.

**The stores.** A3 v2's six files: SHA-256, size and mtime identical before, before the launch,
after the run and after the fingerprint; no journal, `-wal` or `-shm`; fingerprint `0e32f24a…`,
problems none, matching the one taken at `b90752d`
([`06/wider/out/cmd/36_fingerprint_A3v2.txt`](06/wider/out/cmd/36_fingerprint_A3v2.txt)). The copy's
five files are identical from certification to the end, and its fingerprint is again `0e32f24a…`.
