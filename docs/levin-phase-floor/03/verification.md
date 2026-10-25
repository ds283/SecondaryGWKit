## 3. Seven rows, instrumented

**Prompt 03, run 2026-10-25 on the Mac Studio** at `98e68d3` (`handover-remedial`, clean), with no store write.

**Route (a).** A3 v2 was read through a real `datastorekit` `ShardedPool(read_only=True)` whose actors were stood in by `StoreFixtures.standin_pool`.
- `ray.put`, `compute_QuadSource_integral.remote` and the stand-in's serialisation were patched to `ray.cloudpickle` equivalents inside the harness.
- Ray was never initialised, and no lock was taken.
- The route was first proven on a `real_store_fixtures` store ([`state/fixture_proof_final.txt`](03/state/fixture_proof_final.txt)).

**The run.** The seven rows were re-run in-process by the registered, detached run `levin-phase-floor-03-instrument-20261025T194950`, which ended `done`, 7/7, exit 0. Its pickles stay in that run directory on the Mac Studio. The store was fingerprint `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` before and after.

- **Scripts** (all in [`scripts/`](03/scripts/)):
  - [`rebuild.py`](03/scripts/rebuild.py): step 4, `main.py`'s lookups by the same keys and tags;
  - [`run_instrument.py`](03/scripts/run_instrument.py): the wrappers, M0 and the pickles;
  - [`harness.py`](03/scripts/harness.py): route (a), and the pickler;
  - [`launch.py`](03/scripts/launch.py): the registered launch;
  - [`analyse.py`](03/scripts/analyse.py): step 7.

  Each output's header records the SHA-256 of the script that wrote it.
- **Outputs:**
  - [`out/m0_03.txt`](03/out/m0_03.txt): M0, field by field;
  - [`out/instrument_03.txt`](03/out/instrument_03.txt): every group, then M1–M5;
  - [`out/fingerprint_03.txt`](03/out/fingerprint_03.txt).
- **The store** was byte-identical at step 2, at launch, after the run and after the fingerprint: [`before`](03/state/store_files_before.txt), [`after`](03/state/store_files_after.txt). `store fingerprint`, run without `--write`, matches the sidecar.

### 3.1 M0 — the re-runs are production

On all seven rows, everything step 5 lists equals the stored row bit for bit, under `==` and by IEEE-754 bit pattern:
- `total`, `total_abserr`, `total_converged`;
- `WKB_Levin`, `numeric_quad`;
- `WKB_Levin_data.max_depth` and `num_regions`;
- every field of every group record, and of every sub-interval record, in `metadata["WKB_Levin"]["subintervals"]`.

`analytic_rad` is equal too, on every row. The one field not gated is each group record's `elapsed`, a wall-clock time (all 45 differ; the re-run took 0.76–0.96 of the stored time, median 0.83).

The inputs were rebuilt as `main.py` builds them:
- the grid tags came out as `SourceRedshiftGrid_1740_8d1d43b6` and `ResponseRedshiftGrid_145_c51d43ac`;
- every rebuilt z equals the stored z;
- the six parents equal the stored row's;
- the existence query returns each row by its serial.

The wrappers on `adaptive_levin_sincos` and `build_phase_groups` are transparent. Each Levin call was matched to its `PhaseGroup` by the label parsed from `notify_label` and confirmed by object identity.

**Fidelity of the analysis.** From the pickles alone, the driver's own single-region routine reproduces the recorded `abserr` and type of all 580 sampled depth-20 regions exactly.

### 3.2 What the bisection looks like

**Every depth-20 region is a Clenshaw–Curtis fallback region**: 189,196 of 189,196 in the 29 scored groups (at least 10 regions at depth ≥ 18). None of the 45 groups has a Levin region at depth 20.

| row | top group | F | scored groups | median σ_θ/F | median D/σ_θ | slope (real) | slope (control) | D₂₀ real/control |
|---|---|---|---|---|---|---|---|---|
| R1 | `G+r` | 1.69e-6 | 5 | 0.28–0.29 | 0.112 | 0.059 | 0.068 | 3.1e6 |
| R2 | `G+r` | 4.60e-6 | 3 | 0.31–0.32 | 0.093 | 0.067 | −0.020 | 8.8e6 |
| R3 | `G+r` | 9.19e-5 | 2 | 0.26–0.29 | 0.078 | 0.089 | 0.069 | 2.4e8 |
| R4 | `G−r` | 2.60e-6 | 5 | 0.25–0.26 | 0.099 | 0.054 | 0.148 | 5.8e6 |
| R5 | `G−r` | 6.49e-6 | 4 | 0.22–0.23 | 0.100 | 0.051 | −0.002 | 1.1e7 |
| R6 | lone `G` | 1.96e-6 | 5 | 0.34 | 0.113 | 0.027 | 0.023 | 2.0e7 |
| R7 | `G+q+r` | 2.60e-6 | 5 | 0.26 | 0.106 | 0.049 | −0.017 | 1.0e7 |

How to read the table:
- σ_θ/F is the range over the row's scored groups. D/σ_θ, the slopes and the ratio are medians over the row's scored groups.
- The slopes are of ln D on ln h over depths 12–20, from 20 sampled depth-20 regions per group.
- In the control, `theta_mod_2pi` is replaced on each depth-10 span by a degree-12 polynomial fit to it, and nothing else changes.

**The composed phase carries non-smooth jitter at about 0.2–0.35 F**, in every scored group:
- cancelling or not: the class medians are 0.270 (cancelling, 18 groups) against 0.262 (non-cancelling, 11), a ratio of 1.03;
- `G±r`, the lone `G`, and all four `Gqr` signs alike.

**The discrepancy that drives the bisection does not fall with refinement.** D is flat from depth 10 to 20 (slopes −0.005 to 0.17), at about 0.1 σ_θ.

**Remove the jitter and it is gone.** The depth-20 discrepancy falls by 3e6 to 2e8, to the round-off of the nested pair (D ≈ 1e-16 to 1e-13). In the control it is already at that level at depth 10, so it cannot fall further with h.

**The driver cannot see the floor.** `phase_err` reaches `abserr` in 0.85 % of the depth-20 regions (pooled; 0.26–2.1 % by row).

**How far `G+r` cancels.** On R1–R5, |θ'_{G+r}|/|θ'_{G−r}| is 0.26795–0.26885 at five abscissae of the `Gr` sub-interval. The prediction (1−c_s)/(1+c_s) is 0.267949, with c_s² = 0.333333. That is agreement within 0.34 %, so `G+r` oscillates at about 0.27 of `G−r`'s rate and is not stationary. R6 and R7 have no `Gr` sub-interval.

### 3.3 Verdicts

| | Number | Verdict |
|---|---|---|
| **M0** | 7/7 rows bit-identical | **holds** |
| **M1** | 29/29 scored groups in [0.1, 10]; 0.224–0.342 | **holds** |
| **M2** | both parts on 7/7 rows (slopes 0.027–0.089; D/σ_θ 0.078–0.113) | **holds** |
| **M3** | control slope ≥ 2 on 0/7 rows (−0.020 to 0.148); ratio ≥ 100 on 7/7 rows (3.1e6 to 2.4e8) | **fails**, on the slope clause; the clause is **badly posed**, recorded as such and not re-posed (03's `issues.md` I1) |
| **M4** | 1611/189,196 = 0.0085 < 0.1; Clenshaw–Curtis share 1.000 | **holds** |
| **M5** | R1–R5 within 0.34 %; inapplicable on R7 (no `Gr`) | **holds** on R1–R5 |

**The mechanism**, "a non-smooth phase floor, invisible to the driver, drives the bisection to the cap", holds if M1–M4 hold.
- **As written, it does not hold, because M3 fails.** The prompt's clause "M2 holds and M3 fails ⇒ refuted" fires by its letter.
- **What M3 measured points the other way.** Removing the jitter, and nothing else, removes the discrepancy by six to eight orders of magnitude. The slope clause cannot be met by a discrepancy that is already at round-off from depth 10.
- That reading is the user's to make at G1. 03's `log.md` note (a) gives it in full.

**H-anchor holds.** M1 and M2 hold in `G−r` on R4 and R5, in the lone `G` on R6, and in `G+r`; indeed in every scored group.

**H-cancel's condition is met** in the cancelling groups. But the non-cancelling groups carry the same floor at the same level and the same depth, so **it does not account for them**. As in §1, it fails as the whole account.

**For §2.7 layer 2.** The regions at the cap are all in the Clenshaw–Curtis branch (03's `issues.md` I2), and that branch's acceptance test is the one a declared phase floor must reach.
