# levin-phase-floor 03 — orchestrator's review

**Verdict: PASS.** All nine checks of `orchestrator/prompt-03.md` §3 pass. Three of them pass on a reading that the user should confirm:
- check 5: `elapsed` is not gated;
- check 6: M5 is inapplicable on R7;
- check 8: the launch pid is in `status.json`, not `manifest.json`.

Each reading is set out under its check. The mechanism verdict as written does **not** hold, because M3 fails on its slope clause. That is a result of the prompt, not a failed check. It is for the user at G1.

Reviewed on the Mac Studio (mps026298) at `98e68d3`, 2026-10-25, 20:15–20:45 GMT. The orchestrator wrote no code, repaired nothing, and ran no git command that writes.

**Times and identifiers**
- The marker was touched at 19:32:28.47Z, before dispatch, at `<scratchpad>/marker`.
- The run is `levin-phase-floor-03-instrument-20261025T194950`. It launched at 19:49:50Z as pid 87581, and the pid was gone at 20:11:10Z.

**The orchestrator's Gate re-read, 19:32Z, before dispatch.** Every item held:
- `HEAD` was `98e68d3e1b60e6de4378b7d9aded3a120d6088a9`, and `git status --short` was empty;
- `runregistrykit list` showed 10 runs, all finished;
- `rehearsal_lock status` said `lock: free`;
- `store show` reported `problems: none`, digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`;
- `var/levin-phase-floor-03` did not exist, and neither did `~/ScienceData/.rehearsal-lock`.

---

## 1. Scope — PASS

```
$ git rev-parse HEAD
98e68d3e1b60e6de4378b7d9aded3a120d6088a9
$ git status --short
(empty)
$ find var -newer <marker> | grep -v '^var/levin-phase-floor-03' | grep -v '^var/runs/levin-phase-floor-03-instrument-20261025T194950'
var
var/runs
$ find . -path ./var -prune -o -path ./.git -prune -o -newer <marker> -print
(empty)
```

- `var` and `var/runs` appear only because their directory mtimes changed when the two new entries were created.
- Nothing was written outside the two allowed places: no store file, no sidecar, no other run directory, and no file elsewhere in the tree.
- `/tmp/ray-sgk` has an mtime of 17:08, before the marker, and `~/.raypoolkit/` is empty.

## 2. The store — PASS

- `diff state/store_files_before.txt state/store_files_after.txt` differs only in its header line ("before"/"after" and the time). The SHA-256, size and mtime_ns of all six files are identical. The same holds for `store_files_after_fingerprint.txt`.
- **Rehashed independently** with `shasum -a 256 var/datastores/handover-A3-v2-lambdacdm*`, after the review's own reads. All six hashes equal `store_files_before.txt`:
  - primary `1542e05c…`;
  - shard0000 `b311698f…`;
  - shard0001 `c5e341ce…`;
  - shard0002 `175a6731…`;
  - shard0003 `76b83fb3…`;
  - sidecar `3201f6a5…`.
- `./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite`, **without `--write`**, exited 0 and printed:
  - `digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`;
  - `problems: none`;
  - `recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752d…`.
- `ls var/datastores | grep -E 'journal|wal|shm'` matched nothing (exit 1), before and after the review's reads.

## 3. The route — PASS (route (a))

The log's subject line and deviations 3–4 say route (a) was taken. Ray was never initialised and no lock was taken. `state/fixture_proof.txt` (19:41:45Z) and `state/fixture_proof_final.txt` (19:49:35Z, re-run on the final scripts) show:
- the fixture store under `state/fixture_proof_store/` opened `read-only with 2 shards`, with `pool._read_only = True`;
- `object_get` returned a `TkNumericIntegration`, a Tq/Tr input of `QuadSourceIntegral`, with its values;
- `object_store` was refused with `ReadOnlyWrite`, and a lookup that missed was refused with `ReadOnlyMiss`;
- all three fixture files had the same SHA-256, size and mtime_ns before and after;
- `ray.is_initialized() = False`, `FIXTURE PROOF PASSED`, exit 0.

**The order.** The first fixture proof was written at 19:41:45Z. The first opening of A3 v2, the smoke run in `state/smoke/`, was at about 19:48Z, so the proof came first.

**The temporary directory** is under `state/`, not a `TemporaryDirectory`, and the log says so (deviation 3).

The lock was re-checked, as a command of its own: `rehearsal_lock status` gave `lock: free`. Route (b)'s transcript does not apply.

## 4. The wrappers are transparent — PASS

Read in `scripts/run_instrument.py`, `make_wrappers` (`:120-170`) and the row loop (`:385-430`, `:517-525`):
- `adaptive_levin_sincos(*args, **kwargs)` calls `orig_levin(*args, **kwargs)` and returns `result`, the same object. It records references only. `build_phase_groups` does the same with `orig_build`.
- Both module attributes are set just before `work.compute(…)`. They are restored in a `finally`, asserted `is orig_…` after each row, and restored again in the outer `finally`.
- The group label comes from `parse_label(kwargs.get("notify_label"))`. That is a regex on the label `QuadSourceIntegral.py:1067` builds, `… regime {regime_label}, group {group.label}`, so the label is not inferred from call order.
- Each group call is matched to its `PhaseGroup` by (regime, label), and the match is confirmed by the identity of `theta_mod_2pi`. Every row's stdout says `every group call matched by (regime, label) and identity: True`.
- The other 16 Levin calls per row are the `analytic J1…Y4` Bessel calls (`QuadSourceIntegral.py:1325-1391`).

## 5. M0 — PASS (one reading: `elapsed`)

**The comparison code** is `compare()` and `m0()` in `scripts/run_instrument.py:178-293`:
- It uses `==`, `is` for `None`, and, for floats, equality of `struct.pack("<d", …)` as well. A grep for `isclose`, `allclose` and `AlmostEqual` finds none.
- The fields gated are `total`, `total_abserr`, `total_converged`, `WKB_Levin`, `numeric_quad`, `WKB_Levin_data.max_depth` and `num_regions`, every sub-interval field, and every group field.
- The stored metadata's keys, read from the store, are:
  - sub-interval: `z_max, z_min, regime, value, abserr, converged, phase_limited, groups`;
  - group: `label, value, abserr, converged, phase_limited, regions, simple_regions, evaluations, elapsed`.

  Every key is gated except `elapsed`.

**Reading for the user:**
- `elapsed` is a wall-clock time. No re-run can reproduce it to the bit.
- The agent stated this before launch and classified it `STRUCTURALLY REQUIRED` (deviation 6). All 45 values differ; the re-run took 0.76–0.96 of the stored time.
- The orchestrator does not adjudicate this. It is listed for confirmation.

**The output.** `out/m0_03.txt` ends with `R1 … R7: M0 holds; calls matched True` and `all rows M0 holds: True`. The only lines that read `DIFFER` are the 45 `elapsed` lines, all `(not gated)`.

**Checked independently.** Each stored row was read through `sqlite3` with `mode=ro`, and each pickle `row_Rn.pkl` was loaded with `ray.cloudpickle`. `python -I`, with the repository root and `scripts/` put on `sys.path` for the pickle's class references. The two were compared with `float.hex`:

| Row | `total` (stored = pickled) | `total_abserr` (stored = pickled) | conv |
|---|---|---|---|
| R1 (2, 37527) | `-0x1.631c5efe83a46p-78` | `0x1.2f222e3d4dd6dp-95` | 0 |
| R2 (0, 1797) | `-0x1.d7209834862aep-85` | `0x1.e96dafbbbdf4ep-101` | 1 |
| R3 (1, 37670) | `0x1.644dc552ccb4fp-94` | `0x1.54496e45a6b44p-106` | 1 |
| R4 (2, 37720) | `-0x1.5c7f0b6b8e0f0p-83` | `0x1.5aac9bd768287p-96` | 0 |
| R5 (0, 37966) | `-0x1.bc772db669f5bp-87` | `0x1.095b22036083ap-102` | 1 |
| R6 (2, 34728) | `0x1.7d26ed4e62cbap-78` | `0x1.5844c36a8c31fp-99` | 0 |
| R7 (2, 34820) | `0x1.d6c1cb190e1bcp-79` | `0x1.718a77abf74bcp-100` | 0 |

`WKB_Levin` and `numeric_quad` were compared the same way. The result: `all seven bit-equal: True`.

**`analytic_rad`** is reported as `EQUAL (not gated) [== and bits]` on all seven rows.

## 6. The predictions — PASS (one reading: M5 on R7)

- **Order.** The mtime of `state/predictions.md` is 19:48:38.66Z. The run's `manifest.json` is 19:49:50.49Z, so the predictions came first.
- **Word for word.** Prompt §3, lines 181–219 at `98e68d3`, is contained verbatim in both `state/predictions.md` and `log.md` (checked by string containment). `state/predictions_section3.raw` equals those lines exactly.
- **The scoring** is in the log's "Verdicts". Each rule is scored with its numbers:
  - M1 holds;
  - M2 holds, on 7 of 7 rows;
  - M3 fails, with 0 of 7 rows passing because the slope clause fails;
  - M4 holds, at 0.0085;
  - M5 holds on R1–R5 and is inapplicable on R7.

  M3 is scored as written. The agent's argument that the slope clause is badly posed (I1) is recorded beside the score and does not replace it.
- **Reading for the user.** M5's rule names "R1–R5 and R7", but R7 (`q == r == k`) has no `Gr` sub-interval: its regimes are `G` and `Gqr`. So the rule cannot be evaluated on R7. The agent scores M5 "holds on the five rows it applies to" and records R7 as a badly posed part of the rule (deviation 7, `STRUCTURALLY REQUIRED`). Strictly, R7's part is undecided; the orchestrator does not adjudicate it.

## 7. The analysis, re-run — PASS

**The re-run**, into the scratchpad:
```
$ OMP_NUM_THREADS=1 PYTHONPATH=. ./venv/bin/python var/levin-phase-floor-03/scripts/analyse.py \
    var/runs/levin-phase-floor-03-instrument-20261025T194950 > <scratchpad>/instrument_03_rerun.txt
exit=0
$ diff var/levin-phase-floor-03/out/instrument_03.txt <scratchpad>/instrument_03_rerun.txt
(no output; exit 0)
```
The output is byte-identical; it carries no timestamp. `analyse.py` reads only the pickles, through `harness.load_record`, plus the code it imports. It opens no store. Its helper `depth20_floor` reads files only when called on a path, and `eta` is pure.

**The definitions**, as read in the code:
- $E$ is `group.envelope` at the region's midpoint.
- $h$ is the width in $\log(1+z')$.
- $D$ = `abserr`$/(E h)$.
- $\sigma_\theta$ (`sigma_theta`, `:163`) uses 65 `linspace` points, `np.unwrap`, a degree-4 `polyfit` in $t\in[-1,1]$ and the RMS residual.
- The control (`control_phase`, `:175`) replaces `theta_mod_2pi` with a degree-12 `Chebyshev.fit` to the unwrapped phase on the depth-10 span. `theta`, `theta_deriv` and $f$ are unchanged.
- $c_s$ is `sqrt(mf.wPerturbations(expm1(x)))` (`:582`).

**The thresholds**, as read in the code:
- scored means `n18 >= 10` (`:310-311`);
- M1 is `0.1 <= … <= 10`;
- M2 is `-0.5 <= slope <= 0.5` and `0.01 <= D/σ <= 10`, on `>= 6` of 7 rows;
- M3 is `slope >= 2 and ratio >= 100`, on `>= 6` of 7 rows;
- M4 is `share < 0.1`;
- M5 is `abs(dev) <= 0.05` at all five abscissae.

**The ancestry routine.** The log (deviation 14) and `analyse.py`'s docstring say the ancestry used the driver's own `_adaptive_levin_subregion`, not the Clenshaw–Curtis pair for every span. The recomputed discrepancy equals the recorded `region.abserr` at all 580 sampled depth-20 regions. In the event, all 6380 spans were Clenshaw–Curtis.

**Every group is reported.** There are 45 groups, 29 of them scored. The 16 unscored groups appear in the "every group" table.

## 8. The run — PASS (one reading: where the pid is)

- `runregistrykit list` shows `levin-phase-floor-03-instrument-20261025T194950  done  7/7`, with 11 runs, all finished.
- Its `status.json` reads `state: done`, `exit_code: 0`, `pid: 87581`, 7/7 units.
- The manifest names:
  - `campaign = levin-phase-floor`, `prompt = 03`;
  - the slug `instrument`, in `run_id`;
  - `git_head = 98e68d3e…` and `git_dirty = False`;
  - `script_sha256 = f3052aaf…`, which equals `shasum` of the kept `scripts/run_instrument.py`.
- Every checkpoint record carries the same script hash and `git_dirty: false`.
- `stdout.log` (103 kB) and `stderr.log` (empty) are in the run directory.

**Reading for the user.** The manifest has no pid field. RunRegistryKit v0.2.0's manifest schema has none: the manifests of `handover-07-a3-v2` and `py313-venv-02-resumed-c` have the same keys. The detached pid is recorded by `Run.heartbeat(pid=…)` in `status.json`, as `CLAUDE.md` rule 3 says. `state/launch.txt` gives the child pid as 87581 and the launcher pid as 87577.

## 9. The records — PASS

- **`log.md`** follows README §5.1:
  - subject, commit and result;
  - Predictions and verdicts (first);
  - What shipped;
  - Deviations;
  - Verification performed;
  - Observations not acted on;
  - State handed on.

  There are 17 deviations, each classified. Step 1's no-op is deviation 1, `STRUCTURALLY REQUIRED`. Two are `UNINTENDED DRIFT`:
  - deviation 2: the first `stat -f` hashing attempt's transcript was lost. The real hashes were taken by `hash_store.py` before the store was opened, and they equal 01's.
  - deviation 17: a wrong line citation in `analyse.py`'s docstring. The analysis was re-run so that the kept output records the kept script's hash, and the first output is kept in `state/superseded_run1/`.
- **`verification.md`** opens `## 3. Seven rows, instrumented`, and all 12 links are in the form `03/out/…`, `03/scripts/…` or `03/state/…`.
- **`issues.md`** exists, with I1–I5:
  - I1: M3's slope clause cannot be met;
  - I2: every depth-20 region is Clenshaw–Curtis, which touches `[10-levin-wholesale-cc-fallback]` and raises a scope question;
  - I3: `StoreFixtures` cannot carry A3 v2's objects;
  - I4: `PhaseGroup` pickling;
  - I5: "Read first" item 4 is inaccurate about `used_interval`.

---

## Results, for the report

**M0 holds** on 7 of 7 rows, bit for bit. `analytic_rad` is equal on all seven.

| | Number | Verdict |
|---|---|---|
| M1 | 29 of 29 scored groups in [0.1, 10]; σ_θ/F 0.224–0.342 | holds |
| M2 | 7 of 7 rows; slope per row 0.027–0.089; D/σ_θ per row 0.078–0.113 | holds |
| M3 | control slope per row −0.020 to 0.148 (< 2 on 7 of 7); real/control ratio 3.1e6–2.4e8 (≥ 100 on 7 of 7); both clauses on 0 of 7 | **fails** (slope clause) |
| M4 | 1611 / 189,196 = 0.0085; Clenshaw–Curtis share at depth 20 = 1.000 | holds |
| M5 | R1–R5: ratio 0.26795–0.26885 against 0.267949, worst +0.34 %; R7 has no `Gr` | holds on R1–R5; R7 inapplicable |

**The mechanism verdict as written does not hold**, because M3 fails. The clause "M2 holds and M3 fails ⇒ refuted" fires by its letter.

The agent's I1 argues that the slope clause is badly posed:
- with the jitter removed, D is at round-off (1e-16 to 1e-13) from depth 10 onward, so no slope is measurable;
- the ratio clause (a factor of 1e6 to 1e8) points toward the jitter.

The orchestrator records this and does not adjudicate it. It is for G1.

**H-anchor holds.** M1 and M2 hold in every scored group, including `G−r` on R4 and R5, the lone `G` on R6, and `G+r`.

**H-cancel's condition is met** in all 18 cancelling groups. It is met equally in all 11 non-cancelling groups (median σ_θ/F 0.270 against 0.262), so it does not account for them.

**Route:** (a), Ray-free. **Fingerprint:** `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
