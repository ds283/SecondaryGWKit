# Issues found by the rehearsal agent, levin-phase-floor 06 §5 (Mac Studio)

These are proposals for the board. The orchestrator decides whether to open them.

## Proposed

### P1. Two rehearsal rows' totals moved by 10²–10⁴ × the parent's declared error

**Where.** These are R13 → R14, 06's change alone, from
[`out/qsi_extra_counts.txt`](out/qsi_extra_counts.txt) and
[`out/qsi_tabulation.txt`](out/qsi_tabulation.txt):

| $(k, q, r)$, $z_{\rm resp} = 250$ | total A → B | rel. | $/$`abserr`$_A$ | $/$`abserr`$_B$ | depth | regions | converged |
|---|---|---|---|---|---|---|---|
| $(3.162\times10^5)^3$ | 5.9186e-20 → 5.9279e-20 | 1.6e-3 | 1.8e4 | 1.3e4 | 12 → 6 | 31 → 14 | 0 → 0 |
| $(10^5, 3.162\times10^5, 3.162\times10^5)$ | −3.3962e-19 → −3.3963e-19 | 2.8e-5 | 484 | 555 | 14 → 6 | 30 → 15 | 0 → 0 |

- **Two more rows exceed 1 × `total_abserr`$_A$,** both at $k = q = r = 10^6$: 1.25 at
  $z_{\rm resp}$ 755.7, and 1.02 at 996.4. Both are below 1 against `total_abserr`$_B$.
- **What it shows.** On the rehearsal scope, the stored-phase noise moved two unconverged totals far
  beyond the error the parent declared for them. It is also more than R14 declares.
- **How it relates to §6.** It is the rehearsal-scope counterpart of §6's S4 (and S5's question of
  whether the declared error is honest). It is not scored here.
- **Proposed handling:** weigh it beside §6's S4 and S5 before opening anything. If §6 shows the
  same on A3 v2, it belongs with whatever issue S4 or S5 opens. If not, it is a separate note about
  `z_resp = 250`, the end of the range.

### P2. The `!! WARNING` check of a build is not visible in its launch log

- **What happens.** `_build_phase`'s disagreement warning and `_check_phase_sign` run inside Ray
  tasks. In both builds the launch log and `stdout.log` hold only the `Datastore` actor's output;
  no task's output appears there.
- **How it was searched here.** This agent also searched Ray's worker logs,
  `/tmp/ray-sgk/session_*/logs/worker-*`, where every count is 0
  ([`out/ray_worker_logs_check.txt`](out/ray_worker_logs_check.txt)).
- **What is not known.** Whether a task's `print` would have reached the launch log was not tested.
  No task printed anything in either build.
- **Proposed handling:** a note for the next prompt that relies on a warning grep, such as a
  production build. Either show once that a task's `print` reaches the driver's log, or search the
  worker logs as here. The worker logs live under `/tmp`, so they are not durable records.

## Recurrences of open issues (observations)

- **`[05-store-value-diff-compares-volatile-columns]`** recurs as 05 described it: timings,
  labels, `GkSource.parents_digest`, and the GkWKB residual-memo counters (25 rows)
  ([`out/classify_non_qsi.txt`](out/classify_non_qsi.txt)).
- **05's P2, run-to-run nondeterminism of the residual-memo counters,** recurs: the counters
  differ in 25 GkWKB rows here, against 25 in 05 and 31 in 05a.
- **`[04-roundoff-floor-can-be-infinite]`.** 26 `QuadSourceIntegral` rows have `total_abserr` = inf
  in both R13 and R14, and 9 of their `total`s differ.
- **`[10-the-readers-draw-their-work-lists-unseeded]`.** The readers' file counts differ from
  05a's R12 (Gk 396 against 387, GkSource 176 against 162, QSI 70 against 73), with the same
  lookups.
- **`classify_non_qsi.py`'s exemption** names only `QuadSourceIntegral`. A prompt that changes
  more classes reads its exit status per class, as this one did (log, deviation 6). This is a note
  on 05's script, which is a record and is left as written; it is not proposed as an issue.
