## 6f. How two Clenshaw–Curtis floor terms add up across a call's leaves

*Prompt 06f, a measurement. It ran on the Mac Studio on 2026-10-31, Ray-free, with no lock, no
store built and no code changed. The records are under [`06f/`](06f/). What stays on the Mac Studio
is listed in [`06f/KEPT_ON_MAC_STUDIO.txt`](06f/KEPT_ON_MAC_STUDIO.txt).*

### 6f.1 The harness and Gate F0

**The setup:**
- The tree was the worktree at `f4770d13`, whose code is 06e's `f4d8e17`. It was clean throughout,
  and no bytecode was written in it.
- The store was the certified scratch copy of A3 v2, opened read-only. Its fingerprint at the tree
  is `0e32f24a…`, and its five files were EQUAL to their certification by SHA-256 before and after
  ([`06f/out/cmd/15`](06f/out/cmd/15_copy_vs_certified.txt), `18`,
  [`41`](06f/out/cmd/41_copy_after_run_vs_certified.txt)).

**The sample** (seed 20261031, [`06f/out/sample/`](06f/out/sample/)) was drawn from 06e's census
checkpoint `a0dc4d7f…`. The draw reproduced 06e's per-call counts exactly: 491 / 1,323
node-dominated and 2,493 / 2,785 declared-dominated calls.
- **M1** is Z2's call plus 20 node-dominated calls. At `1e-7` only four eligible calls have 100 or
  more Clenshaw–Curtis leaves, and all four were taken.
- **M2** is 20 declared-dominated analytic-block calls spread over J1–Y4.

**The run:** registered run `levin-phase-floor-06f-measure-20261031T204803`, done, 44 of 44 units.
- It captured each call with 06e's `x1_capture.py`.
- It replayed each call in its own process, with pass-throughs at `_adaptive_levin_subregion`,
  `_cc_node_term_parts`, `_cc_roundoff_floor` and `_declared_endpoint_phase_err`.

**F0 holds, with no replacement** ([`06f/out/f0.csv`](06f/out/f0.csv)):
- All 41 captures and replays equal 06e's census records bit for bit: value, abserr, regions,
  maximum depth and converged.
- Every returned leaf's floor is rebuilt from its captured terms in the code's order.
- The summed terms over each call's Clenshaw–Curtis leaves equal the checkpoint's `cc_floor` sums.
- Z2's call, replayed twice in fresh processes, gave the same digest, the same per-leaf table and
  the same $E$ for every one of its Clenshaw–Curtis solves.

### 6f.2 M1: the node errors

On Z2's call (`g-1e-8-0-33995` call 5, 8,600 Clenshaw–Curtis leaves):
- $\sum|E| = 1.312\times10^{-33}$ and $|\sum E| = 2.43\times10^{-35}$, a ratio of 54.
- The call's error with the node errors combined signed is $1.76\times10^{-34}$, against a request
  of $6.69\times10^{-34}$ and a declared $1.33\times10^{-33}$. The leaves' floors without their
  $|E|$ are summed as the code max's them, and $|\sum E|$ is added once.
- Combining within sibling pairs gives $1.11\times10^{-33}$: the cancellation is not local.
- The signs run: 3,129 runs against 4,289 for independent signs, $z = -25$.

Over the 11 sampled node-dominated calls with $N_{\rm CC}\ge100$, the median of
$\sum|E|/|\sum E|$ is 20.8 (range 7.2–80).

**F1, F2 and F3 hold.** F2 is a statement about the returned partition. The driver was not re-run
under another combination.

### 6f.3 M2: the declared phase error

The phase error was measured against `atan2(J, −Y)` from mpmath at 30 digits.
- **The method check passes** ([`06f/out/cmd/31`](06f/out/cmd/31_method_check.txt)): ν = 1/2
  reproduces the 1.11e-16 phase-pair error that `bessel_phase.py` documents.
- **F4 fails.** 49 of 1,177 nodes violate, up to 148×. All of them are on the tail Levin leaves of
  two calls, and the rounding of the group's product $K\eta$ explains each violation to within 0.27
  of `theta_abserr` ([`06f/out/cmd/44`](06f/out/cmd/44_f4_attribution.txt)).
- **F5 fails.** The median of $D/\max_s|\delta I|$ is 6.0. $\delta I$ is converged to $2\times10^{-7}$
  under doubling.
- **F6 fails.** The median of $\sum_\ell|\delta I_\ell|/\max_s|\delta I|$ is 1.0: 18 of the 20 calls
  have one or two Clenshaw–Curtis leaves.

Where the excess lies:
- **The proxy's own slack is small.** Its median over the honest $\sum|w||f|$ is 1.05.
- **The factor of about 6 is cancellation within a leaf.**
- **The declaration is about 2,000× the error the phase actually commits** (median
  $|\delta I_{\rm meas}|/|\delta I|$ is $4.8\times10^{-4}$), except where it under-declares (F4).
- **The term never binds.** Across 06e's census, the declared term over floor-attributed
  Clenshaw–Curtis leaves reaches the request on no call
  ([`06f/out/cmd/45`](06f/out/cmd/45_declared_vs_request.txt)).

The full results, and what a remedy for each term would have to do, are in
[`06f/out/results.md`](06f/out/results.md).
