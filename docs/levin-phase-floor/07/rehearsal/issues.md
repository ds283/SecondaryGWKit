# Issues from 07 §5 (rehearsal agent)

**One to open** (a finding for the user, who is deciding whether to fix it in a new code commit before the
census; the orchestrator holds Part B there):

- **`[07-location-on-an-identically-zero-rate]` — the stationary-point location treats every scan sample of a
  phase rate that is identically zero as a root.**
  - **Where.** `_locate_stationary_points()` (`AdaptiveLevin/levin_quadrature.py` at `ad732887`), step 2:
    every finite sample with `d == 0.0` becomes a `root`. A `q-r` group with `q = r` has
    `theta = theta_q - theta_r` and `theta_deriv` exactly 0.0 everywhere.
  - **What it does (R28, 26 calls; `log.md`, `out/replay_summary.txt`).** Each such call locates 4,097
    points, every sample of the scan grid: 4,095 `root` and 2 `end`. It returns 4,102–5,505 Clenshaw–Curtis
    window pieces, against 1–5 Levin regions at the parent. Summed regions in R28 go from 114,826 to
    227,883, almost all from these 26 calls. One such call takes about 110 s in the replay, against about
    5 s for a G+q+r call. The calls are converged and phase-limited.
  - **What it does not do: it does not make R28 dishonest** (`out/qr_scores.txt`). Three of the 26 calls were
    referenced by direct quadrature, the worst included, and R28 is honest on all three:
    |v28 − ref|/abserr28 = 0.179, 0, 0.069. So is R27 (0.0021, 0.0014, 0.0006), whose declared errors are
    about 10^6 times larger. On the worst call, R28's verdict rests on the Gauss–Legendre sequence
    (m = 8–14 agree to 6e-39), not on QAGS (`verification.md`).
  - **The prompt.** §1.1 does not consider a rate that is identically zero. Whether such a phase should be
    located at all, or whether the group should not be handed to a Levin call, is the user's question.
  - **Next step.** The user's decision (the orchestrator's hold, item 4). How many of A3 v2's calls are of this
    kind, and what they cost in the census (§6), was not measured here.

**For the record, not opened:**
- **05's `classify_non_qsi.py` and job names** (06m's observation 3, again). S0 compares R26
  (`levin-phase-floor-06m`) with R27 (`levin-phase-floor-07`); the labels were normalised in a copy, as 06m did.
- **The registry's manifest names the script's repository.** `p1-references`' launcher lives under the main
  checkout's `var/`, so the manifest's `git_head` is the main checkout's 28ebaf07, though the run's cwd and
  the tool it ran are the code worktree's (74600350). The outputs say which tree; nothing is wrong in
  RunRegistryKit (it records the script's tree, as documented).
- **QAGS on the worst q-r call** is 4.9 times its own error estimate away from the Gauss–Legendre sequence
  (`out/qr_worst_call_detail.txt`). It is a property of the reference's second rule, not of the code measured.
