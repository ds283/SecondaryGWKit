# levin-phase-floor 06h §5 (the census): observations to open or amend on the board

Paths are relative to this directory (`06h/census/` once committed). The census run is
`levin-phase-floor-06h-census-20261101T005037`, and the "before" is 06e's
`levin-phase-floor-06e-census-20261030T081318`. The orchestrator decides whether each one is a new
issue or an amendment.

## D1 — H4 fails at 1e-8: the larger declaration trips the round-off exit's premise

**Measured** (`out/cmd/46`, `47`, `48`): `g-1e-8-3-1072` call 13, "analytic J1", group (+,+,+).
Row `3-1072`: k = 1e5, q = r = 3e8 /Mpc, z_response 1.21e9, band seam, outside the 720.
- **Under 06e:** converged.
  - The premise check failed at the root (2 checks, 1 failed), so the root was split once.
  - Two Levin leaves at depth 1, with floors 4.5e-27 each.
  - abserr 7.16e-27 against a request of 1.63e-26.
- **Under h:** unconverged.
  - The premise check held at the root (1 check, 0 failed), so there were no splits.
  - One floor-limited leaf at depth 0, with `phase_err` 2.36e-26: the eq. (151) floor of 1.94e-26
    plus the declared term of 4.14e-27.
  - The declared term is almost all abscissa (4.136e-27). The constituents' part is 1.9e-30.
- **Effect:** the row's `total` is unaffected. The analytic oracle's value is bit-identical, and the
  oracle is unconverged under both codes (abserr 4.5e-19 against 5.6e-18).

**Impact:** the declared term stayed a quarter of the request, so §0.2's magnitude estimate holds.
But on a span whose eq. (151) floor already sits at or above the request, the larger floor flipped
`_roundoff_exit_premise_holds` from failing to holding. That stopped a bisection which under 06e
reached a floor below the request. H4 says so: "§0.2's estimate of what the declaration costs the
driver is wrong", in this narrow way. It is the only lost call among 2 × 129,621 scored.

**Not recorded:** the halves' floors of this root under $h$ (W3 runs at 1e-7 only). That they summed
to at least ½ the root's floor is inferred from the premise counters.

**Suggested:** amend `[07-roundoff-exit-assumes-bisection-cannot-help]`, or the
`[06b-premise-check-…]` family, with this call: the premise can be satisfied by a declared,
non-bisectable term while the bisectable eq. (151) part would have fallen. Or open
`[06h-declaration-can-trip-the-roundoff-exit-premise]`.

## D2 — A3 v2 reaches the near-singular class; nothing moved there

**Measured** (`out/cmd/49`, `50`): every row has b = 0. 435 rows have
$\min|K|/(k + q c_s + r c_s) \le 10^{-3}$ (Part B's rule):
- all are in the (+,+,−) group with q = r ≫ k, so K = k exactly: rel 2.89e-4 on 145 rows and
  9.05e-4 on 290;
- 301 of them have analytic Levin calls.

The oracle moved by at most 5e-14 of the second reading's bound there, and no call's convergence
changed other than D1's (whose row is one of the 145, though its group is (+,+,+)). The oracle is
**unconverged in 302 of these rows under both codes**.

**Suggested:** none for 06h. Whether the unconverged oracle on the q = r ≫ k rows matters belongs to
the analytic oracle's own record, if anywhere.

## D3 — the declared term now dominates more Clenshaw–Curtis floors, well below the request

**Measured** (`out/cmd/46`): on the analytic calls' floor-attributed fallback leaves, the
declared-dominated calls went from 2,493 to 5,137 (1e-7) and from 2,785 to 5,939 (1e-8). D/request
is at most 0.034 (1e-7) and 0.341 (1e-8); 0 and 2 calls are at ≥ 0.1. This bears on
`[06e-cc-declared-phase-term-is-a-crude-proxy]`: the proxy now multiplies a larger declaration.

**Suggested:** amend `[06e-cc-declared-phase-term-is-a-crude-proxy]` with these counts. No new issue.
