# The orchestrator's addendum to Part A' (received 2026-11-24, in this session; quoted in substance)

The user: "For now I'm happy to do a short run to settle the near-edge leaves." The user doubts that 8, 10 and 12 pi are
genuinely safer than 6, 16 and 20 pi, and suspects the difference comes from which references were looked at. Item B
bears on choosing between V93T8 and V93T10.

Rules: write only under var/levin-phase-floor-07d/measurement-a2/addendum/ (its own scripts/, state/, out/); record this
message in state/ and the addendum in log.md as its own section; every other A'.2 rule holds (--cpus 6, no seventh
process, registered and detached over ten minutes, every selection and rule saved before any reading); recommend nothing.

The orchestrator's own checks at the stop: an independent reference (32-point Gauss-Legendre, 32 to 1,024 panels a
piece, split at the knots, stated error |I_1024 - I_512| + 2 sigma_hat sqrt(sum w^2)) on the 31 leaves whose check at
T in {6, 8, 10, 12} pi under the CC_93 fallback is trusted, undecided on v4, with D < |Q - ref_v4|: none understates;
25 checks become bounded, 24 stay undecided. Several sit at the edge (D/|Q - ref| near 1): band_10_12|group|1e-7#7 and
band_10_12|group|1e-8#24 at 6, 8, 10 pi (D 6.73e-41 against 6.76e-41, eps 8.9e-42); band_16_20|group|1e-8#32 at 10, 12
pi (4.03e-42 against 3.98e-42, eps 3.0e-43); band_6_8|group|1e-7#40 and band_6_8|group|1e-8#59 at 6 pi (3.9e-45 against
3.9e-45, eps 1.5e-46); band_6_8|group|1e-8#8 at 6 pi (1.21e-42 against 1.18e-42). The second reference was taken only on
leaves v4 called understating.

A. Settle the near-edge leaves.
 1. Selection, fixed in addendum/state/ before any reading: every trusted Levin-side check at T in {6, 8, 10, 12} pi,
    under both the CC_93 and the CC_47 fallback, undecided on v4; and every undecided CC-side pair93 and tail93 verdict
    at those T; returned and visited leaves alike, kept apart. Report the counts.
 2. First pass: the second rule (48-point Gauss-Legendre) taken further, to 1,024 panels a piece, or 2,048 where the last
    doubling still moves the value. Record per leaf which term dominates its stated error: the last doubling's change or
    the noise term 2 sigma_hat sqrt(sum w^2).
 3. Second pass, only where the noise term decides the verdict: a reference of the smooth integrand, below the noise of
    theta_mod_2pi: the smooth-phase reconstruction of tools/filon_reference.py (Segment, 07's V4 method, as the call
    references read it), on the leaf's span. Project its cost first; over two hours in all: stop and report the
    projection to the orchestrator.
 4. Report per leaf and T: D, the error on each reference with its stated error, the verdict; by T and route, the counts
    bounded / understate / undecided. A leaf undecided because the integrand's own noise exceeds |D - te|: "at the noise
    floor", with both numbers; no forced verdict.

B. Dissect V93T12's new dishonest call R:a:W2-0-12977-5 (2.79 under V93T12, 0.0485 at the tree).
 1. Compare its V93T12 replay with V93T10's and V93T8's region by region: which regions differ, their V, route, declared
    error, and true error where cheap (v4's rule or the 48-point rule, on the differing leaves only).
 2. What carries the understatement: a CC_93 leaf with V in [10 pi, 12 pi) whose pair93 understates, a change in a trust
    check's sub-solves, or something else.
 3. Whether the same mechanism is present but below budget under V93T10 or V93T8, with each one's ratio of declared to
    true error on the leaves concerned.

Stop when A and B are done or at any stop condition; report the tables, what each found, anything left undecided with
the reason; recommend nothing.
