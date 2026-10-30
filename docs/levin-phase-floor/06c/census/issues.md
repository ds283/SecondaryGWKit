# levin-phase-floor 06c §5 (the census): observations to open or amend on the board

Each observation below gives its numbers and where they are recorded. The paths are relative to
this directory (`06c/census/` once committed). The run is
`levin-phase-floor-06c-census-20261029T130214`; the "before" is 06b's
`levin-phase-floor-06b-census-20261028T161212`. The orchestrator decides whether each is a new
issue or an amendment.

## C1 — X1 fails on one call: an error shared by a leaf and its descendants

**Measured** (`out/cmd/42`–`44`; `out/x1_reference.json`):
- **The failure.** `g-1e-7-0-14127` call 3 (G−q+r, low, `0:202`) is 1.375× its declared error. g
  has 7 leaves, all trusted or floor-limited. The leaf $[14.72, 28.12]$ at depth 1 ($r = 0.079$)
  declares $1.74\times10^{-37}$ against a true error of $2.39\times10^{-37}$. 06b's abserr for this
  call was $+\infty$.
- **The reference is sound.** $S_{10}$, $S_{11}$ and $S_{12}$ agree to $2.6\times10^{-5}$ of
  `abserr`$_g$. The adaptive driver at `rtol` $10^{-11}$ returns g's value to 16 digits.
- **The other 22 calls left in the rule** sit at 0.002–0.99 of their declared errors. One more call
  was dropped by the prompt's agreement test.
- **Against its own request** the failing call is well inside, at $1.3\times10^{-4}$ of it.

**Impact:** this is `[06b-moved-calls-declare-less-than-their-error]` surviving the global budget
and the trust check. Prompt 06c §8 says the check cannot reach this case. The rate is now 1 in 23
sampled calls; 06b's W2 found 17 in 24.

**Suggested:** narrow `[06b-moved-calls-declare-less-than-their-error]` with these numbers.

## C2 — X2 fails: 53 and 127 budget misses do not converge

**Measured** (`out/cmd/44`): of 06b's X2-type calls outside the 720 rows (20,425 at 1e-7, 22,442 at
1e-8), these do not converge under g:
- **at 1e-7, 53:** 49 end at depth 20 with an untrusted leaf, 2 at depth 20 with every leaf
  trusted, and 2 floor-limited;
- **at 1e-8, 127:** 63, 27 and 37 respectively.

The rest (99.7 %, 99.4 %) converge. Every unconverged call is honest in kind: it declares that it
could not meet the request. X2's rule asks for convergence, not for honesty.

**Impact:** the budget does make 06b's misses converge, except where the loop chases a leaf to the
cap. This bears on C4 and on `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.

## C3 — X4 fails, and on the calls checked the fault is 06b's declared error

**Measured** (`out/cmd/44`, `47`, `49`; `out/x4_reference.json`):
- **The violations.** 263 of 8,092 rows at 1e-7 and 302 at 1e-8 violate X4, by up to 29.9× and
  71.5×. They are mostly seam (179, 190) and mid (80, 106) rows.
- **The calls behind them.** In every violating row, some group call (mostly regime G, group G) has
  $|v_g - v_{06b}| > e_g + e_{06b}$, and g reports that call converged.
- **The four worst rows' most-violating calls**, referenced as X1 is:
  - g's error is 0.11–0.61 of `abserr`$_g$;
  - 06b's error is 116–445 of `abserr`$_{06b}$.

**Impact:** X4 assumes both runs declare honestly. On the calls checked, 06b's declared error
understates its error by two orders, which is `[06b-moved-calls-declare-less-than-their-error]` seen
at row level. The rows were not referenced beyond these four calls.

## C4 — X5 fails: depth-20 calls, flat residuals, and the cost

**Measured** (`out/cmd/44`, `50`; `out/deep_lines_06c.json`):
- **The cap.** 109 moved calls at 1e-7 and 153 at 1e-8 reach depth 20: 14 and 12 group calls, the
  rest analytic blocks. 8 and 7 of them are still converged. The largest is `g-1e-8-0-38939`
  call 5 (J1): 74,941 regions, 277,395 solves.
- **The deepest lines.**
  - Each was recomputed after the run, and every recomputed line reproduces its leaf bit for bit.
  - 217 of 262 have a residual that falls by less than 10× over the last five levels.
  - $(e_L + e_R)/e_P$ is often exactly 2 from depth 4–5 on: bisection does not reduce the error,
    and often doubles it.
- **The cost.**
  - Solves rise ×8.2 (1e-7) and ×8.4 (1e-8) over 06b. Wall time is 2.5 h against 51 min.
  - The trust check's quarters and fallback halves account for 55–86 % of all solves.
  - 06b's three depth-20 calls now converge at depth 1–5, with 15–71 solves.

**Impact:** this bears on G2's L2 (the prompt), and on
`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`, whose re-measurement after 07 these
lines are the evidence for.

## C5 — the census budget is not enforced inside a long Levin call

**Measured** (`out/cmd/33`, `44`; log deviation 22): nine units ran 314–1,185 s against the 300 s
budget and finished `done`. 06b's dispatcher (from 04's I9) raises only at the next
`PrimitivePhase.theta_mod_2pi` entry past the deadline, and these units' long Levin calls made none
until they finished. No verdict here depends on it: their full runs are under 06a's 3,600 s pass,
and X5 fails regardless.

**Impact:** 07's census inherits this harness. A unit's budget can be overrun by about 4× without
the unit being marked over budget.

**Suggested:** record it against the harness, for 07's §6.

## D1 — (preparation) X1's reference might drop many calls

This did not happen: only one of 24 calls was dropped. The coordinator gave no ruling, and X1 was
scored as written. Nothing to open.
