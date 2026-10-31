# levin-phase-floor 06f: observations proposed for the board

Paths are relative to this directory (`06f/` once it is committed). The orchestrator decides for
each whether it becomes a new issue or an amendment.

## I1 — `BesselPhaseGroup.theta_abserr` does not declare the rounding of the product $K\eta$ (F4)

**Measured** (`out/m2_calls.csv`, `out/cmd/44_f4_attribution.txt`):
- On 2 of the 20 M2 calls, 49 of 1,177 sampled nodes have $|\delta\theta| > \texttt{theta\_abserr}$:
  - `g-1e-7-3-1054` #20 (Y4): 4 nodes, up to 1.83×;
  - `g-1e-7-0-12689` #19 (J4): 45 nodes, up to 148×.
- 4 of the 49 are Levin-leaf endpoints, where the rule evaluates θ.
- A dense seeded sample (1,000 points per call) finds 473 of 20,000 points above the declaration,
  up to 166×.
- Every violation lies in the tail, where `theta_abserr` is the constituents' $\sim3\times10^{-15}$
  and $|K\eta|$ is between 65 and $8.1\times10^{3}$.
- The rounding of $K\eta$ explains each one to within 0.27 `theta_abserr`. $K$ is exact on both
  calls.
- The class docstring names this "eps |K| x product floor" as a genuine limit, but `theta_abserr`
  sums only the constituents' declared errors, so it is never declared.
- It reaches a value: on `g-1e-7-0-12689` #19, leaf 3 (Levin), the first-order response to the
  measured phase error is $1.7\times10^{-25}$, against that leaf's declared term of
  $1.9\times10^{-26}$. The call's total declared term, $2.1\times10^{-20}$, which its
  Clenshaw–Curtis leaf sets, covers it.

**Impact:** the declared error can understate the phase's real error on leaves where
$\varepsilon|K|\eta$ exceeds the constituents' tail declarations. Here it was covered at the call
level. That was not measured beyond these 20 calls.

**Suggested:** open, for the user to assign. It concerns how `theta_abserr` is constructed, which
is outside 06f (§0.3).

## I2 — narrow `[06e-node-errors-add-in-absolute-value]` with M1

**Measured** (`out/results.md` M1, `out/m1_calls.csv`):
- On Z2's call, $\sum|E|/|\sum E| = 54$. The signed-combination error is $1.76\times10^{-34}$, 0.26
  of the request, so the call would meet its request on its returned partition (F1, F2 hold).
- Combining within sibling pairs leaves $1.11\times10^{-33}$.
- The signs come in runs ($z=-25$).
- Over the 11 sampled calls with $N_{\rm CC}\ge100$ the median ratio is 20.8 (F3 holds).
- On every sampled call but Z2's, the combination moves the declared error by under 1 %.

**Suggested:** narrow the issue, and keep it open for the remedy, which belongs to the user and
06g/07.

## I3 — narrow `[06e-cc-declared-phase-term-is-a-crude-proxy]` with M2, and correct its impact

**Measured** (`out/results.md` M2, `out/cmd/45_declared_vs_request.txt`):
- $D/\max_s|\delta I|$ has median 6.0 (F5 fails).
- The per-leaf combination contributes nothing on these 1–4-leaf calls: the median is 1.0 (F6
  fails).
- The proxy over the honest $\sum|w||f|$ is a median 1.05.
- The declaration exceeds the error actually committed by about 2,000× in the median.
- Across 06e's census, the CC declared term reaches the request on 0 of 11,279 calls with a
  non-zero term, and 0.1 of it on one.

**Impact:** the board's impact line ("may not reach a request its true error meets") was not
observed on 06e's census. Where the term dominates the floor, the floor is far below the request.

**Suggested:** narrow the issue, and record that its impact on convergence is nil on 06e's census.

## O1 — the copied 06e scripts are left as copied, not black-formatted

`scripts/` holds 06e's census scripts. Those 06f ran (`prov.py`, `x1_capture.py`, `worker.py` and
their imports) are byte-identical to 06e's apart from the diff in
`out/scripts_diff_against_06e_at_launch.txt`, and are unformatted, as 06e's own records are. The
seven scripts new in 06f are black-formatted.

**Suggested:** when committed, add `docs/levin-phase-floor/06f/scripts/` to `CLAUDE.md`'s list of
records left unformatted, or format only the new files.
