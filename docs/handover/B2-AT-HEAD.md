# B2's `R` and `E` ladders at `HEAD`, and one Filon reference (2026-11-20)

> **A laptop measurement by the orchestrator, outside any prompt and outside the run registry.**
> It was made to cost B2a ([`prompts/handover/14-self-consistent-realistic-t.md`](../../prompts/handover/14-self-consistent-realistic-t.md))
> and A4's arm P before `handover` resumes. Every file named below is under
> [`b2-at-head/`](b2-at-head/). It changes none of B2's numbers in
> [`SEAM-RESIDUE.md`](SEAM-RESIDUE.md).

**What the numbers support:**
- **At `HEAD`, B2's `R` cells reach the values B2's own probe found once the Levin driver's $T$
  phase derivative was made consistent.** At `together` 980, $N_R+9/8$ is $+3.4714\times10^{-3}$.
  B2's run gave $+3.623\times10^{-3}$, and its probe13 $+3.471\times10^{-3}$
  ([`SEAM-RESIDUE.md`](SEAM-RESIDUE.md) §4.7). At probe13's four places `HEAD` agrees with
  probe13's consistent values to $\le5\times10^{-9}$ in $N$. The $1.5\times10^{-4}$ that §4.7
  attributes to the fixture's mismatch is gone at `HEAD`, without the fixture being changed (§1).
- **Where the mismatch acts, `HEAD`'s declared errors do not cover it.** On `R|together|980`, a
  Filon reference puts the group calls `G+q+r` and `G-q+r` at $5.3\times10^3$ and
  $1.2\times10^3$ times their declared errors from the true integral, and `G+q+r` says it
  converged. With the derivative taken from the phase, the same calls agree with the reference
  to the last digit and to $0.012\times$ the declared error (§2). B2's driver was
  $1.9\times10^8$ and $2.9\times10^6$ times its declared errors from the reference on these calls.
- **The mismatch's effect on $N$ at `HEAD` is $4.5\times10^{-8}$** at `together` 980. That is
  small against every term `handover` attributes, but it is $2\times10^3$ times the declared error
  (§3).
- **Cost.** The `R` ladder takes 821 s of integral time at `HEAD` against B2's 68.5 s. The `E` ladder takes
  33.6 s against 6.0 s. `R|together|1e8` alone takes 481 s, in 20,131 regions at depth 20. B2
  had 286 regions there. **This is not the mismatch:** with the derivative consistent it is 20,032
  regions at depth 20, and the totals agree to $2.6\times10^{-14}$ (§4).

## 0. What was run

| | |
|---|---|
| tree | `handover-remedial` at `c1a91d73` (clean). Its code is `d3b973d7`'s, `levin-phase-floor` 07c's |
| host | the laptop, Python 3.13.16, Ray-free, no datastore, no lock. Load averages 4–13 during the run, recorded per cell, so wall and integral times are rough |
| harness | B2's `docs/handover/b2/scripts/b2_cells.py`, unchanged: SHA-256 `0a8ae0eb…`, the same as in every one of B2's checkpoint records. `REF_ATOL, REF_RTOL = 1e-45, 1e-12` |
| cells | B2's `R` and `E` objects on all 16 rungs (32 cells), one at a time, through `b2_cells.py --cell`. All 32 `ok` |
| B2's run | `var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl`, at `28ade78` (rebased as `d88faec`) |

**Scripts** (`b2-at-head/scripts/`). `run_ladder.sh` ran the 32 cells. It wrote into a
session scratchpad, and its outputs were copied to `cells/`. `summarise.py` compares them with
B2's checkpoint (`summary.md`). `ref_gqr.py` captures a cell's group calls and computes a Filon
reference for named ones (`tools/filon_reference.py`, route B), against `HEAD` and against B2's
checkpoint. Its output is in `reference/`. `cell_phase.py` runs a cell with probe13's "phase" arm,
and `run_phase.sh` ran it at `together` 980 and $10^8$ (`cells-phase/`). `compare_phase.py`
compares those with `cells/` (`cells-phase/comparison.txt`). `ref_gqr.py` and `summarise.py` were
formatted with `black` after they ran; nothing else changed. `ref_gqr.py` gained its arm argument
between its two runs; its "base" path is the first run's code.

**`ref_gqr.py`'s edges.** `tools/filon_reference.production_edges` reads tables the fixture's
objects do not all carry (`ph._leading._table`, `ph._spline`, `f._f._friction_F._table`,
`G._Gk.sin_amplitude._spline`). `ref_gqr.py` runs a copy that skips the missing ones: 973 edges
from 3 tables. Its four readings agree to $3.3\times10^{-28}$ on values of order $10^{-13}$.

## 1. The ladders against B2

`summary.md` holds the full table. The cell totals:
- **`E`**: every total moved by $\le2.8\times10^{-10}$ relative, and no group call moved beyond
  its errors except one at `q-smooth` $10^7$.
- **`R`**: totals moved by $3\times10^{-12}$–$5\times10^{-7}$ relative, except `together` 980
  ($1.3\times10^{-4}$). On `together` and `T-first`, 1–4 group calls per cell moved beyond the sum
  of the two declared errors, 22 of them by more than 100×. Every such call is one of
  `G±q±r`, the groups in which both $T$s are oscillatory. On `q-smooth` nothing moved beyond its
  errors except one call at $10^7$.

| place | $N_R+9/8$, B2 as run | B2's probe13, derivative from the phase | `HEAD` as run |
|---|---|---|---|
| `q-smooth` 980 | $-5.638\times10^{-5}$ | $-5.638\times10^{-5}$ | $-5.6384\times10^{-5}$ |
| `q-smooth` $10^7$ | $+7.46\times10^{-7}$ | $+7.08\times10^{-7}$ | $+7.128\times10^{-7}$ |
| `together` $10^7$ | $-1.0903\times10^{-3}$ | $-1.0897\times10^{-3}$ | $-1.0897\times10^{-3}$ |
| `together` 980 | $+3.623\times10^{-3}$ | $+3.471\times10^{-3}$ | $+3.4714\times10^{-3}$ |

So B2's driver turned the fixture's mismatch into up to $1.5\times10^{-4}$ in $N$. `HEAD`'s driver
subdivides further and recovers the true integral almost entirely. On `together` 980 its
`G+q+r` and `G-q+r` calls take 11 and 27 regions where B2's took one.

## 2. The Filon reference on `R|together|980`

The sub-interval $z\in(6.0696, 294.106)$, where $G$ and both $T$s are oscillatory. Two arms: as
run (the Levin $T$ derivative is $\omega(1+z)$, Liouville–Green), and "phase" (the derivative is
`phase.theta_deriv`, consistent with the phase the driver oscillates with). The integrand, and so
the reference, is the same in both arms: the two references agree to every digit. The B2 column is
B2's checkpoint value, run as run.

| call | reference | arm | `HEAD`: error / declared | regions | converged | B2: error / declared |
|---|---|---|---|---|---|---|
| `G+q+r` | $-4.17264753622703\times10^{-13}$ | as run | $5.33\times10^3$ (relative $4.9\times10^{-9}$) | 11 | yes | $1.87\times10^8$ (relative $4.2\times10^{-5}$), 1 region |
| | | phase | 0 | 22 | yes | |
| `G-q+r` | $-3.92789907767564\times10^{-14}$ | as run | $1.16\times10^3$ (relative $1.4\times10^{-9}$) | 27 | no | $2.88\times10^6$ (relative $3.0\times10^{-6}$), 1 region |
| | | phase | 0.0118 | 73 | no | |

With the derivative consistent, `HEAD` meets the harness's $10^{-12}$ honestly on both calls.
The dishonesty is the mismatch: the driver's error estimate cannot see a derivative that
disagrees with the phase, as `[13-the-tlg-term-depends-on-the-levin-subdivision]` says. It is not
the $\varepsilon x$ phase floor, about $10^{-13}$ at this $x$, nor a defect of the driver on a
consistent integrand.

## 3. The mismatch's size at `HEAD` (`cells-phase/comparison.txt`)

| cell | $N_R+9/8$ as run | phase | $\lvert\Delta{\rm total}/{\rm total}\rvert$ | `G±q±r` moved, in summed declared errors |
|---|---|---|---|---|
| `together` 980 | $+3.471375\times10^{-3}$ | $+3.471330\times10^{-3}$ | $4.0\times10^{-8}$ | 618–44,000 |
| `together` $10^8$ | $-3.249084\times10^{-4}$ | $-3.249084\times10^{-4}$ | $2.6\times10^{-14}$ | $\le0.09$ |

The groups `r` and `G±r` return identical values in both arms, in both cells. At $10^8$ the mismatch has no visible effect. That is consistent with the Liouville–Green
truncation falling as $x^{-4}$, but that explanation is not tested.

## 4. Cost

| | integral time, B2 | `HEAD` | ratio |
|---|---|---|---|
| `R` ladder (16 cells) | 68.5 s | 821.1 s | 12.0 |
| `E` ladder (16 cells) | 6.0 s | 33.6 s | 5.6 |

Growth of Levin regions from 980 to the top rung:

| shape (top rung) | `R` at `HEAD` | `R` in B2 | A4's arm P (`28ebaf0`) | `E` at `HEAD` | `E` in B2 |
|---|---|---|---|---|---|
| `together` ($10^8$) | 125 → 20,131 (161×) | 44 → 286 (6.5×) | 46 → 288 (6.26×) | 11 → 52 (4.7×) | 7 → 9 (1.3×) |
| `T-first` ($10^7$) | 49 → 432 (8.8×) | 34 → 114 (3.4×) | 34 → 112 (3.29×) | 12 → 48 (4.0×) | 7 → 13 (1.9×) |
| `q-smooth` ($10^7$) | 1,473 → 2,078 (1.4×) | 686 → 1,244 (1.8×) | 686 → 1,480 (2.16×, to $10^6$) | 11 → 45 (4.1×) | 17 → 30 (1.8×) |

**`R|together|1e8`.** 20,131 regions in 481 s; B2 took 286 in 4.5 s. Almost all of it is the
four `G±q±r` calls on $z\in(6.07, 3.01\times10^7)$:

| call | regions | max depth | phase-limited regions | regions at the deepest level | time |
|---|---|---|---|---|---|
| `G+q+r` | 669 | 19 | 239 | 6 | 39 s |
| `G+q-r` | 7,531 | 20 | 4,255 | 3,276 | 171 s |
| `G-q+r` | 5,599 | 20 | 3,151 | 2,448 | 125 s |
| `G-q-r` | 6,293 | 20 | 3,585 | 2,708 | 141 s |

None of the four is converged. With the derivative consistent the cell takes 20,032 regions at
depth 20, so the growth is the driver's on a consistent integrand at `rtol = 1e-12`, not the
fixture's mismatch. Whether those values are honest is **not measured**: no reference was taken
at $10^8$. The `E` ladder also grows 4–5× at `HEAD`, where it was nearly flat in B2 (1.3–1.9×).

## 5. What it means for B2a and A4's arm P

- **B2a's Part A** makes the fixture's $T$ self-consistent. On this record that removes the one
  thing that makes `HEAD`'s declared errors fail on this fixture at $10^{-12}$. Until Part A
  lands, a declared error on an `R`-like cell with oscillatory $T$s understates its true error by
  up to $5\times10^3$ at 980, and its effect on $N$ is $\le5\times10^{-8}$.
- **B2a's Part B** should cost about 6,000 s of cell time at `HEAD` on the laptop, an estimate
  from the ratios above applied to B2's 208 cells: roughly 1–1.5 h of wall time at two cells at
  once. The `together` $10^8$ cells dominate.
- **A4's arm P** took nearly the same regions at A4's tree as B2's `R` did at B2's (the two
  columns above). This record re-measures the `R` ladder at `HEAD`, in 821 s, which is what a
  re-run of arm P was for.

## 6. What this does not cover

- One cell was referenced, and two of its calls; `together` $10^8$ was not.
- The `E` cells were not referenced. Their totals are of order $10^{-13}$–$10^{-15}$ in $N$.
- Every number is at the harness's `REF_RTOL = 1e-12`, not production's tolerance pair.
- The cost growth since A4 is not attributed to any one `levin-phase-floor` prompt.
