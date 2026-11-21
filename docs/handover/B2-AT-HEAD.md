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
- **Added 2026-11-21 (§7): the cost is one commit's, and on this cell it buys neither honesty nor
  $N$.** Against Filon references, `HEAD`'s four `G±q±r` calls at `together` $10^8$ are honest,
  at 0.06–0.49 times their declared errors. Over a ladder of the 13 code trees since B2,
  `levin-phase-floor` 06e's `f4d8e173` takes the cell from 1,085 regions to 25,639. The tree
  before it, 06d's, was honest on all four calls in 30 s, and its $N$ agrees with `HEAD`'s to
  $2.2\times10^{-13}$. What the cost buys is declared errors 50 times smaller at 06e and 100–330
  times smaller at `HEAD`, which still do not meet `rtol = 1e-12` on any of the four calls.

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

## 7. The references at $10^8$, and the commit that raised the cost (2026-11-21)

> Added on 2026-11-21, on the laptop, answering two of §6's bullets: `together` $10^8$ is now
> referenced, and its cost is attributed. Sections 0–6 are as written on 2026-11-20.

### 7.1 What was run

| | |
|---|---|
| tree | `handover-remedial` at `5c57be96` (clean), for the references and the scoring |
| host | the laptop, Python 3.13.16, Ray-free, no datastore, no lock. One-minute load averages 9–24 during the references (recorded per reading, and per cell in the ladder), so times are rough; region counts are deterministic |
| references | `scripts/ref_1e8.py`, with `scripts/build_1e8.py`: `tools/filon_reference`, route B, the four readings (QAWO and FCC at resolutions 1 and 2) of each `G±q±r` call of `R\|together\|1e8`, with 6 workers. Each call is rebuilt in each worker by running the cell with those calls cut to `depth_max = 1`; the integrand and phase do not depend on depth. Edges from `ref_gqr.py`'s tolerant copy of `production_edges`: 3,978 from 3 tables. 16 readings, 46–338 s each, 43 min in all |
| ladder | `scripts/run_commit_ladder.sh`: B2's `R` cells `together` $10^8$, `together` $10^7$ and `T-first` $10^7$, one at a time, on each of 13 trees: B2's `d88faec8` and the 12 code commits after it to `67b5af74` (07c). `89c368f2` changes docstrings only and was skipped. Each tree is `git archive` of everything but `docs/` and `prompts/`, in a session scratchpad, with `HEAD`'s harness added: `realistic_large_x.py`, B2's `b2_cells.py` (SHA-256 `0a8ae0eb…`) and the three `docs/radiation-oracle/` modules it imports, none changed since `d88faec8`. `ladder/<sha>/import_check.txt` shows each tree imported its own `levin_quadrature.py` and `QuadSourceIntegral.py`, matching the commit's blobs. 39 of 39 cells `ok` |
| scoring | `scripts/score_1e8.py`, output `reference-1e8/score.md` |

**Determinism.** `d88faec8`'s cell reproduces B2's checkpoint record of `R|together|1e8` exactly:
the same total, and every group's value and region count. `67b5af74`'s reproduces
`cells/R-together-1e+08.json` (§4, `HEAD`) the same way. So the ladder's trees compare like with
like, and `HEAD`'s row is 07c's.

**Kept in git.** `reference-1e8/readings-without-segments.jsonl` is the readings file with each
reading's per-segment fit diagnostics (`segments`, 3,977–3,978 entries a reading) removed and their
count kept. The full file was 23 MB, SHA-256 `a8728787…`. It is not kept: the per-reading `tmap`
summary it also holds (largest fit residual, all segments monotone, no direct segments) is kept,
and `ref_1e8.py` regenerates the file. The readings record `ref_1e8.py`'s and `build_1e8.py`'s
SHA-256, which match the committed files, and `score_1e8.py` was formatted with `black` after its
first run; its output is unchanged. `run_commit_ladder.sh` is as run on its second launch. The first
launch wrote no cell output, because its output path was relative while each cell runs inside its
tree; it was stopped after two minutes, and its partial directory was removed.

### 7.2 The references

| call | value (FCC, resolution 2) | scatter of the four readings / value | scatter / `HEAD`'s declared error |
|---|---|---|---|
| `G+q+r` | $+3.993365754008674\times10^{-28}$ | $1.3\times10^{-14}$ | 0.012 |
| `G+q-r` | $-5.851859547510000\times10^{-29}$ | $4.4\times10^{-14}$ | 0.015 |
| `G-q+r` | $-7.979593690725447\times10^{-30}$ | $5.5\times10^{-13}$ | 0.029 |
| `G-q-r` | $-3.886555916951814\times10^{-30}$ | $1.4\times10^{-12}$ | 0.026 |

The scatter is between the resolutions: at each resolution QAWO and FCC agree more closely. A call
is scored **honest** when $|{\rm value}-{\rm ref}|+{\rm scatter}\le$ its declared error, and
**dishonest** when $|{\rm value}-{\rm ref}|-{\rm scatter}>$ it. No call in the ladder is between
the two, so every verdict below holds whichever reading is the truest.

### 7.3 The ladder at `together` $10^8$

Per call, true error over declared error (`score.md` holds the regions, depth, declared and true
relative errors of each). No call on any tree is converged.

| tree | regions | depth | time | `G+q+r` | `G+q-r` | `G-q+r` | `G-q-r` | cell: true error / total | cell declared |
|---|---|---|---|---|---|---|---|---|---|
| `d88faec8` B2 | 286 | 13 | 5.5 s | 0.012 | **7.9** | 0.021 | 0.65 | $8.3\times10^{-8}$ | $3.9\times10^{-8}$ |
| `4ddc3c2c` 06b | 1,059 | 14 | 21 s | 0.064 | 0.24 | 0.19 | **21** | $-3.6\times10^{-12}$ | $4.7\times10^{-9}$ |
| `ad9a5f48` 06c | 6,651 | 18 | 226 s | 0.016 | 0.084 | 0.062 | 0.23 | $2.8\times10^{-14}$ | $4.7\times10^{-9}$ |
| `3d70db93` 06d | 1,085 | 13 | 30 s | 0.0002 | 0.0006 | 0.009 | 0.0014 | $2.5\times10^{-13}$ | $4.8\times10^{-9}$ |
| `f4d8e173` 06e | **25,639** | 20 | 665 s | 0.036 | 0.024 | 0.021 | 0.11 | $5.9\times10^{-14}$ | $4.7\times10^{-9}$ |
| `7172a8f5` 06h | 25,639 | 20 | 668 s | as 06e | | | | $5.9\times10^{-14}$ | $4.7\times10^{-9}$ |
| `3ae6f3a4` 06g | 20,220 | 20 | 527 s | 0.042 | 0.16 | 0.12 | 0.49 | $6.9\times10^{-14}$ | $4.7\times10^{-9}$ |
| `48cf87c0` 06m to `67b5af74` 07c (6 trees) | 20,131 | 20 | 475–489 s | 0.055 | 0.16 | 0.12 | 0.49 | $6.4\times10^{-14}$ | $4.7\times10^{-9}$ |

The cell's true error is the four calls' summed signed error against the references, in the cell's
units, over its total; the reference's own scatter is $1.3\times10^{-14}$ of the total.
$N_R+9/8$ is $-3.2500211\times10^{-4}$ in B2's tree, $-3.24908366\times10^{-4}$ in 06b's, and
$-3.249083699\ldots\times10^{-4}$ from 06c's on, where it varies by $2.6\times10^{-13}$ across the
11 trees.

The four calls' declared errors relative to their values:

| tree | `G+q+r` | `G+q-r` | `G-q+r` | `G-q-r` | regions in the four calls |
|---|---|---|---|---|---|
| `3d70db93` 06d | $1.1\times10^{-10}$ | $9.8\times10^{-10}$ | $6.4\times10^{-9}$ | $1.2\times10^{-8}$ | 1,047 |
| `f4d8e173` 06e | $2.3\times10^{-12}$ | $2.0\times10^{-11}$ | $1.1\times10^{-10}$ | $2.4\times10^{-10}$ | 25,603 |
| `67b5af74` 07c | $1.1\times10^{-12}$ | $3.0\times10^{-12}$ | $1.9\times10^{-11}$ | $5.3\times10^{-11}$ | 20,092 |

The true errors move far less: 06d's $2.5\times10^{-14}$, $6.1\times10^{-13}$,
$5.7\times10^{-11}$ and $1.7\times10^{-11}$ of the values, against 07c's $6.1\times10^{-14}$,
$4.8\times10^{-13}$, $2.3\times10^{-12}$ and $2.6\times10^{-11}$.

### 7.4 The `together` and `T-first` $10^7$ cells

Regions / depth / integral time; not referenced.

| tree | `together` $10^7$ | `T-first` $10^7$ |
|---|---|---|
| `d88faec8` B2 | 313 / 14 / 5.5 s | 114 / 9 / 2.0 s |
| `4ddc3c2c` 06b | 811 / 14 / 15 s | 230 / 11 / 2.9 s |
| `ad9a5f48` 06c | 4,366 / 16 / 152 s | 816 / 15 / 50 s |
| `3d70db93` 06d | 799 / 11 / 22 s | 356 / 11 / 11 s |
| `f4d8e173` 06e | 1,284 / 20 / 45 s | 414 / 16 / 15 s |
| `3ae6f3a4` 06g | 1,160 / 20 / 36 s | 440 / 16 / 17 s |
| `48cf87c0` 06m to `67b5af74` 07c | 1,205 / 20 / 38 s | 429–432 / 16 / 15–16 s |

06e raises these too, by 1.6× and 1.2×, and takes both to depth 20 and 16. The 161× growth of §4
is a property of the $10^8$ rung.

### 7.5 What it shows

- **`HEAD` is honest at `together` $10^8$.** Its four `G±q±r` calls are within 0.06–0.49 times
  their declared errors of the references, and the cell's true error is $1.4\times10^{-5}$ times
  its declared one. §4's open question is answered for this cell.
- **The cost is 06e's commit.** `f4d8e173`, "Pin Chebyshev end nodes and measure the node
  displacement", takes `R|together|1e8` from 1,085 regions at depth 13 to 25,639 at depth 20, and
  its time from 30 s to 665 s. 06g's `3ae6f3a4` and 06m's `48cf87c0` bring it down to 20,131,
  through `G+q+r` alone (6,180 → 757 → 669); the other three calls keep 06e's regions to `HEAD`.
  06h, and every commit after 06m's, leave this cell unchanged. 06c's `ad9a5f48` had raised it to
  6,651 before 06d's `3d70db93` brought it back to 1,085.
- **On this cell the cost buys neither honesty nor $N$.** 06d's tree was honest on all four calls.
  Its cell's true error, $2.5\times10^{-13}$ of the total, is $5\times10^{-5}$ times its declared
  $4.8\times10^{-9}$, and its $N$ agrees with `HEAD`'s to $2.2\times10^{-13}$. From 06e on, the
  driver spends 19–24 times the regions, and the four calls' declared errors fall 50-fold at 06e
  and 100–330-fold by `HEAD`. Their true errors stay of the same order, and none of the four
  reaches the harness's `rtol = 1e-12`. The three calls at depth 20 are subdividing to the cap for
  a per-call `rtol` they do not meet, on groups that are 0.3%, 0.6% and 4.3% of the cell total
  (`G+q+r`, at 29%, stops at depth 19).
- **What this cell cannot show** is whether 06e's change is needed elsewhere. 06e was made for
  its own measured reason, the displacement of the Chebyshev end nodes, on its own census; this
  is one cell of one fixture. Whether a call should go on subdividing to the depth cap for a
  per-call `rtol` it cannot reach is `levin-phase-floor` 08's question (G2's L2 as reopened, the
  depth cap, and Q5, `rtol`), and this cell is a measured case of it.
- **B2's own cell was outside its declared error.** In B2's tree `R|together|1e8` is
  $8.3\times10^{-8}$ of its total from the truth against a declared $3.9\times10^{-8}$, through one
  call, `G+q-r`, integrated in one region at 7.9 times its declared error. The effect on $N$ is
  $9.4\times10^{-8}$. In 06b's tree the cell is honest, $3.6\times10^{-12}$ of its total from the
  truth, though its `G-q-r` is 21 times its declared error. From 06c's on every call is honest.
  B2a's Part B re-runs B2's cells at `HEAD`, so no B2 number is carried forward from this tree.

### 7.6 What it means for B2a

Part B's estimate in §5 stands: at `HEAD` the `together` $10^8$ cells dominate it, at about 480 s
each on the laptop, and this record shows those cells are honest. A tree at 06d's cost would run
them in about 30 s, but 06d is not the code `handover` runs at, and the trade is 08's to decide.
