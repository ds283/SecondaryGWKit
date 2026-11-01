### The phase re-measurement (§5, H1 and H2), on the Mac Studio, 2026-11-01

06f's M2 items 1 and 4 were repeated at 06h's code on 06f's 20 M2 calls.
- **Setting:** Ray-free, with no lock, on 06's certified scratch copy of A3 v2, read-only.
- **The trees:** $h$ = `c1ec3b2`, whose code is `7172a8f` + `ac1c6f4`, and the parent $p$ = `2c325f5`, whose code is 06e's. Each is a clean worktree.
- **The run:** `levin-phase-floor-06h-phase-20261101T033514` finished done, 84 of 84 units, exit 0.
- **The records:** [`06h/phase/`](06h/phase/). The log is [`06h/phase/log.md`](06h/phase/log.md).

**The captures reproduce the censuses** ([`out/cmd/30`](06h/phase/out/cmd/30_capture_p.txt), [`31`](06h/phase/out/cmd/31_capture_h.txt)).
- Each call was captured afresh at both trees, as `x1_capture.py` captures.
- 20 of 20 at $p$ equal 06e's census checkpoint (`a0dc4d7f…`) bit for bit, and 20 of 20 at $h$ equal 06h's (`3104c18e…`), on value, abserr, regions, maximum depth, converged and label.
- All 40 replays hold 06f's F0 against their own census.
- On every call:
  - both trees return the same partition;
  - the spans, groups and dense samples are equal.

**The reference** is each constituent's phase at $m_i e^u$, at the exact $e^u$ (`mpmath.exp` of the double $u$), at 30 digits, composed with the group's signs.
- **The method check passes** ([`out/cmd/32`](06h/phase/out/cmd/32_method_check.txt)). It ran on $h$'s capture of `g-1e-7-0-12689` #19, after the captures and before any scoring:
  - the ν = 1/2 phase-pair error is $9.1\times10^{-18}$ and $5.9\times10^{-17}$, against the documented $1.11\times10^{-16}$;
  - the group's $\delta\theta$ at the exact $e^u$ is the same at 30 and 50 digits;
  - atan2(J, −Y) and arg $H^{(1)}$ + π/2 agree to $\le 10^{-31}$.
- **The 50-digit check** ([`out/cmd/33`](06h/phase/out/cmd/33_refcheck_50_digits.txt), 06f's review method, Hankel formula), at $h$'s worst point and four others: $\delta\theta$ agrees with the stored value to a relative 0 at the worst point, and to $\le 5.5\times10^{-15}$ at the others.
- **The parent's replay reproduces 06f's M2 item 1 bit for bit** under 06f's $\hat x$ convention: 49 of 1,177 nodes violate, worst 148. At the exact $e^u$, the node 06f's review put at 246 gives 245.76.

| | Rule (prompt §5) | Number | Verdict |
|---|---|---|---|
| H1 | $\lvert\delta\theta(u)\rvert \le \texttt{theta\_abserr}_h(u)$ at every point sampled, on all 20 calls | **21,177 points:**<br>– 1,177 nodes, every node of every returned leaf;<br>– 20,000 dense points, 06f's seed.<br>**0 violate.** Worst ratio **0.493**, on #19's CC leaf 1, where the constituents' term dominates.<br>**The parent on the same points:** 561 violate, worst **376**. | **holds** |
| H2 | the first-order response to the measured phase error ≤ the leaf's declared term, on every returned leaf | **27 leaves.**<br>**0 violate.** Worst **0.268** (#19, Levin leaf 3); median $4.5\times10^{-5}$.<br>**The parent:** 2 violate, worst **41.6** on the same leaf. | **holds** |

**Descriptive** ([`out/cmd/43`](06h/phase/out/cmd/43_analyse.txt)). Measured over declared at $h$'s points, split by the dominant part of the declaration:
- **Constituents:** 20,446 points, median $6.8\times10^{-4}$, maximum 0.493.
- **Abscissa:** 731 points, median 0.134, maximum 0.459.
- **$\phi_{\rm eval}$:** dominant at no point.
- $\varepsilon_{\exp}$ is `EXP_RELERR` = $2.23\times10^{-16}$.
- On the tail Levin leaves of #19 the declared term rose 13× and 121×; the call's $D$ moved 0.9 %.

**The store.** The scratch copy is EQUAL to its certification before and after ([`11`](06h/phase/out/cmd/11_copy_files_before.txt)/[`12`](06h/phase/out/cmd/12_copy_vs_certified.txt), [`40`](06h/phase/out/cmd/40_copy_files_after.txt)/[`41`](06h/phase/out/cmd/41_copy_after_vs_certified.txt)).
- Nothing was created under `var/datastores/`.
- No Ray was started, and the lock was free throughout.
