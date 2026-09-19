# Primary sources for the general-$b$ oracle

**Versions are pinned: arXiv `1912.05583v3` (23 Feb 2020, latest of three) and `2109.01398v2` (5 Nov 2021, latest of two).** Always cite the version. An equation number alone is not a citation in this pair of papers — see **Version hazard** below.

Fetched 2026-09-19 from arXiv (the bare `arxiv.org/pdf/<id>` and `/e-print/<id>` serve the latest version, which is what is recorded here). **These are third-party papers, retained so that a transcription
is reproducible against the exact text it was made from** — `prompts/radiation-oracle` found three
errata in Kohri & Terada, one of which turned on the *rendering* of an equation rather than its
content (KT audit §2 erratum 1). The LaTeX source removes that failure mode entirely.

## What is committed, and what is not

**Committed:** the extracted LaTeX and bibliography — the text a transcription is actually made
from, small, diffable, and unambiguous where a rendered equation is not.

| File | SHA-256 |
|---|---|
| `1912.05583-src/inducedSGWBwarxiv_revised_2.tex` | `2e40edf6c54a5f518284797f1a41224a229096fd66c17c434495d678ed530cf3` |
| `1912.05583-src/inducedSGWBwarxiv_revised_2.bbl` | `f1f0dd0c1bd922b824e5a040971a1dd818b44382b3de1df241574f9d65f8815b` |
| `2109.01398-src/template_review.tex` | `7af45aef9ff0b6688ddfcf9959790bcb359f2a3d17650def80124ade8aeab0f7` |
| `2109.01398-src/template_review.bbl` | `8ca862eb0f3625ea6f581a7994581ec092d11bbacadb97d16f2e8c6bbe75725a` |

**Not committed** (`.gitignore` in this directory): the PDFs and the e-print tarballs. They are
large binaries and they are reproducible from arXiv. Their checksums are recorded here so that a
refetch can be verified rather than trusted.

| Artefact | arXiv | SHA-256 |
|---|---|---|
| `1912.05583.pdf` | [1912.05583](https://arxiv.org/abs/1912.05583) | `32e24ac6f939227a66321246ca9fe7221b82c78901acf4cc28858319a92a629d` |
| `1912.05583-src.tar.gz` | e-print | `f70db08768d27a9e486684f1d5c9d5cce404671990c54a520cecc5876cca6f5a` |
| `2109.01398.pdf` | [2109.01398](https://arxiv.org/abs/2109.01398) | `c095e844edf57a3a70d0faa2e93901824681afaeb949a42afb98c837917405a2` |
| `2109.01398-src.tar.gz` | e-print | `fa2be5e1d62ea2f5681fc136a193866b8810db86aabe5d299fcef1e5c33d764c` |

Restore them, from this directory:

```bash
for id_v in 1912.05583v3 2109.01398v2; do
  id="${id_v%v*}"
  curl -sSL -o "${id}.pdf"        "https://arxiv.org/pdf/${id_v}"
  curl -sSL -o "${id}-src.tar.gz" "https://arxiv.org/e-print/${id_v}"
done
shasum -a 256 -c CHECKSUMS
```

`CHECKSUMS` beside this file carries all eight lines in `shasum -c` format, so a refetch is
verified in one command. **A checksum mismatch on a `.tar.gz` is expected if arXiv has issued a new
version** — it is not necessarily corruption. Check the version against the committed `.tex`, whose
checksum above is the binding one; the equation numbering in this document was resolved against
*that* text.

The figures were deleted from the extracted trees before committing; re-extracting a tarball
restores them.

**The authoritative text for any transcription is the `.tex`**, not the PDF and not a rendered
HTML view:

- `1912.05583-src/inducedSGWBwarxiv_revised_2.tex` — G. Domènech, *Induced gravitational waves in
  a general cosmological background*, Int. J. Mod. Phys. D **29** (2020) 2050028.
- `2109.01398-src/template_review.tex` — G. Domènech, *Scalar Induced Gravitational Waves Review*,
  Universe **7** (2021) 398.

Figures were deleted from the extracted trees; the `.tex` and `.bbl` are kept. The two `.tar.gz`
are kept verbatim so the deletion is reversible.

## Version hazard — `2109.01398` (4.7) changed between v1 and v2

**`1912.05583` is stable**: its Green's function (2.23) and its exact kernel (3.1) carry
$\{Y_\beta{\cal I}^x_J - J_\beta{\cal I}^x_Y\}$, the correct order, in **all three** of its versions.
Only the review moved.

**The review's Green's function differs between its two arXiv versions, in sign and in a factor of
$k^2$.** Everything in this directory, and every citation in `prompts/handover` and in
[`DOMENECH-KERNEL-RECON.md`](../DOMENECH-KERNEL-RECON.md), is **v2**.

| | arXiv v1 (3 Sep 2021) | arXiv v2 (5 Nov 2021) — **pinned here** |
|---|---|---|
| (4.7) `eq:hgreen` | $\dfrac{k\pi}{2}\dfrac{(k\tilde\tau)^{b+3/2}}{(k\tau)^{b+1/2}}\big(J_{b+1/2}(k\tau)Y_{b+1/2}(k\tilde\tau)-J_{b+1/2}(k\tilde\tau)Y_{b+1/2}(k\tau)\big)$ | $\dfrac{\pi}{2k}\dfrac{(k\tilde\tau)^{b+3/2}}{(k\tau)^{b+1/2}}\big(J_{b+1/2}(k\tilde\tau)Y_{b+1/2}(k\tau)-J_{b+1/2}(k\tau)Y_{b+1/2}(k\tilde\tau)\big)$ |

**v1's (4.7) is $-k^2$ times v2's**, verified at $b \in \{0, 0.2, 0.5\}$ and several $k$. **v2 is the
correct one**: it gives $G \to +(\tau-\tilde\tau)$ just after the source, the causal normalisation,
and it agrees with `1912.05583v2` `eq:green2` and with `docs/spec/02-greens-function.md` R10. v1
fails that sign test and carries a spurious $k^2$.

**(4.10) `eq:Isimple` and (4.12) `eq:Isimple2` are byte-identical between v1 and v2.** So the v2
paper is internally inconsistent: (4.7) was corrected and the correction was not propagated. That
is the mechanism behind
[`DOMENECH-KERNEL-RECON.md`](../DOMENECH-KERNEL-RECON.md) §1's first finding.

v1, for checking a citation made against it (not committed, and not the reference for any work
here):

| Artefact | SHA-256 |
|---|---|
| `2109.01398v1.pdf` | `813d6160d5884a9efbc8d86a70315aa0d18668c13d1781b5696f9b6bda54b5f2` |
| `2109.01398v1-src.tar.gz` | `2cc361df0fddde03d1573d8abe0c1d2d5a8824b161e9416220371014d2610ddd` |

Fetch with the explicit version suffix, `https://arxiv.org/pdf/2109.01398v1` and
`https://arxiv.org/e-print/2109.01398v1`.

## Equation numbering

Equation numbers in the *published* numbering, resolved from the `.tex` by counting numbered
environments per section (comment-stripped; starred and `\nonumber`-only environments excluded).
**The `\label` is the reliable identifier — cite both.**

### 1912.05583 §3 "Semianalytical calculation"

| № | `\label` | What |
|---|---|---|
| (3.1) | `eq:kernel2` | **Exact** kernel: $I(x,u,v)=4^{\beta}\frac{3\pi}{2\alpha^3}\frac{1+w}{1+3w}\Gamma^2[\beta+2](uvwx)^{-\beta}\{Y_\beta(x){\cal I}^x_J-J_\beta(x){\cal I}^x_Y\}$ |
| (3.2) | — | Definition of ${\cal I}^x_{J,Y}$ as $\int_0^x$ |
| (3.3) | `eq:IJ` | ${\cal I}^\infty_J$ — the $x\to\infty$ coefficient, Ferrers $\mathsf{P}$ |
| (3.4) | `eq:IY` | ${\cal I}^\infty_Y$ — the $x\to\infty$ coefficient, Ferrers $\mathsf{Q}$ and off-cut ${\cal Q}$ |
| (3.5) | — | $y$, $Z^2$, $\tilde y$, $\tilde Z^2$ |
| (3.6) | `eq:kernel3` | $I(x\gg1)$ — **doubly** asymptotic (Bessels also expanded) |
| (3.7), (3.8) | `eq:IJJ`, `eq:IYY` | $I_J$, $I_Y$ — (3.3)/(3.4) with constants stripped into the prefactor |
| (3.23) | `eq:kernelradiation` | $\overline{I^2_{RD}}(x\gg1)$ — radiation oscillation average |

The paper calls **(3.6), (3.7), (3.8)** "the main result of this paper".

### 2109.01398 §4 "Analytical transfer functions"

| № | `\label` | What |
|---|---|---|
| (4.8) | `eq:x` | $x\equiv k\tau$ |
| (4.9) | `eq:fsimple` | the source $f(x,u,v)$ |
| (4.10) | `eq:Isimple` | **Exact** kernel, in $b$ and $c_s$: $I(x,u,v)=\pi4^{b}\Gamma^2[b+3/2]\frac{2b+3}{b+2}(c_s^2uvx)^{-b-1/2}(J_{b+1/2}(x){\cal I}_{Y}-Y_{b+1/2}(x){\cal I}_{J})$ |
| (4.11) | `eq:Isimpledef` | Definition of ${\cal I}_{J/Y}$ as $\int_0^x$, in $b$ and $c_s$ |
| (4.12) | `eq:Isimple2` | $I(x\gg1)$ — **doubly** asymptotic |
| (4.13) | `eq:y` | $y = 1-\frac{1-c_s^2(u-v)^2}{2c_s^2uv} = -1-\frac{1-c_s^2(u+v)^2}{2c_s^2uv}$ |
| (4.14) | `eq:kernelaverage` | $\overline{I^2(x,u,v)}$ — **the general-$b$ oscillation average** |
| (4.42) | `eq:kernelsuperhave2` | $\overline{I^2_{RD}(k\ll k_{\rm rh},\tau\gg\tau_{\rm rh})}$ — a *reheating-transition* limit, not the general-$b$ average |
