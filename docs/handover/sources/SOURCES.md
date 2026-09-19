# Primary sources for the general-$b$ oracle

Fetched 2026-09-19 from arXiv. **These are third-party papers, retained so that a transcription
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
for id in 1912.05583 2109.01398; do
  curl -sSL -o "${id}.pdf"        "https://arxiv.org/pdf/${id}"
  curl -sSL -o "${id}-src.tar.gz" "https://arxiv.org/e-print/${id}"
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
