# The one-loop SIGW calculation, written up

`one-loop-sigw-derivation.tex` is a narrative derivation of the one-loop ($P_{22}$)
scalar-induced gravitational-wave spectrum in the form the pipeline implements: from the
metric and field equations, through the scalar transfer function, the sourced tensor equation,
the tensor Green's function and the source time integral, to the two-dimensional loop integral
that `OneLoopIntegral` is to compute. It was compiled on 2026-09-29 from the author's
handwritten notes (`MAIN` 10–14, `NUM` 01–11) via their signed-off transcriptions in
[`docs/spec/`](../spec/), and every displayed result that is not a definition is
machine-checked.

The document is the *derivation*; the spec files remain the *record of what is on the pages*.
Appendix C maps each equation to its spec result and page, and Appendix B lists the slips found
on the pages and how the author resolved them (`docs/spec/REVIEW-QUEUE.md`). Where the two
disagree, the spec files and the review queue are authoritative for what the notes say; this
document is authoritative only for the derivation it presents.

## Building

```bash
cd docs/derivation && latexmk -pdf one-loop-sigw-derivation.tex
```

MacPorts TeX Live (`/opt/local/bin/latexmk`) builds it with no warnings other than a few
paragraph-level overfull boxes of under 15 pt. Build products are ignored by the local
`.gitignore`; the PDF is not committed.

## Verification scripts

Run from the repository root with the project `venv`:

```bash
./venv/bin/python docs/derivation/scripts/einstein_tensor.py
./venv/bin/python docs/derivation/scripts/check_formulas.py
```

`einstein_tensor.py` expands the Einstein tensor of the metric in §1.3 to second order in the
scalar potential and first order in a transverse-traceless tensor mode. It confirms the
first-order equations of §2.1, the tensor operator of §3.1 (coefficient exactly ½) and the
second-order geometric source of §3.2 (coefficient exactly −2, no $\phi\,\partial\partial\phi$
term) — the one input that the transcribed specs do not derive, because `MAIN` 01–08 were not
transcribed.

`check_formulas.py` re-derives symbolically the redshift forms of the transfer-function and
Green's-function equations, both effective frequencies, the $a''/a$ identity, the polarisation
factors, the Bessel recurrences and the coefficient arithmetic ($8\pi^2$, $648\pi^2$), and checks
numerically the analytic transfer function, the retarded Green's function, the three forms of the
source kernel $f$, and the closed-form source time integral against direct quadrature.
