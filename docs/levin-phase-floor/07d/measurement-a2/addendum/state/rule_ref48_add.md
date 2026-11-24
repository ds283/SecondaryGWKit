# Addendum A.2 and A.3: the references and the verdicts, fixed before any is read (2026-11-24)

## A.2, the first pass: ref48 (the second rule taken further)
- Pieces: the leaf split at the knots v4 used (leafref_v4.call_knots, leafref_v4.interior); the whole leaf where none
  are known. The integrand: leafref_v4.Integrand (the captured call's own callables), as v4 and the second reference.
- Rule: composite 48-point Gauss-Legendre (numpy leggauss(48)), n uniform panels on every piece, summed by math.fsum.
- n = 512 and 1,024. c = |I_1024 - I_512|; noise = 2 sigma_hat sqrt(sum w^2 of the 1,024 rule) (sigma_hat the leaf
  record's v3 sigma_hat, as v4 and the second reference).
- "The last doubling still moves the value": c > noise. Then n = 2,048 as well: c = |I_2048 - I_1024|, noise from the
  2,048 rule. A leaf whose c still exceeds its noise at 2,048 is recorded "moving at 2,048"; its stated error is as below.
- ref48 = the finest value; eps48 = c + noise; the dominant term is "change" if c >= noise, else "noise".

## Verdicts (every reference)
- te = |value - ref|; bounded if D >= te + eps; understates if D < te - eps; else undecided (v4's rule, unchanged).

## A.3, the second pass: where the noise term decides
- "The noise term decides the verdict": undecided on ref48 with eps48, and decided with eps = c alone
  (|D - te48| > c). Only these leaves' checks go to the second pass; a leaf goes if any of its checks does.
- The reference: tools/filon_reference.reference on the captured call with x_span replaced by the leaf's span
  ([start, end], the leaf record's hex), route by route_of (A for an analytic Bessel block, B for a cosmological group),
  edges for route B the leaf's pieces as in A.2 (production_edges' walk, restricted), workers 1. Four readings (FCC and
  QAWO at resolutions 1 and 2); the value is FCC at resolution 2 (the tool's choice); its stated error
  eps_s = the four readings' scatter + 2 x fit_noise_1sigma of the FCC resolution-2 reading (route B; 0 for route A).
- Its cost is projected (a timing probe on the first three leaves, in selection order) before it runs; over two hours
  in all: stop and report the projection.

## "At the noise floor" (A.4)
- A check whose final verdict (the last reference taken on its leaf) is undecided is "at the noise floor" when the
  integrand's own noise, s_47 = sigma_hat sqrt(sum w^2 of the leaf's 47-point Clenshaw-Curtis rule) (leaf record
  cc.c47.sum_w2), times 2, is at least |D - te|. Both numbers are reported. No verdict is forced.
