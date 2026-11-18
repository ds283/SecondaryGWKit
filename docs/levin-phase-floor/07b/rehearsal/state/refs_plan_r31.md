# The references for "which store understates" (R31 rehearsal agent, written 2026-11-16 ~20:10Z, before any reading)

Descriptive, not a prediction and not scored against any rule (the user's word at stop 1, relayed: "Yes, continue, and
settle which store understates").

## The calls

The 7 QuadSourceIntegral group records whose |v31 - v30| > e30 + e31 (exactly, in Fractions), all rtol 1e-7, Route B
(cosmological group phase), each captured from R31's store by the R31 replay harness (run
levin-phase-floor-07b-r31-replay-20261116T195034, stage refcap; out/refcap.json; pickles state/inputs/ref-*.pkl):
- checked bit for bit against R31 (refcapcheck; and re-run from the pickle at the tree, out/refs/refcap_replay_h.json);
- checked bit for bit against R30: the pickle's call re-run with 07a's module (git show 1ec734d7, 4ea1948b..., R30's
  tree's) installed by Part A's variant_loader.py gives R30's stored value, abserr, regions and converged
  (out/refs/refcap_replay_a07.json). So the one captured integrand is both stores' integrand.

An eighth capture is available: the record whose converged changed (G+q+r at (1e5, 1e6, 1e6), z 144627.7; |dv| is 0.37
of e30 + e31). Optional; about 100 s more.

**No reference Part A took serves:** Part A's references are A3 v2 census calls (k grid 1e5, 3.14e5, 9.85e5, ...); the
rehearsal's k grid is 1e5, 3.162e5, 1e6, and every one of the seven has q or r (or k) at 3.162e5 or 1e6, which no Part A
call has (grep of the selection's and calls' notify_labels).

## The method (07's V4 / 06g's, the tool's own four readings)

`python -B -m tools.filon_reference PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6 --out JSON`, the
tree's tool (tools/filon_reference.py d37e633b..., the tool 07's V4 and Part A read with), PYTHONPATH the tree then
the rehearsal's scripts/capture (for the pickles' harness names). Independent of the code under test: it imports nothing
of AdaptiveLevin's driver; Route B linearises the captured phase and integrates the captured amplitude by FCC (mpmath)
and QAWO (double), each at two resolutions.
- **Reference:** FCC r2. **Stated error:** the four readings' scatter (largest minus smallest).
- **T** = min(e30, e31)/100 in the call's own units (07a's W4 rule, Part A's "scored if stated error <= min abserr/100"):
  the tool's `counts` is then exactly "the reference is good enough to decide".
- Part A's split-edge FCC (`leafref_07b.py --split-ratio 1.8`) is not needed: every segment of every call is monotone and
  none has a slope ratio above 2 (n_nonmono 0, n_steep 0, D_f 0), so the tool's own FCC reads every segment in t.

## The decision, per record and store (fixed before the readings)

With ref, s (stated error), and each store's call value v and abserr e (from the bit-checked replays, in mpmath):
- the reference **does not decide** if s > min(e30, e31)/100: reported, no verdict for that record;
- a store **understates** if |v - ref| - e > s; is **honest** if e - |v - ref| > s; otherwise **undecided** for that store;
- the record's answer: R30, R31, both or neither understates (or undecided, said so).

## Cost

Projected by 07a's refitted model (Part A's deviation 16: with the gamma D term, which is 0 here), with its margins
(out/refs/refs_projection_r31.json): QAWO r1 + r2 + FCC r1 + r2 per call, one call at a time, six worker processes:
**the seven 842 s with margin (0.23 h), x1.5 for load 0.35 h; all eight 943 s, x1.5 0.39 h.** Every QAWO reading projects
under 30 minutes (largest 152 s).

## The run

One registered, detached run (slug r31-refs), Part A's run_06j.py driver (unchanged) via scripts/launch_r31.py: one
command per call, one at a time, outputs out/refs/readings/<sid>.json, the run backstop at 4 h (about 10 x the
projection). Ray-free (the pickles are loaded with ray.cloudpickle; Ray never initialised), no lock (no Ray), no store,
nothing written in a tree. Then a scoring script (out/refs/understates.json, .txt).
