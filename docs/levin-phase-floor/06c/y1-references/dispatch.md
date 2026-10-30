# Y1 references: dispatch brief (Mac Studio orchestrator, 2026-10-29T17:12Z, host mps026298)

You are a fresh agent on the Mac Studio. The orchestrator sends this brief. The user asked for
this work through the laptop orchestrator. The user's words: "We probably also want to know which
code was correct on those Y1 rows from 06c Part B, if that can be done in parallel."

## Background

Prompt 06c of the `levin-phase-floor` campaign (SecondaryGWKit) ran two rehearsal builds of the
same scope:
- **R17** at tree `fef401c`, which is 06b's Levin code;
- **R18** at tree `e5764a6`, which is 06c's global error budget plus 06d's φ_CC floor.

Outside `QuadSourceIntegral` (QSI) every value is bit-identical between the two stores. Inside
QSI, the rule Y1 says |T18 − T17| ≤ e17 + e18, where T is a row's `total` and e its
`total_abserr`. **Y1 fails on 30 of the 838 rows scored.**

- **The list:** `var/levin-phase-floor-06c/review/checks/check4_y1.txt`, worst first.
- **The calls that move, per row:** `var/levin-phase-floor-06c/rehearsal/out/y1_violations_worst_calls.txt`
  and `.../rehearsal/out/y1_violations_groups.txt`.

**The twelve worst rows violate by 336–16,763×.** All are equilateral (k=q=r). In each, the
`G+q+r` group call moves by 0.03–1%, while both codes call it converged.

The question is which code is right. In at least one code, the call is converged and its declared
error is dishonest by 10³–10⁵×. These rows are also the leading candidate for the
"near-singular, converged-dishonest" class that 06d's floor was meant to address. That class has
never been measured.

## The task

**Scope.** For each of the 30 rows, compute an independent reference for every Levin call whose
result differs between R17 and R18. A call differs if its value, abserr, regions or depth is not
bit-identical. Then compute the reference for the row's QSI total.

- Do the twelve 336–16,763× equilateral rows first.
- Then do as many of the other 18 as is affordable. Say which you did not do, and why.

**Per row, report:**
- for each moved call: its label, its span, both codes' value and declared abserr in hex, the
  reference value, and the reference's own error estimate;
- for each code: |value − ref| / abserr, and whether the code is within its declared error;
- the row's total: T17, e17, T18, e18, the reference total and the verdict. The verdict is one of
  R17 within, R18 within, both, or neither.

**Also report:** whether these rows are 06d's near-singular class. For each `G+q+r` call:
- how close the phase derivative θ′ comes to zero on the span, against its scale;
- where on the span that happens;
- what the driver's regions look like there.

Give the evidence.

## What counts as a reference

**It must not be the Levin driver judging itself.** That driver is what is in question. That
rules out:
- `adaptive_levin_sincos` at tighter atol or rtol;
- equal-piece Levin sums (06b's W2, 06c's X1).

You may report those as a secondary comparison, labelled as such.

**Use an independent quadrature of the same double-precision integrand:** ∫ f(x)·sin θ(x) dx and
∫ f(x)·cos θ(x) dx, in whatever combination the driver forms. Look at `adaptive_levin_sincos` in
`AdaptiveLevin/levin_quadrature.py` for exactly what it integrates. Acceptable methods:
- composite Gauss–Legendre or Clenshaw–Curtis on the captured callables, with enough nodes per
  oscillation of θ and compensated summation (`math.fsum`);
- `mpmath` quadrature over the interpolants.

Count the oscillations, (θ(b) − θ(a)) / 2π, first, and size the grid from that count.

**Establish accuracy by convergence.** Take two resolutions, at least doubling the nodes per
panel or panels per oscillation. Their difference must lie well below the smaller of the two
codes' declared errors for that call: aim for 1/100 of it. If you cannot reach that, say so and
give what you reached.

**Prove that the reference integrates the function each driver saw.**

1. **Capture each moved call's inputs** in each tree: span, `f`, `theta` and keyword arguments.
   Wrap `adaptive_levin_sincos` while re-running that row's QSI in-process, the way the census
   harness does.
2. **Replay the captured call in its own tree.** The replay must reproduce that store's call
   value and abserr **bit for bit**, compared by `float.hex`. Use R17's values for `fef401c` and
   R18's for `e5764a6`. A call that does not reproduce bit for bit gets no reference: report it.
3. **Show that both trees' captured `f` and θ agree bit for bit** at the reference's own nodes,
   or at a dense sample of them. Then one reference serves both codes.
4. **Check the limits and the phase convention** against the code, and say how you checked them.

## Machinery to reuse (read; do not edit)

**Ray-free harness.** `var/levin-phase-floor-06c/census/scripts/harness.py` runs
`ComputeTargets.QuadSourceIntegral`'s `compute_QuadSource_integral` in-process, on a read-only
`datastorekit` `ShardedPool`, with Ray's `remote`, `put` and `wait` stood in. Its
`open_readonly_pool(primary, cluster)` opens a store by its primary `.sqlite`.

**Capture and replay examples:**
- `.../census/scripts/x1_capture.py` and `worker.py` (capture);
- `.../census/scripts/x1_reference.py`;
- the orchestrator's `var/levin-phase-floor-06c/review/checks/x1_ref_mine.py` (replay, and the
  equal-piece sums).

**Copy these into your own scripts directory and change them there.** The census and review
scripts are records: never edit them in place.

**How to read the stores:** `var/levin-phase-floor-06c/review/checks/y1_reader.py` keys the QSI
rows. It uses sqlite3 `mode=ro` over every shard, keyed by k, q, r (through
`wavenumber_exit_time` joined to `wavenumber.k_inv_Mpc`), by `redshift.z` and by
`tolerance.log10_tol`, all as hex.

**The stores.** Their primaries are `var/datastores/levin-phase-floor-06c-r17.sqlite` and
`...-r18.sqlite`, and each has `-shard0000..0003`.

**Do not open the originals through any `ShardedPool`.**

1. Before anything else, record the SHA-256 of all ten original files.
2. Copy R17's and R18's five files each into
   `var/levin-phase-floor-06c/y1-references/store_copies/`.
3. Point every pool at those copies.

At the end, re-hash the originals and the copies, and show that none of them changed. Plain
sqlite3 `mode=ro` reads of the originals are allowed, as `y1_reader.py` does. The copies are
about 150 MB, and you delete them at the end, after hashing.

## Trees and commands

- **R17's tree:** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c`, at `fef401c`.
- **R18's tree:** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6`, at `e5764a6`.
  Both are detached and clean.
- **The main checkout:** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `28ebaf0`. It holds the
  venv, `var/` and the stores. No tree code runs from it.

**Every Python command** is
`/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`, with
`PYTHONPATH=<that tree>[:<your scripts dir>]`.

**Records go in `var/levin-phase-floor-06c/y1-references/`:**
- `scripts/`;
- `out/` (every command's output, numbered);
- `log.md`;
- `results.md`, holding the table and conclusion;
- `results.json`, holding the per-call numbers in hex.

**Every output starts with:**
- the host;
- the UTC time;
- each tree's `git rev-parse HEAD` and `git status --short`;
- the script's SHA-256;
- the SHA-256 of every input.

## Binding rules

- **No git command that writes, in any tree. That includes fetch.** Nothing is committed.
- **No Ray.** The `ray` package may be imported, as production modules do, but Ray is never
  initialised. Never `ray stop`. Never touch the rehearsal lock.
- **No store writes.** Pools open only the copies, and read-only.
- **Never touch:**
  - A3 v2;
  - its scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`;
  - any run directory under `var/runs/`;
  - any earlier prompt's records;
  - the census and review records under `var/levin-phase-floor-06c/`.

  Read them only. Write only inside `y1-references/`.
- **At most 6 CPUs at once.** Single-process numpy is fine; use no more than six worker
  processes.
- **Wall time:** run anything over about two minutes in the background, with `run_in_background`,
  and wait on it once. **Never poll, never `pgrep -f`.**
  - Size the work before you launch it.
  - If any single computation would take longer than about 45 minutes, **stop before launching
    it**. Report the estimate, and what you would run, in your final message. Such a job has to
    be registered in the run registry, and the laptop orchestrator has to be told first.
- **Write no production code.** Your scripts are readers and quadratures.

## Deviations and stopping

**Log deviations.** Record every deviation from this brief in `log.md`. Classify each as
`STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.

**Stop and report verbatim if:**
- a replay fails to reproduce its store;
- the two trees' `f` or θ differ;
- the reference does not converge;
- anything else would force you outside these rules.

Do not work around any of these.

## Your final message

Your final message is read by the orchestrator, who will re-check your numbers. It holds:
- the per-row table: row, moved calls, the codes' |value − ref| / abserr, and the verdict;
- how you established each reference's accuracy;
- how you proved it integrates the driver's function;
- the near-singular evidence;
- which rows you did not do;
- every deviation;
- the paths of your records.
