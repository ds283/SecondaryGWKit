# 07c Part A — deviations, drafted as they occur (copied into log.md at the end)

1. **07b's census harness copied whole into `scripts/`** from `var/levin-phase-floor-07b/census/scripts/` (230 files,
   byte-identical: `out/scripts_07b_census_sha256_{source,as_copied}.txt`), all but its `rec.sh`; `rec.sh` is written
   fresh from 07b Part A's extension `scripts/ext/rec.sh` (paths only). 07b Part A's scripts are in
   `scripts/from_07b_measurement/` (276 files, byte-identical). `prov.py` is changed where marked "prompt 07c" (the tree,
   07b's census as read-only input). IMPLEMENTATION CHOICE.
2. **The copies were made with the shell's cwd in the records directory** (`cd` into it for `cp`), not the worktree, for
   one command. Nothing was written in the worktree. UNINTENDED DRIFT.
3. **`populations_07c.py`'s first run read 07b's comparison pairs as [07a, 07b]**; they are [07b, 07a] (07b's
   `compare_07b.py`: `[s, s7]`). It found 5 chase calls instead of 198. Its output is in `state/superseded/` and its
   first script in `scripts/superseded/`; the corrected run reproduces the census agent's 198 as a set, and 324 / 18 /
   306. UNINTENDED DRIFT (a failed first attempt).
4. **R31's record is compared with the record 07b's rehearsal agent captured** (`var/levin-phase-floor-07b/rehearsal/
   out/refcap.json`, which that agent showed to reproduce R31's store), not with R31's store: the brief forbids touching
   R31's store. It runs at its own rtol (1e-7) only: no record exists at 1e-8 to reproduce. STRUCTURALLY REQUIRED.
5. **The Clenshaw-Curtis pair is CC_N and CC_{2N-1} at the calls' order N = 24** (CC_24 and CC_47), not CC_16 and
   CC_31: the census calls run at `chebyshev_order` 24. The prompt's labels are the audit's (N = 16). STRUCTURALLY REQUIRED.
6. **The unit drawn for R4 and R5 is an integrand**, and every call of it in the population is scored from its one reading
   (R5: 30 integrands, 43 of the 306 calls). The rule is in `scripts/draw_refs_07c.py` and `state/r5_rule.txt`,
   written before the draw. IMPLEMENTATION CHOICE.
7. **Every integrand of section 2.1 runs at both rtols:** where a population holds one rtol only, the other is replayed
   from the same capture with the other census record's (rtol, atol) (07b Part A's deviation 8) and must reproduce 07b's
   census record there (`x-...` records, `extract_census_refs_extra_07c.py`). IMPLEMENTATION CHOICE.
8. **The leaf reference gained a noise-averaging rule after its first test.** On R31's record's 122 chase leaves every
   interpolatory rule (CC_93, CC_185, composite Gauss-Legendre) disagreed with every other at about 1e-37, above the leaf's
   floor: the integrand as evaluated carries noise (about 3-6e-12 of |g| per sample) the floor does not model. leafcc_07c.py
   now also fits a 600-point Chebyshev least-squares polynomial (degrees 24 and 32), and uses whichever reference states
   the smaller error. The first test's output is `out/leafcc_test/superseded/`. IMPLEMENTATION CHOICE.
9. **D2's population is the Clenshaw-Curtis leaves of section 2.1's calls**, not of the whole census: a leaf can be measured
   only with its call's integrand, and only those calls are captured (x1 capture takes group calls). Its analytic strata
   hold 13 leaves (6, 5 and 2), below the 20 a stratum the prompt asks for; there are no analytic window pieces and no
   guarded leaf. STRUCTURALLY REQUIRED.
10. **06d's numbers are not reproduced on their own case:** the records found name no case for "2.7e-6 against 5.4e-17 at
    6 pi". The audit's table (section 2.2, its own integrand at N = 13) is reproduced to the digits it prints; the same
    integrand at N = 16 gives 9.8e-8 at 6 pi and 3.8e-12 at 3 pi (out/d2/reproduce.json). STRUCTURALLY REQUIRED.
11. **(E) is run with tail_sum4, which does not qualify as an estimate** by section 2.3's rule (no candidate does; the
    smallest ratio to CC_47's true error is 0.20 on the gate leaves sampled). It is run, flagged, so that the tables show
    what a sharper estimate does; its variants (Eg, Ea, SEg, SEa) are reported but carry no bound basis.
    IMPLEMENTATION CHOICE.
12. **register_variant_07c.py's first form failed on 14 registrations** (its pattern wanted double quotes; the generator
    writes the name with repr's single quotes); v1 is in scripts/superseded/, the instrument's registration under it
    stands. UNINTENDED DRIFT (a failed attempt).
13. **The references run two readings at a time** (four processes), not 07b's three, so that D2 and the variants could run
    beside it in the other two slots; projected 17.7 h elapsed instead of 11.8 h. IMPLEMENTATION CHOICE.
14. **(S) is run at two noise factors,** C = 4 (S, with the combinations) and C = 2 (S2), for the sensitivity of the one
    constant D1's distribution does not fix sharply. Seven candidates in all, within eight; the instrument and the seven
    instrumented copies (for table 4's descent column) are not candidates. IMPLEMENTATION CHOICE.
15. **D2's leaves are classed noise- or truncation-dominated** (d2_regimes_07c.py: true error at most 3 sigma sqrt(sum w^2)),
    added after the first table showed every candidate below 1 somewhere. IMPLEMENTATION CHOICE.
16. **Test replays and a test leaf run** are kept in out/replays_test/ and out/leafcc_test/; no table reads them.
    IMPLEMENTATION CHOICE.
17. **Which leaves a variant changes is measured by comparing its replay with the tree's, leaf by leaf**
    (`variant_reach_07c.py`, `out/variants/reach.{json,md}`), not from the variant's own counters (`num_s07c_*`,
    `num_n07c_trusted`). The replay harness records the result's standard keys and the refusal counts, not a variant's
    added keys. The variants' designs say which leaves each may change; the comparison says which it did. IMPLEMENTATION
    CHOICE.
18. **The instrumented copies (`<v>_i`) run on the chase's 259 calls** (the 198 chase calls, the R2 set at both rtols,
    which holds R2-28, and R31's record), not all 454: table 4's descent column needs them only there. On all 259 each
    equals its plain variant bit for bit (`out/d1/variants_descent.json`, `_identity`). IMPLEMENTATION CHOICE.
19. **Every variant's window stops are compared against 07b's,** not only (E-all)'s (`out/cmd/41_window_changes_summary.txt`).
    Most changes are only the split count at which the same stop fell (`stopped_after_split`). IMPLEMENTATION CHOICE.
20. **score_07c.py was corrected once before it scored anything.** A dry run on the references then present (07b's, with
    none of this Part's readings taken) showed that the R5 verdict line could not format a call with no 07a record (an
    `x-` call at the other rtol). v1 is in `scripts/superseded/score_07c_v1.py`; the dry run's output is
    `out/tables_test/` (`out/cmd/42_score_dryrun.txt`), and no table reads it. IMPLEMENTATION CHOICE.
21. **R2-28's two calls are scored twice in table 3:** as 07b's item 2 calls `a:R2-28` (1e-7) and `a:R2-28~o` (1e-8), and
    as the chase calls `c-1e-7-0-13757-5` and `c-1e-8-0-13757-5`, whose replays are the same calls, bit for bit. Section
    2.6 lists 07b's 187 and the chase separately, so both are kept, and the duplication is stated with the table.
    IMPLEMENTATION CHOICE.
22. **The registered runs' driver runs with its cwd in the records directory** (`launch_07c.py`, from 07b's
    `launch_07b.py` unchanged in this), and the manifests record the main checkout's HEAD (`28ebaf07`). The launching
    shell's cwd was the worktree. Every job is given `PYTHONPATH` = the worktree, and every replay records the module it
    ran and its SHA-256. Nothing was written in the worktree. IMPLEMENTATION CHOICE (inherited from 07b's launcher).
23. **The first references run took no reading in 12.4 h.** `leafref_07c.py` v1 named its worker loader `leafref_07b`
    (07b's module name, kept from the copy), which is not on the reading's path, so every pool worker failed in its
    initializer and was respawned (190,485 `ModuleNotFoundError`s in each job log, `out/cmd/refs/failed_first/`, 268 MB
    each). The run's monitor stopped it at its backstop, at 23:48:26Z (`levin-phase-floor-07c-refs-20261118T112553`, state
    killed, 0 of 123). The script was never test-run before the launch. v2 names `leafref_07c`; v1 is in
    `scripts/superseded/leafref_07c_v1.py`. A smoke reading on the smallest integrand (`out/refs_test/`,
    `out/cmd/43_refs_smoke_fccsplit.txt`) took 105 s against 101 s projected. 13.7 h of Part A's 36 h were used when it was
    found. The orchestrator was asked how to relaunch. UNINTENDED DRIFT.
24. **The references were relaunched three at a time** (six worker processes; nothing else was running), on the laptop
    orchestrator's ruling (option (b)), as `levin-phase-floor-07c-refs2-20261118T235323` (`state/plan_refs2.json`
    `cd35f76b…`, from `plan_refs.json` by `plan_refs2_07c.py`). It reads the same 123 readings, with the same rule, seeds
    and projection model: 11.8 h projected with the 1.2 margin, against 17.7 h at two at a time. Its backstop is
    74,784 s: Part A's 36 h less 1.5 h, from the plan's writing. To make that backstop wall-clock, `run_06j.py` v2 (07b's
    driver, changed where marked "prompt 07c"; v1 in `scripts/superseded/run_06j_v1.py`) divides the remaining
    projections in P by the plan's `backstop_divisor` (3). v1's P sums them serially, so at three at a time it would
    have stopped the run at its start. IMPLEMENTATION CHOICE (on the orchestrator's ruling).
