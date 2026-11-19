# levin-phase-floor 07c, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07c Part A, the chase below a refused leaf: section 2.1's calls, section 2.2's D1, section 2.3's D2,
section 2.4's variants, section 2.5's references R4 and R5, and section 2.6's tables. Run on the Mac Studio (mps026298).
The worktree is `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f`, at the dispatch SHA
`2fd979fd95fa4d4ff3ac750e0c4094d83d744f7f`, whose code is 07b's (`5e19c4b6`, `AdaptiveLevin/levin_quadrature.py`
`8956c7b6…`). Records are in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/measurement/`.
**Commit:** none. The orchestrator commits the records, on the laptop.
**Result (2026-11-19): the planned stop of section 2.7.**
- **One stop of section 8's kind came first.** It was put to the orchestrator, ruled on, and is recorded as deviations 23
  and 24: the first references run took no reading in 12.4 h.
- **Three of the seven variants qualify:** N, Eg and Ea.
- **S, S2, SEg and SEa fail** on R2-28 alone. That call converges under them and is dishonest, at ratios 1.12–1.14.
- **Table 4:** chase calls left unconverged, of 201:

  | tree | S | S2 | N | Eg | Ea | SEg | SEa |
  |---|---|---|---|---|---|---|---|
  | 201 | 5 | 6 | 21 | 186 | 186 | 4 | 4 |
- **The gate** holds before and after (`out/gate/`).
- **Section 2.1:** every replay at 07b's code reproduces its record bit for bit: 454 of 454 (the 198 chase calls and
  their 8 counterparts at the other rtol, 07b Part A's 187, R31's record against R31's captured record, R5's 43 and
  their 17 counterparts). Captures: 198 of 198 and 43 of 43 equal 07b's census.
- **D1:**
  - The census agent's descent analysis is reproduced on all 198 calls. Every depth-20 leaf is below a refused leaf in
    192, and 6 have no depth-20 leaf.
  - 24,109 bisections below refused leaves are tabulated by route and cause: Levin 15,445, Clenshaw-Curtis 8,664.
  - The chase's 12,520 Clenshaw-Curtis leaves are re-solved with their true errors. 6,615 of them were bisected with a
    residual above their floor. On 2,968 of those, CC_47's true error is at or below the floor; on 1,159 it still is
    with the reference's own error added.
  - **Honesty with converged set aside:** on the 61 chase calls referenced (R4's 58, R2-28's 2, R31's record), the
    tree's |v − ref|/abserr is at most 0.416, except R2-28's two at 1.107.
- **D2:** no candidate estimate bounds CC_47's true error on every sampled leaf; none qualifies.
- **Variants:** seven run on all 454 calls (S, S2, N, Eg, Ea, SEg, SEa), each with its instrumented copy on the chase's
  259 calls, bit-identical to it.
- **References** (`levin-phase-floor-07c-refs2-20261118T235323`): 123 readings, each with exit 0, in 6 h 22 min.
  - They cover R4 (30 chase integrands), R5 (30 integrands, 60 calls) and R31's record.
  - Every call they serve meets the scoring rule: 121 of 121.
  - **R5:** all 43 calls dishonest at 07a's code are honest at 07b's, at ratios 0.00014–0.11. Their 17 counterparts are
    honest too, at ≤ 0.198.
- **Qualification** (`out/tables/scores_07c.md`):
  - N, Eg and Ea qualify.
  - S, S2, SEg and SEa do not. Each makes R2-28's four rows converged and dishonest. Those are two calls, each scored
    twice (deviation 21).
  - **Eg and Ea carry no bound basis:** D2 found that tail_sum4 does not qualify as an estimate (deviation 11).

## What shipped

Nothing in the tree. Its `git --no-optional-locks status --short --ignored` was empty at every check, and the main
checkout stayed at `28ebaf07`. The records, all under `measurement/`:
- **`scripts/`:**
  - 07b's census harness, copied whole from `var/levin-phase-floor-07b/census/scripts/` (230 files,
    `out/scripts_07b_census_sha256_{source,as_copied}.txt`). Changes are marked "prompt 07c": `prov.py`, `rec.sh` (written
    fresh from 07b Part A's `scripts/ext/rec.sh`, paths only) and `x1_capture_07c.py` (07b Part A's `x1_capture_07b.py`,
    its title).
  - 07b Part A's scripts, in `scripts/from_07b_measurement/` (276 files, byte-identical,
    `out/scripts_07b_measurement_sha256_{source,as_copied}.txt`).
  - The new scripts: `gate_hashes_07c.py`, `populations_07c.py`, `extract_census_refs_07c.py`,
    `extract_census_refs_extra_07c.py`, `capture_selection_07c.py`, `launch_07c.py` (from 07b's `launch_07b.py`),
    `make_calls_07c.py`, `replay_07c.py`, `run_replays_07c.py`, `plan_replays_07c.py`, `make_instr_07c.py`,
    `register_variant_07c.py`, `make_variants_07c.py`, `d1_descent_07c.py`, `leafcc_07c.py`, `leafcc_specs_07c.py`,
    `d2_table_07c.py`, `d1_tables_07c.py`, `d2_reproduce_07c.py`, `d2_population_07c.py`, `d2_regimes_07c.py`,
    `draw_refs_07c.py`, `slice_refs_07c.py`, `leafref_07c.py` (07b's `leafref`, its `TREE`), `refs_projection_07c.py`,
    `plan_refs_07c.py`, `plan_refs2_07c.py`, `variants_descent_07c.py`, `variant_reach_07c.py`, `score_07c.py` (v2). `run_06j.py` (07b's
    run driver) is changed to v2 where marked "prompt 07c" (deviation 24), and `leafref_07c.py` is v2 (deviation 23).
  - Superseded first forms in `scripts/superseded/`: `populations_07c.py` v1, `register_variant_07c.py` v1,
    `score_07c_v1.py`, `leafref_07c_v1.py`, `run_06j_v1.py` (deviations 3, 12, 20, 23, 24).
- **`scripts/variants/`:** the instrument `levin_quadrature_07c_instr.py` (`7c0a5b86…`, +83/−2 against the tree), and
  `levin_quadrature_07c_<v>.py` for S `09088683…`, S2 `b7819ee2…`, N `fbbdcbed…`, Eg `7d32bc1e…`, Ea `fd715e3d…`,
  SEg `50672dd7…`, SEa `33fedbdf…`, with their instrumented copies `<v>_i`. Each diff against the tree is in
  `out/variants/<v>.diff`, each SHA-256 in `out/variants/variants.json`, each basis in `state/variants/<v>.json`.
- **`state/`:** `seeds_07c.json` (`05f428f3…`, written 10:24:13Z, before any draw); `selection_r4.json`
  (`01918f2b…`), `selection_r5.json` (`e901adff…`), `selection_d2.json` (`c4beb666…`), each saved before any reading or
  solve; `refs_rule.txt`, `r4_rule.txt`, `r5_rule.txt`, written before any reading; `populations_07c.json`
  (`33910d71…`); `calls_07c.json` (`fcdbeffc…`, 394 calls) and `calls_07c_r5.json` (`7a469f20…`, 60); the census
  records (`census_refs_07c*.pkl`); the captures (`capture_chase/`, `capture_r5/`); the leaf specs (`leafcc_d1/`,
  `leafcc_d2/`); the reference inputs and specs (`refs/`); the plans with their purpose and scope texts; and
  `deviations_draft.md`.
- **`out/`:** `cmd/` and `commands.log` (every command and its output); `gate/`; `captures/`; `replays/` (tree,
  instrument, each variant and each `<v>_i`, with summaries); `d1/` (`d1_descent.json`, `d1_cc_leaves.{json,md}`,
  `d1_tables.{json,md}`, `variants_descent.json`); `leafcc_d1/`, `leafcc_d2/`; `d2/` (`reproduce.json`,
  `population.json`, `d2_table.{json,md}`, `d2_regimes.json`); `variants/` (`variants.json`, the diffs,
  `reach.{json,md}`); `refs/` (`inputs.json`, `refs_projection.json`, `readings/`); `tables/` (`scores_07c.{json,md}`);
  `cmd/refs/failed_first/` (the failed first references run's two job logs, 268 MB each, deviation 23); and the test
  outputs, which no table reads: `replays_test/`, `leafcc_test/`, `tables_test/`, `refs_test/`.
- **Registered runs** (`var/runs/`): `levin-phase-floor-07c-capture-chase-20261118T101845`,
  `-capture-r5-20261118T105127`, `-base-20261118T110242`, `-refs-20261118T112553`, `-variants-20261118T113152`,
  `-refs2-20261118T235323`. Every one has a recorded end. Five are done; `-refs-` was killed by its own backstop, with
  0 of 123 (deviation 23).

### The variants (section 2.4), each with its basis, and the leaves it changes

The noise level is the leaf's own trust-check yardstick ν_B: a leaf is *at the noise level* if δ₁ ≤ C ν_B and δ₂ ≤ C ν_B.
**Basis** (`out/d1/d1_descent.json`): of the 8,664 Clenshaw-Curtis bisections below refused leaves (the 198 chase calls,
R2-28 at both rtols and R31's record), both δ ≤ ν_B in 1,288 (15%), ≤ 2ν_B in 6,342 (73%), ≤ 4ν_B in 7,665 (88%),
≤ 10ν_B in 8,428 (97%). Their δ₁/(φ_P + φ_L + φ_R) has median 0.68 and 90th percentile 1.01. D2 finds their CC_47 true
errors noise-dominated in 22,700 of the 22,876 chase leaves measured. Of the 15,445 Levin bisections, δ₁/ν_B has median
47, and both δ ≤ 4ν_B in 2,460. **Where the basis does not reach:** the analytic calls have no chase in section 2.1's
sample, and the limit 6 is dqage's iroff1 limit (07a's window stop at the same noise level), not measured here.

| variant | what it is | SHA-256 | constants and margin |
|---|---|---|---|
| **S** | (S): below a refused leaf, bisections at the noise level count against the subtree's root; at 6 the subtree stops; every leaf of it at the noise level, and every one made in it later once found so, is floor-limited by `_floor_limit_stopped_window_leaf` | `09088683…` | C = 4 (88% of the CC bisections; the 90th percentile of δ₁/φ is 1.01, so 4 is about 4× it); limit 6 |
| **S2** | S at C = 2 | `b7819ee2…` | C = 2 (73%), the sensitivity of C |
| **N** | (N): a checked leaf with δ₁ ≤ ν_B, δ₂ ≤ ν_B and r ≥ 1 trusted, declaring max(residual, δ₁ + δ₂) | `fbbdcbed…` | none added. D1 found 1,288 CC and 983 Levin such leaves |
| **Eg** | (E-gate): the C2 gate's CC leaves declare tail_sum4 in place of the pair | `7d32bc1e…` | none. **tail_sum4 does not qualify** (D2): flagged, no bound basis |
| **Ea** | (E-all): every CC solve, window pieces included | `fd715e3d…` | as Eg |
| **SEg** | S with Eg | `50672dd7…` | as S and Eg |
| **SEa** | S with Ea | `33fedbdf…` | as S and Ea |

**What each changes** (`out/variants/reach.md`, `out/cmd/41_window_changes_summary.txt`; leaves compared by span):

| variant | calls touched of 454 | leaves made (CC / Levin) | tree leaves not made (CC / Levin) | same span, record differs | windows changed: stop / leaves at stop / split count only |
|---|---|---|---|---|---|
| S | 328 | 2,787 / 1,280 | 13,418 / 3,517 | 4,388 / 2,036 | 0 / 0 / 41 |
| S2 | 311 | 2,817 / 935 | 11,670 / 2,347 | 5,724 / 2,161 | 0 / 0 / 31 |
| N | 438 | 1,548 / 855 | 6,280 / 1,984 | 1,695 / 2,410 | 0 / 7 / 68 |
| Eg | 394 | 20,549 / 155 | 4,838 / 112 | 19,543 / 0 | 0 / 5 / 74 |
| Ea | 404 | 21,459 / 155 | 5,786 / 112 | 20,924 / 0 | 1 / 107 / 14 |
| SEg | 405 | 14,052 / 1,475 | 15,568 / 3,696 | 8,813 / 2,009 | 0 / 5 / 77 |
| SEa | 411 | 14,962 / 1,475 | 16,516 / 3,696 | 10,194 / 2,009 | 1 / 107 / 14 |

"Split count only" is a window whose stop fell, with the same leaves, at a different `stopped_after_split`: the global
split counter moves when a variant changes the splits elsewhere. **Under Ea and SEa one window stop is lost:**
`a:R3w-0-2006-0~o`, window [0x1.7dd87bff0dd28p+4, 0x1.2aaf213520fc0p+5], stopped at split 27 with 17 leaves at the
tree, never stopped under (E-all) (iroff1 count 3).

**Untouched calls, bit-identical to the tree** (replay and region arrays' SHA-256; one per set in `reach.md`), for
example: S and S2 `c-1e-8-1-2229-4` (`da85f758…`); N `a:R2-02` (`ffc9e1ca…`); Eg, Ea, SEg and SEa `a:R2-00`
(`8afa1d28…`). N touches every chase call, and Eg, Ea, SEg and SEa every chase call too, so their untouched examples
come from other sets.

## Predictions and verdicts

Part A fixes no prediction (section 6's are Part C's). **The qualification rule** (section 2.6): a variant qualifies if,
on table 3, no referenced call is converged and dishonest that is not so at the tree, and, on table 4, it leaves fewer
chase calls unconverged than the tree.

| variant | new converged & dishonest (table 3) | chase unconverged of 201 (table 4) | qualifies | regions / tree, all 454 (table 5) | wall / tree |
|---|---|---|---|---|---|
| tree | – | 201 (195 with every depth-20 leaf below a refused leaf) | – | 1.000 | 1.000 |
| S | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 (ratios 1.143) | 5 (0) | no | 0.745 | 0.726 |
| S2 | the same four (1.120) | 6 (0) | no | 0.796 | 0.773 |
| N | none | 21 (19) | yes | 0.884 | 0.879 |
| Eg | none | 186 (182) | yes | 1.313 | 1.435 |
| Ea | none | 186 (182) | yes | 1.312 | 1.432 |
| SEg | the same four (1.139) | 4 (0) | no | 0.926 | 1.003 |
| SEa | the same four (1.139) | 4 (0) | no | 0.925 | 1.036 |

**Table 3** (305 referenced; 6 not scored, 07b's `a:R2-27`, `a:R2-29` and `a:R3a-0-33666-a13` at both rtols):

| variant | converged & honest | converged & dishonest | unconverged |
|---|---|---|---|
| tree | 235 | 0 | 64 |
| S | 292 | 4 | 3 |
| S2 | 292 | 4 | 3 |
| N | 285 | 0 | 14 |
| Eg | 228 | 0 | 71 |
| Ea | 228 | 0 | 71 |
| SEg | 287 | 4 | 8 |
| SEa | 287 | 4 | 8 |

Under Eg and Ea, seven calls that converge at the tree are unconverged; under SEg and SEa, five (`issues.md` item 8).
Each unconverged call's ratio is listed in `out/tables/scores_07c.md`. **R2-28** is off by the same −2.30×10⁻³⁷ under
every code (`issues.md` item 7). Table 5's "calls touched" uses a narrower definition than `reach.md` (`issues.md` item
10).

## Deviations from the prompt

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

## Verification performed

See `verification.md`. In brief:
- the gate is equal before and after (records 01 and 47);
- the tree's status was empty at every check, and the main checkout stayed at `28ebaf07`;
- 454 of 454 replays reproduce 07b's census at the tree and at the instrument;
- each `<v>_i` equals its variant on 259 of 259;
- the captures are 198 of 198 and 43 of 43;
- every Result-line number was checked against `out/tables/scores_07c.json`.

## Observations not acted on

See `issues.md`, items 1–10, proposed and none opened:
- integrand noise above the CC floor;
- the CC abserr below its true error;
- window pieces' noise far above their floor;
- the chase mostly Levin bisection;
- 06d's case not found;
- the "nearest" column;
- R2-28's common error;
- (E)'s newly unconverged calls;
- the projection's overestimate;
- table 5's "touched" definition.

Also, as the brief said, `var/levin-phase-floor-07b/rehearsal/out/commands.log` is 1,402 bytes longer than its list
says. It was not acted on.

## State handed on

- **No process of this Part is running.** Every registered run has a recorded end. Ray and the rehearsal lock were
  never used.
- **The worktree** is unchanged at `2fd979fd`.
- **The records** are under `var/levin-phase-floor-07c/measurement/`.
  - The pickles and every file over 2 MB stay on the Mac Studio, listed with SHA-256 and size in
    `KEPT_ON_MAC_STUDIO.txt`, which includes the failed first run's logs in `out/cmd/refs/failed_first/`.
- **For Part B and the orchestrator:**
  - the tables, in `out/tables/scores_07c.{json,md}`;
  - the variants and their SHA-256s, in `scripts/variants/` and `out/variants/variants.json`;
  - this Part's references, in `out/refs/readings/`;
  - the seeds, in `state/seeds_07c.json`: R4 707304, R5 707305, D2 707302.

## Extension: R2-28

**Subject:** the laptop orchestrator's extension of Part A (2026-11-19), on the user's ruling at the stop: find the cause of
R2-28's error before the fix is chosen. It asks five questions, answered below in its order. The extension used the
same worktree at `2fd979fd` and the same records directory, with no Ray, no lock and no store. Its records are
48–53 in `out/cmd/`. Its scripts are new: `r2_28_spec_07c.py`, `r2_28_table_07c.py`, `r2_28_probe_07c.py`,
`r2_28_mech_07c.py` and `r2_28_summary_07c.py`. Its outputs are in `out/r2-28/` and its inputs in `state/r2-28/`.
**Commit:** none.
**Result (2026-11-19):**
- **One leaf carries R2-28's error, and every 07b-lineage code makes it.** The leaf is the trusted Levin leaf
  [22.3566866, 23.5393356], at depth 4: the call's last sixteenth.
  - Under every code it has the same span and the same hex value (`-0x1.b0e16966a7238p-96`).
  - Its error is −2.338×10⁻³⁷ against a reference stated to 2.9×10⁻⁴², which is 1.247 of its e′ (1.874×10⁻³⁷).
  - The rest of every partition errs by +3.5 to +3.7×10⁻³⁹: 0.017–0.018 of the call's abserr, and 0.16–0.27 of the
    abserr less this leaf's e′.
- **The mechanism is 07b's M2**, the error common to a trusted Levin leaf, its halves and its quarters:
  - the leaf's own Levin solve errs by −4.8×10⁻³⁸;
  - its halves' sum and its quarters' sum both err by −2.34×10⁻³⁷;
  - its eighths' sum errs by +1.0×10⁻³⁸.

  r = 0.0048 trusts the leaf, and the leaf returns its quarters' sum. e′ is built from δ₁ = |V_P − S1|, the difference of
  two errors of the same sign, so it is smaller than the quarters' error.
- **It is not the CC fallback, and not a leaf the rate rules made by refusing it.**
  - The CC leaves' errors sum to 7×10⁻⁴¹.
  - The leaf passed its own range rule (q = 3.26). Its one refused ancestor is the root, refused by the range rule at
    q = 5.76×10⁷; its depth 1–3 ancestors failed the trust test on r ≥ 1.
- **07a's one region:** its value errs by −4.82×10⁻³⁸, 0.21 of 07b's error.
  - The error sits in one quarter, the last, [18.81, 23.54], which errs by −4.82×10⁻³⁸. The other three quarters err
    by less than 4×10⁻⁴³ each. So there is no cancellation.
  - The value is honest because its e′ is 8.68×10⁻³⁴, the root's own floor, 18,000× its error. Its error is of the same Levin
    scatter, about 10⁻³⁷, not below it.
- **One call, not a class, as far as this Part's data reach.** R2-28 is the only referenced call on which every
  07b-lineage code errs above its abserr (4 rows: its two calls, each scored twice). Of R2-28's trusted Levin leaves,
  only the carrying leaf is above its e′ by more than 1.05×. The others above e′ are at 1.008–1.041, plus one (tree
  leaf 18) at the reference's own error level.
- **The size against the request:** R2-28's error, 2.30×10⁻³⁷, is 1.6×10⁻⁴ of its request. The request is atol,
  1.446×10⁻³³, which is above rtol·|I| at both rtols.

### Question 1: where the error sits

**Method.**
- One reading of `leafref_07c.py` (record 49) over R2-28's whole span: split FCC at resolutions 1 and 2, split ratio 1.8
  (R2-28's call reference's method), 6 workers.
- Its points are every leaf end of a:R2-28 and a:R2-28~o under the tree and the seven variants, and of 07a's region,
  together with the halves', quarters' and eighths' break points of every trusted leaf of the tree's, S's and 07a's
  partitions of a:R2-28. That gives 1,999 points and 1,998 pieces (`state/r2-28/spec.json` `7336643e…`, record 48b).
- The reading took 81 s at r1 and 287 s at r2 (`out/r2-28/leafref.json` `79c0c9c1…`).
- A span's reference is the sum of its r2 pieces. Its stated error is √(Σ fit-noise variance) + |r2 − r1|, 07b's
  x1_table rule.
- The whole span's sum is −1.76168148866605873×10⁻²⁹, stated to 2.2×10⁻⁴¹. That is 8.3×10⁻⁴² from 07b's call
  reference.

**The table** (`out/r2-28/leaves.{json,md}`, record 50). In every partition the leaves' errors sum to the call's error,
−2.301×10⁻³⁷. By kind, in the tree's partition of a:R2-28:

| kind | n | sum of errors | sum of stated reference errors | sum of e′ |
|---|---|---|---|---|
| trusted Levin | 46 | −2.301×10⁻³⁷ | 1.2×10⁻⁴⁰ | 1.98×10⁻³⁷ |
| trusted direct (CC by the C2 gate) | 188 | −1.8×10⁻⁴² | 6.2×10⁻⁴² | 9.6×10⁻³⁹ |
| floor-limited direct | 44 | +7.1×10⁻⁴¹ | 4.5×10⁻⁴² | 4.0×10⁻⁴⁰ |
| floor-limited Levin | 16 | −2.3×10⁻⁴² | 1.9×10⁻⁴¹ | 2.7×10⁻⁴¹ |
| untrusted direct (depth 20) | 2 | −3.5×10⁻⁴⁵ | 3.5×10⁻⁴⁵ | 3.8×10⁻⁴⁴ |

**The leaves:**
- **The carrying leaf** is tree leaf 295 (S's 99).
  - It is a trusted Levin leaf at depth 4, [0x1.65b4fd067cd45p+4, 0x1.78a11e65a7f80p+4] = [22.3566866, 23.5393356],
    with value kind quarters.
  - Its error is −2.338×10⁻³⁷ against its reference −2.13426569258546191×10⁻²⁹, stated to 2.9×10⁻⁴²; its e′ is
    1.874×10⁻³⁷, so error/e′ = −1.247.
  - Its |θ′| runs from 372.7 to 1,216 (ratio 3.26), and its phase variation V is 844 rad.
  - **Its rule history** (from the instrumented replays, tree and S_i alike):
    - the root, [4.617, 23.539], passed the trust test (r 0.025) and was refused by the range rule, q = 5.76×10⁷;
    - its ancestors at depths 1, 2 and 3 ([14.08, 23.54], [18.81, 23.54], [21.17, 23.54]) were untrusted by the trust
      test, clause r ≥ 1 (r 18.1, 1.16 and 5.17);
    - the leaf itself passed the range rule (q 3.26) and was trusted by r < 1/4 (r 0.0048, δ₁/ν_B 2.7×10⁴);
    - no edge rule acted on R2-28 (0 edge refusals).
- **Its neighbour**, leaf 294 (depth 5, [21.765, 22.357], trusted Levin, |θ′| 1,216–2,197), errs by +3.71×10⁻³⁹, which
  is 0.42 of its e′. Every other leaf errs by at most 3.6×10⁻⁴¹.
- **The same leaf, with the same hex value, is in every 07b-lineage partition** (`out/r2-28/summary.md`, record 53):

  | code | regions | carrying leaf's e′ / call abserr | rest of the call: error | rest / abserr | rest / (abserr − e′) |
  |---|---|---|---|---|---|
  | tree | 296 | 0.902 | +3.70×10⁻³⁹ | 0.0178 | 0.181 |
  | S | 100 | 0.931 | +3.69×10⁻³⁹ | 0.0183 | 0.267 |
  | S2 | 133 | 0.912 | +3.69×10⁻³⁹ | 0.0180 | 0.205 |
  | N | 263 | 0.903 | +3.63×10⁻³⁹ | 0.0175 | 0.180 |
  | Eg, Ea | 359 | 0.897 | +3.53×10⁻³⁹ | 0.0169 | 0.165 |
  | SEg, SEa | 100 | 0.928 | +3.69×10⁻³⁹ | 0.0183 | 0.254 |

  The ~o calls give the same figures to the digits shown.

### Question 2: why

**Method.** `r2_28_probe_07c.py` (record 51, `out/r2-28/probe.json`) re-solves the leaf, its neighbour and 07a's region
at the tree's module, which is imported unchanged. Each span is re-solved as P, its halves, its quarters and its eighths,
under four rules:
- the driver's solve;
- the forced Clenshaw-Curtis pair;
- Levin at N + 8 = 32;
- Levin at N − 8 = 16.

Each probe reproduces its replay's leaf by hex (V_P = `value_own`, Q = `value`). `r2_28_mech_07c.py` (record 52,
`out/r2-28/mech.{json,md}`) sets every piece against its own reference.

**The carrying leaf's sums**, each against P's reference:

| rule | V_P | S1 = L + R | Q (returned) | S3 (eighths) |
|---|---|---|---|---|
| driver (N = 24) | −4.82×10⁻³⁸ | −2.347×10⁻³⁷ | **−2.338×10⁻³⁷** | +1.01×10⁻³⁸ |
| Levin N + 8 | −1.50×10⁻³⁷ | −1.98×10⁻³⁷ | −2.04×10⁻³⁷ | −9.95×10⁻³⁸ |
| Levin N − 8 | −8.86×10⁻³⁸ | −7.12×10⁻³⁸ | −1.10×10⁻³⁷ | −2.58×10⁻³⁷ |
| forced CC | −6.1×10⁻²⁸ | +6.4×10⁻²⁸ | +3.6×10⁻²⁹ | −5.7×10⁻³⁰ |

**What separates the candidates:**
- **07b's M2, an error common to the leaf's halves and quarters, fits.**
  - S1 and Q carry the same error, −2.35 and −2.34×10⁻³⁷. That makes δ₂ = |S1 − Q| = 9.0×10⁻⁴⁰ and r = 0.0048, so the
    leaf is trusted.
  - The returned Q is wrong by 1.25 e′. e′ ≈ δ₁/(1 − r), and δ₁ = |V_P − S1| = 1.86×10⁻³⁷ is the difference between
    V_P's error (−4.8×10⁻³⁸) and S1's (−2.35×10⁻³⁷). Both have the same sign, so δ₁ is smaller than S1's error.
  - The eighths break the pattern (S3 +1.0×10⁻³⁸), as 07b found for M2.
  - Piece by piece, most of the error is in the right half R (−1.84×10⁻³⁷) and its right quarter RR = [23.244, 23.539]
    (−1.91×10⁻³⁷), at the call's upper end, where |θ′| is lowest. RR's eighths do not carry it: RRL −5.7×10⁻³⁹, RRR
    +7.8×10⁻³⁸.
  - δ₂ is small partly by cancellation between halves: L − (LL + LR) = −4.7×10⁻³⁸ and R − (RL + RR) = +4.7×10⁻³⁸.
- **The error is not truncation, and the floor does not model it.**
  - At the next orders the same pieces err by different amounts, at both N + 8 and N − 8. Over the leaf's 15 pieces
    and the three Levin rules (45 solves), the median |error| is 2.5×10⁻³⁸ and the largest 2.4×10⁻³⁷.
  - The median error is 2.0×10⁴ × the piece's floor (`phase_err`, 0.4–3.6×10⁻⁴²). The stated reference errors are at
    most 2.9×10⁻⁴², so the reference does not limit this.
  - On the neighbour, leaf 294, at |θ′| 1,216–2,197, the same measure is 8.4×10⁻⁴⁰, which is 2.3×10³ × the floor.
  - So the Levin solves on this span scatter at about 10⁻³⁷, or 5×10⁻⁹ of the leaf's value. That is above their
    floors, and does not fall with order or depth.
  - Its source within the solve (the integrand's noise, `issues.md` item 1, or the collocation) was not separated in
    this extension.
- **The CC fallback's estimate** is excluded: the carrying leaf is Levin. The CC leaves' errors sum to +7×10⁻⁴¹, against
  their e′ sum of 1.0×10⁻³⁸. The forced CC pair is no second rule here (V 59–844 rad, errors up to 10⁻²⁷); it shows
  only that CC cannot serve on these spans.
- **An edge or range rule's leaf**, in the sense of a leaf the rule made, is excluded:
  - there was no edge refusal;
  - the leaf passed its own range rule (q 3.26);
  - its depth 1–3 ancestors were untrusted by the trust test, not by a rate rule.

  The range rule's one part is the root's refusal (q 5.76×10⁷). That refusal sends the call into a partition at all;
  without it the call is 07a's one region (question 3).

### Question 3: why 07a was right

07a's region is [4.617, 23.539], trusted at r 0.025, returning its quarters' sum Q. It is reproduced by the tree's module
by hex, both V_P and Q. The subregion solve is unchanged between `1ec734d7` and `5e19c4b6` (deviation 27).

| piece | span | driver's error | stated reference error | V |
|---|---|---|---|---|
| V_P (own) | whole | +2.77×10⁻³⁸ | 2.2×10⁻⁴¹ | 3.2×10¹⁰ |
| S1 | | −4.64×10⁻³⁸ | | |
| **Q (returned)** | | **−4.82×10⁻³⁸** | | |
| quarters LL, LR, RL | [4.617, 18.81] | −3.1, −2.5, −3.6×10⁻⁴³ | 1.4–5.9×10⁻⁴² | 4.7×10⁶–3.2×10¹⁰ |
| quarter RR | [18.81, 23.54] | **−4.82×10⁻³⁸** | 2.3×10⁻⁴¹ | 4.2×10⁴ |
| RR's eighths RRL, RRR | | −2.0×10⁻⁴², −8.14×10⁻³⁸ | | |

- **The value's error is in one piece, the last quarter. There is no cancellation:** the other quarters err by less than
  4×10⁻⁴³ each.
- That quarter is solved as one Levin solve over [18.81, 23.54], which contains the carrying leaf's span. It errs by
  −4.8×10⁻³⁸, a fifth of the error 07b's partition returns on [22.36, 23.54] alone.
- **07a is honest because its e′ is large, not because its value is exact.** Its e′, 8.68×10⁻³⁴, is the root's floor
  (`phase_err`, from V = 3.2×10¹⁰). That is 1.8×10⁴ × its error, and 0.60 of the request.
- Its value is more accurate than 07b's because one Levin solve happened to land closer. By the second rules on the same
  quarter, RR errs by −8.1×10⁻³⁸ at N + 8 and −5.2×10⁻³⁸ at N − 8. 07a's accuracy is of the same scatter, about 10⁻³⁷,
  not below it.

### Question 4: one call or a class

From what this Part holds (`out/r2-28/summary.md`):
- **Calls.** Of the 305 referenced calls, including R31's record and the chase calls, 4 rows have every 07b-lineage
  code's error above its abserr. All four are R2-28: a:R2-28, a:R2-28~o, c-1e-7-0-13757-5 and c-1e-8-0-13757-5, which
  are two calls each scored twice.
- **Shared errors below the line.** Eight more rows show a shared error (the eight codes agree to a tenth of it) with
  every ratio above 0.5, none above 1:

  | call | rows | ratio |
  |---|---|---|
  | W2-2-14345-3 | 2 | 0.828 |
  | R2-08 | 2 | 0.783–0.805 |
  | R3a-3-843-a5 | 2 | 0.556 |
  | R2-00 | 2 | 0.542 |

  They were not attributed here.
- **Leaves.** This Part has references on trusted Levin leaves only for R2-28 (D1's and D2's are of CC leaves).
  - Of R2-28's 26–55 trusted Levin leaves in each partition, those above their e′ and resolved are:
    - the carrying leaf, at 1.247;
    - two at depths 11–12, at 1.008 and 1.041 (−3.4×10⁻⁴¹, −9.8×10⁻⁴²);
    - in the tree, N, Eg and Ea, one at depth 9 at 11.9× a tiny e′ (error 1.58×10⁻⁴², reference error 1.42×10⁻⁴²: at
      the reference's limit).
  - 07b's extension records the same kind elsewhere (07b `log.md`, observation 14): trusted Levin carriers on the pairs
    and v09, 1.17–2.66 e′ wrong, at 07b's variants. That was not re-measured here.
- **Count: one call (R2-28) in the scored calls; one carrying leaf above 1.05 e′ in the leaves read.**

### Question 5: table 3, R2-28's rows annotated

Table 3 is unchanged (`out/tables/scores_07c.md`). Section 2.6's rule and the qualification are not re-scored.

| variant | converged & honest | converged & dishonest | unconverged |
|---|---|---|---|
| tree | 235 | 0 | 64, including R2-28's 4 rows at 1.107ᵃ |
| S | 292 | 4: R2-28's rows, 1.143ᵃ | 3 |
| S2 | 292 | 4: R2-28's rows, 1.120ᵃ | 3 |
| N | 285 | 0 | 14, including R2-28's 4 rows at 1.109ᵃ |
| Eg | 228 | 0 | 71, including R2-28's 4 rows at 1.102ᵃ |
| Ea | 228 | 0 | 71, including R2-28's 4 rows at 1.102ᵃ |
| SEg | 287 | 4: R2-28's rows, 1.139ᵃ | 8 |
| SEa | 287 | 4: R2-28's rows, 1.139ᵃ | 8 |

ᵃ **Cause, under every code:** the trusted Levin leaf [22.3566866, 23.5393356] at depth 4. It is the same span and the
same hex value under all eight codes, and it carries −2.338×10⁻³⁷ = 1.247 of its e′ by 07b's M2. The rest of each
code's partition errs by +3.5 to +3.7×10⁻³⁹, which is 0.16–0.27 of the code's abserr less that leaf's e′. The ratios
differ between codes only in the abserr each sums. No code's own change adds to the error.

### Deviations (extension)

25. **A slip in `issues.md` item 7:** it says R2-28's reference was read "by fccsplit and QAWO at both resolutions".
    The scored reference has two readings, fccsplit r1 and r2 (`scores_07c.json`: basis `fccsplit-r1+fccsplit-r2`,
    spread 8.42×10⁻⁴²). The orchestrator found it. The earlier text is left as written, on the orchestrator's
    instruction. UNINTENDED DRIFT.
26. **`r2_28_spec_07c.py`'s first run (record 48) refused the ~o calls,** because their capture carries an rtol
    override. The blob and so the integrand are the same, so the reference is read from the 1e-7 capture, and the
    rerun (48b) checks the blob only. UNINTENDED DRIFT (a failed first attempt; nothing written).
27. **07a's region is read from 07b Part A's replay at 07a's code**
    (`var/levin-phase-floor-07b/measurement/out/replays/tree/R2-28.json`, read only). It is probed at the tree's module,
    because 07a's worktree no longer exists. The tree's module reproduces 07a's region (V_P and Q) by hex, and the
    tree's instrumented root check records the same own value and quarters' sum. 07a (`1ec734d7`) and 07b (`5e19c4b6`)
    differ by the rate rules' commit alone. STRUCTURALLY REQUIRED.
28. **One reference reading serves every partition.** The halves, quarters and eighths are read for the trusted leaves
    of the tree's, S's and 07a's partitions only, not the other variants'. The other variants' carrying leaf is the same
    leaf. IMPLEMENTATION CHOICE.
29. **The second rules are the forced Clenshaw-Curtis pair and Levin at N ± 8.** No rule independent of the Levin
    collocation and accurate at these phases was run on the pieces. The leaf references (split FCC) are the independent
    measure. IMPLEMENTATION CHOICE.
30. **Question 4's leaf-level count covers R2-28's partitions only.** This Part has no references on trusted Levin
    leaves elsewhere, and the orchestrator asked for no new population. STRUCTURALLY REQUIRED.

### Verification (extension)

- **The worktree and the gate:** the worktree's git status was empty after the extension, and the main checkout stayed
  at `28ebaf07`.
- **The probes reproduce their leaves by hex:** each probe's driver solve reproduces its leaf, V_P and Q, on all three
  spans (`mech.md`).
- **The reading:** in every partition the leaf errors sum to the call's error against 07b's reference to four digits.
  The reading's whole span agrees with 07b's call reference to 8.3×10⁻⁴², within its stated 2.2×10⁻⁴¹.
- **The numbers:** every figure in this section's Result was checked against `leaves.json`, `mech.json` and
  `summary.json`.

### Observations not acted on (extension)

`issues.md` items 11 and 12.

### State handed on (extension)

- **Nothing is running.** The extension launched no registered run; its longest command took 6.1 min (record 49).
- **Records:**
  - the outputs are `out/r2-28/` (`leafref.json`, `leaves.{json,md}`, `probe.json`, `mech.{json,md}`,
    `summary.{json,md}`);
  - the inputs are `state/r2-28/` (`spec.json`, `probe_spec.json`);
  - the commands are records 48–53.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` gains one file in an appended section, `out/r2-28/leaves.json`
  (2,861,998 bytes).
