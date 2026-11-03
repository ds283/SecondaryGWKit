**Verdict: Part B is complete and its records stand.**
- **The §5 stop on V8 fired.** It was put to the user, who ruled to score the rest for the record. V8's verdict, fails, stands.
- **No §7 stop fired.** Gate V holds on all three parts.
- **V7 holds.** V1, V2, V5, V6 and V8 fail.
- **V3:**
  - holds by its rule as written: 66 of 90 count, and the predictions require every Route B reading to be all_monotone;
  - fails with the new tool's readings accepted;
  - fails as the user's ruling scores it.

  In both failing readings the only violation is z12, at 1,681 × its abserr.
- **V4** holds both ways.
- **All ten review checks pass.** The slips found are recorded below. One question is open for the user: whether to rerun three stopped QAWO readings.

# levin-phase-floor 06g, Part B: the orchestrator's review (Mac Studio)

Written 2026-11-03 by the orchestrator on `mps026298.phys.susx.ac.uk`.

**Its sources:**
- the prompt, `prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md`;
- the note, `orchestrator/prompt-06g.md` §B and §B.2;
- the user's rulings during the run, each quoted below as relayed by the laptop orchestrator.

**Where the evidence is:**
- every command I ran myself, with its output, is under `review/checks/`;
- my readers are under `review/scripts/`;
- the briefs and the six addenda to the references brief are under `review/`, with their SHA-256 in
  `review/dispatch_sha256.txt`;
- the rulings, in order, are in `review/rulings_log.md`.

I wrote no code in the trees, and ran no `git` command that writes. The worktrees `ff3e839` and
`b5626ec` were fetched and added by the laptop, not by me.

## The trees and the runs

| | Tree | Run | Result |
|---|---|---|---|
| R23 | parent `4f4f9ed` (06h's code) | `levin-phase-floor-06g-r23-parent-20261101T110641` | done, exit 0, `1b332ffa…` |
| R24 | `1c5336e` (code `3ae6f3a`) | `levin-phase-floor-06g-r24-leafvalues-20261101T111215` | done, exit 0, `1b332ffa…` |
| Census | g = `1c5336e` | `levin-phase-floor-06g-census-20261101T114315` | done, exit 0, 18,560/18,560, **5 h 40 min** wall (06h: 2 h 33 min) |
| References, part 1 | tool at `1c5336e` | `levin-phase-floor-06g-references-20261101T175231` | done, exit 0, 60/60 |
| References, part 2 | tool at `1c5336e` | `levin-phase-floor-06g-references-20261101T201415` | done, exit 0, 380/380: 363 done, 9 error, 8 stopped |
| Rerun batch | tool at `ff3e8394` | `levin-phase-floor-06g-references-rerun-20261102T185324` | done, exit 0, 67/67: 64 done, 3 stopped |
| FCC batch (laptop) | tool at `b5626ecb` | `levin-phase-floor-06g-fcc-laptop-20261102T233229` (laptop's registry) | done, exit 0, 32/32; copied to `references/out/readings_fcc_b5626ec_laptop/` |

**Worktrees** (`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-…`):
- `4f4f9ed`, `1c5336e` and `b5626ec` are clean including ignored files;
- `ff3e839` holds 71 `.pyc` files in 16 `__pycache__/` directories, see Slips.

The main checkout stayed at `28ebaf0`, clean. The stash list is unchanged: the two unrelated
Mac Studio stashes, and none of 07's. Python 3.13.16.

## The predictions

| | Rule as written | Numbers | Verdict |
|---|---|---|---|
| **V1** (§4) | R23/R24 consistency on `total` and the oracle | On `total`, 12 rows violate, worst 13,101.5 (k = 1e5, z = 5236.55), each confined to the sub-interval-1 G+q+r call. On the oracle, 0, worst 0.4986. 294 rows bit-identical on `total` | **fails** (a finding, not a stop) |
| **Gate V** | the parent reproduces 06h; the integrand is 06h's; the new code is deterministic | P1–P3 identical to 06h on result, oracle and every call. The diff touches only `AdaptiveLevin/`, `tools/`, tests and CLAUDE.md. G1 = G2, with 2 CC calls with ΣE ≠ 0 and 107 trusted leaves | **holds** |
| **V2** | rows agree, `total` and oracle | The same 5 rows at each rtol violate on `total`, worst 2.164 (`0-12959`), all from the `G+r` call. The oracle: 0, worst 0.499 | **fails** |
| **V3** | honest where referenced | See below | **holds as written; fails** with clauses (ii)/(iii) accepted, and as ruled (z12) |
| **V4** | Z3 samples 2, 8, 9, 14 ≤ 0.01 | z02 0.00121, z08 2.1e-7, z09 1.44e-4, z14 0.00121 | **holds** both ways |
| **V5** | no call at `depth_max` outside the 720 | 966 calls in 577 rows (1e-7) and 996 in 582 (1e-8), against 06h's 0 and 1; 165 and 168 of them still converge; no unit over budget | **fails** |
| **V6** | `g-1e-8-0-33995` #5 below `depth_max`, converged or node_limited | node_limited, but at depth 20; 963 regions, ΣE −5.71e-35, Σ\|E\| 6.51e-34 | **fails** |
| **V7** | `g-1e-8-3-1072` #13 converges | abserr 1.544e-26 against a request of 1.629e-26; 2 regions; depth 1 | **holds** |
| **V8** | no convergence lost outside the 720 | 801 calls in 470 rows (1e-7); 830 in 476 (1e-8). All but 3 are (b) refined to depth 20; 3 are node_limited (1e-8). One gained: V7's | **fails**: the §5 stop |

**V3 in full** (`references/out/scores_addendum6/`, with the stage-3 addition in
`scores_stage3/`):

| Reading of the rule | Counted | Verdict |
|---|---|---|
| As written (the predictions require every Route B reading all_monotone) | 66 of 90 | holds, no violation |
| Clauses (ii) and (iii), the new tool's direct segments, accepted in place of all_monotone | 68 | fails, z12 |
| As the user's ruling scores it, every reference scored (82 with two implementations, 8 FCC-only) | all 90 | fails, z12, also on two-implementation references alone |

**z12** is `g-1e-8-1-2174` #1, one of the 720 stationary-point rows.
- **The miss:** g's value is **1,681.35** × abserr_g from the reference (06h: 1,678.07). g reports
  it **converged**, at depth 18 with 135 regions.
- **The reference is sound:** its four readings agree to 2.6e-4 of the tolerance.
- **It is the same integral as z00.** By the Route B criterion and the integrand at 65 points, z00
  and z12 are equal; they differ only in rtol.
- **z00 at 1e-7** is unconverged at depth 20 and lands 0.018 abserr_g from the same reference.

The next largest ratio in V3 is 0.58.

## The user's added samples (scored apart from V3)

**V1's 12 group calls** (addendum 1). Their references, after the FCC fix at `b5626ecb`, agree to
3.1e-6 to 3.5e-4 of the tolerance.
- **R24 (06g) is nearer than R23 on all 12.** R23 is 4.5e3 to 6.0e5 of its abserr away.
- **R24's declared error covers its distance on all 12**, at 0.0018 to 0.942.
- My own recomputation agrees (`checks/v1_a6_spotcheck.txt`).

**V8's 12 lost calls** (addendum 2, selected by me before any reading; `review/v8_sample.json`):
- g, unconverged, is within its abserr on 11 of 12; v03 is at 1.20;
- 06h, converged, is within on 9 of 12; v03 2.31, v08 4.29 and v11 1.07 are not.

## The descriptive items (the census agent's, from `census/out/`)

**The values.**
- On the same partition, no value moves by more than abserr_g.
- On a different partition, the median ratio is 0.61; 1,805 calls at 1e-7 move by more than
  abserr_g.

**The slow leaves behind V8** (addendum: the user's request):
- **r:** 4,342 leaves at 1e-7, all at depth 20, clustered at **r ≈ 1/2**: 1,763 lie within 0.01 of
  it.
- **Δ1/φ_P:** median 2.27, maximum 44.6.
- **Ancestor chains:** r wanders and settles only in the last levels.
- **Position:** a median of 2 clusters per call, none at the call's interval ends.
- **By group:** every group has the same r distribution.

**The node sum, the exit and cost.**
- **The node sum:** Σ\|E\|/\|ΣE\| has median 1, p99 about 100, maximum 1.76e4.
- **The exit:** firings rose from 90,509 to 402,760 (1e-7).
- **Cost:** unit wall ×3.43 (1e-7) and ×2.96 (1e-8); group-call regions ×4.95 and ×3.81.

**The stationary-point rows and the near-singular class.**
- **The 720 stationary-point rows:** every total moved; 316 and 340 calls were lost.
- **The near-singular class:** 435 rows, the same as 06h; 43 group calls were lost at each rtol.

**The correction −ΣE** (references): it moves the value nearer the reference on 12 of 25, and is
about neutral on most. My recomputation on r51 gives 0.3495, the agent's figure.

## The ten checks

1. **Scope: passes.** `var/` against the pre-dispatch listing of 71,558 files:
   - **0 removed, 0 changed, 184 added** (`checks/var_files_*.txt`), every one under
     `var/runs/levin-phase-floor-06g-*` or `var/datastores/levin-phase-floor-06g-r2{3,4}*`;
   - outside `var/levin-phase-floor-06g/`, no pre-existing file is newer than the dispatch marker;
   - the A3 v2 scratch copy is EQUAL to its certification after (`checks/a3v2_copy_after.sha256`);
   - 06h's checkpoint is `3104c18e…` after;
   - 07's work is untouched.
2. **The runs: pass.** The registry holds 46 runs, none running or stale; the only failed or killed
   runs predate Part B. Every 06g run is `done`, exit 0, on this host. The census's and the
   references' manifests carry the main checkout's `28ebaf0`, their cwd; their records carry the
   trees.
3. **The fingerprints: pass.** `runregistrykit store fingerprint`, no `--write`, from `1c5336e`: R23
   and R24 both `1b332ffa…`, problems none. The store bytes are unchanged by my reads
   (`checks/fingerprint_*.json`, `rehearsal_stores_*.sha256`).
4. **The values: pass.**
   - **The store diff:** my `tools.store_value_diff --json` is byte-identical to the agent's
     (`36cd46a4…`).
   - **Outside `QuadSourceIntegral`:** 05's classifier (`26690e61…`) on it finds the 7 value tables
     identical (360,462 rows) and 0 unclassified differences.
   - **V1:** `v1_rescore.py`, my own reader of the stores, gives every number above.
5. **The readers and the lock: pass.** Readers 6/6 exit 0; R24 is byte-identical before and after.
   My own `status` and `ray-processes`, each a command of its own, after the rehearsals, after the
   census and at the end: the lock is free and no Ray process is left (`checks/lock_*.txt`).
6. **Gate V: passes**, by my own reader against 06h's checkpoint and between G1 and G2
   (`checks/gate_v_check.txt`), and by my own `git diff` (`checks/gate_v_part2_and_trees.txt`).
7. **The trees: pass.**
   - **The census:** all 18,560 census records are at `1c5336e`, clean, with script `5f21f072…`, and
     every unit loaded its modules from the code worktree (`checks/census_provenance.txt`).
   - **The tools:** the readings record their tool's SHA-256: `d4bfa7c0…` (1c5336e), `5ed56fc2…`
     (ff3e839) and `d37e633b…` (b5626ec). Each matches its worktree's file.
8. **The predictions: pass.**
   - census `state/predictions.md` at 11:42:36Z, before its launch at 11:43:15Z;
   - references `predictions_20261101T175120Z.md` at 17:51:20Z, before part 1 at 17:52:31Z;
   - the addenda's rule files before their readings;
   - my V8 selection at 17:33:53Z, before any reading (`checks/predictions_mtimes.txt`).
9. **The verdicts: pass.**
   - **The census verdicts:** `census_rescore.py`, mine and independent of the agents' scripts,
     reproduces V2, V5, V6, V7 and V8 from the two checkpoints exactly
     (`checks/census_rescore.txt`).
   - **V1, z00 and z12:** I recomputed them from the readings.
   - **One reference re-made another way:** r51, by FCC at a resolution finer than either of the
     tool's (fit (28, 361), 5 pieces of N = 64), agrees with its reference to **3.4e-5 of the
     tolerance**. Its −ΣE ratio is 0.3506 against 0.3495 (`checks/remake_r51_and_sumE.txt`).
10. **The records: pass.** Each agent wrote `log.md`, `verification.md`, `issues.md` and
    `KEPT_ON_MAC_STUDIO.txt`, with every deviation classified: rehearsal 6, census 28, references
    33. All kept-file SHA-256s verify: rehearsal 11, census 29, references 419
    (`checks/kept_verify.txt`).

## Stops, questions and how they were answered

1. **The §5 stop on V8.** The (b) threshold cost convergence on 470 and 474 rows. The user ruled:
   "(b) with the V8 sample, please". Score the rest for the record, and add 12 of V8's lost calls to
   the references.
2. **V1's 12 added to the references,** scored apart. The user: "Happy to have the Filon references
   added – that sounds like a good idea."
3. **The slow-leaf breakdown,** requested by the user: "Yes, please do that".
4. **The references' cost gate** (part 2 projected at 20 h). The user: "Quibbling over ~8 hours of
   compute that only has to be run once strikes me as a false economy". The 30-minute cap became a
   check, not a cutoff, and every reference is scored.
5. **r52 and part 2's backstop.**
   - **What happened:** backstop (b) stopped six readings at projection + 1 h. The projection was
     per segment, whereas QAWO's cost follows phase span (0.057–0.10 µs/rad).
   - **The user:** "Leave the backstop alone for the rest of part 52". A rerun batch of the
     stopped readings followed: "Happy with this".
6. **The ZeroDivisionError.** It was a stationary point of the phase inside a Route B segment, in 14
   samples (w00–w11, z00, z12). Their readings were held.
   - **The user** approved the laptop's tool fix: "I agree with your implementation (1) and (2)".
     The result is `ff3e8394`, with all 56 readings rerun.
   - **The FCC fix:** FCC near a stationary point was "worth fixing", and was re-run "because we
     don't trust them". The result is `b5626ecb`, run on the laptop: "Can you run the FCC batch
     here, leaving Stage 3 still running on the Mac Studio?"
7. **r24 qawo r2: not added.** It is added only if r24 r1 disagreed with FCC. It agreed to 5.6e-5
   of the tolerance.
8. **When to score:** after stage 2. "we can use the result to inform our view on whether we should
   let Stage 3 run to completion."
9. **Open: whether to rerun r47 qawo r2, r53 qawo r2 and r26 qawo r1.**
   - The rerun batch stopped them on a projection that ignored the chunks in flight.
   - Corrected (phase-span model, checked on the finished readings), they need 3.0, 8.9 and 13.7 h.
   - No verdict depends on them.
   - The report is `review/rerun_batch_end_report.md`. My two attempts to send it to the laptop
     failed on a hook timeout.

## Slips, and what they touched

**Mine:**
- **The references brief and 06i's inputs.** My brief said 06i's 13 inputs were 06e's captures at
  `6d81fad`. They are 06c depth20-tiling captures from `e5764a6`, which 06e replayed. This did not
  affect the comparison.
- **Part 2's backstop (b).** I passed on the agent's backstop without checking it against the
  user's "only when absurd" intent. Six readings were stopped that would have finished in hours.
- **The rerun batch's projection.** My addendum 4 asked for progress "counted out of order" but not
  for the chunks in flight. Three readings were stopped wrongly as a result.
- **The ier flag rate.** In the r52 report I quoted the laptop's "92% of panels" before correcting
  it to 23% of QUADPACK calls.

**The agents' slips,** each recorded in its log:
- **UNINTENDED DRIFT:** rehearsal 1, census 4 and references 5. They include:
  - a lock listing piped through `sed`;
  - a harness script first run from a scratchpad;
  - an early `sqlite3 -readonly` query before a hash. The hashes still matched.
- **The scorer's outputs:** the references' scorer overwrote its own first-run outputs (deviation
  27).
- **The `.pyc` files:**
  - 71 `.pyc` files in the `ff3e839` worktree, and one in `references/scripts/__pycache__/`, all
    dated 2026-11-02T23:12:03–05Z;
  - not from the rerun batch, which ran with `-B`;
  - most likely the laptop's read-only slope survey run on this host;
  - harmless to values, recorded, and nothing deleted.

  The stray `.pyc` must not be copied as a record.

## For the laptop: what to copy back, and what stays

**Copy back** everything under `var/levin-phase-floor-06g/` except the files each
`KEPT_ON_MAC_STUDIO.txt` lists:
- `rehearsal/`, 224 MB in all;
- `census/`, 39 MB;
- `references/`, 398 MB, without its `scripts/__pycache__/`;
- `review/`, 10 MB;
- this file.

**Also copy** the small run directories:
- `var/runs/levin-phase-floor-06g-r23-parent-20261101T110641/` (28 K) and
  `…-r24-leafvalues-20261101T111215/` (32 K), with `…-r2{3,4}.launch.log`;
- `…-references-20261101T175231/` (64 K), `…-references-20261101T201415/` (384 K) and
  `…-references-rerun-20261102T185324/` (176 K).

**Kept on the Mac Studio only,** until the user says otherwise:
- **The R23 and R24 stores,** `var/datastores/levin-phase-floor-06g-r2{3,4}*.sqlite`, ten files
  (primaries `88cc355f…` and `6e4815c8…`).
- **The census run directory in full:** `var/runs/levin-phase-floor-06g-census-20261101T114315/`,
  4.5 GB, checkpoint `03adcbba…` and 16 region pickles. It is 07's "before" and the references'
  source.
- **`rehearsal/out/store_value_diff.json`** (`36cd46a4…`, 42 MB) and `rehearsal/extract/`
  (182 MB).
- **The census's pickles,** `state/rows.json`, `out/v8_calls.json` (`1e9a62b0…`) and
  `out/v8_slow_leaves.json`.
- **The references' 404 pickles and files over 2 MB** (218 MB).

The three lists are:

| List | SHA-256 |
|---|---|
| `rehearsal/KEPT_ON_MAC_STUDIO.txt` | `f1489bf0…` |
| `census/KEPT_ON_MAC_STUDIO.txt` | `c8e24de6…` |
| `references/KEPT_ON_MAC_STUDIO.txt` | `284034e7…` |

**The worktrees:** `SecondaryGWKit-lpf06g-{4f4f9ed,1c5336e,ff3e839,b5626ec}` stay, for the
references' reproducibility, until the user says otherwise.

**Candidate issues for the board,** with full text in each agent's `issues.md`:
- **Rehearsal:**
  - `[06g-r24-g+q+r-group-moves-past-both-errors]`, now answered by V1's references: R24 is
    honest, and R23 was wrong by 4.5e3 to 6e5 of its error.
- **Census:**
  - the C1–C7 findings: V8's slow leaves at depth 20, V5, V2, V6, the cost, and the 3 node-limited
    losses to add to `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`, plus r
    clustering at 1/2.
- **References:**
  - z12, converged but 1,681 × abserr off;
  - the two tool defects, closed by `ff3e8394` and `b5626ecb`;
  - the two backstop defects;
  - the 9 error readings.
