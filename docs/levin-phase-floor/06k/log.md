# levin-phase-floor 06k — three ways to stop refining the slow leaves, measured on A3 v2 (a measurement)

**Subject:** prompt 06k (`prompts/levin-phase-floor/06k-slow-leaf-trust-routes.md`), on the Mac Studio
(mps026298), 2026-11-04.
**Commit:** none; records committed by the orchestrator from the laptop.

## Predictions and verdicts

*Copied in at 2026-11-04T09:00Z, before any K0 unit, Part L probe or census. The record is
`state/predictions_and_rules_20261104T0915Z.md` (read-only, SHA-256 `e315d74d…`, mtime 08:59:36Z),
which also holds K0's draw rule and seed and the operational definitions. Prompt 06k §4, verbatim:*

README §5 rule 3. Copy them into the log unchanged. Each is scored per route, **holds**, **fails**
or **undecided**, against the rule as written. None is a stop. In each rule, "outside the 720"
means outside the 720 stationary-point rows.

- **L1 — the route does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves
  (L-b), the route trusts or leaves none.
- **L2 — the route reaches the slow leaves.** **Rule:** the route trusts or leaves at least 80 % of
  06j's 142 sampled slow leaves (L-a).
- **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
  each rtol, converge under the route.
- **C2 — nothing converged is lost.** **Rule:** no call outside the 720 that 06g converges is
  unconverged under the route.
- **H1 — no referenced call gets worse.** **Rule:** on every referenced call outside the 720,
  $|v_{\rm route} - {\rm ref}|/e_{\rm route} \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
- **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than
  $e_{\rm route} + e_g$ against 06g's census.
- **K1 — routes 1 and 2 fire only on noise.** **Rule:** every leaf the route trusts that 06g
  refined has $\Delta_1/\phi_P < 5.8$ (c07's carrying leaf, the smallest of 06h's kind). Scored for
  routes 1 and 2. Route 3 decides by size, not by noise. Its firings are reported by
  $\Delta_1/\phi_P$ and by $D$ against the request, and K1 is not scored for it.

**The basis.**
- **L1, H1, S1 and K1** are the safety constraint of §0.1, stated four ways: at leaf level on the
  known bad leaves, on the references, census-wide, and on every firing. Route 3's safety rests on
  L1, H1 and S1 alone, and on $\eta$'s margin.
- **L2 and C1** are what the remedy is for. The 80 % follows 06j; 90 % leaves room for calls held
  by something other than slow leaves.
- **C2** holds by construction if the route only adds trust, and is a check that it does.

A route passes as a candidate remedy if L1, C2, H1 and S1 hold, and K1 for routes 1 and 2. L2 and
C1 say how much it recovers. Every number behind each verdict is reported, so that a different
reading can be made from the record.

<!-- verdicts are added below this line when scored; the block above is not edited -->

| Prediction | route 1 | route 2 (as built) | route 2 literal | route 3 |
|---|---|---|---|---|
| **L1** | **holds**: 0 of 17 carrying leaves trusted or left | **holds**: 0 | **holds**: 0 | **fails**: 1 (c01, left at D = 1.7e-4 of η × request) |
| **L2** | **holds**: 114 of 142 (80.3 %) | **fails**: 57 (40.1 %) | **fails**: 57 (40.1 %) | **holds**: 142 (100 %) |
| **C1** | **holds**: 792/801 (98.9 %) at 1e-7, 818/830 (98.6 %) at 1e-8 | **holds**: 801/801, 828/830 | **holds**: 794/801, 821/830 | **holds**: 801/801, 829/830 |
| **C2** | **fails**: 7 calls lost (1e-8, node_limited) | **fails**: 5 (1e-8, node_limited) | **fails**: 53 (51 untrusted leaf, 2 node_limited) | **fails**: 1 (1e-8, node_limited) |
| **H1** | **holds**: 115 of 115 references outside the 720 at ≤ max(1, 06g's ratio); worst 1.185 (v03, 06g 1.203) | **holds**: worst 1.202 (v03) | **holds**: worst 1.200 (v03) | **fails**: 2 of 115: c01 3.935 (06g 0.0014), r56 1.492 (06g 0.0059) |
| **S1** | **holds**: 0 calls | **fails**: 10 calls in 10 rows, worst 1.323 | **fails**: the same 10, worst 1.323 | **fails**: 46 calls in 44 rows, worst 1.645 |
| **K1** | **fails**: 89,171 of 229,338 trust firings outside the 720 at Δ1/φP ≥ 5.8 (max 1.1e4) | **fails**: 82,565 of 167,281 (max 3.7e5) | **fails**: 88,186 of 191,623 trust additions (max 3.7e5) | not scored (the rule) |

**Notes on the scoring.**
- **H1 is scored on the 124 references the censuses can match** (115 outside the 720): 06g's 102 and 06j's 22.
  V1's 12 are calls of 06g's rehearsal store R24, not of A3 v2's census. They are unscored, on the user's ruling
  (deviation 24).
- **K1** counts every firing that trusts a leaf 06g's rule finds untrusted (for route 2 literal, the `trust_added`
  firings), in group and analytic calls. The verdict is the same inside the 720: 15,491 of 38,240 (route 1), 13,781 of
  29,318 (route 2) and 15,047 of 35,256 (route 2 literal).
- **No route passes as a candidate remedy** by §4's rule (L1, C2, H1 and S1, and K1 for routes 1 and 2). Route 1
  fails C2 and K1 only.
- **Where the numbers are:** `out/results.md`, `out/partC/summary.json`, `out/partC/h1.json`,
  `out/partL_summary.json` and `out/k0/summary.json`.

## Result

- **Gate K0 holds.**
  - The identity variant and the tree's own module reproduce 06g's census bit for bit on all 100 units.
  - Every route's calls with no firing are bit-identical to 06g's: 1,855 calls for route 1, 1,837 for route 2,
    1,032 for route 2 literal and 683 for route 3.
- **Every route recovers convergence.** C1 holds for all four, at 98.6–100 % of V8's lost calls outside the 720.
- **Every route costs less than 06g.** Unit wall is 0.78 (route 1), 0.60 (route 2), 0.62 (route 2 literal) and 0.30
  (route 3) of 06g's; 06h's was 0.32.
- **Every route also loses some calls that 06g converges** (C2), and none passes as a candidate.
  - **Route 1** keeps L1, H1 and S1. Its noise yardstick passes leaves far above the noise level (K1).
  - **Routes 2 and 2 literal** move 10 calls past both errors. Their declaration leaves out the node error of the Q
    they return.
  - **Route 2 literal** also withdraws trust (30,158 withdrawals) and loses 53 calls.
  - **Route 3's** left leaves can be wrong by more than the call's error: 46 moves, 2 references missed, and z00
    in the 720 at 1,681×.

## What shipped

No code. The records are in `var/levin-phase-floor-06k/`: `log.md` (this file), `verification.md`, `issues.md`,
`KEPT_ON_MAC_STUDIO.txt`, and the directories below.

**`out/`**
- every command with its output (`out/cmd/`, `out/commands.log`);
- the variants' diffs and AST checks (`out/variant_*_diff.txt`, `out/cmd/11b`, `91`, `92`);
- K0's records (`out/k0/`);
- Part L's tables (`out/partL_leaves.csv`, `out/partL_summary.json`, `out/partL_route2lit_withdrawals.csv`,
  `out/partL_route2lit_by_solve_kind.json`, `out/partL/`);
- the draws (`out/lg_draws/`);
- Part C's tables (`out/partC/`): convergence, depth, moves with their carrying leaves, firings, cost, and the
  references' scores;
- `out/results.md`;
- the scripts' diffs against their sources (`out/scripts_diff_at_launch_k0partL.txt`, `out/scripts_diff_final.txt`).

**`scripts/`** holds 06j's scripts (06g's census scripts with 06j's), 06g's calibration scripts and 06k's own, plus
`scripts/variants/` with the five variant files:

| variant | SHA-256 |
|---|---|
| identity | `d89c959e…` |
| route 1 | `23cd69e4…` |
| route 2 | `017fae1c…` |
| route 2 literal | `2fd10b78…` |
| route 3 | `e139ec71…` |

**`state/`**
- the predictions, rules and seeds, and the K0 correction (both read-only);
- the references' copies of the scripts;
- the launch transcripts and their checks;
- the captures (`state/capture`, `state/inputs`) and K0's worker directories (`state/k0/`);
- the census extracts (`state/extract/`);
- the pre-change copies of edited scripts;
- the development runs (`state/dev/`).

**Seven registered runs, every one finished:**
- `levin-phase-floor-06k-k0partL-20261104T091640`: `failed`, for its draws command alone (deviation 8);
- `levin-phase-floor-06k-lgdraws-20261104T095904`: `done`;
- `levin-phase-floor-06k-route2lit-20261104T101930`: `done`;
- `levin-phase-floor-06k-census-route1-20261104T103009`, `…-census-route2-20261104T151120`,
  `…-census-route2lit-20261104T185927` and `…-census-route3-20261104T225339`: each `done`, 18,560 of 18,560.

## The stages, in order

1. **08:49–09:16Z.**
   - Read the brief, the prompt and its reading list.
   - Made the Gate checks (`out/cmd/00`–`07b`).
   - Wrote the predictions, rules and seeds (08:59:36Z) and the log's block (08:59:47Z).
   - Built the variants and their AST check; wrote the harness.
   - Ran the development checks outside the scored units, drew K0's units, and wrote the K0 correction (09:11Z).
2. **09:16–09:54Z.** Run `k0partL`: captures, K0 under five configurations, Part L's probes, and the draws (which
   failed).
3. **09:54–10:10Z.** Scored K0 and Part L; reran the draws (run `lgdraws`); planned stop 1.
4. **10:15–10:30Z.** On the user's ruling: built the route 2 literal variant; ran its K0 and draws (run
   `route2lit`); stop 1b.
5. **2026-11-04T10:30Z – 2026-11-05T01:00Z.** The four censuses, one at a time, each cleared by the orchestrator.
6. **2026-11-05.**
   - Part C's analyses (`out/cmd/100`–`103`, `122`).
   - Stopped on V1's references; ruled (a).
   - Checks after (`110`–`121`); the records.

## Deviations from the prompt


1. **The predictions file's name** says "T0915Z" (`state/predictions_and_rules_20261104T0915Z.md`), but it was
   written at 08:59:36Z (its mtime), before any K0 unit. The name was a planned label. **UNINTENDED DRIFT**
   (cosmetic).
2. **The first Ray-process check was piped through `tail`**, which the lock then listed (`out/cmd/07`, exit 1). It
   was redone as a command of its own with `>>` (`07b`: none listed, exit 0). **UNINTENDED DRIFT.**
3. **`index_06g.py` ran with PYTHONPATH `<tree>:`**, because an unset shell variable left an empty second entry
   (`out/cmd/12`). An empty entry adds the working directory (`var/levin-phase-floor-06k/`, with no module at its
   top), so this was harmless. **UNINTENDED DRIFT.**
4. **The variants were rebuilt once** (`out/cmd/10` → `10b`, AST check `11` → `11b`), to rename route 3's helper
   locals so that every name the variants bind ends in `_06k`. Routes 1 and 2 and the identity variant are
   byte-identical across the two builds. The first build's script is `state/make_variants_as_first_run.py`.
   **IMPLEMENTATION CHOICE.**
5. **`cache_built` added to K0's excluded keys**, by `state/k0_comparison_correction_20261104T0911Z.md`, written
   after one development unit and before any K0 unit. It records what a worker's producer cache built while the
   unit ran, which depends on which units share a worker. It differs on 95 of 100 units under the tree's own module
   as under the identity variant, and it is still reported. **IMPLEMENTATION CHOICE.**
6. **K0 adds a control:** the tree's own module run on the same 100 units, with the same worker grouping
   (`state/k0/tree`, `out/k0/compare_tree.json`). **IMPLEMENTATION CHOICE.**
7. **The copied scripts' changes**, each marked "prompt 06k". Diffs: `out/scripts_diff_at_launch_k0partL.txt`, and
   the final diff at the end. **STRUCTURALLY REQUIRED.**
   - `prov.py`, `refs_common.py` and `calibration/prov.py`: the tree and the paths (and `--no-optional-locks`);
   - `worker.py`: the variant installed before anything imports AdaptiveLevin, the tree evidence and module roots
     that recognise it, and the route record `route_06k`;
   - `check_captures.py`: the changed scripts and the new key;
   - `calibration/lg_seeded_draws.py`: the five modes, each variant loaded by `variant_loader`, the route record, and
     the working directory moved out of the tree (`state/lg_cwd/`).
8. **The LiouvilleGreen draws failed in run `k0partL`** (`out/cmd/k0partL/70_lg_draws.txt`, exit 1 after 0.5 s,
   nothing written). My edit to `calibration/prov.py` (deviation 7) had put a comment on `subprocess.run`'s
   argument line, which commented out `check=True, capture_output=True, text=True`, so `git()` returned None in
   the provenance header. The edit is corrected (the as-run copy is `state/calibration_prov_as_k0partL.py`).
   The draws were rerun alone as the registered run `levin-phase-floor-06k-lgdraws-20261104T095904` (plan
   `lgdraws` added to `launch_06k.py`; its as-run copy for `k0partL` is `state/launch_06k_as_k0partL.py`), done
   with exit 0. Run `k0partL` is recorded `failed` for this command alone. **UNINTENDED DRIFT.**
9. **`score_l.py` first merged the L-b leaves by span across different calls** (11 "distinct" leaves: the `G+r`
   calls share the span [23.539, 25.255] on different integrands). The key is now the row, the call and the
   span, giving the prompt's 17 (`out/cmd/80` → `80b`). No verdict changes between the two. **UNINTENDED DRIFT.**
10. **`lg_compare_06k.py`'s first run failed** (`out/cmd/84`), because `prov.header` names its inputs relative to
    the tree. The input's hash is now written by the script itself (`84b`). **UNINTENDED DRIFT.**
11. **For Part C, prepared and not run:** `run_census.py` gains `--variant` and the backstop (`--projection-s`),
    `launch_census_06k.py` is new, and `state/rows.json` is a read-only copy of 06g's (`aa523d28`), which
    `run_census.py` reads from its records directory. **STRUCTURALLY REQUIRED.**
12. **Development checks outside the scored units:**
    - `g-1e-7-2-1449` under each variant (`out/cmd/14`–`17b`);
    - two captures, s00 and f00, with Part L probes on them (`18`, `19`, into `state/dev/`);
    - one draws unit per mode for tree and route 3 (`20`);
    - a one-unit census smoke test under route 2 (`83`).
    They score nothing. **IMPLEMENTATION CHOICE.**
13. **Route 2 as built** evaluates the corrected rule only where rule (b) finds a leaf untrusted. A leaf 06g trusts
    keeps 06g's decision and declaration. The literal reading is measured at leaf level (L-c: it would untrust 326
    of 1,882 trusted leaves). Put to the user at planned stop 1. **IMPLEMENTATION CHOICE.** The ruling
    (deviation 17) keeps it, and adds the literal reading as a fourth variant.
14. **Route 3 as built** ranks only the untrusted refinable leaves, and an untrusted leaf at `depth_max` keeps the
    call unconverged. η_min is recorded both ways. Put to the user at planned stop 1. **IMPLEMENTATION CHOICE.** The
    ruling (deviation 17) keeps it as built.
15. **A firing at `depth_max`** (a leaf 06g's rule returns untrusted, since it cannot refine it) is counted as a
    firing, reading "would have refined" as "finds untrusted". Firings are reported by depth, so those at
    `depth_max` can be read apart. **IMPLEMENTATION CHOICE.**
16. **Part L's route 3 request** for L-b is the call's request from 06g's census value, max(atol, rtol |v_g|),
    although the leaf is 06h's: the code under test is 06g's. Under 06h's value the request would move by
    rtol |v_g − v_06h|. That is not computed per call here; the nearest L-b leaf to the cut-off, c01, has D at
    1.7e-4 of η times the request. **IMPLEMENTATION CHOICE.**
17. **The user's rulings at planned stop 1** (relayed by the orchestrator, 2026-11-04):
    - on Q1 (route 2): *"OK, so this isn't really a contradiction, it's just that the note said that trust would be
      preserved if the leaf had `r < 1/4`, and it might not be. We had better run that fourth census."*;
    - on Q2 (route 3): *"stick with what has been built."*

    As the orchestrators read the rulings, route 2 is measured both ways, each as its own variant and census:
    route 2 as built (deviation 13) and **route 2 literal**. That makes four censuses: route 1, route 2 as built,
    route 2 literal, route 3. Route 3 stays as built (deviation 14, now ruled). **STRUCTURALLY REQUIRED** (the
    user's ruling).
18. **The fourth variant, `scripts/variants/levin_quadrature_route2lit.py`** (SHA-256 `2fd10b78…`), built by
    `make_variants.py` with its `route2lit` addition (`out/cmd/90`; the script as at stop 1 is
    `state/make_variants_as_stop1.py`). The other four variants are byte-identical to before.
    - **What it does:** every checked leaf is decided by rule (b) on the values corrected by −E, and declared on
      them. A trusted leaf declares max(residual, Δ1c)/(1 − r_c); an untrusted leaf at `depth_max` declares
      max(residual, Δ1c)/(1 − r_c) if r_c < 1, else max(residual, Δ1c + Δ2c).
    - **Its firings, by kind:** `trust_added`, `trust_withdrawn`, `declaration_changed` (trusted under both) and
      `declaration_changed_untrusted` (untrusted at `depth_max` under both). A leaf whose seven solves carry no
      E is corrected by exactly 0, so it never fires. *(Corrected 2026-11-04T10:30Z: the first text said "(every
      Levin leaf)", which is wrong. The correction is applied per solve, so a Levin parent whose halves or quarters
      are Clenshaw–Curtis solves does carry their E in Δ1c and Δ2c. Only a leaf whose seven solves are all Levin is
      unchanged. Counts:*
      - *Part L: of 1,009 L-c Levin parents, the 621 with all seven solves Levin are unchanged; all 388 with a
        Clenshaw–Curtis solve among the six others are corrected, and 23 of those change decision;*
      - *K0: 6,144 of route 2 literal's 12,091 firings are on Levin parents (5,005 `declaration_changed`, 1,029
        `trust_added`, 108 `trust_withdrawn`, 2 `declaration_changed_untrusted`). The firing record keeps the
        parent's branch, not the six others', but by construction each of these has at least one Clenshaw–Curtis
        solve among them.)*
    - **The diffs:** `out/variant_route2lit_diff.txt` (against the tree) and
      `out/variant_route2lit_vs_route2_diff.txt` (against route 2 as built).
    - **The AST checks:** against the tree, only `_adaptive_levin` differs (`out/cmd/91`). Against route 2 as
      built, only `_check_leaf` differs (`92`). Both are CONFINED.

    **STRUCTURALLY REQUIRED.**
19. **K0's clause 2 for route 2 literal.** The orchestrator rules that clause 2 cannot apply to this variant, since it
    changes the declaration of every checked Clenshaw–Curtis leaf; that is recorded here as a deviation, not a
    failure. Because every such change is counted as a firing (deviation 18), the calls with no firing of any kind
    are still compared with 06g bit for bit and reported. **STRUCTURALLY REQUIRED.**
20. **Drivers and harness extended for `route2lit`:**
    - `run_k0.py`, `run_census.py` and `launch_census_06k.py` accept it;
    - `lg_seeded_draws.py` gains `--modes` and `--out`, so that only the `route2lit` mode is run, into
      `out/lg_draws/lg_seeded_draws_route2lit.jsonl`, and compared with run `lgdraws`'s `tree` mode (the identity
      mode reproduced the tree bit for bit there);
    - `launch_06k.py` gains the plan `route2lit`.

    The pre-change copies are `state/*_as_stop1.py`. A development run of the dev unit under `route2lit`
    (`out/cmd/93`, `94`) was equal to 06g on its 6 calls with no firing. **IMPLEMENTATION CHOICE.**
21. **Scoring scripts extended for `route2lit`:**
    - `k0_summary.py` gains the firings by kind and by the parent's branch;
    - `lg_compare_06k.py` reads route 2 literal's draws file beside run `lgdraws`;
    - `withdrawals_l.py` is new: every L-c withdrawal, and the corrections by the solves' kinds.

    The pre-change outputs are `state/k0_summary_as_stop1.json` and `state/lg_compare_as_stop1.txt`; the other
    variants' rows of both are unchanged. **IMPLEMENTATION CHOICE.**
22. **Part C reads compact extracts of the six census checkpoints** (06g's, 06h's and the four routes'), made by
    `scripts/extract_census.py` into `state/extract/` (`out/cmd/100`). Each extract records its checkpoint's
    SHA-256; 06g's is `03adcbba…` and 06h's `3104c18e…`. **IMPLEMENTATION CHOICE.**
23. **Part C's comparisons are new scripts** (`analyse_c.py`, `movers_c.py`, `score_h1.py`), not reruns of 06g's
    `v8_06g.py` and `values_beyond06g.py`. Those two name 06g's and 06h's runs in their code. The new scripts apply
    their rules to each route against 06g:
    - calls matched by kind, index, label and span;
    - V8's lost calls from `v8_calls.json`, each one's hold-back classed as an untrusted leaf, node_limited or
      other;
    - V2's measure e_a + e_b in exact arithmetic.

    **IMPLEMENTATION CHOICE.**
24. **V1's 12 references are not scored.** The prompt (§3) names V1's 12 among H1's references. They are calls of
    06g's rehearsal store R24 (k = q = r = 1e5, 3.16e5 and 1e6), not of A3 v2's census: they carry no census unit,
    and their R24 values occur nowhere in 06g's census checkpoint. So no census can score them (prompt §7, "a
    reference cannot be matched to its call").
    - I stopped and asked. The user ruled: *"(a) as recommended, please"*.
    - The recommendation, as relayed: listing V1 was an error in prompt 06k §3 (06g's VERIFICATION §6g.4 records
      them as captured at R24's tree); V1 is dropped from H1; H1 is scored holds or fails on the 124 matched
      references, whose 115 outside the 720 decide it; V1's 12 are recorded as unscored and unmatched.
    - No store other than the scratch copy was read, and no new computation was made.

    **STRUCTURALLY REQUIRED.**
25. **The scoring scopes the rules leave open:**
    - **C2 and S1** count group and analytic calls alike. Every lost or moving call outside the 720 is a group call,
      except routes 2 and 2 literal's eight single-region analytic movers.
    - **K1** counts every trust firing in group and analytic calls. For route 2 literal it counts the
      `trust_added` firings, not withdrawals or declaration changes. It is scored outside the 720, with the counts
      inside reported; both readings fail.
    - **H1's 22 from 06j** include the 1e-8 twins f05–f09, each scored as its own call against its 1e-7 twin's
      reference, as 06j scored them.

    **IMPLEMENTATION CHOICE.**
26. **One input left out of the first after-check.** The after-check of the checkpoints and inputs (`out/cmd/115`)
    left out `v8_slow_leaves.json`. It was hashed in a command of its own (`115b`, `b4643033…`, equal to the Gate's).
    **UNINTENDED DRIFT.**
27. **A wrong in-session count, never saved as a record.** My first count of route 3 calls whose two η_min readings
    differ compared a sorted list with an unsorted one, and found 173,205. The recorded script, `eta_route3.py`
    (`out/cmd/122`), gives 1. **UNINTENDED DRIFT.**
28. **The cost projections were too high for routes 2, 2 literal and 3, and slightly low for route 1.** The census
    plan's P was 06h + (06g − 06h) × K0's unit-wall ratio. The censuses took:

    | route | took | P |
    |---|---|---|
    | 1 | 16,758 s | 16,183 s |
    | 2 | 13,621 s | 14,332 s |
    | 2 literal | 13,968 s | 15,037 s |
    | 3 | 7,626 s | 11,866 s |

    Only route 1 passed P. The backstop reported once, at 15:00:12Z (P_o 16,261 s, far below a day), and stopped
    nothing. **IMPLEMENTATION CHOICE** (recorded for the cost).

## Verification performed

`verification.md` has the full text. In brief:
- **The Gate:** every clause passes (`out/cmd/00`–`07b`).
- **The store and the inputs:**
  - the scratch copy is EQUAL to its certification before, after, and after the final fingerprint (`03`/`04`,
    `110`/`111`, `113`/`114`), and its fingerprint is `0e32f24a…` before and after (`05`, `112`);
  - 06g's census checkpoint, its 16 region pickles, `v8_calls.json`, `v8_slow_leaves.json`, `rows.json` and 06h's
    checkpoint match by SHA-256 after (`115`, `115b`);
  - the `var/datastores/` listing is unchanged.
- **No Ray:**
  - every worker of the four censuses and the six K0 configurations reports Ray not initialised and no foreign
    module (`out/cmd/120`, `121`);
  - every census worker's tree evidence holds, with its variant file and SHA-256;
  - the lock is free and no Ray process is listed after (`118`, `119`).
- **The trees:**
  - the worktree is at `89370609`, clean including ignored files, before and after (`00`, `116`), and no
    `__pycache__` was written anywhere;
  - the main checkout is at `28ebaf0`, clean (`116`);
  - the registry holds 55 runs, all finished, seven of them 06k's (`117`).
- **The variants:**
  - only `_adaptive_levin` differs from the tree's file (`11b`, `91`);
  - route 2 literal differs from route 2 as built only in `_check_leaf` (`92`);
  - the identity variant is byte-identical to the tree's file.
- **K0 and Part L:** as in the Result; the probes reproduce the census on every leaf where the census holds the check.

## Observations not acted on

1. **Route 3's declaration D can understate a left leaf's error by about 10³ or more.**
   - 46 calls outside the 720 move past both errors, each carried wholly by one coarse leaf left at
     D ≤ 10⁻³ × request.
   - Two references are missed: c01 at 3.9× and r56 at 1.5×.
   - In the 720, z00 goes from 0.018 to 1,681×.
   - η's margin assumed D within a factor of 1,000 of the leaf's error; on these leaves it was not.
2. **Route 2's declaration is short where it trusts a leaf on corrected numbers and returns Q.** Δ1c leaves out the
   node error by which Q differs from V_P. On single-region analytic calls this moves the value by up to 1.32 ×
   (e_route + e_g).
3. **Route 1's ν1 includes the halves' floors.** When a Levin parent's halves are Clenshaw–Curtis solves, these are
   hundreds of times φP, so route 1 trusts leaves up to Δ1/φP ≈ 1.1 × 10⁴.
4. **Every route loses a few 1e-8 calls to the node-limited stop.** The losses are 7, 5, 2 and 1 for routes 1, 2,
   2 literal and 3. In each, Σe′ meets the request while |ΣE| holds the call above it. The two issues in this
   family overlap: `[06e-node-errors-add-in-absolute-value]` and 06g's node-limited stop.
5. **Route 2 literal withdraws trust at r < 1/4**: 30,158 withdrawals in the census, 265 of the 326 at leaf level.
   This is the case §0.2's "06g's trust for any leaf with r < 1/4" unchanged excludes; it is measured here on the
   user's ruling.
6. **The capture pickles' bytes differ from 06j's.** The pickled code names the tree's path, and this is a
   different worktree. The captured calls reproduce 06g's census on every value.

## State handed on

- Nothing is running. The lock is free and no Ray process is listed.
- The four census run directories, the capture and K0 pickles and the extracts are kept on the Mac Studio
  (`KEPT_ON_MAC_STUDIO.txt`, with SHA-256 and sizes).
- **The board:**
  - `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` gains 06k's measurement, which proposes no
    remedy: no route passes as a candidate.
  - `issues.md` proposes two new issues and a narrowing.
- Which route, if any, becomes code is for the user.
