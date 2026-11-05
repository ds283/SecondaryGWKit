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

*(Verdicts are scored after the censuses. Until then, the leaf-level numbers are in `out/partL_summary.json`, and K0's
in `out/k0/summary.json`.)*

## Deviations from the prompt

*Kept up to date during the run; last brought up to date 2026-11-04T10:13Z, before any census.*

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
    of 1,882 trusted leaves). Put to the user at planned stop 1. **IMPLEMENTATION CHOICE**, pending the ruling.
14. **Route 3 as built** ranks only the untrusted refinable leaves, and an untrusted leaf at `depth_max` keeps the
    call unconverged. η_min is recorded both ways. Put to the user at planned stop 1. **IMPLEMENTATION CHOICE**,
    pending the ruling.
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
