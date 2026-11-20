# 07c Part C — orchestrator's checks at planned stop 2 (2026-11-19, ~15:45 UTC)

1. **Re-runs** (`../stop2_reruns.sh`, its output `../stop2_reruns.out`; the census agent's harness, run by me with
   output here; 15:33:40Z–15:37:15Z; the worktree's `status --short --ignored` empty after):
   - two K0 units at the parent (07b's code, via the harness's variant loader): `h-1e-8-0-2108` (23 calls) and
     `h-1e-8-3-14505` (21 calls). Each equals 07b's census unit (`ck07b_lines.jsonl`, grepped from checkpoint
     `396784bc…`) in every Levin call's value, abserr (hex), `converged`, `num_regions`, `max_depth`, and in the unit's
     `total` and `total_abserr` (hex);
   - one unit where the fix acts, `h-1e-8-0-33929`, run twice at the tree in fresh workers: identical in every call
     and both totals; equal also to the agent's Y0(4) run 1. Its calls #2 and #3 each have one chase stop, and each
     equals (S)'s replay record `c-1e-8-0-33929-{2,3}.json` in all five outputs. Against 07b's census those two calls
     move (unconverged at 40 regions, depth 20 → converged at 30 regions, depth 15), as the fix intends.
   - Reader: `scripts/compare_stop2.py` (my own); results `compare_stop2.txt`. Its first version mis-read hex strings
     in the (S) replay files; fixed and re-run, both outputs kept in the file.
2. **The predictions** (`census/state/predictions.md`, `cad4fd7f…`, mode 0444): mtime 15:30:24.71Z, after the latest
   Y0 output (`out/y0/y0a_imports.txt`, 15:29:50Z) and before any census run (none exists). Its four verbatim blocks
   (§6 lines 502–521, §9.5 663–697, §9.7 720–724, §8 538–539) each equal the prompt's lines exactly. Its part 5
   readings were read; they follow the prompt (Y9 (i) sets aside budget misses with no depth-20 leaf, as §9.2 and
   §9.5 treat them).
3. **Protected files at stop 2** (`checks/protected_stop2_*`, `checks/extra_stop2.sha256`): the 160 recorded files all
   OK, 150 files in 07b's census run directory, the 17,277 further files unchanged against the before-hashes, no
   `-wal`/`-shm`.
