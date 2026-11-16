# Deviations noted as they happened (draft for log.md)

1. 07a's census harness copied whole into `scripts/` from the tree's `docs/levin-phase-floor/07a/census/scripts/`
   (byte-identical to the Mac Studio copy), with 07a Part A's replay/attribution scripts in
   `scripts/from_07a_measurement/`; `prov.py` changed where marked "prompt 07b"; `x1_capture_07b.py` from
   `x1_capture_07a.py` (title). IMPLEMENTATION CHOICE.
2. The first capture (`out/cmd/06_capture_h.txt`) failed before starting: `state/rows.json` (07a's copy of 06g's,
   aa523d28) had not been copied. Copied (SHA checked) and re-run as `06b`. UNINTENDED DRIFT (failed attempt).
3. `make_calls_07b.py`'s first manifest took every no-window call to be a group call; `n4` is analytic J2 (index 7),
   and the census extraction refused it. The kind is now read from 07a's checkpoint; the first manifest is kept in
   `state/superseded/`. UNINTENDED DRIFT.
4. The first tree replay command did not run (zsh does not word-split a variable); the first 07 replay of
   `i3-34599-07` failed on `own_err` (07's module has none) and wrote no record; its log is in
   `out/replay_logs_failed_first/`. Fixed (`getattr`) and re-run. UNINTENDED DRIFT (failed attempts).
5. `i3-34599-07` (07's capture, to be compared with 07's census) was included in the tree run, where it differs by
   design; the record is moved to `out/replays/not_a_test/`. It shows that 07's capture of 1-34599 at 07a's code
   returns 07a's census value bit for bit. UNINTENDED DRIFT (cosmetic).
6. R3's four shrinkers counted as four distinct (row, kind, index): 0-33666 J4, 3-39138 #2, 3-39138 #3, 2-21486 J2.
   The literal four smallest calls (0-33666 J4 at both rtols, 3-39138 #2 and #3) are all covered, the second rtol by
   the shared integrand. IMPLEMENTATION CHOICE.
7. R1's captures are 07's cross-check captures, re-checked by replay at 07a's code at both rtols, not captured afresh
   (the prompt allows either). IMPLEMENTATION CHOICE.
8. The identity of a (row, index)'s integrand across rtols is shown by replaying one capture at the other rtol's
   (rtol, atol) and reproducing 07a's census record there bit for bit, in place of 07a's 1001-point comparison of two
   captures. IMPLEMENTATION CHOICE.
9. `refs_projection_07b.py` was started while the pilot's six workers ran (a seventh busy process, ~11 min) and was
   stopped; it is re-run after the pilot. UNINTENDED DRIFT.
10. The capture workers and the registered runs' jobs run with cwd = the records directory (07a's harness design);
    every script is invoked by absolute path with PYTHONPATH starting with the tree, never `python -m`, so the cwd is
    not on sys.path. Interactive commands ran with cwd in the worktree. IMPLEMENTATION CHOICE.
11. M3's phase perturbation: production declares no theta_abserr; the stated proxy is the ulp of the raw phase at the
    window edge (2^-13 rad at |theta| ~ 6.8e11 for v08). IMPLEMENTATION CHOICE.
12. Variant (d) compares the leaf's own Levin antiderivative across a 4 pi sub-span beside the window (p by
    barycentric interpolation on the leaf's nodes) with the nested Clenshaw-Curtis pair there, not the sub-span's own
    Levin solve, which resolves the edge by its narrowness and so would not test the leaf. IMPLEMENTATION CHOICE.
13. Variants (e) and (f) written from §2.2's first findings (v08's carrier; items 3 and 4 without windows), before
    §2.2 was complete. IMPLEMENTATION CHOICE.
