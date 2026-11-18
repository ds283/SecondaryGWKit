# Deviations draft (R31 rehearsal agent), carried into log.md

1. `rec.sh` is R30's (`065afcc6`) with its record path changed by sed (`out/rec_sh.diff`). IMPLEMENTATION CHOICE.
2. S0' is scored by a new script, `s0prime.py`, with Part A's `strip_docstrings` (07b `measurement/scripts/gate_code_07b.py`)
   copied verbatim; the raw `git diff --stat` as section 5 item 1 writes it is also in `out/00-gate.txt`.
   IMPLEMENTATION CHOICE.
3. The lock commands run with cwd in the worktree and `PYTHONPATH` set to it (the dispatch's rule for every Python command),
   where R30's ran from the main checkout; `raypoolkit` is an installed package, so the module run is the same. Each lock
   command's `lock.log` entry starts with a header line written by `echo` (a shell builtin; no process). IMPLEMENTATION CHOICE.
4. A listing of every protected file (mtime, size, path; `scripts/protected_listing.sh`) is taken before and after, beside
   the store hashes, to show no protected file changed. IMPLEMENTATION CHOICE.
5. A first protected-listing one-liner failed: zsh does not word-split an unquoted `$P`, so `find` saw one path that does not
   exist and the file held only the store files. It was read-only, its output was overwritten by the script's, and nothing
   used it. UNINTENDED DRIFT.
6. R31's fingerprint is checked twice: from `status.json`, and by `runregistrykit store fingerprint` without `--write`
   (read only; the store re-hashed equal after it). IMPLEMENTATION CHOICE.
7. `normalise_jobname_labels.py` is 07a's (`41da5c72`) with its two prefixes (`07a-` -> `07b-`) and docstring changed
   (`out/normalise_jobname_labels.diff`). IMPLEMENTATION CHOICE.
8. `classify_non_qsi.py` is run on both the raw and the normalised JSON; the raw gives 5,148 UNCLASSIFIED, every one a
   `label` whose job-name prefix differs; the verdict is the normalised copy's. STRUCTURALLY REQUIRED (the job names
   differ by the brief).
9. The classifier prints no provenance header, so the shell wrote one. IMPLEMENTATION CHOICE.
10. Stdlib-only scripts run with `-B` and `PYTHONPATH` set to the worktree, cwd the worktree (the dispatch's rule), where
    R30's ran with `-I -B` from the main checkout. None imports tree code. IMPLEMENTATION CHOICE.
11. `tabulate_qsi_r30_r31.py` is 07a's `tabulate_qsi_r29_r30.py` with the store tags shifted (29 -> 30, 30 -> 31), TREES,
    RUNS, the DEPTH_MAX comment and the docstring changed; the verbatim reader untouched (`out/tabulate_qsi_r30_r31.diff`).
    IMPLEMENTATION CHOICE. Its first generation aborted on its own assertion before writing anything (the regex had
    shifted the run paths' tokens). UNINTENDED DRIFT, harmless.
12. `qsi_summary_r30_r31.py` is 07a's `qsi_summary_r29_r30.py` with the tags shifted and a new function `additions_07b`.
    IMPLEMENTATION CHOICE. One printed label ("grp sp/wp/gr 29") was missed by the regex and corrected, and the function
    was extended once; the summary was re-run each time and only the last output is kept. UNINTENDED DRIFT.
13. One read-only look at the converged-down row's four group records printed output that was not saved; the same fields
    are in `qsi_records_r30_r31.csv` and `qsi_summary_r30_r31.txt`. UNINTENDED DRIFT.
14. S3 is scored by 07a's s3_score.py with the tags shifted and the reader module renamed. IMPLEMENTATION CHOICE.
15. The capture harness is R30's on-disk copy (hashes equal to 07a's records). prov.py points at the tree h = 0a103b5a and
    R31's store links; worker.py adds `refusals_07b` per call; route_07a is kept, descriptive only. IMPLEMENTATION CHOICE.
16. replay07b_r31.py is generated from R30's replay07a_r30.py by scripts/gen/make_replay07b_r31.py: tags shifted; the 49
    abserr-only rows replaced by the 8 records of interest (|v31 - v30| > e30 + e31 exactly, or converged changed); the
    check reports refusals; blockcheck compares with R30's captures; two new subcommands capture the group records of
    interest. A first generation aborted before writing. IMPLEMENTATION CHOICE (the abort: UNINTENDED DRIFT, harmless).
17. The replay pipeline is a registered run (slug r31-replay), driven by Part A's run_06j.py (unchanged) and a new
    launcher launch_r31.py, where R30's pipeline was an unregistered nohup shell script. The brief asks that long jobs be
    registered. run_06j.py appends each command's output to out/commands.log. IMPLEMENTATION CHOICE.
18. blocks1 (the pilot) ran in session, as R30's did. IMPLEMENTATION CHOICE.
19. S1, S2 and S4 are R30's scripts adapted (diffs in out/): R31's store and replay; S1's keys renamed v31/abserr31 (R30
    kept R29's names); each adds R31's refusals and R30's numbers for comparison (S1 against R30's s1_scores.json, S2 against
    R30's s2_scores.json, S4 against R30's block_scores.json). S4's integrand is compared with R29's capture, on which the
    references were taken, as R30's was. block_reference_r30.py is R30's, unchanged. IMPLEMENTATION CHOICE.
20. R30's capture_compare.py (why the block captures are not byte-identical to the earlier store's) was not re-run: the
    integrand's equality with R29's capture at 1001 points is shown by block_score_r31.py, as R30's S4 rested on.
    IMPLEMENTATION CHOICE.
21. A read-only `tail -3` loop failed (option form) and was repeated as `tail -n 3`. UNINTENDED DRIFT.
22. The per-call total of the four one_reading-kind projections was printed by a one-liner and not saved; it is
    re-derivable from out/refs/refs_projection_r31.json, and the plan records it. UNINTENDED DRIFT.
23. The readers ran in session (about 2 min), not as a registered run, as R30's did. IMPLEMENTATION CHOICE.
24. The descriptive addition (the user's word at stop 1): the 8 records' captures, made by the replay run, each re-run
    from its pickle at the tree (against R31) and at 07a's module by Part A's variant_loader (against R30)
    (scripts/refs/refcap_replay.py); a projection (scripts/refs/refs_projection_r31.py, 07a's model with its gamma D
    term, Part A's deviation 16). New scripts. IMPLEMENTATION CHOICE.
25. The launcher's first r31-refs edit aborted on its own assertion before writing (an anchor occurring twice); the
    launcher was unchanged (cmp). UNINTENDED DRIFT, harmless.
26. The references are the tool's own command line (`python -m tools.filon_reference`, all four readings in one process
    per call), where Part A read each reading by one_reading.py for its monitor; the readings are the same calls of
    `reading()` (the tool's docstring and one_reading.py's). The job runs with cwd the rehearsal records directory, which
    holds no `tools` package, so `-m` resolves to the tree's through PYTHONPATH. IMPLEMENTATION CHOICE.
