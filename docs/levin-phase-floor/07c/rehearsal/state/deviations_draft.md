# Deviations draft (07c rehearsal, R32), turn 1
1. rec.sh is 07b's (9a80a313) with its record path changed by sed (43c59f3b). IMPLEMENTATION CHOICE.
2. S0' scored by a new s0prime.py (after 07b's; "nothing the build imports" = no *.py, *.toml, requirements.txt or config/ path listed); the raw git diff --stat is also in out/00-gate.txt. IMPLEMENTATION CHOICE.
3. Lock commands ran from the worktree with PYTHONPATH set, each with an echo header (a builtin), output >> state/lock.log. IMPLEMENTATION CHOICE (as 07b).
4. The acquire ran in the same zsh command as a preceding heredoc `cat >> out/commands.log`, which had exited before the acquire started; the acquire was still the only process running when it ran. UNINTENDED DRIFT, harmless.
5. A protected listing (mtime size path) taken before; protected paths include var/levin-phase-floor-07c/review/ (the orchestrator's, expected to change). IMPLEMENTATION CHOICE.
6. The R32 launch line: 07b's form, run in zsh with the purpose in a shell variable passed as sh's $0; its text is also in state/purpose_r32.txt. IMPLEMENTATION CHOICE.
# turn 2
7. The fingerprint was also re-derived by `runregistrykit store fingerprint` without `--write`. Read only; the store re-hashed equal. IMPLEMENTATION CHOICE.
8. normalise_jobname_labels.py is 07b's (bdc2d44d) with its prefixes (07a- -> 07b-, 07b- -> 07c-) and docstring changed. IMPLEMENTATION CHOICE.
9. The classifier was run on the raw and the normalised JSON. The raw gives 5,148 UNCLASSIFIED, every one a `label` whose job-name prefix differs; the verdict is the normalised copy's. STRUCTURALLY REQUIRED.
10. The classifier prints no provenance header, so the shell wrote one. IMPLEMENTATION CHOICE.
11. Stdlib-only scripts ran with -B and PYTHONPATH, cwd the worktree, not -I (the generators ran with -I). None imports tree code. IMPLEMENTATION CHOICE.
12. tabulate_qsi_r31_r32.py is generated from 07b's (a7dc4941) by gen/make_tabulate_qsi_r31_r32.py: store tags shifted (72), TREES, RUNS, the DEPTH_MAX comment and the docstring; the verbatim reader asserted unchanged. IMPLEMENTATION CHOICE. The generator's first run aborted on its own assertion (a wrong line index) before writing anything; the generator was corrected by two asserted replacements and re-run. UNINTENDED DRIFT, harmless.
13. qsi_summary_r31_r32.py is generated from 07b's (50d33417): tags shifted (161), tree path, docstring; `additions_07b` kept with its name; nothing new added. Its "record-level S3 form" is float arithmetic and descriptive; S3 is scored exactly after stop 1. IMPLEMENTATION CHOICE.
14. A read-only look at the unconverged rows was saved (out/stop1_unconverged_look.txt) rather than printed. IMPLEMENTATION CHOICE.
15. The stage timings of R31 and R32 were diffed into out/stage_timings_r31_r32.diff (bash process substitution). IMPLEMENTATION CHOICE.
# turn 3
16. S3 is scored by 07b's s3_score.py generated with the tags shifted and section 9.5's named row added (listed whether or not beyond the bound; not counted as a violation). IMPLEMENTATION CHOICE.
17. The capture harness is 07b's on-disk copy; prov.py points at the tree and R32; worker.py adds chase_07c; replay07c_r32.py is generated from 07b's (79 tag shifts), records of interest by the same rule (1 record: the converged change). IMPLEMENTATION CHOICE.
18. make_s3_score.py's first output had the reader module shifted twice (tabulate_qsi_r32_r32); caught on reading the diff before any run, the generator corrected and re-run. UNINTENDED DRIFT, harmless.
19. The replay ran as a registered run (slug r32-replay) by 07b's run_06j.py and a new launch_r32.py, as 07b's r31-replay; run_06j.py appends each command's output to out/commands.log. IMPLEMENTATION CHOICE.
20. Block pass 1 (the pilot) ran in session, as 07b's did. IMPLEMENTATION CHOICE.
21. S3's named record is scored against 07c Part A's reference ref-R31-3-20363-g1 (Part A's readings of 07b's R31 capture of that call), not a reference in 07b's rehearsal: 07b's rehearsal captured the call but read no reference for it. The R32 capture is compared byte for byte with that R31 capture. IMPLEMENTATION CHOICE (no new reference).
22. The two phase_limited flips are tabulated by a new phase_limited_flips.py (the orchestrator's note). IMPLEMENTATION CHOICE.
23. S1, S2, S4 are 07b's scripts generated with tags shifted, the parent's context read from 07b's score files, R32's block serials, and chase_07c recorded. IMPLEMENTATION CHOICE.
