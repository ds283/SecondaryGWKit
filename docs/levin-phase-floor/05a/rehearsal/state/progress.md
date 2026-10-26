# Progress, rehearsal agent, levin-phase-floor 05a section 5

- 12:20Z gate re-check: holds (out/gate.txt).
- 12:20Z lock acquired (exit 0).
- 12:20:40Z R11 launched from the worktree c4eaefd: run levin-phase-floor-05a-r11-parent-20261026T122041,
  sh pid 22677, driver (Python) pid 22684; set-pid 22684 exit 0. One check: running, stage
  CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS, 0 Tracebacks, cwd of driver and Datastore actor 22808 = worktree,
  manifest git_head c4eaefda…, git_dirty false (out/r11_launch_check.txt).
- Turn 1 ended; waiting for the orchestrator to resume with R11's final status.json.
- Next: R11 end checks (fingerprint 1b332ffa…), worktree unwritten, then R12 launch from the main checkout.
- 12:24Z R11 end checks: done, exit 0, 1b332ffa…, 21/21 class digests = R8's, taken.git_head c4eaefda…, git_dirty false;
  worktree unwritten (status --ignored and find -newer empty); 0 Tracebacks; ray-processes empty (out/r11_end_check.txt).
  R11 store hashed: state/r11_store_sha256_after_build.txt.
- 12:24:33Z R12 launched from the main checkout 149c393: run levin-phase-floor-05a-r12-reanchor-t-20261026T122434,
  sh pid 24894, driver pid 24897; set-pid 24897 exit 0. One check: running, 0 Tracebacks, cwd main checkout,
  git_head 149c3938…, git_dirty false (out/r12_launch_check.txt).
- Turn 2 ended; waiting for R12's final status.json.
- Next: R12 end checks, hashes, store_value_diff, classify, tabulate, readers, release.
- 12:27Z R12 end checks hold (out/r12_end_check.txt). 12:28Z comparison exit 1; 7 value tables bit-identical; 0 unclassified;
  QSI total differs in 424/864, max rel 2.9e-11, max 8.8e-5 abserr. Readers 6 x exit 0, R12 unchanged. Lock released 12:32Z.
- Records written: log.md, verification.md, issues.md.
