# Orchestrator — prompt 09, `analytic_G_w` outside its domain

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/handover/09-analytic-g-w-outside-its-domain.md`](../09-analytic-g-w-outside-its-domain.md)
**Board item:** 09 · **Closes:** `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`;
**narrows** `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`
(prompt 10 closes it) · **Model:** Opus, as the prompt recommends.

**Gate**, checked by the orchestrator on 2026-10-25 at `4a3e848`:
- `rpk-v030-adoption` 01 has landed and been reviewed, and the campaign is closed (`88742d4`). The
  prompt's hold is released.
- `handover-remedial`, clean; `git worktree list` shows the main checkout only.
- Nothing is `running` or stale in the registry (52 runs: 50 finished, 2 `unknown` pre-registry).
- `~/.raypoolkit/rehearsal-lock` and `~/ScienceData/.rehearsal-lock` do not exist;
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
- The rehearsal's inputs exist: `docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py`,
  `run_readers.sh` and `var/runs/rrk-v020-adoption-03-reference-20261018T232838/`. Neither
  `var/datastores/handover-09-reference.sqlite*` nor `var/handover-09/` exists. 50 GB free.

## 0. What makes this prompt unusual

It is the campaign's first change to a value that `main.py` stores. A green suite is not enough for
the review. The review turns on three things a suite cannot see: that the values inside the domain
are **bit-identical**, that the rehearsal at `--zend 250` still fingerprints `39809dca…`, which is
the evidence that §1 changes nothing inside the domain, and that every deliberate breakage was
caught by a named test.

**One ruling at dispatch (user, 2026-10-25).** `extract_Gk_data.py` has a signed, linear-$y$
`analytic_G_w` plot (`plots/Gk-linear/`), which prompt §2.2 did not foresee. The helper returns
$|y|$, so following §2.2 there would contradict §2.5. The user chose to filter it by the same rule
and keep its sign. That is now a dated addendum after prompt §2's item 5, committed with this note.

**Facts checked by the orchestrator at `4a3e848`.** Pass them on.
- **Line drift.** `extract_Gk_data.py`'s mislabelled CSV columns are at `:286-287`, not
  `:282-283`; the CSV dict is at `:303-304`. `safe_fabs` is at `extract_common.py:399` and
  `set_loglog_axes` at `:426`, as stated. `set_loglinear_axes` is at `:418`.
- **More callers than the reading list names.** Besides `convergence_reference.py:490-510`,
  `compute_analytic_G` is called by `test_phase_groups.py` (`:266`, `:347`, `:564`),
  `test_background_tau.py` (`:657-658`) and `test_quadsource_integral.py` (`:481`), each at a
  stand-in model's constant $w$. These bear on prompt §7's last stop: the agent reads them and
  records whether any can now receive `None`.
- **`analytic_G_rad` calls the same function** at $w = 1/3$, so §1.2's overflow rule reaches it too.
  "Untouched" in §1.4 means its call site, not its value on overflow. If the rehearsal still
  fingerprints `39809dca…`, no rehearsal value changed either way. The agent records how it read
  this.
- **`GkSource.py:498-506`** chooses numeric or WKB by `is not None` **on the object**, not on the
  value, so a `None` value passes through unchanged. Both factories' columns are `nullable=True`
  (`ObjectFactories/GkWKBIntegration.py:1019-1020`, and likewise in `GkNumericIntegration.py`); the
  agent confirms both, as §1.4 says.
- **The rehearsal's recipe.** Its launch form is `docs/rrk-v020-adoption-verification.md` **§1.1**
  (§3.5 is what A3 v2 inherits). The most recent run under the current lock is `rpk-v030-adoption`
  prompt 01 §2.8 and its log, `## The rehearsal`. The agent follows that, with this prompt's names:
  `--job-name handover-09`, `--register r5-reference`, `--campaign handover --prompt 09`, the
  launch log `var/runs/handover-09-reference.launch.log`, and a `--purpose` naming this prompt, the
  dispatch SHA and `39809dca`.
- **The manifest will say `git_dirty: true`.** The rehearsal has to run on 09's edits, before the
  one commit. It is the same as `rpk-v030-adoption` 01's deviation 4, `STRUCTURALLY REQUIRED`.
- **Lock commands: one per shell command, no pipe.** `>> file 2>&1`, then `echo $?`. Never `| tee`,
  and never in the shell command that launches, waits or runs the readers. `--purpose` names no
  temp dir.
- **The known flakes** (`test_tk_wkb_phase … test_wall_time_per_object` and its kin) are re-run
  alone and recorded. They are not a stop.

## 1. Before you dispatch

1. The Gate, as above.
2. **Baselines**, the seven roots serially at `4a3e848`, in the session scratchpad. They are
   recorded in the board's dispatch record when 09 lands, and compared with the agent's counts.
3. **The `var/` marker:** `touch` a file in the scratchpad immediately before dispatch, for check 8.
4. **Commit this note and the prompt's addendum**; that commit is the dispatch SHA.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own**; orchestration commits land on this branch.
- **One commit, at the end**, staged by explicit path. It covers `ComputeTargets/analytic_Gk.py`,
  `extract_common.py`, the four readers, the new test modules under `ComputeTargets/tests/`, the
  log, the board and `docs/OPEN_ISSUES.md`. A path outside that list is a stop.
- **The rehearsal:** lock, launch detached, make the one check, then **end the turn**. Report the
  driver's Python pid, the `sh` pid, the run id and the lock's owner file. The orchestrator waits
  on the pid, once, in the background, and resumes the agent with the final `status.json`. The agent
  does not wait or poll.
- **Stop and ask** at any of prompt §7's conditions. A stop with the lock held leaves it held and
  prints the owner file.

## 3. The review — nine checks

1. **Scope.** `git show --stat <09>`: only §2's paths. No existing test module is modified, and
   nothing under `ObjectFactories/`, `docs/` or `RunDrivers/` changes.
2. **The domain and bit-identity, read.** The predicate is `1 + 3w > 0` exactly, named once. Both
   functions test it before any `jv`/`yv`, and the in-domain expression is the old one, token for
   token (`git diff -U0` on `analytic_Gk.py`). The test's in-domain reference is computed inline
   from `scipy.special` and compared with `==`, not `assertAlmostEqual`. The log states the value
   of `1 + 3w` at `w = -1.0/3.0`.
3. **The readers, read.** Every log–log `analytic_G_w` series in the four readers goes through the
   helper with that plot's $|G|$ as reference. The signed `Gk-linear` series goes through its signed
   form. `extract_GkWKB_data.py` has no bare `fabs(` on an `analytic_` attribute. `extract_Gk_data`'s
   CSV `analytic_G_w`/`analytic_Gprime_w` read `value.analytic_G_w`/`value.analytic_Gprime_w`. No
   other line in the readers changed.
4. **The roots, re-run here at `<09>`.** Six equal to the baselines; `ComputeTargets` up by exactly
   the test methods added, counted from the new modules. All OK, with any flake re-run alone.
5. **The breakage record.** Prompt §3's (a)–(e) and the signed form's own breakage, each naming a
   failing test, each reverted (`git status` clean of them). If the record of one is thin, re-apply
   it by hand, run the module, and revert.
6. **The rehearsal.** The lock transcript: `acquire` 0, `set-pid` 0 with the Python pid,
   `ray-processes` listing the instance while it runs, `release` 0 at the end, `status` free. The
   run `done`, exit 0. **The fingerprint `39809dca…`**, recomputed here with `store fingerprint`
   (no `--write`), and `compare_with_recorded.py`'s verdict `equal`.
7. **The readers on the rehearsal store**: every reader exits 0, the store's files are unchanged
   (SHA-256 before and after), and no Ray process is left.
8. **Nothing else moved.** `find var -newer <marker>` lists only the run directory, its launch
   log, the rehearsal store and its sidecar, and `var/handover-09/`. `git status --short` is clean.
9. **The records.** The log has every heading of the previous handover logs and classifies each
   deviation (§0's `git_dirty` and the signed-form addendum among them). The board's 09 row and
   item; `[08-extract-gk-data-…]` in §4, closed; `[08-the-gkwkb-reader-fails-…]` narrowed, not
   closed; `docs/OPEN_ISSUES.md`'s row deleted and the other's hook narrowed, with the count and
   date correct by counting. `black --check` is clean on every touched `.py` file. The commit
   message follows `CLAUDE.md`'s form.

## 4. After the review

If it passes: residue first, in a follow-up commit of the orchestrator's own ("this commit" → the
SHA on the board and in the log; this README's 09 row). Then report to the user: the fingerprint,
the roots, the breakage record, that `handover-09-reference` is theirs to retire, and that 10 is
unblocked. 10 needs the Mac Studio's checkout moved to 09's tree, so the branch has to be pushed
first.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §7 conditions, or any failed check in §3: report
it, do not repair it.
