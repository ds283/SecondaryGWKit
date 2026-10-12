# Log — runregistry-extraction prompt 01, freeze the run registry

- **Subject:** [`01-freeze-the-run-registry.md`](../01-freeze-the-run-registry.md), board items F1–F3.
- **Commit:** this commit (parent `d94d66b`).
- **Result:** landed. Every acceptance item passes; no stop condition fired.
- **Model:** Sonnet 5.5.

## What shipped

- `CLAUDE.md`: one section, "`RunRegistry/` is frozen — it is moving to RunRegistryKit", directly
  after "Long-running jobs — the run registry". It says the freeze covers the package
  `RunRegistry/` only (not its callers), nothing under it changes until prompt 06, defects go to
  the campaign board and `docs/OPEN_ISSUES.md` and are fixed in RunRegistryKit, and the six rules
  still bind. No other line of `CLAUDE.md` changed.
- Board §2.1: 25 rows from `git ls-tree -r HEAD RunRegistry/`, reformatted to two columns by one
  `awk` line, no hash edited. Checked against `git ls-tree -r --name-only HEAD RunRegistry/ | wc -l`
  (25) and against `c8a7f14` (`git diff --stat c8a7f14 HEAD -- RunRegistry/` is empty).
- `measure/out/01-list.txt`, `01-store-show.txt`, `01-roots.txt`, each headed by the `HEAD` it was
  taken at (`d94d66b`).
- Board: header Status, the §1 row, F1–F3, §2.1, `### 5.1`.

## Deviations from the prompt

1. **Board header Status corrected** from "none landed" to "1 landed (01)". **IMPLEMENTATION
   CHOICE** (README §6.2 C7; the orchestrator's brief). The prompt's file list allows the board.
2. **`HEAD` is `d94d66b`, not `c8a7f14`.** The gate says the plan's commit is `HEAD` or an
   ancestor; the three commits since touch only `docs/` and `prompts/`. **STRUCTURALLY REQUIRED**:
   `c8a7f14` is not `HEAD`. Acceptance 1's `git diff --stat c8a7f14 HEAD -- RunRegistry/` is empty.
3. **Extra header line in each measurement file** (`HEAD`, command, exit status), and the roots
   file records the load average before each root and after the last. **IMPLEMENTATION CHOICE**:
   the prompt asks for `HEAD` recorded and the brief for the load average and exit statuses.
4. **Each `store show` block ends with `# exit status: N`** (all 0). **IMPLEMENTATION CHOICE**, per
   the brief.
5. **Section wording of the freeze** adds that it covers `RunRegistry/` only and names the board's
   §2.1 as where the hashes are. **IMPLEMENTATION CHOICE**, per the brief's scope note.

No `UNINTENDED DRIFT`.

## Verification performed

- `git diff --stat c8a7f14 HEAD -- RunRegistry/`: empty. `git ls-tree -r HEAD RunRegistry/`: 25 rows;
  `--name-only | wc -l`: 25.
- `01-list.txt`: "29 run(s) ...: 27 finished, 2 unknown", none `running`; the two unknown are
  `a3-pilot` and `a3-pilot-resume`. Exit 0. (README §0.2: 29, 27, 2.)
- `01-store-show.txt`: 14 blocks (13 under `var/datastores/`, one under
  `backup-pre-resume-20260921T091011/`), each `kind: registry`, `problems: none`, `state: retired`,
  exit 0.
- `01-roots.txt`, serial, foreground, README §8's command, at `d94d66b`:

  | Root | Ran | Result |
  |---|---|---|
  | AdaptiveLevin | 32 | OK |
  | CosmologyModels | 39 | OK |
  | ObjectFactories | 271 | OK |
  | RunRegistry | 286 | OK |
  | tools | 49 | OK |
  | LiouvilleGreen | 148 | OK (skipped=1) |
  | ComputeTargets | 693 | OK |

  Every count is README §0.2's. No failure, so no known wall-clock test was re-run. Load average
  rose from about 7 to about 29 during the run (another session); it ended at 14.
- `var/`: only `list` and `store show` were run; no `begin()` and no store-modifying subcommand.
  The registry's and the factories' test suites ran under `unittest` as usual.

## The deliberate-breakage record

None. This prompt adds no code and no test (acceptance 6).

## Observations not acted on

None.

## State handed to the next prompt

`RunRegistry/` is frozen at the 25 blobs in board §2.1. The baselines for 06 and 08 are in
`measure/out/`. Prompt 02 (R0) is unblocked; the registry shows nothing running.
