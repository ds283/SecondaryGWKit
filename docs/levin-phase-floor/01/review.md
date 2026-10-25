# levin-phase-floor 01 — orchestrator's review

**Reviewer:** the prompt-01 orchestrator session on the Mac Studio (mps026298), 2026-10-25, after
the executing agent (Opus, fresh context) reported. Dispatch SHA `757ef01`. Every command below was
run by the reviewer in `/Users/ds283/Documents/Code/SecondaryGWKit`; nothing was repaired.

**Verdict: STOP — check 9 fails (records).** Checks 1–8 pass, and every number the review
recomputed agrees with the agent's. The failure is in `verification.md` only:
1. it gives the census script's SHA-256 as `068ce023…`, which is the header of the superseded run 2
   (`state/superseded_run2/census_01.txt`). The script on disk, and the header of every final output
   (`census_01.txt`, `groups_01.csv`, `candidates_01.txt`), is `aac0b4ed…`;
2. it holds no links. Its file references are code spans relative to the run directory
   (`scripts/group_census.py`, `out/census_01.txt`, `state/store_files_{before,after}.txt`, …),
   which from `docs/levin-phase-floor/VERIFICATION.md` would need to be `01/…`. The note asks for
   links relative to `docs/levin-phase-floor/`.

Per the note's §5 this is reported and not repaired.

## Before dispatch — the Gate (§1)

- `git rev-parse HEAD origin/handover-remedial` → `757ef016fc37be996c342d162c05f17b3c430e4d` twice.
  No fetch run.
- `git status --short` → empty.
- `./venv/bin/python -m runregistrykit list` → 10 runs: `done` ×5, `killed` ×4, `failed` ×1. None
  `running` or stale.
- `runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite` → `kind: registry`,
  `problems: none`, digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, no
  retirement recorded.
- `ls -d var/levin-phase-floor-01` → `No such file or directory`.
- Marker touched at 2026-10-25 18:12:04.256 UTC (session scratchpad `var_marker`), immediately
  before dispatch.

## Check 1 — scope: PASS

```
git rev-parse HEAD           → 757ef016fc37be996c342d162c05f17b3c430e4d
git status --short           → (empty)
find var -newer <marker> | awk -F/ '{print $1"/"$2}' | sort | uniq -c
      1 var/
     24 var/handover-11
     49 var/levin-phase-floor-01
find var -newer <marker> | grep -vE '^var/(levin-phase-floor-01|handover-11)(/|$)'
var
```
The only other entry is the directory `var` itself, whose mtime moved when the two run
directories were created. No store file and no sidecar appears.

## Check 2 — the store: PASS

- `diff state/store_files_before.txt state/store_files_after.txt` differs only in the comment
  header's timestamp (18:14:12Z vs 18:18:37Z). With `#` lines removed, the data lines are
  identical: `DATA LINES IDENTICAL`. All six files (primary, four shards, sidecar) have the same
  SHA-256, size and mtime before and after.
- `./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite`
  (no `--write`), exit 0:
  ```
  digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56
  problems: none
  recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752dda46e57d588f8107cbd4d4557134fadb2
  ```
- `ls -la var/datastores | grep -E 'A3-v2.*(journal|wal|shm)'` → none.

## Check 3 — the population: PASS

- `census_01.txt` (a): `9280 (expected 9280: yes)`, `rows at any other pair: 0`, by shard
  `[2320, 2320, 2320, 2320]`, `metadata parse failures / missing WKB_Levin: 0`.
- My count from `groups_01.csv` (the reviewer's script, `review_recompute.py`, below): **6791**
  distinct `(shard, serial)`, 37,325 records. This agrees with census (b): `37325 records from 6791
  rows with a Levin sub-interval (2489 rows have none)`, and 6791 + 2489 = 9280. Depth-20 rows with
  group records: **2265**, i.e. every depth-20 row (`depth-20 rows with no group records: 0`).

## Check 4 — the reconnaissance reproduced: PASS

From `groups_01.csv`, depth-20 rows with `compute_time > 100`:
```
depth-20 rows over 100 s: 629; compute_time 282.93 CPU-hr; group elapsed 282.88 CPU-hr
   G+r                    149.02 CPU-hr  52.7%
   G-r                     94.68 CPU-hr  33.5%
   Gqr                     21.48 CPU-hr   7.6%
   G                       17.70 CPU-hr   6.3%
```
Equal to `prompts/levin-phase-floor/measure/out/recon_groups.txt` (629 rows, 282.93 CPU-hr;
149.02 / 94.68 / 17.70 / 9.87+4.60+3.73+3.27 = 21.47 → 21.48 by rounding; 52.7 / 33.5 / 6.3 / 7.6 %).

## Check 5 — the sign convention: PASS

The log's "Before the census" reading cites code, and the reviewer read each citation:
- `ComputeTargets/GkSourcePolicyData.py:136-142`: the comment and `GK_PHASE_SIGN = -1`; `:184`,
  `:193` pass `sign=GK_PHASE_SIGN` to `PrimitivePhase`.
- `ComputeTargets/primitive_phase.py:276-283` (`raw_theta`: `self._sign * self._k * leading + …`)
  and `:291-310` (`theta_deriv`: `dtheta_dz_leading = self._sign * self._k * self._rate(raw_x)`).
- `ComputeTargets/TkSourceFunctions.py:184` `TK_PHASE_SIGN = +1`, used at `:393`, `:403`.
- `ComputeTargets/phase_groups.py:395-397` (`_OscillatoryT.theta_deriv` = ω(1+z)), `:436-437`
  (`_OscillatoryG.theta_deriv` delegates to the G phase), `:553-557` (`_composed_theta_deriv`, the
  signed sum).

So θ_G falls with z′, θ_q, θ_r rise, and the same-signed group (`G+r` on `r == k`) is the slower
one, as prompt §1 states. The log notes it does not rest on the source-remediation note alone.

## Check 6 — the predictions: PASS

- **Verbatim.** Prompt §1's body (between `## 1. Predictions…` and `## 2. Steps`, `---` removed)
  diffed against the log's "### The rules, as the prompt wrote them" with `diff -B`, trailing
  whitespace stripped → `RULES VERBATIM`.
- **Before the census.** `state/predictions_written.txt`: `2026-10-25T18:14:25Z … group_census.py
  exists: no`. The first census output (`state/superseded_run1/`) is from 18:17:06Z.
- **Scored** holds / fails / undecided, each with its number, and the numbers equal T4
  (`census_01.txt:188-215`).
- **The cancelling flag in code.** `scripts/group_census.py:105-114`: `parse_signs(label)`; `None`
  without G; `True` iff a factor `q` or `r` is present with the same sign as G and that factor's
  wavenumber equals k (`q_is_k = qs == ks`, `r_is_k = rs == ks`, cross-checked against float
  equality by an `assert` at `:281`). This is prompt §1's definition.
- **The reviewer's recomputation**, with an independent flag derived from the label and the CSV's
  `k, q, r` (records where it differs from the CSV's `cancelling`: **0**):
  ```
  P1: depth-20 elapsed 285.6504 CPU-hr; non-cancelling 125.2704; cancelling 160.3800; no-G 0.000000
      S = 0.4385
  P2 cancelling      n=2946 slope=0.9682 median e/F=2.117 p10=0.257 p90=13.52
  P2 non-cancelling  n=6943 slope=0.8216 median e/F=1.489 p10=0.1481 p90=20.42
  P3 non-cancelling  share F>1e-7 = 0.9991
  P3 cancelling      share F>1e-7 = 0.9997
  ```
  The verdicts follow from the rules as written:
  - **P1:** S = 0.4385 ≥ 0.25 → **fails H-cancel**; H-anchor "holds only if P2 holds", and P2
    holds → **holds** for H-anchor.
  - **P2:** both n ≥ 100; slopes 0.968, 0.822 ∈ [0.6, 1.4]; medians 2.117, 1.489 ∈ [0.1, 10];
    ratio 1.42 < 10 → **holds for H-anchor**. The non-cancelling class meets the conditions →
    **fails H-cancel**.
  - **P3:** 0.9991, 0.9997 ≥ 0.8 → **holds for H-anchor**; descriptive for H-cancel.

## Check 7 — findings §2.5: PASS

Recomputed from `groups_01.csv`:
```
2.5: converged depth-20 rows 1086; rel>1e-6 257; rel>1e-4 132; max 0.0503; unconverged 1179
non-finite total_abserr among rows with group records: 558; at depth 20: 5
group records with non-finite abserr: 803
group-level converged (depth-20): 8408; e_g>1e-6 2541; e_g>1e-4 1004
group-level converged (all): 30580; e_g>1e-6 15560; e_g>1e-4 7635
```
1086 / 257 / 132 / 5.0e-2 agree with findings §2.5 item 1. The non-finite rows are identified to
sub-interval and group in `census_01.txt` (d) (e.g. `:256` by sub-interval index 0: 392, 1: 316,
2: 95; per-row listings from `:262`), with how SQLite returns them (`typeof 'real'`, text `'Inf'`,
JSON `Infinity`). The count is 558, not findings' 5; the 5 at depth 20 are the findings' rows, and
the difference is explained in the log and opened as `issues.md` I2.

## Check 8 — the candidates: PASS

`awk` over `out/candidates_01.txt`, rows per class heading:
```
8 == G+r (slow, cancelling) top on r == k
8 == G-r (fast) top on r == k
3 == lone G top
8 == a Gqr group top
```

## Check 9 — the records: **FAIL**

- `log.md` follows README §5.1: subject/commit/result, `## Predictions and verdicts` (first),
  `## What shipped`, `## Deviations from the prompt`, `## Verification performed`,
  `## Observations not acted on`, `## State handed on`. Six deviations, each classified; step 1's
  no-op is deviation 1, `STRUCTURALLY REQUIRED` (parallel dispatch); "No `UNINTENDED DRIFT`".
  **Pass.**
- `issues.md` exists, with I1–I5. **Pass.**
- `verification.md` is written as `VERIFICATION.md` §1 ("## 1. The phase-group census") and its
  numbers agree with T4 and (d) (`:110`, `:118-119`, `:132-133`, `:150`, `:166`). **But:**
  - `verification.md:8`: "**Script:** `scripts/group_census.py`, SHA-256 `068ce023…` in the output
    header." `shasum -a 256 scripts/group_census.py` →
    `aac0b4ed90d3b2913f07a21ad05cdfde9e97759f41dde086381c31ef5af92eab`, and that is the header of
    the final `census_01.txt`, `groups_01.csv` and `candidates_01.txt`. `068ce023…` is the header
    of `state/superseded_run2/census_01.txt` (run 1's is `c6325a0b…`). **Wrong provenance stamp.**
  - `grep -oE '\]\([^)]+\)' verification.md` → nothing: there are no links. File references are
    code spans relative to the run directory (`verification.md:8-17`, `:41`, `:184`, `:186`,
    `:208`: `scripts/…`, `out/…`, `state/…`), not relative to `docs/levin-phase-floor/`, where they
    would read `01/…`. **Not as the note asks.**

## The reviewer's script

`review_recompute.py`, in the orchestrator's session scratchpad, run as
`./venv/bin/python -I review_recompute.py var/levin-phase-floor-01/out/groups_01.csv`. It reads
only the CSV, derives the cancelling flag itself, and prints the blocks quoted under checks 3, 4,
6 and 7.

## Report

- **Stop**, at check 9: `verification.md` cites the superseded run-2 script SHA `068ce023…` instead
  of the final `aac0b4ed…`, and its paths are not links relative to `docs/levin-phase-floor/`.
  Checks 1–8 pass.
- **P1:** S = 0.4385 → fails H-cancel; holds for H-anchor (via P2).
- **P2:** cancelling n 2946, slope 0.968, median 2.117; non-cancelling n 6943, slope 0.822, median
  1.489; ratio 1.42 → holds for H-anchor, fails H-cancel.
- **P3:** 0.9991 (non-cancelling), 0.9997 (cancelling) → holds for H-anchor; descriptive for
  H-cancel.
- **S = 0.4385.**
- **Fingerprint:** `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, matching the
  sidecar; store files byte-identical before and after.
