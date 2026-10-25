# Review — handover prompt 03, the policy-geometry census

**Verdict: STOP at check 5** (the §5.5 shape comparison). Prompt §7's fourth condition is not cleared. Checks 1–4, 6 and 7 pass.
This is relayed to the user for a ruling. Nothing was repaired.

- **Orchestrator:** Claude Opus 5.5, on the Mac Studio (`mps026298`), in `/Users/ds283/Documents/Code/SecondaryGWKit`.
- **Executing agent:** one fresh-context Sonnet subagent, dispatched 2026-10-25 22:35 UTC.
- **Dispatch SHA:** `c8ca1559fcc53fdb88998f170257635431fb85ca` on `handover-remedial`. It contains the refresh commit `f1a3888`.
- **The checkout was not moved.** `levin-phase-floor` 04's session had already started.
- **Nothing was committed, and no git command that writes was run,** by the orchestrator or by the agent.

## Before dispatch — the Gate, re-read by the orchestrator

| Item | What was run | What it printed | Verdict |
|---|---|---|---|
| 11's board row is "reviewed" | `grep -n '^\| *11 ' prompts/handover/IMPLEMENTATION_STATE.md` | ":130 … ✅ 2026-10-25, reviewed on the Mac Studio by a remote orchestrator (7 / 7 checks)" | pass |
| Dispatch commit is pushed, checked without fetching | `git rev-parse @{u}`; `git branch -r --contains HEAD` | `c8ca1559…`; `origin/handover-remedial` | pass |
| `HEAD` and a clean tree | `git rev-parse HEAD`; `git status --short` | `c8ca1559fcc53fdb88998f170257635431fb85ca`; empty | pass |
| `HEAD` contains `f1a3888` | `git merge-base --is-ancestor f1a3888 HEAD` | yes | pass |
| The Gate diff | `git diff 98e68d3 HEAD --stat -- ComputeTargets/GkSourcePolicyData.py ComputeTargets/GkSource.py ObjectFactories/GkSourcePolicyData.py ObjectFactories/GkSource.py` | empty | pass |
| The registry | `./venv/bin/python -m runregistrykit list` | 12 runs: "12 finished; 0 with no host recorded". No `levin-phase-floor-04-*` run was registered, so the exception did not arise | pass |
| The store | `runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite` | kind `registry`; problems none; history `create` only, no retirement; digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`; run `handover-07-a3-v2-20261022T211155` `done` | pass |
| No `var/handover-03/` | `ls -d var/handover-03` | "No such file or directory" | pass |

**The marker:** `touch` of `<orchestrator scratchpad>/var_marker_03` at 2026-10-25 22:35:51 UTC, immediately before dispatch.

**No suite baselines** were taken: the prompt changes no tracked file.

## Check 1 — Scope: pass

- `git rev-parse HEAD` printed `c8ca1559fcc53fdb88998f170257635431fb85ca`.
- `git status --short` was empty.
- `find var -newer <marker>` listed:
  - `var/` itself and `var/handover-03/`, with 34 entries under it;
  - `var/levin-phase-floor-04/`, the other session's: its `out/`, `state/`, `scripts/`, `log.md`, `issues.md` and `review.md`.
- It listed no store file, no sidecar and no run directory.
- The agent's own files are all under `var/handover-03/`: `out/` (with `dev_runs/`), `scripts/` (with `exploration/`), `state/`, `POLICY-GEOMETRY.md`, `log.md` and `issues.md`.

## Check 2 — The store: pass

**Before and after.** `diff state/store_files_before.txt state/store_files_after.txt` showed them identical.

**Now.** `shasum -a 256` of the six files, re-taken at review, equals both:

| File | SHA-256 |
|---|---|
| primary | `1542e05c…` |
| shard 0 | `b311698f…` |
| shard 1 | `c5e341ce…` |
| shard 2 | `175a6731…` |
| shard 3 | `76b83fb3…` |
| sidecar | `3201f6a5…` |

**Fingerprint.** `./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite`, run without `--write`, printed:
- `digest: 0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`;
- "problems: none";
- "recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 … at b90752d…".

**No journal.** `ls var/datastores | grep -iE 'journal|wal|shm'` found nothing.

## Check 3 — The script: pass

The script is `scripts/policy_geometry.py`: 1948 lines, SHA-256 `fa010147f92397853a6d19049424117f6d29afd4b5b378cbf4942985db390b4c`.

- **Imports:** `ast`, `collections`, `hashlib`, `json`, `math`, `os`, `sqlite3`, `subprocess`, `sys`, `time` and `numpy`. Nothing comes from this repository's packages. The constants are read from source text with `ast.literal_eval`.
- **Opens:**
  - every database through `sqlite3.connect(f"file:{…}?mode=ro", uri=True)` (`:63`);
  - the other `open()` calls (`:68`, `:76`, `:237`) read files for hashing, read source text, and read the sidecar.
- **The header's SHA equals the file's.** `out/policy_geometry_03.txt` records `fa010147…`, the same as `shasum`.
- **`black --check` is clean:** "1 file would be left unchanged".

**It counts by node index.** I read `:287-288`, `:429-441`, `:517-541` and `:826-831`:
- `order` is the redshift serials sorted by stored `z`, and `gidx` maps a serial to its node index;
- each `GkSource`'s nodes are its `GkSourceValue.z_source_serial`s, checked to be the slice of the global grid from `z_response` up;
- `i_num` and `i_wkb` are the indices of `numeric_smallest_z_serial` and `primary_WKB_largest_z_serial`;
- `i_x` is found by `zmap[s] == r["crossover_z"]`, exact equality, and anything other than exactly one hit is flagged;
- `numeric_intervals = i_x - i_num` and `WKB_intervals = i_wkb - i_x`.

`log(1+z)` appears only in the forward direction: for $c_{\rm num}$, $c_{\rm WKB}$, widths and the e-fold interpolation. No `exp` or `expm1` back to $z$ appears in any comparison.

**The exit-time row** is taken from `wavenumber_exit_serial` (`exits[r["exit_policy"]]`, `:511`), never by tolerance. It is checked against the `GkSource` parent's on every row (`:815-819`).

## Check 4 — The counts, recomputed by the orchestrator: pass

**The type/quality count** came from `select type, quality, count(*) from GkSourcePolicyData group by type, quality` on each shard (`mode=ro`), summed:

| Code | Decoded | Rows |
|---|---|---:|
| `0\|0` | `numeric`/`complete` | 363 |
| `1\|0` | `WKB`/`complete` | 725 |
| `2\|0` | `mixed`/`complete` | 71 |
| `2\|3` | `mixed`/`minimal` | 1 |

That is 11's table in every cell, with 72 `mixed` rows.

**Three `mixed` rows recounted with `sqlite3`.** Counted directly from `GkSourceValue ⋈ redshift` on the stored `z`, with no code of the agent's:
- `numeric_intervals` is the number of nodes with `z_num < z ≤ crossover_z`;
- `WKB_intervals` is the number with `crossover_z < z ≤ z_wkb`;
- `xhits` is the number of nodes whose `z` equals `crossover_z`.

| Row | `source_serial` | `xhits` | `numeric_intervals` | `WKB_intervals` | Agent's table |
|---|---:|---:|---:|---:|---|
| shard 2, serial 460 (`minimal`) | 708 | 1 | **6** | **1** | 6 / 1 ✓ |
| shard 3, serial 510 (max) | 147 | 1 | **664** | **82** | 664 / 82 ✓ |
| shard 2, serial 352 (min `complete`) | 671 | 1 | **301** | **82** | 301 / 82 ✓ |

**The distributions** are over 72 `mixed` rows (`out/policy_geometry_03.txt` §3).

The `minimal` row's `metadata` is `{"Levin_z_dtheta_dlogz": -33.73828518372407}`, re-read with `sqlite3`. It has no `comment` key, as the agent reports.

## Check 5 — The §5.5 comparison: **STOP**

**What is in order:**
- The comparison is stated as one of shape.
- The 1e5–1e7 restriction is tabulated (table (b)), and so is the by-k-decade table (table (c)).
- Each cell has an attribution.
- `fail`/`incomplete` is recorded as **unexplained**, correctly.

**Why it does not clear prompt §7's fourth condition.** Restricted to 1e5 ≤ k ≤ 1e7, `mixed` is **6.34 %** (45 + 1 of 725) against §4.2's **13.42 %**. That is less than half, so "materially scarcer". The census accounts for the gap with a **denominator**: 145 response redshifts per k on A3 v2, against an inferred 66 on §5.5's store. It supports this with:
- `mixed` rows per k being about equal, 9.0 against 8.71;
- a probe at $z_{\rm response} \ge 10^7$, which restores §5.5's rank order (`mixed` 11.80 %).

It assigns that to "differences 1 and 2 jointly". Neither of §2.2's three differences is that:
- Difference 1 is "its datastore no longer exists".
- Difference 2 is "the grid criterion changed": the source-grid density criterion of `qcd-background-audit` 15.
- The probe's reading is a different **response range**: a lower end at $10^7$ against A3 v2's 0.1. That is a run-configuration difference, a fourth reason.

The census says itself that the account "leans on 462 = 7 × 66, an inference from a division and not a recorded fact, and on the probe, which is a hypothesis". It adds: "If the reviewer finds that account insufficient, this is the condition."

Note §3 item 5 makes "a fourth reason invented to explain a cell" a stop. I do not adjudicate whether the denominator account is acceptable. That is relayed to the user.

**The agent's judgement call (a), verbatim:**

> **Judgement call (a):** I did not treat the shape difference as the second stop condition. The account leans on 462 = 7 × 66, which is an inference from a division, and on the probe. If you find it insufficient, that condition applies.

## Check 6 — §2.1: pass, with one observation for the user

**Item 1.** The overlap was set against $z_{e3}$, $\sqrt{z_{e3}z_{e4}}$ and $z_{e4}$ on all 72 `mixed` rows (`POLICY-GEOMETRY.md` §5), with landmarks from the policy row's own exit-time row.
- Every row has nodes carrying both representations in $z_{e4} \le z < \sqrt{z_{e3}z_{e4}}$: 21 to 26 nodes on 61 rows, and 7 to 20 on 11.
- On those 11 rows $z_{\rm response} > z_{e4}$ cuts the source grid from below. I read that as "present" within the grid that exists, since $z_{\rm source} \ge z_{\rm response}$.
- So the stop condition, the band's absence, is not met.

**The observation for the user.** The census also measures that:
- `primary_WKB_largest_z` is the **top of the source grid** on 71 of 72 rows, so the overlap is 8.8 to 17.0 e-folds wide;
- on all 71 `complete` rows `crossover_z` is **one node**, $3.121867\times10^{15}$, super-horizon at every k.

The seam therefore does not sit in the half-e-fold band. That bears on B4's lever map and on README §2 (q). Per prompt §2.1, the correction is the user's. The census did not make it, and it is not a stop condition as written.

The agent's judgement call (b), verbatim:

> **Judgement call (b):** this bears on B4's lever map, and I did not stop because the stop condition is the band's absence.

**Item 2.** It is stated as a proxy. `POLICY-GEOMETRY.md:558` says: "**A3 v2 has no `incomplete` row** (§3), so that test is **vacuous here**. What follows is a proxy for it". §10 repeats the point.

## Check 7 — The records: pass, with two minor notes

**`log.md`** has every heading of `logs/11-a3-v2-seam-record.md`: What shipped, Deviations from the prompt, Verification performed, Observations not acted on, and State handed to the next step. It adds the datastore, the §5.5 comparison cell by cell, the non-contiguity counts and the exit-time check. All eleven deviations are classified:
- 1 `STRUCTURALLY REQUIRED`: the checkout was checked, not moved;
- 6 `IMPLEMENTATION CHOICE`;
- 3 `UNINTENDED DRIFT`:
  - three development runs before the recorded one, kept in `out/dev_runs/`;
  - two unrecorded first runs of the exploration scripts;
  - one rejected `Edit` to a mistyped path.
- Deviation 2 needed no classification: the `levin-phase-floor` 04 exception did not arise.

**`POLICY-GEOMETRY.md`:**
- It has the header block, with the source-grid tags.
- Its §0 is the D5 answer in one bold paragraph. *Minor:* a second, unbolded paragraph of caveats follows it inside §0. The bookkeeping session may judge whether that meets "one paragraph".
- It has "Set beside prompt 11" (§9), citing `A3-V2-VERIFICATION.md` §5.3, and the closing "What this does not cover" (§10).
- Its links are relative to `docs/handover/`: the `a3-v2/policy-geometry/…` and `../../prompts/handover/…` forms.

**`issues.md`** drafts the three issues under the prompt's names and proposes no fix. I spot-checked at the dispatch tree:

| Citation | What is there |
|---|---|
| `MetadataConcepts/GkSourcePolicy.py:6` | `["maximize-numeric", "maximize-WKB"]` |
| `MetadataConcepts/GkSourcePolicy.py:25-30` | reassigns before formatting the message |
| `ComputeTargets/GkSourcePolicyData.py:457` | `"maximize-WKB"` |
| `ComputeTargets/GkSourcePolicyData.py:465` | `"minimize-WKB"` |
| `ComputeTargets/GkSourcePolicyData.py:472-475` | the `raise` |
| `MetadataConcepts/QuadSourcePolicy.py:6` | `"maximize_numeric"` |
| `MetadataConcepts/QuadSourcePolicy.py:60-65` | the same reassign-then-format |
| `main.py:3224-3225` | the ungated `dump_incomplete_GkSourcePolicy.remote(obj)` |
| `main.py:2801-2804` | `dump_incomplete` and `dump_incomplete_path` |

All are as cited.

*Note:* items 4 to 7 of `issues.md` are candidate observations, not among the prompt's three. Opening them is the bookkeeping session's call.

## Observations for the bookkeeping session (not acted on)

- **Stale citations.** The agent reports that README §2 (q)'s L1 and L2 citations are stale: L1 is now `main.py:2056` and L2 is `main.py:466-468`. Not checked here.
- **Holes above the primary run:**
  - 12 of 1160 rows have a WKB hole with $z_{\rm response} \ge z_{e4}$: 11 `numeric` and the `minimal` row;
  - none has one with $z_{\rm response} < z_{e4}$;
  - 17 rows have WKB nodes above their primary run.

  This is not a material fraction, and the stop condition does not apply.

## The user's ruling on check 5 (2026-10-25) — check 5 PASS; overall verdict PASS

The ruling reached this session from the laptop session (`local_284b59f6…`):

- **The ruling:** the user accepted the census's denominator account of the §5.5 shape difference. Check 5 is cleared.
- **The grounds:** the difference is documented, not invented. [`docs/source-remediation-verification.md:224-238`](../../docs/source-remediation-verification.md) records §5.5's run:
  - its command is `--k-min 1e5 --k-max 1e7 --k-count 7 … --zend 1e7`;
  - the text reads "the grid runs from `z_exit_suph_e5 = 6.8788e14` down to `z_end = 1e7` in **784 source redshifts** and **66 response redshifts**";
  - it adds "`z_end = 1e7` rather than the production `0.1` was chosen deliberately".
- **What follows:**
  - 462 = 7 × 66 is a recorded fact, not an inference.
  - The census's table (g) assumed the old response range began at $10^7$. That is the recorded configuration, not a hypothesis.
  - The response range (`z_end` 1e7 against A3 v2's 0.1) is a fourth known difference, which the refreshed prompt's §2.2 omitted.

The orchestrator re-read lines 220–240 of that file at the dispatch tree, and the quotations above are from it.

**Check 5: PASS, on the user's ruling.** `fail`/`incomplete` (7 rows on §5.5, none here) stays recorded as unexplained, as the census states.

**Overall verdict: PASS** (7 / 7; check 5 on the user's ruling).

The agent's records (`log.md`, `POLICY-GEOMETRY.md`) are left as written. The laptop adds the citation to them when it makes the bookkeeping commit.
