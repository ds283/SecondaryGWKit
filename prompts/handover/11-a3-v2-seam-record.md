# Prompt 11 — A3 v2's seam record: the policy summary and the clamp-gap census

**Campaign:** [`README.md`](README.md) · **Board item:** **11**, outside workstreams A–E. It is the
input A3's census and **D1** are scored with ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, after
[`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §3.3 items 2–3 (its
workplan item 4).
**Where it runs:** on the Mac Studio, in a Claude Code session there, as 10 did. **It commits
nothing there.** Everything it writes is under `var/handover-11/`.
**Uses:** no Ray, no lock, no store write. SQLite `mode=ro` on A3 v2's primary and four shards, and
a read of the A3 v2 run's `stdout.log`.
**Closes:** nothing. **Narrows:**
- `[12-handover-clamp-error-in-production]` (`source-remediation` board): the gap census on all
  9280 rows, where the 2026-09 figures came from one run's 1666 / 1798 sub-intervals;
- `[03-gk-classification-is-diagnosed-and-ignored]` (no board row yet; README §2 (p)): whether any
  `fail` or `incomplete` row exists on the shipping grid.

**Recommended model:** **Sonnet**. A census, read mechanically, like 03.

**Read first:**
1. [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §3, in full.
2. This campaign's [`README.md`](README.md) §2 (a) and (b): the gap by construction, and the
   2026-09 production figures.
3. `ComputeTargets/QuadSourceIntegral.py`: `_check_gap` and `build_partition` (`:291-570`). What a
   row's `metadata["partition"]` holds.
   - **A correction to findings §3.3 item 2.** The field it names, `max_clamp_gap_log1pz`, is the
     **allowance**: `HANDOVER_CLAMP_MAX_GRID_STEPS` × the mean source-grid step, the most the clamp
     will bridge. It is one number per row.
   - **The measured gaps** are in `partition["subintervals"][i]["clamp_gaps_log1pz"]`, a dict with
     keys among `Tq`, `Tr` and `source`. Each value is a shortfall in $\log(1+z)$, `0.0` when there
     is none.
4. The `source-remediation` board, `[12-handover-clamp-error-in-production]`
   (`prompts/source-remediation/IMPLEMENTATION_STATE.md:437`): the figures this census is set
   against, and how `total` was compared with `analytic_rad` by gap and by regime.
5. `main.py:3277-3290`, which prints the `GkSourcePolicyData SUMMARY STATISTICS` block;
   `ObjectFactories/GkSourcePolicyData.py:13-41`, the integer codes of `type` and `quality`.
6. [`docs/handover/a3-v2/after/scripts/depth_census.py`](../../docs/handover/a3-v2/after/scripts/depth_census.py),
   the read-only pattern to follow.

**Gate** (check and save to `out/gate_11.txt`):
- 10's board row is "reviewed";
- the dispatch commit is pushed;
- on the Mac Studio, `runregistrykit list` shows nothing `running` or stale;
- the checkout's `git status --short` is empty;
- `store show` gives A3 v2 live, fingerprint `0e32f24a…`.

No Gate condition on the code's drift since `b90752d` is needed: nothing here imports the pipeline.
The script reads stored values with `sqlite3` and `json`.

---

## 0. What this prompt is

Findings §3.2 argues that the seam governs A3 v2's accuracy, and that the stored `total`s are
probably good to a few percent. That is **inferred** from A2 and the 2026-09 figures, not measured
on A3 v2. This prompt takes the measurements A3 v2 makes possible without a rebuild:
- the **policy summary**: does the shipping grid have any `fail` or `incomplete` $G_k$?
- the **clamp-gap census**: how often, and by how much, each factor is clamped across the seam, on
  every row;
- the **gap's effect** on `total` against `analytic_rad`, in the band where that oracle is close.

**Reconnaissance, 2026-10-25.** The orchestrator looked at `stdout.log` while writing this prompt.
At lines 89–102 the block reads:
- 1160 instances;
- `numeric` 363, all `complete`;
- `mixed` 72: 71 `complete`, 1 `minimal`;
- `WKB` 725, all `complete`.

So there is no `fail` or `incomplete`. This prompt transcribes the block formally, checks it
against the table, and identifies the `minimal` row.

## 1. Steps

Make `var/handover-11/{out,scripts,state}`. Save every command and its output. Run commands with
variable arguments through `bash`.

1. **Check out the dispatch commit.**
   - `git fetch origin && git merge --ff-only origin/handover-remedial`;
   - `HEAD` is the dispatch SHA, and `git status --short` is empty;
   - `./venv/bin/pip freeze` is unchanged by the move (save it before and after).
2. **Hash the store.** The SHA-256, size and mtime of the primary, the four shards and the sidecar:
   `state/store_files_before.txt`. The same for `var/runs/handover-07-a3-v2-20261022T211155/stdout.log`.
3. **The policy summary** (`out/policy_11.txt`).
   - Copy the `GkSourcePolicyData SUMMARY STATISTICS` block from that `stdout.log` **verbatim**,
     with its line numbers.
   - Count the `GkSourcePolicyData` rows directly from the shards, `mode=ro`, by decoded
     `(type, quality)`.
   - **The two must agree in every cell.**
   - List every row that is not `complete`, with its $k$, $z_{\rm response}$, `type`, `quality` and
     `crossover_z`.
4. **Write `scripts/seam_census.py`**, read-only and Ray-free, following `depth_census.py`. It
   writes a header with:
   - its own SHA-256;
   - each store file's SHA-256;
   - the store's fingerprint;
   - `git HEAD`.

   Then:
   - **(a) The population.** QuadSourceIntegral rows at `(1e-32, 1e-7)`. **9280 expected.** Every
     row's `metadata["partition"]` must parse.
   - **(b) One record per sub-interval**, in `out/subintervals_11.csv`:
     - the row's keys, $k$, $q$, $r$, $z_{\rm response}$ and band (low $<9$, mid $9$–$10^5$, seam
       $>10^5$);
     - the sub-interval's `z_max`, `z_min`, regime and method;
     - its `clamp_gaps_log1pz` for `Tq`, `Tr` and `source`;
     - the row's `max_clamp_gap_log1pz`, and the mean source-grid step it implies, the allowance
       divided by `HANDOVER_CLAMP_MAX_GRID_STEPS`. Read the constant from the source and record its
       value.
   - **(c) The gap census** (`out/seam_census_11.txt`). For each of `Tq`, `Tr` and `source`:
     - the number of sub-intervals with a non-zero gap, and the number of rows with at least one;
     - the gap's quartiles, median and maximum, both in $\log(1+z)$ and in mean source-grid steps;
     - the number at or within 1 % of the allowance;
     - the same, by $k$ and by band.

     Set the result **beside** the 2026-09 figures: `Tq` on 1666 sub-intervals, `Tr` on 1798,
     median 1.2e-02, max 2.2e-02, about 1.0 mean grid step, and the source spline clamped on 108.
     Also count rows with any `skipped` breakpoint.
   - **(d) The gap's effect, against the oracle**, as
     `[12-handover-clamp-error-in-production]` did it.
     - **Population:** seam-band rows only ($z_{\rm response} > 10^5$, where LambdaCDM is
       radiation-dominated).
     - **Measure:** $|{\rm total}/{\tt analytic\_rad} - 1|$.
     - **Strata:** the number of clamped factors in the row (0, 1, 2), and the number of
       oscillatory factors in the row's last Levin sub-interval (0 to 3), each tabulated by
       $z_{\rm response}$ decade, with median and maximum. Rows with `analytic_rad` zero or
       non-finite are counted and excluded.
     - **State in the output what this does not separate.** The oracle is constant-$w$, built at
       $b = 0$. Its own deficit on LambdaCDM grows towards matter–radiation equality. The 2026-09
       comparison was taken at $z_{\rm response} = 7.63\times10^9$ and $1.45\times10^9$ for that
       reason. So decades below about $10^8$ are reported but not read as the clamp's error.
5. **Run it** with `./venv/bin/python -I scripts/seam_census.py <primary>`. If it may exceed ten
   minutes, run it in the background and wait for the notification. Do not poll.
6. **Hash again.** `state/store_files_after.txt` and the `stdout.log` hash, both identical to
   step 2. Run `store fingerprint` (without `--write`); it must equal the sidecar's.
7. **Write the records.**
   - `log.md`, in `qcd-background-audit` README §5.1's form, as 10's was;
   - `verification.md`: the text of **§5 "The seam record"** of
     `docs/handover/A3-V2-VERIFICATION.md`;
   - `issues.md`.

## 2. Acceptance

1. Every store file and `stdout.log` are byte-identical before and after, and the fingerprint is
   `0e32f24a…`.
2. The summary block is transcribed, and it agrees cell by cell with the table count. Every
   non-`complete` row is listed.
3. The gap census covers 9280 rows, with the three factors' statistics set beside the 2026-09
   figures.
4. The oracle comparison is tabulated as step 4 (d) says, its limits stated.
5. Records under `var/handover-11/`: `out/`, `scripts/`, `state/`, `log.md`, `verification.md` and
   `issues.md`.

The orchestrator then:
- copies everything to `docs/handover/a3-v2/seam/`, with `subintervals_11.csv` gzipped if it is
  over 10 MB;
- appends §5 to `A3-V2-VERIFICATION.md`;
- writes `logs/11-a3-v2-seam-record.md`;
- narrows `[12-handover-clamp-error-in-production]` on the `source-remediation` board;
- records the policy result against README §2 (p);
- commits.

## 3. Stop conditions — stop and ask

- Any Gate item not as stated.
- The summary block is missing, or disagrees with the table in any cell.
- A population other than 9280, or a row with no `partition`.
- Any store file, or `stdout.log`, changes.
- A gap above the row's allowance. `_check_gap` raises on that, so such a row should not exist.
