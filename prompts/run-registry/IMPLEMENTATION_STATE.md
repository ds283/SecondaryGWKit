# Run registry campaign — implementation state

**Last updated:** 2026-09-22 · **Status: STARTED — 1 of 2 prompts landed (01).** Prompt 01 landed
the `RunRegistry/` package, the `var/runs/` layout, the lister, and the six rules as a `CLAUDE.md`
section. **The self-match regression is in the tree and bites**: the same test fails against a
`pgrep`-based implementation and passes against the shipped one. Nothing has adopted the registry
yet — that is prompt 02, which carries the `realistic_large_x.py` hash hazard. One issue is open in
§3; §4 is empty.

**Campaign:** [`README.md`](README.md) ·
**Package:** `RunRegistry/` · **Layout:** `var/runs/<campaign>-<prompt>-<slug>-<timestamp>/` ·
**Rules:** [`CLAUDE.md`](../../CLAUDE.md), "Long-running jobs — the run registry" ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.11

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit** —
> the row added, moved or deleted, and the count and date in its header corrected. An issue owned
> by another board is moved to **that** board's §4 and its row deleted from the index. The index is
> an index: one line per issue, pointing here. Where the two disagree, this board is right.
> See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [The run registry](01-the-run-registry.md) | **G1** | Opus 5 | ✍️ yes | ✅ 2026-09-22 | *"Record long-running jobs on disk instead of in a conversation"* | [`logs/01-…`](logs/01-the-run-registry.md) |
| 02 | [Adopt the registry](02-adopt-the-registry.md) | **G2** | — | ✍️ yes | ⬜ not run | — | — |

The two items below are the two prompts' own: prompt 01's header names **G1**, and README §2 gives
the campaign two prompts and no more. The four deliverables of G1 are kept in one row rather than
split, because they land or fail together — a helper without the `CLAUDE.md` rules is the thing
README §0.3 calls worse than nothing.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| G1 | **FACILITY + RULES** | The `var/runs/` layout, the helper, the lister, the launch and liveness rules, and the `CLAUDE.md` section that makes them binding. | 01 | ✅ **Done, 2026-09-22.** Four parts. **(a) The helper** — `RunRegistry/` (534 lines over two modules, 333 statements, standard library only, no production compute path in the diff): `begin()` / `known()` / `record()` / `heartbeat()` / `finish()` / `list_runs()`, an immutable `manifest.json`, an atomically-replaced `status.json`, an append-only `checkpoint.jsonl` stamped with the provenance triple. **Every manifest field traces to a README §0 failure** (log, "Every manifest field"); `status.json` carries §1.3's six fields and nothing else. Script-hash mismatch is **discard with a notice**, as `realistic_large_x.load_checkpoint` does, and a record is **never re-stamped** — asserted byte for byte. The hash is taken once in `begin()`, which is the source the interpreter is actually running. **(b) The regression** — `RunRegistry/tests/test_run_registry.py`, **16 test methods**, 2.0 s, no Ray, no datastore, everything in a temporary directory; **demonstrated to bite**, the same module giving `AssertionError: 'alive' != 'stale'` against a `pgrep -f`-based `liveness()` and `OK` against the shipped one (log §1). It refuses to pass vacuously — it asserts the trap is armed before asserting the verdict, and that guard **fired twice during development**. **Finding:** macOS `pgrep` excludes itself *and its ancestors* unless `-a` is given, so a lone poller cannot match itself; the real mechanism is **cohort**-matching between concurrent pollers, which is why nineteen deadlocked where one would have worked, and the test reproduces it that way. **(c) The lister** — `python -m RunRegistry list`, 87 lines, read-only; against the real `var/runs/` it reports **both A3 pilot directories**, which have no manifest, and ignores the four loose files beside them. **(d) The rules** — `CLAUDE.md`, "Long-running jobs — the run registry": discovery, liveness, launch, do not babysit, durability, provenance. Rule 1 is the load-bearing one and has **no mechanism behind it** (log, "Limits"). |
| G2 | **ADOPTION** | The existing long-running entry points register themselves, **without invalidating any existing checkpoint**. | 02 | ⬜ **Written, not run.** Nothing uses `RunRegistry` yet. The hazard is unchanged and untouched: `docs/handover/realistic_large_x.py` gates reuse on a hash of its own source, so editing it discards the 60 cells in `var/runs/realistic_large_x_cells.jsonl` behind `REALISTIC-LARGE-X.md`'s tables. |

---

## 3. Active and unresolved issues

- **[01-var-runs-holds-unattributable-loose-files]** *(opened 2026-09-22 by prompt 01)* — four
  files sit at the top level of `var/runs/`, beside the two A3 pilot directories, with nothing
  saying which run they belong to: `run.out` (23,616 bytes, 2026-09-20T17:08), `run.pid` (6 bytes,
  2026-09-20T14:09), `run.progress` (3,504 bytes) and `realistic_large_x_cells.jsonl` (60,484
  bytes, 60 cells). Three of the four are exactly the artefacts the registry now gives a home to —
  a stdout capture, a pid and a progress file — and the mtimes suggest, but do not establish, that
  they belong to the `realistic_large_x.py` run of 2026-09-20 rather than to either A3 attempt.
  This is README §0 item 2 in its residual form: the files survived, the attribution did not.
  **Impact:** low and bounded. The lister ignores them correctly (it lists directories), nothing
  reads `run.pid` or `run.progress`, and `realistic_large_x_cells.jsonl` is load-bearing and
  correctly placed — its path is in `realistic_large_x.py`'s own docstring, which is its
  attribution. What is lost is the other three's. **Next step:** prompt 02, which is the prompt
  entitled to touch `realistic_large_x.py` and to decide where a registered run's outputs go, may
  record in its log which run each file came from if it can establish it from the file contents —
  **but must not move, rename or delete any of them**, per campaign README §4 and prompt 01 §3.
  Deliberately not acted on here: prompt 01 may only *read* `var/`. Measurement:
  [`logs/01-the-run-registry.md`](logs/01-the-run-registry.md), "Observations not acted on" item 1.
  Indexed at `docs/OPEN_ISSUES.md` §1.11.

---

## 4. Resolved issues

None yet. This campaign has opened one issue and closed none.
