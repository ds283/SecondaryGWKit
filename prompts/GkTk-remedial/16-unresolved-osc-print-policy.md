# Prompt 16 — The `has_unresolved_osc` print policy: a per-$k$ summary

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[00-unresolved-osc-print-policy]`; README §7 **D2**, decided by the user 2026-09-11
**Review sections:** §13.1 (the flag is a live warning), §10.2; no new findings.
**Design facts:** README §2 **(h)** — read the note in §1 below before you touch anything.
**Depends on:** 11 (which implemented the faithful test and measured the fire rate).
**Recommended model:** Opus — `main.py` plumbing on two work queues, which cannot be imported and
whose failure mode is silent.
**Files you may touch:** `Quadrature/integrators/numeric_with_phase_cut.py`,
`ComputeTargets/GkNumericIntegration.py`, `ComputeTargets/TkNumericIntegration.py`, `main.py`,
`ComputeTargets/tests/test_numeric_phase_cut.py`, `ComputeTargets/tests/test_main_plumbing.py`,
plus the log, the status board and `docs/OPEN_ISSUES.md`.
**Do not touch:** the Datastore factories; the *logic* of
`scan_sample_grid_for_unresolved_osc`'s test (which pair fails, and the values of
`has_unresolved_osc` / `unresolved_z` / `unresolved_efolds_subh`) — prompt 11 settled that and it
must not move by a bit; `LiouvilleGreen/integration_tools.py`; the stop-search window attributes
`z_exit_subh_e3`/`_e6` and the $\sqrt{z_{e3}z_{e4}}$ limit (README §0.3); `main.py`'s
`delta_logz=` arguments, its `0.85` truncation constants and its tolerance plumbing; everything
README §5 rule 8 lists.

Read first: README §2 (h), §7 D2; `IMPLEMENTATION_STATE.md`'s `[00-unresolved-osc-print-policy]`
entry in §3 and rows M17–M18; `logs/11-numeric-diagnostics-and-units.md` — its Result section
(the D2 measurement) and its "State handed to the next prompt"; review §13.1.

---

## 1. The one thing that could go wrong, first

README §2 (h) — *"preserving the `has_unresolved_osc` warning"* — is a campaign stop condition, and
`README.md` §4.3 lists deleting it among the deviations the orchestrator must stop on. **This
prompt does not delete the warning. It relocates it**, which is the decision README §7 D2 reserved
for the user and which the user took on 2026-09-11, option (ii): *store the flag and print a
per-$k$ summary in `main.py`*.

So: the flag must still be computed exactly as prompt 11 computes it, still be returned in the
payload, still be persisted by the (untouched) factories, and the information must still reach a
human running the pipeline — as one summary per $k$ instead of two lines per object. If you find
yourself removing the test, changing which sample pair trips it, or dropping the fields from the
payload, stop and say so: that is not this prompt.

**Why the change is wanted.** Prompt 11 made the test faithful, and it consequently fires on
**2,149 of 2,149** $G_k$-like objects (0 of 6 $T_k$-like). At ~65,000 `GkNumericIntegration`
objects per model that is ~$1.3\times10^5$ printed lines per model, where production prints none
today. The signal is real and worth keeping — it is the same condition as the hand-over campaign's
`[05-numeric-region-is-now-the-accuracy-floor]` and `[06-source-spline-residual-vs-handover]`
(`docs/OPEN_ISSUES.md` §1.1) — but per object it is unreadable.

---

## 2. What to build

### 2.1 Make the per-object print opt-out, and opt out of it

`scan_sample_grid_for_unresolved_osc` currently prints its two warning lines whenever the flag
trips. Give it a keyword-only `warn: bool = True` parameter that gates **only** the two `print`
calls, and thread an equivalent through `numeric_with_phase_cut`
(`warn_unresolved_osc: bool = True` is the suggested spelling). The two production integrators
pass the value that suppresses the per-object line, because `main.py` now summarises.

The default stays `True`, so any other caller — a test, a script under `docs/`, a future
integrator — keeps today's behaviour and the warning is still one function call away. Deleting the
`print`s outright instead is **not** acceptable: it would make the relocation irreversible and
would leave a direct caller of the integrator with no warning at all.

Nothing else about the scan changes. Its return value must be bit-identical to prompt 11's for
every input.

### 2.2 Accumulate and report, in `main.py`

`main.py` builds the two numeric work queues with `store_results=False`
(`RayWorkPool(... title="CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS" ...)` and its
transfer-function twin), so the computed objects are **not** retained in `RayWorkPool.results` and
there is no list to sweep afterwards. The seam that does exist is `post_handler`: `RayWorkPool`
calls it once per completed task, in the driver process, at the task's exit point
(`RayTools/RayWorkPool.py:361-363, :398-400, :417-419, :489-491, :513-515`), and a `post_handler`
returning `None` is safe with `store_results=False`. Neither numeric queue passes one today.

Use it. Per sector: an accumulator created before the queue runs, a `post_handler` that records
each completed object into it, and a summary printed after `.run()` returns.

**Two constraints fix the shape of this code.** `main.py` cannot be imported (README §5 rule 11),
and `ComputeTargets/tests/test_main_plumbing.load_main_py_functions` extracts **module-level
`ast.FunctionDef` nodes only** — not classes, and not functions nested inside `run_pipeline`. So
the recording and the formatting must be **module-level functions of `main.py`**, and the
formatter must **return** its lines rather than print them, or it cannot be tested. Suggested,
and an `IMPLEMENTATION CHOICE` to be recorded if you depart from it:

```python
def record_unresolved_osc(summary: dict, obj) -> None:
    """Record one completed numeric-integration object into `summary`, keyed by wavenumber."""

def format_unresolved_osc_summary(summary: dict, sector_label: str) -> List[str]:
    """Render `summary` as the lines to print. Returns [] when nothing flagged."""
```

with the queue wired as `post_handler=lambda obj: record_unresolved_osc(Gk_unresolved_osc, obj)`
and the lines printed after the queue completes.

**What the summary must contain**, per wavenumber that flagged at least once: the wavenumber, how
many objects flagged out of how many were recorded for that $k$, and the range of
`unresolved_efolds_subh` (equivalently of `unresolved_z`) over them — the depth inside the horizon
at which the grid first failed, which is the quantity the hand-over campaign wants. A single
closing line giving the totals across all $k$. When nothing flagged, say so in one line rather
than printing an empty block; a run that prints nothing at all is indistinguishable from a run
where the wiring was dropped, which is the silent failure this prompt is most exposed to.

Guard the accessors: `has_unresolved_osc` raises `RuntimeError` when it has not been populated
(`TkNumericIntegration.py:235-238` and the `GkNumericIntegration` twin). An object that reaches
`post_handler` should be populated, but record defensively and count anything unpopulated
separately rather than letting the pipeline die in a reporting path.

### 2.3 Do not change what is stored

The payload keys, the factories and the persisted columns are untouched. A datastore written
before this prompt is readable after it, and vice versa: this prompt changes **only** where the
information is printed. Say so explicitly in the log.

## 3. Tests

In `test_numeric_phase_cut.py` (extending prompt 11's module, not rewriting it):

1. **The flag is unchanged.** With `warn=False` the returned dict — all three keys — is equal to
   the `warn=True` result on the same run, and prompt 11's existing bit-identity constants still
   pass. The scan's decision must not depend on the new parameter.
2. **The print is gated, both ways.** On a deliberately coarse grid (prompt 11 has one):
   `warn=True` prints the two warning lines; `warn=False` prints nothing at all; in both cases
   `has_unresolved_osc is True`. Capture stdout.
3. **The integrators opt out.** A production-geometry `Gk` run through `numeric_with_phase_cut`
   prints nothing, and still returns `has_unresolved_osc is True`.

In `test_main_plumbing.py`:

4. **The summary functions work**, exercised through `load_main_py_functions` on stand-in objects
   in the pattern already in that module: several objects across two wavenumbers, a mix of flagged
   and unflagged, produce a summary naming both wavenumbers with the right counts and the right
   e-folds range; an all-clear accumulator produces the one-line "nothing flagged" form; an object
   whose flag is unpopulated does not raise.
5. **The wiring exists — an `ast` guard, in the spirit of prompt 12's.** Both numeric
   `RayWorkPool` constructions in `main.py` pass a `post_handler`, and both are followed by a call
   to the formatter. This is the test that catches the silent failure: if a later edit drops the
   `post_handler`, the pipeline still runs and simply never reports.

## 4. Verification and acceptance

- New and existing tests pass; `discover -s ComputeTargets/tests -t .` passes with **no test
  lost** (prompt 11 left it at 273).
- `grep -n "print" Quadrature/integrators/numeric_with_phase_cut.py` shows the two warning lines
  still present, and reachable — gated, not deleted.
- `git diff HEAD~1 -- main.py` touches only the two queue constructions, the accumulators, the
  summary calls and the two new module-level functions. The `delta_logz=` arguments, the `0.85`
  constants and the tolerance plumbing are untouched.
- `git diff HEAD~1 --stat` shows no `Datastore/`.
- `black --check` clean on every file you touched.
- Report, in the log, the summary block a production run would now print for one $k$ — the actual
  formatted text, from the test's stand-ins — so a reader can see what replaced the 1.3e5 lines.

## 5. Log and commit

Close `[00-unresolved-osc-print-policy]` in `IMPLEMENTATION_STATE.md` §3, move it to §4 with the
decision and its date, and update `docs/OPEN_ISSUES.md` in the same commit (count 51 → 50). In the
§4 entry, record that the *semantic* question the flag now detects — whether the response grid
should resolve the mode through the numeric→WKB seam at all — is **not** closed by this prompt and
is owned by the hand-over campaign (`docs/OPEN_ISSUES.md` §1.1,
`[05-numeric-region-is-now-the-accuracy-floor]` and `[06-source-spline-residual-vs-handover]`);
this prompt settles only where the information is printed. Update board row 16 and M17/M18.

"State handed to the next prompt": the new parameter names and defaults; the two module-level
`main.py` function names and signatures; the summary format.

Commit subject, or something equally specific:
`Summarise unresolved-oscillation warnings per wavenumber`.
