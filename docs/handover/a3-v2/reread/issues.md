# Issues — handover prompt 10

No stop was hit, and nothing here blocks acceptance. Three observations are offered for the
orchestrator to open, narrow or drop. None was acted on.

## 1. The readers' work lists are unseeded random draws

- **Found:** `extract_GkWKB_data.py:472` draws `z_subsample` with `random.sample`, with no seed, at
  about 10 % of the source redshifts: 174 here. Whether the pair 08 died on
  (`z_source = 1.3633e16`, serial 19) is plotted at all is therefore a draw with odds of about
  174 / 1740, roughly 1 in 10.
- **It was in this run's sample, so acceptance 2 holds.** But a re-run would not reproduce that
  in general. Prompt 10's procedure cannot by itself guarantee that "the pair 08 died on has its
  PDF, PNG and CSV".
- **The other readers draw the same way.** `extract_Gk_data.py:416` and `:432`, `extract_GkSource_data`,
  `extract_TkWKB_data`, `extract_tensor_source_data` and `extract_QuadSourceIntegral_data` all use
  `sample`, so their file counts vary between runs on the same store. Here, `extract_Gk_data` made
  1251 files with `--models` and 1980 without, from the same 522 lookups.
- **Impact:**
  - A reader re-run cannot be compared count-for-count with an earlier run.
  - A check that names a specific pair passes only when the draw includes it.
  - The read-back is complete for the drawn sample, not for the store.
- **Next step, if wanted:** a `--seed`, or an explicit `--pair` / full-grid mode, in the readers.

## 2. The lock's `status` exits 0 while held with no pid recorded

- **Expected:** the orchestrator's note §0, citing RayPoolKit `docs/rehearsal-lock.md` §4 at
  `v0.3.0`, expected `status` to exit **3** while the lock is held with `pid: -`.
- **Found:** `status --ray-temp-dir /tmp/ray-sgk` exited **0** in that state. It printed
  `lock: HELD`, `owner pid: - (no pid recorded yet)` and "Ray processes of /tmp/ray-sgk: none"
  (`out/lock_10.txt`, 10:49:41Z).
- **Not checked:** the form without `--ray-temp-dir`, while the lock was held. The documented table
  was not read on this host.
- **Next step:** either the note or the package's documentation is wrong about this exit status.
  This is for the orchestrator to check against RayPoolKit `v0.3.0`. If the package is at fault,
  it is reported there, per `CLAUDE.md`.

## 3. `extract_GkWKB_data.py` leaks its theta figure

- **Found:** the GkWKB log now carries matplotlib's "More than 20 figures have been opened"
  `RuntimeWarning`, twice, from `:144` and `:322`. 08's run failed before any worker reached 20
  figures, so its log did not have it.
- **Cause:** `plot_Gk` opens a second figure for the theta plot (`:322`), and the
  `plt.close()` at `:319` comes before it. The theta figure is therefore never closed, and each
  worker accumulates one figure per pair.
- **Impact:** memory in the plotting workers only; all 1392 pairs were written. This is
  pre-existing, and not part of 09's change.
