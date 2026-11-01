# Files of prompt 06h's Part B kept outside this directory

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06h/` (paths below are relative
to it), and those in the hashed list at the end also in the laptop's `var/levin-phase-floor-06h/`.
They were left out of this directory as 06e's were: every pickle, every file over 2 MB, and the
readers' `rehearsal/extract/`. Each agent's own list, with SHA-256 and sizes, is its
`KEPT_ON_MAC_STUDIO.txt`: [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt),
[`census/`](census/KEPT_ON_MAC_STUDIO.txt) and [`phase/`](phase/KEPT_ON_MAC_STUDIO.txt).

**Kept on the Mac Studio only, never copied to the laptop:**
- `rehearsal/extract/` (3,099 files, 172 MB), the readers' output: unseeded random subsamples, not
  reproducible file for file;
- the census's Gate H and smoke pickles, `census/state/H/`, `census/state/H2/` and
  `census/state/smoke/` (14 files);
- the phase measurement's 80 capture pickles, `phase/state/capture_h/` and `phase/state/capture_p/`,
  the inputs and regions of 06f's 20 M2 calls at $h$ and at the parent.

**The census's run directory** `var/runs/levin-phase-floor-06h-census-20261101T005037/` (149 files,
2.3 GB) stays there in full: the checkpoint, 16 region pickles, the events and the worker logs. It
is 06g's "before". Its `checkpoint.jsonl` has SHA-256
`3104c18ed8b119f1798aec75be98d5b39bce641f3b5623b4c1cdc5b18b5ef372`. The phase measurement's run
directory `levin-phase-floor-06h-phase-20261101T033514` stays in `var/runs/` as the registry's
record.

**The rehearsal stores** `var/datastores/levin-phase-floor-06h-r21*` and `-r22*` are kept until the
user retires them; on 2026-11-01 the user said they will (*"I'll retire R21 and R22 to keep things
clean."*). Their fingerprint, `1b332ffa…`, can be taken only from a checkout whose code declares
`theta_residual`, such as the worktree `SecondaryGWKit-lpf06h-c1ec3b2`, not from the Mac Studio's
main checkout at `28ebaf0` (`review/checks/fingerprint_note.txt`). The run directories
`levin-phase-floor-06h-r21-parent-20261101T001852` and
`levin-phase-floor-06h-r22-groupphase-20261101T002423` stay in `var/runs/`, with their launch logs
`levin-phase-floor-06h-r2{1,2}.launch.log` beside them.

**Added past the laptop's `*.csv` exclusion:** `rehearsal/out/qsi_rows.csv` and
`rehearsal/out/qsi_analytic_rows.csv` (the per-row tabulations of all 864 rows), and
`phase/out/phase_calls.csv` and `phase/out/phase_leaves.csv` (H1's and H2's tables).

The scratch copy of A3 v2 that the census read,
`var/levin-phase-floor-06/wider/state/a3v2_copy/`, is 06's (`06/KEPT_ON_MAC_STUDIO.md`), and is
still kept until the user says otherwise.

**Left out of this directory, on both machines**, by SHA-256 taken on the Mac Studio on 2026-11-01 and equal on
the laptop:

```
68b9c378166368df621d21c0aac12d5a4edcb548a43cc719f67e7421c841fd0f  rehearsal/out/store_value_diff.json
aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417  census/state/rows.json
```

`census/state/rows.json` is byte-identical to 06e's (`06e/KEPT_ON_MAC_STUDIO.md`).
