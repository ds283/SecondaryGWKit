# Files of prompt 06j kept outside this directory

These were kept in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06j/` (paths below are relative
to it), where the rest of 06j's records also stay. They were left out of this directory as 06f's
were: every pickle and every file over 2 MB. There was no file over 2 MB other than the pickles.

> **Removed 2026-11-03, on the user's word** (*"If we don't need the pickles they can also be
> cleaned up."*). The 138 files below were checked against their SHA-256 (all 138 equal) and
> deleted. The worktree was removed at the same time. They can all be recreated: every capture
> and every replay reproduced 06g's census bit for bit (Gate J0) from the scratch copy and 06g's
> census checkpoint, both still kept, and run 1, which made them, took 11 minutes. The run
> directories are kept: they are the registry's record of the runs.

**Kept on the Mac Studio only, never copied to the laptop, until 2026-11-03:** the 138 pickles that
[`KEPT_ON_MAC_STUDIO.txt`](KEPT_ON_MAC_STUDIO.txt) (SHA-256 `6cd7b72a…`) lists with SHA-256 and
size, 137,574,385 bytes in all. The Mac Studio review re-checked every hash and size against the
files, with 0 mismatches, and found the list complete (`review/checks/kept_listed.sha256`,
`kept_found.txt`). They are:
- the 92 capture pickles of the 46 sampled calls, `state/capture/C00/` to `C45/`
  (`inputs_CNN.pkl` and `regions_CNN.pkl`), which `x1_capture.py` wrote and which unpickle with the
  harness's `scripts/` on `PYTHONPATH`;
- the 46 per-call inputs under `state/inputs/` (`s00`–`s23`, `f00`–`f09`, `c00`–`c11`), which
  `slice_06j.py` cut from the captures and checked by SHA-256; the replays, the probes and the
  reference tool read them.

**Added past the laptop's `*.csv` exclusion:** the three tables `out/partS_leaves.csv`,
`out/partS_nodes.csv` and `out/w_calls.csv`. Python's `csv` wrote them with CRLF line endings, and
git stores them with LF (`core.autocrlf=input`, as for 06f's tables). So the committed bytes
differ from the files whose SHA-256 the command transcripts print. Those originals are still on
the Mac Studio, and the laptop's working copies are byte-identical to them.

**The run directories** stay there and are also copied to the laptop's `var/runs/`, checked by
SHA-256:
- `var/runs/levin-phase-floor-06j-run1-20261103T091504/`: the captures, Gate J0, Part S's probes
  and Part W's preparation; `done`, 76 of 76, exit 0;
- `var/runs/levin-phase-floor-06j-readings-20261103T093537/`: the 68 reference readings; `done`,
  68 of 68, exit 0.

**Read, not written, by 06j:**
- 06g's census checkpoint
  `var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl` (`03adcbba…`) and its
  16 region pickles;
- 06h's census checkpoint (`3104c18e…`);
- 06g's `census/out/v8_calls.json` (`1e9a62b0…`) and `v8_slow_leaves.json` (`b4643033…`);
- 06's scratch copy of A3 v2, `var/levin-phase-floor-06/wider/state/a3v2_copy/`.

Each was unchanged by SHA-256 before and after (`review/checks/`), and each is still kept until the
user says otherwise (`06g/KEPT_ON_MAC_STUDIO.md`, `06/KEPT_ON_MAC_STUDIO.md`).

**Checked from the laptop on 2026-11-03:** the 961 files copied back equal the Mac Studio's by
SHA-256, and so do the two run directories' 10 files. S1, S3 and W1–W4 were re-scored from
`out/partS_leaves.csv` and `out/w_calls.csv` by the laptop orchestrator's own reader, giving the
same verdicts and numbers.

**The worktree** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9` (detached at
`dfa19d98`, clean with `--ignored`) was removed on 2026-11-03 with `git worktree remove`, on the
user's word.
