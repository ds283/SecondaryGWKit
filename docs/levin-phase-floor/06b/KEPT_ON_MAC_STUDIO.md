# Files of prompt 06b's Part B kept on the Mac Studio only

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06b/` (paths below are relative
to it). They were left out of this directory as 06a's were: every pickle, every file over 2 MB, and
the readers' `rehearsal/extract/` (180 MB, 3,220 files).

**The census's run directory** `var/runs/levin-phase-floor-06b-census-20261028T161212/` (1.7 GB,
138 files) stays there in full: the checkpoint, the per-worker pickle streams of region arrays, the
events and the worker logs. Prompt 07's §6, through 06c's census, compares against it. Its
`checkpoint.jsonl` has SHA-256
`f934ea245c0086529fb9372a9ad457d4d8668114ce346dca5440404c766646ad`.

**The rehearsal stores** `var/datastores/levin-phase-floor-06b-r15*` and `-r16*` (ten files) are
kept until the user says otherwise. Their SHA-256 are in `rehearsal/state/r15_store_sha256_after_build.txt`
and `rehearsal/state/r16_store_sha256_after_reads.txt` in this directory. The run directories `levin-phase-floor-06b-r15-parent-20261028T135349` and
`levin-phase-floor-06b-r16-premise-20261028T135745`, and the two launch logs beside them, stay in
`var/runs/`.

**Added past the laptop's `*.csv` exclusion:** `rehearsal/out/qsi_rows.csv` (the per-row tabulation
of all 864 rows), `census/out/moved_calls.csv` and `census/out/w3_regions.csv`.
`rehearsal/out/qsi_rows.attempt1.csv`, a superseded first attempt, is not committed.

The scratch copy of A3 v2 that the census read, `var/levin-phase-floor-06/wider/state/a3v2_copy/`,
is 06's (`06/KEPT_ON_MAC_STUDIO.md`), and is still kept until the user says otherwise.

The SHA-256 of each was taken on the Mac Studio on 2026-10-28 at 17:44 UTC, when the records were
copied back.

```
918d304a8d93bc0c55f95b1ae0da15f16e7dd7835603ded4789dd962e676f450  census/out/cmd/40_convergence_kinds.txt
8835477bc75e697680540e54e6f801a4aa96811208d05c796b02094986fd9641  census/out/commands.log
24d197283b75d62df5634ee8937fb251a7177f13505daa2c142ec0ae314fd920  census/state/W0/B1/inputs_B1.pkl
ccf70d72f36aa30d7185ca667dc6c95bb542ed305aaa529b0568c4827238cc0d  census/state/W0/B1/regions_B1.pkl
ddec6e410d18266ed99ebd3b8b68b0ec574cb32f95505c96f731243673c517de  census/state/W0/B2/inputs_B2.pkl
6536fcd0c2af9f1233e9f9e06816bd26e3973ac8a56e48d02ed7bb53f36cfb6d  census/state/W0/B2/regions_B2.pkl
163fa08113f22b267ad38f3af7d8338c3e95a097106bc91460e9000205b6604b  census/state/W0/P1/regions_P1.pkl
1f69daa5099fddad4038af1c1d8d3fa8c049327a8cbaa76a670edc01b7a72512  census/state/W0/P2/regions_P2.pkl
e61dd7a6148ec0736ea55914c8ab1bfae96857a7007dd30e24c314120a9e4b3f  census/state/W0/P3/regions_P3.pkl
d57ac2a6a826b15c315bc89071ce3f0198bff21ceb6aa7b07b6dd5575b553e7b  census/state/dev/smoke1/inputs_S1.pkl
a622f89d39ede9b763569e5897e684a28cae2818e3e9cf3829e0811ac685bf8d  census/state/dev/smoke1/regions_S1.pkl
946fd068cca6a29e086cdfbca73160cb808ae27ae0cb655adeba81155eddc673  census/state/dev/smoke2/inputs_S2.pkl
8128d05891580de6386ebd79352ee858daefbb25f918f3649bebf7821718b7bf  census/state/dev/smoke2/regions_S2.pkl
bdb1e9eb7bf565be2a99683ec01f84dfe013bb494f07b9d4a44ac54c8eedd6c1  census/state/dev/smoke3/inputs_S3.pkl
b1d8ce6af7ee5f024321e81c227b6890609852e56ffb0e119698e3afdcf3c676  census/state/dev/smoke3/regions_S3.pkl
aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417  census/state/rows.json
48fa4ec1d057b1101c60ed027058e7fc2aed5758367bc2e3a745c816b086656d  census/state/smoke/inputs_1e-7-k1_102.pkl
4a550123b8063b01bb5028aa2f3644c13a63c35b96c311cc70ea686c176aa7ee  census/state/smoke/inputs_1e-7-k3_1.pkl
51a04e9ce94532e895dffe7e0a99507134c09ed78682239f4903ba777fbd84bc  census/state/smoke/regions_1e-7-k1_102.pkl
61c4e80beaa28c2a21a75d0ed6a5ff71164a4a5636d64c17c5a434c6c8b27730  census/state/smoke/regions_1e-7-k3_1.pkl
0d5d8977eb9fafc902084fa7a69305b8c2e230db1123b47f0ee697c2d1556a20  census/state/smoke/regions_1e-8-k1_102.pkl
5495e96b5de20a787fc26203f74515830eec3e9ed5b895f2dc4512a3b658c5c8  census/state/smoke/regions_1e-8-k3_1.pkl
51b9b94a4581e7e5d58b00ea4e8940f7c91f28f5e85a2dcf8fbaf7b85f3cffe6  rehearsal/out/store_value_diff.json
```

`census/out/cmd/40` is the full output of the changed `convergence_kinds_06a.py`; this directory's
`census/out/convergence_kinds_b.txt` holds its summary. `rehearsal/out/store_value_diff.json` (40 MB)
is the value comparison's full output; its summary is `rehearsal/out/store_value_diff.meta.txt`, and
the Mac Studio's review re-ran it byte-identically.
