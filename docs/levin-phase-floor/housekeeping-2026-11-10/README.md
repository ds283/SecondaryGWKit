# Housekeeping after levin-phase-floor 07, 2026-11-10

On the user's word of 2026-11-10 (*"Yes, please go ahead with the cleanup as listed"*), after 07's
Part B review passed and its records were copied back to the laptop and checked by SHA-256.
Every file deleted is listed here with its SHA-256, taken immediately before deletion.

## Laptop

- **07's four stashes**, drafts of 07 Part A superseded by the landed commits, were dropped. Their
  SHAs and messages are in `laptop_stashes_dropped.txt`.
- **`var/levin-phase-floor-07/wip/`** (patches of the same drafts) and the copy-back directories
  `var/levin-phase-floor-{06,06c,06e,06f,06g,06g-fcc-laptop,06h,06i,06j,06k,06l,06m,floor-breakdown,07-leaves}/`
  were deleted: 9,593 files, in `laptop_var_deleted.sha256`. Their records are committed under
  `docs/levin-phase-floor/`. The 139 files with no byte-identical committed copy are captured
  inputs (pickles) and intermediate outputs over 2 MB.

## Mac Studio

- **Stores retired** with `runregistrykit store retire`, each sidecar kept as its record:
  R13, R14 (06), R15, R16 (06b), R23, R24 (06g), R25, R26 (06m), R27, R28 (07). None had a
  registry sidecar, so each was given one (`store create`), its fingerprint recorded
  (`store fingerprint --write`), and then retired; every fresh fingerprint matched, `1b332ffa…`.
  R13 was read from the main checkout (`28ebaf07`). The others carry `theta_residual` and were read
  from 07's tree `e182bc7a`. The full transcript is in `macstudio_stores_retired.txt`. **Kept:** A3 v2
  and R29.
- **Census run directories trimmed** to `manifest.json`, `status.json`, `stdout.log` and
  `stderr.log`, plus a `TRIMMED_2026-11-10.txt`. The runs trimmed are 06a, 06b, 06c, 06e, 06g, 06h,
  06k's four routes, 06l and 06m. 1,665 files were deleted, listed in
  `macstudio_census_trimmed.sha256` (06m's checkpoint is `ea5bc3ab…`, as committed). The registry
  still lists all 69 runs. **Kept whole:** 07's census and references run directories.
- **Working directories deleted:** `var/levin-phase-floor-{01,03,04,05,05a,06,06a,06b,06c,06e,06f,06g,06h,06j,06k,06l,06m,floor-breakdown,07-leaves}/`,
  40,290 files (18 of them symlinks to stores), listed in `macstudio_working_dirs_deleted.sha256`.
  **Kept:** A3 v2's scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, with its
  `copy_files_*.txt` and `store_files_*.txt` certification lists, for 08. After the deletion its five
  files still equal `copy_files_certified.txt`. Also kept: `var/levin-phase-floor-07/`.
- **Worktrees removed:** `SecondaryGWKit-lpf07-91472db`, `-7460035` and `-e182bc7`, all clean
  (`macstudio_worktrees_removed.txt`). The main checkout stays at `28ebaf07`.

## What this means for older records

The `KEPT_ON_MAC_STUDIO` lists committed by 01–06m name files that no longer exist. They are left as
written, because they record what was kept when each prompt landed. This note and its SHA-256 lists
are where those files went. Under `var/levin-phase-floor-07/rehearsal/` on the Mac Studio, the
symlinks to the R27 and R28 shards now dangle.
