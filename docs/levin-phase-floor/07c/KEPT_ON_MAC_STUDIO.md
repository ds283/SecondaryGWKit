# Files of prompt 07c kept outside this directory

These stay in the Mac Studio's main checkout, `/Users/ds283/Documents/Code/SecondaryGWKit/`.

## Part A

The copy-back list was made on the Mac Studio before anything was copied: every file under
`var/levin-phase-floor-07c/measurement/` and the six registered run directories
`var/runs/levin-phase-floor-07c-*/` except pickles and files over 2 MB. Every copied file matches the Mac
Studio's SHA-256.

**Left out by the agent's rule (468 files):** every pickle and every file over 2 MB. These are listed with SHA-256
and sizes in the agent's [`measurement/KEPT_ON_MAC_STUDIO.txt`](measurement/KEPT_ON_MAC_STUDIO.txt) (`366cad95…`).
The files left out are exactly the ones it lists. One listed file, `out/commands.log`, grew after the list was
written, as the extension appended its commands. Its first 3,177,402 bytes, the length listed, still hash to the
listed `beab0152…`.

**Left out by the orchestrator (7,234 files, 1,093,862,743 bytes):** fourteen folders of `out/replays/`. The user
asked whether the replay JSON was needed, and these folders are not:
- the other variants' plain replays: `N/`, `S2/`, `Eg/`, `Ea/`, `SEg/` and `SEa/`;
- every instrumented copy: `S_i/`, `S2_i/`, `N_i/`, `Eg_i/`, `Ea_i/`, `SEg_i/` and `SEa_i/`;
- the instrument's own replays, `instr/`.

The tables they produced are committed: `out/tables/scores_07c.{json,md}`, `out/d1/`, `out/variants/reach.{json,md}`,
and each folder's `*_summary_*.json` beside it. Each is listed with SHA-256 and size in
[`KEPT_REPLAYS_ON_MAC_STUDIO.txt`](KEPT_REPLAYS_ON_MAC_STUDIO.txt) (`efa0886f…`). All 7,234 were re-hashed on the
Mac Studio against that list on 2026-11-19, and all matched.

**Kept in the repository:**
- `out/replays/tree/`: the 454 replays at 07b's code, each reproducing its census record;
- `out/replays/S/`: the replay records of the chosen fix, which §9.5's Y8 and Part B's review compare the code with
  bit for bit.

**Also kept on the Mac Studio, from earlier prompts:** R29's, R30's and R31's stores; 07a's and 07b's census run
directories; A3 v2's scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, for 08.
