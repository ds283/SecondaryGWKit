# levin-phase-floor 07d, Part A′ — the measuring agent's part of VERIFICATION.md §7d (section 9.2)

Mac Studio (mps026298), Python 3.13.16. The worktree is `SecondaryGWKit-lpf07d-c2c57a3`, at the amendment SHA
`c2c57a36`, whose code is 07c's (`67b5af74`; `AdaptiveLevin/levin_quadrature.py` `d141a80b…`). Every command's output is
in `out/cmd/`. These measurements were taken on this tree. A later re-run adds a subsection and leaves this one as written.

## 7d.A′.1 The tree and the gate

- **The gate held at both ends.** `out/cmd/01_gate.txt` (02:13Z) and `out/cmd/99_gate_end.txt` (05:29Z) each read:
  - worktree HEAD `c2c57a36`, status `--short --ignored` empty;
  - `git diff --stat 67b5af74 HEAD -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` empty;
  - `levin_quadrature.py` `d141a80b…`, equal;
  - main checkout at `28ebaf07`, status empty;
  - Part A's 246 kept files re-hashed, 0 differ;
  - no other `var/levin-phase-floor-07d/*` entry;
  - no run running or stale.

  The two bodies are identical except the registry's run count, 103 → 107: this Part's four runs.
- **No Ray started, no lock was taken, and no store was opened.** Every replay records `ray_initialized` false, and
  the replay script exits if Ray is initialised.

## 7d.A′.2 The instrument and the reproduction

- **I6 (pass-through records only) equals Part A's tree replay** on 4,220 of 4,220 calls: value, abserr, regions, depth,
  converged, the refusal and chase counts, and every leaf. Run `levin-phase-floor-07d-a2-instrument-20261123T022323`.
- **P0** (every V93-family edit except the Clenshaw–Curtis one) **equals the tree** on 211 of 211 calls
  (`out/cmd/08_p0_check.txt`).
- **The reproduction of `_check_leaf`** (`scripts/checkrepro_a2.py`) **equals I6 bit for bit on 488 of 488 checked
  leaves**, 338 group and 150 analytic, drawn by `state/repro_selection_rule.txt` (`out/repro/verify_repro_a2.json`).
  - The first comparison differed on the `rule` field only, on 80 leaves (log deviation 7). Its output is kept.

## 7d.A′.3 The leaf run and the references

- **The leaf run:** `levin-phase-floor-07d-a2-leaves-20261123T023302`, 1,803 specs, 0 errors.
- **v4 against Part A:**
  - Part A's CC₄₇ is reproduced on 2,873 of 2,873 leaves, and its σ̂ and s_own likewise;
  - Part A's declarations D on 10,542 of 10,542;
  - v3's verdicts re-derived with 0 mismatches.
- **CC₉₃ by `np.dot`:**
  - It equals V93's own solve on 125 of 125 (`out/repro/check_cc93_a2.json`), and on 3,642 of 3,642 in
    `out/repro/decl93_V93.json`.
  - The largest |dot − fsum|/|CC₉₃| is 4.89 × 10⁻¹³.
- **The second reference** (48-point Gauss–Legendre, `out/repro/second_ref_a2.json`), on the 84 leaves where a table 2
  or 4 declaration understates:
  - agrees with v4 within eps_v4 + eps2 on 77;
  - disagrees on 7, by 1.11–2.29 eps_v4.

## 7d.A′.4 The variants

- **Each variant was specified in `state/variants_a2.json` before it was built.** Its SHA-256 and diff are in
  `out/variants/`. Each replay records that the module it ran is that file at that SHA.
- **Runs:**
  - batch 1 (`a2-variants1-20261123T030234`): seven variants, 4,220 of 4,220 written each, exit 0;
  - KT (`a2-variants2-20261123T045048`): 4,220 of 4,220, exit 0.
- **KT's threshold** (8π) was chosen by `state/kt_rule_a2.md`. The rule was written at 03:25Z, before batch 1's results
  were read, and the choice was appended after table 7 was read.
- **At most six worker processes ran during every registered run,** with one lapse of about two seconds, logged as
  deviation 13.

## 7d.A′.5 Records

- The worktree's `git --no-optional-locks status --short --ignored` is empty at the end, and the main checkout's
  `status --short` is empty.
- `KEPT_ON_MAC_STUDIO.txt` was written last.
