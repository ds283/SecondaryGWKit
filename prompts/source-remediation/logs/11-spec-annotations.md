# Log 11 — Record in the specs what the audit and this campaign settled (audit §6)

**Prompt:** prompts/source-remediation/11-spec-annotations.md
**Commit:** *(SHA intentionally not embedded, following the precedent of log 01 deviation 4 — the log
and board are inside the commit they would name)* — "Record the spec-code audit's three recommended
annotations", the single commit that adds this log
**Model:** Claude Sonnet 5
**Date:** 2026-09-09
**Result:** COMPLETE

## What shipped

Three docs-only annotations plus a disposition table, each marked **"Audit note (2026-09), not
author sign-off"** as the prompt requires, added to the existing §0/head blocks — no `R##` formula,
`-B.md` duplicate, or `REVIEW-QUEUE.md` touched.

1. **`docs/spec/02-greens-function.md`, §0.2** — new bullet after item 2.8, before "Tier 3, $k$ vs
   $k_{\rm phys}$": records that the code keeps $\omega_{\rm eff}(z)$ unstarred in the phase-shifted
   amplitude (`GkWKBIntegration.py:458-459`, the R34/R35/R46 form), that the audit's GK-4
   (`docs/spec-code-audit/GK-report.md`, script `GK_04_wkb_matching.py`) showed the starred R37 form
   would drift the reconstruction by $(\omega^*/\omega(z))^{1/2}\approx4$ over the test range against
   $3.8\times10^{-12}$ agreement unstarred, and recommends closing Q9 with "the stars on `NUM` 11 p.5
   are a slip". R37's own transcription is untouched.
2. **`docs/spec/04-source-integral.md`, §0.2** — new bullet after the "Tier 3, $w$ vs $w_0$" item:
   records the code's LG convention $J_\nu=m\sin\vartheta$, $Y_\nu=-m\cos\vartheta$
   (`LiouvilleGreen/bessel_phase.py`), i.e. `NUM` 07's R22 with $\gamma=\Theta-\pi/2$ relative to
   `NUM` 06 p.10's $\cos\gamma$ (spec 04 Q7), self-consistent and reproducing R14 to
   $10^{-8}$–$10^{-6}$ relative on seven configurations. Cites `docs/spec-code-audit/QI-report.md` §1
   row R17 and §3 note 6.
3. **`docs/spec/01-transfer-function.md`, Tier 3 author notes (head block)** — one sentence appended
   to the "$c_s^2=w(z)$" bullet: the literal R16 denominator (with $\Omega_{cc}$) had in fact been
   transcribed into `LambdaCDM_GenericEOS.wPerturbations`, which is audit finding A1, fixed in commit
   `0f50782` ("Exclude Lambda from the GenericEOS perturbation sound speed"); the sign-off's
   "$\Lambda$ unperturbed" reading is the binding one, and `LambdaCDM.wPerturbations` had it right
   throughout.
4. **`docs/spec-code-audit-2026-09.md`** — new "§8 Disposition (2026-09)" section at the end: a table
   mapping every A1–A7, B1–B11 finding to the commit SHA(s) and subject line(s) that discharged it
   (recovered from `git log --oneline` by subject, per the campaign's SHA-omission convention — see
   below), rows for §4.1–§4.4 (not yet run; prompt 12) and §6 (this commit), and a closing pointer to
   `prompts/source-remediation/IMPLEMENTATION_STATE.md` and the out-of-scope list. No finding's text
   was rewritten.

All ten prompt-01–10 commit SHAs were recovered from `git log --oneline` by matching each board row's
quoted subject line against the log, then confirmed present with `git cat-file -e`:

| Prompt | SHA | Subject |
|---|---|---|
| 01 | `0f50782` | Exclude Lambda from the GenericEOS perturbation sound speed |
| 02 | `3199c7b` | Fix WKB value, policy and label hygiene slips |
| 03 | `f8c75f5` | Remove the grid-end bias in the background derivative splines |
| 04 | `154126b` | Filter QuadSourceIntegral work items to triangle-closing triples |
| 05 | `e3348e4` | Add a two-region LG representation of T_k for source consumers |
| 06 | `3df5604` | Restrict QuadSource to the region where both T_k are numeric |
| 07 | `f06f587` | Add the phase-group decomposition of the source integrand |
| 08 | `4afd531` | Partition the source time integral and Levin-integrate its phase groups |
| 09 | `ffc50ae` | Record b, an error bound and honest tolerances on QuadSourceIntegral |
| 10 | `815217b` | Supply the transfer functions to the source integral stage |

## Deviations from the prompt

### 1. The commit SHA for this commit is not embedded in the log or the board — IMPLEMENTATION CHOICE

Same reasoning as log 01 deviation 4, which this prompt's own dependency line ("needs their commit
SHAs") does not override for *this* commit: the log and `IMPLEMENTATION_STATE.md` are themselves
inside the commit that would carry the SHA, so no value written into them can be correct without a
second, undocumented amend. The board and this log identify the commit by subject; every *other*
commit's SHA (the ten this prompt cites) is a real, external, already-existing SHA and is recorded
verbatim, which is exactly what the prompt asked for.

### 2. A2 and A4 map to more than one commit in the §8 table — IMPLEMENTATION CHOICE

The prompt's §8 instruction is "a table mapping A1–A7 and B1–B11 to the commit that fixed each". The
board's own item-level table (§2) already records A2 as discharged across prompts 05/06/08 and A4
across 07/08/10 — each prompt did one genuinely distinct part (the LG representation, the grid
truncation, the phase-group algebra and its wiring) and no single one of the three commits is "the"
fix. Rather than pick one commit and misrepresent the other two as uninvolved, the disposition table
lists all commits that contributed to each of these two findings, in the same order the board uses.
Every other finding maps to exactly one commit.

### 3. B9 is annotated "evaluated, left as-is" in the §8 table rather than left bare — IMPLEMENTATION CHOICE

The board (§2, item B9) records B9 as "✅ (left as-is, commented)" rather than fixed outright — prompt
02 evaluated it (its `Levin_z` chunking rationale) and documented a reason to leave the code as it is,
per README §7's explicit allowance ("Audit B9 is *evaluated* in prompt 02 but may legitimately be left
as-is with a recorded reason"). Listing it identically to the seven genuine one-line fixes in the same
commit would misstate what happened, so the table carries a parenthetical matching the board's own
wording.

### 4. No STRUCTURALLY REQUIRED or UNINTENDED DRIFT deviations

The prompt's shape matched the repository exactly: all three spec files have the §0.2 (or head-block
Tier 3) structure the prompt assumed, all cited commits exist with the subjects the board records, and
all cited file paths and line numbers (`GkWKBIntegration.py:458-459`, `bessel_phase.py`,
`GK-report.md`, `QI-report.md`) resolve as stated.

## Verification performed

**Ran, passed.** For every one of the ten cited SHAs: `git cat-file -e <sha>` (exit 0) and
`git log -1 --format='%s' <sha>` matched the board's quoted subject exactly — output reproduced in
"What shipped" above.

**Ran, passed.** File-existence checks for every path cited in the three spec annotations:
`ComputeTargets/GkWKBIntegration.py`, `LiouvilleGreen/bessel_phase.py`,
`docs/spec-code-audit/GK-report.md`, `docs/spec-code-audit/QI-report.md` all exist; `sed -n
'455,462p' ComputeTargets/GkWKBIntegration.py` confirms `norm_factor = sqrt(H_ratio / omega)` at the
cited lines (unstarred $\omega$).

**Ran, passed.** `$$`-pair parity check (`grep -o '\$\$' <file> | wc -l`, must be even) on all four
edited files: spec 01 → 94, spec 02 → 138, spec 04 → 74, the audit document → 2. All even.

**Ran, passed.** `git diff --stat` and the full `git diff` for the four edited files: every hunk in
the three spec files falls inside their existing `## 0` head/§0.2 blocks (no `## 1`+ section or `R##`
display touched); the audit-document diff is a pure append after the existing "Reproduction" §7,
adding only the new "## 8. Disposition" section.

**Reasoned, not run.** The GK-4 and QI §3 note 6 numbers quoted in the two spec bullets
($3.8\times10^{-12}$, $(\omega^*/\omega(z))^{1/2}\approx4$, $10^{-8}$–$10^{-6}$) were not re-derived;
they are transcribed verbatim from `docs/spec-code-audit/GK-report.md` (GK-4 section, lines 167–190)
and `docs/spec-code-audit/QI-report.md` (§1 row R17, §3 note 6, lines 68 and 478–481), which this
prompt instructs to be read and cited, not re-verified.

## Observations not acted on

- **`docs/spec-code-audit/scripts/TK_05_background_derivatives.py` section (d) still prints the
  pre-A1 "discrepancy"** (log 01 observation 1) — an audit artefact, correct as a record, but
  potentially misleading to a reader who does not know the fix landed. Left alone: the prompt's remit
  is the three named spec files and the audit document's new §8, not the frozen scripts (README §5
  item 8 places `extract_*.py`/scripts out of touching-scope generally, and no line of this prompt
  names the scripts directory).
- **Spec 01's Q4 (§6, "Open questions") itself is not marked closed.** The prompt names "R16 / Q4"
  but its instruction is specifically "add a sentence to the Tier 3 $c_s^2$ bullet", not to append a
  "Closed" line under Q4 in §6 the way Q2/Q7/Q10 were closed elsewhere in this same file. The Tier 3
  bullet already states the binding reading ("$\Lambda$ unperturbed... $\Lambda$ excluded"), so Q4's
  substance was resolved at sign-off, before this campaign; this commit records where the *literal*
  R16 reading leaked into code and when it was fixed, which is a narrower, code-facing fact than Q4's
  physics question. Left as the prompt specifies; a later editor may want a "Closed" line at Q4 for
  symmetry with the file's other closed questions, but that would be rewriting an author-authored
  section, which this Sonnet-run, docs-only prompt does not have standing to do.

## State handed to the next prompt

- All ten prompt commit SHAs are now recorded in one place a machine can read without re-deriving
  them from subjects: `docs/spec-code-audit-2026-09.md` §8. Prompt 12 (verification) can cite this
  table rather than re-running `git log --oneline` itself.
- The three spec files' §0/head blocks now each carry exactly one new "Audit note (2026-09), not
  author sign-off" bullet; no other campaign prompt should add a second one to the same file without
  checking these first, to avoid duplicate annotations of the same finding.
- Nothing here changes any board row's status; row 11 is set to ✅ with this commit's subject, mirroring
  the SHA-omission convention of every deviated row above it.
