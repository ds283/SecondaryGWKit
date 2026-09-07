# Prompt 11 — Record in the specs what the audit and this campaign settled (audit §6)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §6 (three recommended spec edits); §0.2 A1 (origin in spec 01 R16/Q4)
**Depends on:** 01–10 (needs their commit SHAs). Docs only.
**Recommended model:** Sonnet
**Files you may touch:** `docs/spec/01-transfer-function.md`, `docs/spec/02-greens-function.md`,
`docs/spec/04-source-integral.md`, `docs/spec-code-audit-2026-09.md`, plus the log and the status
board. **Do not** edit any `R##` formula, any `-B.md` duplicate, or `REVIEW-QUEUE.md`.

---

## 1. Character of this commit

Annotations, not physics. Each spec already has a "§0 Author sign-off notes" block (specs 02–05)
or a head block of author notes (spec 01); add to those, in the same voice and format, so that the
next reader of the spec sees what the code settled. Mark each note **"Audit note (2026-09), not
author sign-off"** — the author has not signed these; they are the audit's findings and the user
may edit them.

## 2. The edits

1. **Spec 02, R37 and open question Q9** (`docs/spec/02-greens-function.md`). Add to §0.2: the
   code keeps $\omega_{\rm eff}(z)$ *unstarred* in the phase-shifted amplitude (R34/R35/R46 form);
   the audit (GK-4) showed the starred form of R37 would drift the reconstruction by
   $(\omega^*/\omega(z))^{1/2}\approx4$ over the test range while the unstarred form agrees with the
   exact solution to $3.8\times10^{-12}$ in radiation. Recommend Q9 be closed with "the stars on
   `NUM` 11 p.5 are a slip". Cite `docs/spec-code-audit/GK-report.md` GK-4 and its script
   `GK_04_wkb_matching.py`. Leave R37's transcription as it is (it records the page).
2. **Spec 04, open question Q7** (`docs/spec/04-source-integral.md`). Add to §0.2: the code's LG
   convention for the analytic branch is $J_\nu = m\sin\vartheta$, $Y_\nu = -m\cos\vartheta$
   (`LiouvilleGreen/bessel_phase.py`), i.e. `NUM` 07's R22 with $\gamma = \Theta - \pi/2$ relative to
   `NUM` 06 p.10's $\cos\gamma$; the code is self-consistent in it and reproduces R14 to
   $10^{-8}$–$10^{-6}$ (QI-1). Cite `QI-report.md` §1 row R17 and §3 note 6.
3. **Spec 01, R16 / Q4 and the Tier 3 $c_s^2$ note** (`docs/spec/01-transfer-function.md`). Add a
   sentence to the Tier 3 $c_s^2$ bullet: the literal R16 denominator (with $\Omega_{cc}$) had been
   transcribed into `LambdaCDM_GenericEOS.wPerturbations`, producing audit finding A1, fixed in
   commit `<SHA of prompt 01>`; the sign-off's "$\Lambda$ unperturbed" reading is the binding one
   and `LambdaCDM.wPerturbations` had it right throughout.
4. **The audit document** (`docs/spec-code-audit-2026-09.md`). Add a short "§8 Disposition
   (2026-09)" at the end: a table mapping A1–A7 and B1–B11 to the commit that fixed each (from the
   board), with "not in scope" for the one-loop-layer items and a pointer to
   `prompts/source-remediation/IMPLEMENTATION_STATE.md`. Do not rewrite any finding.

## 3. Verification

- Every SHA you cite exists (`git cat-file -e <sha>`), and the file paths you cite exist.
- The spec files still render: no broken `$$` pairs (count them), no edits outside the §0/head
  blocks (`git diff` shows changes only there).

## 4. Log and commit

Log to `logs/11-spec-annotations.md`. Board: row 11, item §6. One commit; body lists the three spec
notes in one line each.
