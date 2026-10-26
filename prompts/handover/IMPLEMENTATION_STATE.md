# Hand-over campaign — implementation state

**Last updated:** 2026-10-26 · **Status: STARTED — 5 of 10 groupings landed (00, A1, A2, A3, B1), and prompt 05,
which is outside them.** Prompt
00 landed the reconnaissance as
[`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md); **prompt
01 (A1) has landed the general-$b$ oracle** as `ComputeTargets/tests/domenech.py` with
`ComputeTargets/tests/test_domenech_oracle.py`; **prompt 02 (A2) has landed the realistic-flavour
large-$x$ harness** as [`docs/handover/realistic_large_x.py`](../../docs/handover/realistic_large_x.py)
with [`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md), and **has separated the
clamp term from the representation term** — the measurement
`[12-handover-clamp-error-in-production]` records as impossible. A3 is written and not run; B1, B2,
B3, B4, C1, C2, D1, D2, E1 and E2 are groupings only. **Seven** issues are open in §3 — four from
prompts 01 and 02, and **three opened 2026-09-23 by the A3 baseline run**, which was stopped by the
user after a read of its cost showed the shipped quadrature tolerance pair unreachable on
production's squeezed triples. An eighth, **opened 2026-10-16 by a measurement of a hard-killed run
ahead of A3 v2**, a `runregistrykit` defect drafted for RunRegistryKit, was closed on 2026-10-19 by
`rrk-v020-adoption` prompt 03 and is in §4; one is closed in §4 on another board. **Prompt 05 (2026-10-16)** is not seam work: it closed
`[00-main-and-the-readers-default-to-ray-address-auto]`, which the user assigned here on
2026-10-12 and the `runregistry-extraction` board holds, so a second issue is recorded in §4 as
closed on another board. **One more is assigned here (2026-10-19)** and held on another board:
`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`, on `rrk-v020-adoption`'s (§3 below
points at it). **Prompts 06 and 07 were written 2026-10-22** (not seam work): the A3 v2 host, the
Mac Studio, made ready and rehearsed there (R4), then the launch; 08, after the build, is held. Their
planning opened one issue, `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`. **06 landed
2026-10-22: R4 reproduced `39809dca…` on the Mac Studio**, and opened `[06-nothing-tracked-ignores-venv]`,
so §3 holds **nine**. **07 launched A3 v2 on 2026-10-22 at 21:11:54**
(`handover-07-a3-v2-20261022T211155`, estimated 39–44 h). **It ended `done` on 2026-10-24 at
21:56 (about 48 h 45 m), fingerprint `0e32f24a…`. 08 landed 2026-10-25: the store is kept and
verified on the Mac Studio, the lock released, and its depth census shows the depth-20 cliff survives
at `rtol = 1e-7`.** 08 opened `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`,
so §3 held **ten**; the review closed
`[a3-baseline-quadsource-integrals-are-1680-short]` and narrowed the other two `a3-baseline` issues
(2026-10-25), so it held **nine**; `[06-nothing-tracked-ignores-venv]` closed the same day with a
`.gitignore` line, so it holds **eight**. **09 and 10 were written 2026-10-25** for the GkWKB
failure, on the user's decision to fix both the column and the readers; their planning opened
`[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`, so §3 holds **nine**.
**An analysis of A3 v2 (2026-10-25, not a prompt)** is
[`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md). It proposes a
phase-precision floor as the cause of the depth-20 cliff, says what A3 v2 makes measurable at the
seam, recommends README §7 D1 (b), and sets out a workplan. It narrowed three §3 entries; the
count is unchanged. **09 landed 2026-10-25 on the laptop:** `compute_analytic_G`/`Gprime` return
`None` outside $1+3w>0$ and on overflow, the four readers plot `analytic_G_w` only through
`extract_common.sensible_log_points`, and a rehearsal at its tree fingerprinted `39809dca…`. It
closed `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`, narrowed the GkWKB
issue to prompt 10's re-read, and opened
`[09-two-readers-plot-the-radiation-form-under-the-w-label]` and
`[09-extract-gk-data-writes-redshift-objects-into-its-z-response-column]`, so §3 holds **ten**.
**10 ran 2026-10-25 on the Mac Studio, orchestrated there for the first time:** at 09's tree every
reader exits 0 on A3 v2, the GkWKB read-back is complete (1392 pairs, the one 08 died on among
them), and the store is byte-identical, fingerprint `0e32f24a…`. It closed
`[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`, and its review opened
`[10-the-readers-draw-their-work-lists-unseeded]` and
`[10-extract-gkwkb-data-never-closes-its-theta-figure]`, so §3 holds **eleven**. The second was
closed the same day by a direct commit, on the user's decision, so §3 holds **ten**. **`py313-venv` 01
(2026-10-25)** moved this branch to Python 3.13.16 with `ray` 2.59.0 and closed
`[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` (§4), so §3 holds **nine**.
**On 2026-10-25 the user opened [`prompts/levin-phase-floor`](../levin-phase-floor/README.md)**
from the findings' §2. Three §3 entries were assigned to it the same day, and stay here: the two
`a3-baseline` entries on the quadrature tolerance and the unconverged rows, and
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`. **11 was written the same day**: A3 v2's
seam record, read-only on the Mac Studio. The count is unchanged at **nine**. **README §7 D1 was
decided the same day: (b), overlap by construction, with the initialisation kept at the extremum;
B1 is written as prompt 12.** **11 ran the same
day on the Mac Studio, in parallel with `levin-phase-floor` 01, and passed review.** It narrowed
`[12-handover-clamp-error-in-production]` on the `source-remediation` board, and answered README
§2 (p)'s question for the shipping grid: no `fail` and no `incomplete`. It opened
`[11-seam-band-rows-deviate-from-the-oracle-by-up-to-two-orders]`, so §3 holds **ten**.
**12's code (B1, Part A) landed the same day on branch `handover-12-b1`, cut at `98e68d3`, in a
worktree on the laptop; rehearsal R8 is to come on the Mac Studio.** The consumers now switch at the
first grid node below $z_{\rm init}$ with two numeric overlap nodes, and the clamp is gone. It opened
`[12-realistic-large-x-reads-the-retired-clamp-records]`, so §3 holds **eleven**. The issues
B1 closes move only after R8.
**12's rehearsal R8 ran the same day on the Mac Studio, and B1's records commit lands it.** R8 ended
`done`, fingerprint `1b332ffa…`, which differs from R3's in `QuadSource` only. Every
`QuadSourceIntegral` row carries `partition.handover` and none a clamp. The review passed 7 of 7,
check 3 on the user's ruling: `QuadSourceIntegral`'s fingerprint cannot see its values, so a
`QuadSource`-only difference is what B1 should show. The ruling opened
`[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]`, so §3 holds **twelve**. On the
`source-remediation` board B1 closed `[08-handover-clamp-error]` and
`[12-handover-clamp-error-in-production]`, and narrowed `[05-numeric-region-is-now-the-accuracy-floor]`
and `[06-source-spline-residual-vs-handover]`. R8's fingerprint is the rehearsal reference from here
on; its store was retired on 2026-10-25 at 21:33 GMT with `store retire`, after `store create` gave it a sidecar and `store fingerprint --write` recorded `1b332ffa…`; the dry run matched that fingerprint and found only R8's own run naming the store. Its five files went, and its sidecar stays as the `retired` record ([`b1/r8/out/retire_r8.txt`](../../docs/handover/b1/r8/out/retire_r8.txt)).
**A3 and A4 were refreshed the same day** (prompts 03 and 04, written in September and never run),
on the user's order of 2026-10-25: A3 on the Mac Studio, read-only on A3 v2 at `HEAD`; A4 on the
laptop, in a worktree at `98e68d3`; both in parallel with `levin-phase-floor` 04. A3 now has an
orchestrator note, and A4 a row on this board for the first time. The refresh records on
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` that A4 is its only planned answer; no
issue is opened or closed, and §3 still holds **twelve**.
**A3 landed the same day** ([`POLICY-GEOMETRY.md`](../../docs/handover/POLICY-GEOMETRY.md)): on
A3 v2's 72 `mixed` rows, `crossover_z` has 6–664 grid intervals of numeric clearance and 1–82 of
WKB clearance, and only the one `minimal` row has fewer than two on a side. It answers README §7
**D5**. The review passed 7 of 7, check 5 on the user's ruling: the §5.5 shape difference is the old
run's response range (`z_end = 10^7`, 66 response redshifts), which is recorded and which the
prompt's §2.2 omitted. A3 opened the three README §2 (o)/(p) issues and two of its own, so §3 holds
**seventeen**.

**Campaign:** [`README.md`](README.md) ·
**Background:** [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) ·
[`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.1

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit** —
> the row added, moved or deleted, and the count and date in its header corrected. An issue owned
> by another board is moved to **that** board's §4 and its row deleted from the index. The index is
> an index: one line per issue, pointing here. Where the two disagree, this board is right.
> See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 00 | [Domènech reconnaissance](00-domenech-reconnaissance.md) | input to **A1** | Fable 5.1 | ✍️ yes | ✅ 2026-09-19 | *"Read the Domenech papers and write down what the kernel says"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | **none, by its own §8** — it lands no code, and its whole output is [`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md), which is the record. §11 of that document is a later addendum by Claude Opus 5. |
| A1 | [The Domènech general-$b$ oracle](01-domenech-general-b-oracle.md) | **A1** | Opus 5 | ✍️ yes | ✅ 2026-09-20 | *"Land the Domenech general-b oracle with its tests"* | [`logs/01-…`](logs/01-domenech-general-b-oracle.md) |
| A2 | [The realistic-flavour large-$x$ harness](02-realistic-flavour-large-x-harness.md) | **A2** | Opus 5 | ✍️ yes | ✅ 2026-09-20 | *"Separate the hand-over clamp from the representation floor"* | [`logs/02-…`](logs/02-realistic-flavour-large-x-harness.md) |
| A3 | [The policy-geometry census](03-policy-geometry-census.md) | **A3** — refreshed 2026-10-25: on A3 v2 on the Mac Studio, read-only and Ray-free, at `HEAD`, orchestrated there as 10 and 11 ([`orchestrator/prompt-03.md`](orchestrator/prompt-03.md)); reconciled with 11's policy count | Sonnet, on the Mac Studio | ✍️ 2026-09-19; refreshed 2026-10-25 | ✅ 2026-10-25 on the Mac Studio at `c8ca155`, in parallel with `levin-phase-floor` 04; reviewed there, 7 / 7, check 5 on the user's ruling; three rows recounted from the laptop | records in the bookkeeping commit | [`logs/03-…`](logs/03-policy-geometry-census.md) |
| A4 | [The $G_k$ phase decision test](04-gk-phase-decision-test.md) | **A4** — refreshed 2026-10-25: on the laptop in a worktree at `98e68d3` (its harness raises at `HEAD` since B1); now the only planned answer to `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` | Opus | ✍️ 2026-09-21; refreshed 2026-10-25 | ⬜ not run; dispatchable, in parallel with A3 | — | — |
| B1 | [Remove the $T_k$ gap, by overlap](12-remove-the-tk-gap.md) (prompt 12) | **B1** — README §7 D1 (b), decided 2026-10-25; the code on the laptop, rehearsal R8 on the Mac Studio | Opus 5.5 | ✍️ 2026-10-25 | ✅ 2026-10-25: code reviewed 8/8 in its worktree; R8 on the Mac Studio reviewed 7/7, check 3 on the user's ruling | `049c7c8` (code, rebased from `73f4c4d`); records: `ad23a29` | [`logs/12-…`](logs/12-remove-the-tk-gap.md) |
| B2 | Score the unmasked residue | **B2** | — | ⬜ no | ⬜ | — | — |
| B3 | Give the seam a contract | **B3** | — | ⬜ no | ⬜ | — | — |
| B4 | Make `incomplete` unreachable | **B4** | — | ⬜ no | ⬜ | — | — |
| C1 | The stop point as an identity | **C1** | — | ⬜ no | ⬜ | — | — |
| C2 | The policy vocabulary | **C2** | — | ⬜ no | ⬜ | — | — |
| D1 | The $G_k$ consumer spline | **D1** | — | ⬜ no | ⬜ | — | — |
| D2 | The phase as carried producer→consumer | **D2** | — | ⬜ no | ⬜ | — | — |
| E1 | Re-take the trade-off on the current grid | **E1** | — | ⬜ no | ⬜ | — | — |
| E2 | Choose the depth | **E2** | — | ⬜ no | ⬜ | — | — |
| 05 | [The Ray address default](05-ray-address-defaults-to-local.md) | **05** — not seam work; closes `[00-main-and-the-readers-default-to-ray-address-auto]` | Opus 5.5 | ✍️ yes | ✅ 2026-10-16 | *"Start a fresh Ray instance by default in main.py and the readers"* | [`logs/05-…`](logs/05-ray-address-defaults-to-local.md) |
| 06 | [The A3 v2 host and R4](06-a3-v2-host-and-rehearsal.md) | **06** — not seam work; the Mac Studio's checkout and venv, and rehearsal R4 there | Opus 5.5, on the Mac Studio | ✍️ 2026-10-22 | ✅ 2026-10-22, reviewed | `319b14f` | [`logs/06-…`](logs/06-a3-v2-host-and-rehearsal.md) |
| 07 | [Launch A3 v2](07-launch-a3-v2.md) | **07** — not seam work; the A3 v2 launch on the Mac Studio | Opus 5.5, on the Mac Studio | ✍️ 2026-10-22 | ✅ 2026-10-22, launched and reviewed; the build runs | `a6902b1` | [`logs/07-…`](logs/07-launch-a3-v2.md) |
| 08 | [After the A3 v2 build](08-after-the-a3-v2-build.md) | **08** — the store's sidecar and fingerprint, the after-build checks, the readers, the depth census against the A3 baseline, the lock's release | Opus 5.5, on the Mac Studio | ✍️ 2026-10-25 | ✅ 2026-10-25, reviewed; closed on a §8 stop by the user's ruling | `9ef8a19` | [`logs/08-…`](logs/08-after-the-a3-v2-build.md) |
| 09 | [`analytic_G_w` outside its domain](09-analytic-g-w-outside-its-domain.md) | **09** — `NULL` outside $1+3w>0$ and on overflow; readers plot only sensible points; the `extract_Gk_data` CSV columns | Opus 5.5 | ✍️ 2026-10-25 | ✅ 2026-10-25, reviewed (orchestrator's baselines at `4a3e848` and re-run at `1c19998` equal the agent's; fingerprint `39809dca…` recomputed) | `1c19998` | [`logs/09-…`](logs/09-analytic-g-w-outside-its-domain.md) |
| 10 | [Re-read A3 v2](10-reread-a3-v2.md) | **10** — the fixed readers on A3 v2, on the Mac Studio | Opus 5.5, on the Mac Studio | ✍️ 2026-10-25 | ✅ 2026-10-25, reviewed on the Mac Studio by a remote orchestrator; check 9 (link form) failed and was corrected on committing, by the user's decision | `423aefd` | [`logs/10-…`](logs/10-reread-a3-v2.md) |
| 11 | [A3 v2's seam record](11-a3-v2-seam-record.md) | **11** — the policy summary block against the table; the clamp-gap census on all 9280 rows; the gap's effect against `analytic_rad` in the seam band | Sonnet, on the Mac Studio | ✍️ 2026-10-25 | ✅ 2026-10-25, reviewed on the Mac Studio by a remote orchestrator (7 / 7 checks), in parallel with `levin-phase-floor` 01; the table count re-checked from the laptop | records in the bookkeeping commit (it commits nothing on the Mac Studio) | [`logs/11-…`](logs/11-a3-v2-seam-record.md) |

`README.md` §3 groups B into four (B1, B2, B3, B4) where the orchestration prompt's §8 listed
three; §4 item 5 of that README is the argument for splitting the contract (B3) from the invariant
(B4), and both rows are carried here.

**This table is the status, not `README.md` §3.** That README's §3 rows and its "Status" blockquote
carry the annotations they were written with — A1 is still described there as *"written 2026-09-19,
not run"* — and they are the record of what was **asked**. They are not maintained per prompt; this
board is (campaign README §5 rule 4). Where the two disagree, this one is right.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| 00 | **INPUT** | Read both Domènech papers from the committed LaTeX and say what the general-$b$ kernel is, which convention every symbol is in, where the branch cuts are, what the two papers disagree about, and what a correct implementation must reproduce. Lands no code. | 00 | ✅ **Done, 2026-09-19.** [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md). Found **two sign errors in the published review** — (4.10) `eq:Isimple` prints $\big(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J\big)$ where its own (4.7) and (4.9) give the opposite order, and (4.12) `eq:Isimple2`'s $\cos$ term is signed against its own $\sin$ terms — resolved both of README §2 (i2)'s apparent inter-paper discrepancies ($I_{2020} = 2c^2I_{\rm rev}$; the asymmetric factor of 2 is $\Gamma[3]$ and is correct), confirmed §2 (i3)'s resonance structure, and **derived** $N(b) = -\frac{(3+2b)^2}{2(2+b)^2}$ rather than leaving it predicted. §11 was added afterwards from the arXiv version history: the review's (4.7) was corrected between v1 and v2 and **the correction was not propagated** to (4.10) or (4.12). Two defects in the document itself are open in §3 below, both found by A1. |
| A1 | **FACILITY** | The general-$b$ oracle in the tree beside `kohri_terada.py`: the **corrected** (4.10) with the $x\to\infty$ coefficients (3.3)/(3.4) substituted and the outer Bessels kept exact; the doubly asymptotic (4.12) with its $\cos$ sign flipped; a quadrature of the finite-$x$ (4.11); $A$, $B$, $C$ separately callable; and a head helper. Tied to the nine $b = 0.2$ fixture cases and bridged to Kohri & Terada at $b = 0$. | 01 | ✅ **Done, 2026-09-20.** `ComputeTargets/tests/domenech.py` (782 lines) with **22 test methods** in `test_domenech_oracle.py`, covering prompt §3's eight groups plus two acceptance tests. **$N(b)$ is now measured:** $-1.194214876033$ on all nine $b = 0.2$ cases, spread **5.285e-13** (4.4e-13 relative), worst deviation from the derived $-\frac{(3+2b)^2}{2(2+b)^2}$ **4.130e-13**, each inside a per-case bound built from the quadrature's declared error (1.7e-14 to 6.9e-12) and the pipeline's own (1.0e-12 to 1.1e-11); against $I_{\rm rev}$, $\lvert N+1\rvert \le$ **3.461e-13**. The $b = 0$ tie to `kohri_terada` holds at **$\le4.0\times10^{-15}$ except at $u = 0.01$, where eq. (22)'s own rounding floors it at 3.9e-10**. Recon §2.4's, §4.1's and §4.2's tables are reproduced to the digit. **The $y>1$ region is refused, not continued** (recon §6.2's rule is inferred, not read — recon §10 item 2); `I_quadrature` is defined there and ties to $\tfrac98$ eq. (22) at 2.18e-15 on `T-first`. All four of prompt §3's deliberate breakages were applied and caught by named tests (log, "The deliberate-breakage record"); breakage 4 exposed a missing anchor in the two Wronskian tests, which was added. `ComputeTargets` 530 → **552** (+22, exactly the methods added), `CosmologyModels` **39**, `LiouvilleGreen` **148** (skipped=1); no existing file in the diff. |
| A2 | **FACILITY** | Extend `docs/radiation-oracle/large_x.py` from the exact flavour to the realistic one, with and without `drop_first_WKB_sample`, at $x_{\rm resp}$ to $10^7$–$10^8$. The single measurement that separates the clamp term from the phase re-spline term. | 02 | ✅ **Done, 2026-09-20.** `docs/handover/realistic_large_x.py` (908 lines) and [`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md); **60 cells in 10,772 s**, no Ray, no datastore, checkpointed per cell. **The control reproduces KT §8 Table 8.1 on all sixteen rows and ten columns, digit for digit**, independently re-checked against a separately-taken `large_x.py` run. **The two terms are separated and the $2\times2$ is additive:** clamp term **3.91e-03 – 9.12e-01**, representation term **4.08e-05 – 2.68e-03**, ratio **11.6 – 1034**, interaction $\le$ **2.82e-05** of the clamp term and $\le$ **1.03e-02** of the representation term; every term exceeds its own error bar by $\ge$ **3.5e+03**. **Both of the prompt's $x$-scaling guesses are refuted by measurement:** the clamp term is $x$-independent (slopes −0.07/+0.03/+0.03) because the fixture pins the hand-over at $x_T = 19.1$, giving the campaign README's 0.44 rad held phase at every rung; and the representation term is *also* flat (−0.06/−0.15/−0.12) rather than growing like $h^4x/384$, which over-predicts by $10^5$ in trend — `TkSourceFunctions` no longer re-splines the growing phase, so `GkTk-remedial` prompt 10's fix is confirmed end-to-end in `total`. `drop_first_WKB_sample` proved **inert in the exact flavour** (bit-identical totals, Table 6), so the exact half uses a matched `WKB_region` truncation; the two clamp terms then agree to five significant figures. Gap opened: **1.00 fixture grid step = 1.00 source-grid step = 1.92× production's median, 1.05× its maximum**. Set-up cost is **flat** (0.02–0.10 s, ratio 1.1–2.0), refuting prompt §3 item 1; the **integral** is the binding cost (33–3239×), growing 1.7×/3.6×/**8.6×** per decade by shape, so `q-smooth` realistic stops at $x_{\rm resp} = 10^5$ — prompt §7's second stop condition met by measurement. Suites unchanged. Two issues opened in §3. **Note added 2026-09-22 by `prompts/run-registry` prompt 02: `realistic_large_x.py` predates the run registry by design and is deliberately not adopted into it.** It gates checkpoint reuse on a SHA-256 of its own source, so **editing it by one character discards all sixty cells** in `var/runs/realistic_large_x_cells.jsonl` and costs a three-hour recomputation; that hash is what proved, in four seconds, that the committed script produced the published numbers. Its value as a fixed artefact exceeds its value as a registry client. Anyone proposing to touch it should read `prompts/run-registry/IMPLEMENTATION_STATE.md` `[02-realistic-large-x-is-outside-the-registry]` first. |
| A3 | **MEASUREMENT** | Read-only census over stored `GkSourcePolicyData` rows: how many **source-grid intervals** `crossover_z` has on each side, against the stored `quality` band, plus the type/quality census and the `fail` rows. Needs a datastore. Answers README §7 **D5**. | 03 | ✅ **Done, 2026-10-25**, at `c8ca155` on the Mac Studio, read-only and Ray-free; the store byte-identical, fingerprint `0e32f24a…`. **D5's answer:** on the 72 `mixed` rows, `crossover_z` has 6 / 308 / 481 / 664 / 664 numeric and 1 / 82 / 82 / 82 / 82 WKB intervals (min / 5th / median / 95th / max), by node index. Only the `minimal` row (shard 2, serial 460) has fewer than two on a side (6 and 1). **On all 71 `mixed`/`complete` rows `crossover_z` is one node, $3.121867\times10^{15}$, super-horizon at every $k$**, and `primary_WKB_largest_z` is the grid top, so the overlap is 8.8–17.0 e-folds wide: the 82 is `CLEARANCE_GOOD` = 0.05 measured in intervals at the top of this grid (`[03-the-g-crossover-is-one-super-horizon-node-on-every-complete-row]`). The half-e-fold band of README §2 (q) carries both representations on all 72 rows, so the lever map is not falsified. 11's policy count is reproduced in every cell. **The §5.5 shape:** restricted to $10^5$–$10^7$/Mpc, `mixed` is 6.34 % against 13.42 %, the same per $k$ (9.2 against 8.86) over 145 response redshifts per $k$ against 66. The review stopped at check 5 because that is not one of §2.2's three differences. **The user ruled (2026-10-25)** that it is a fourth known one, recorded at `docs/source-remediation-verification.md:224-238` (`--zend 1e7`, 66 response redshifts), which §2.2 omitted. §5.5's 7 `fail` rows and their absence here are unexplained. Record: [`POLICY-GEOMETRY.md`](../../docs/handover/POLICY-GEOMETRY.md), [`docs/handover/a3-v2/policy-geometry/`](../../docs/handover/a3-v2/policy-geometry/). ⬜ *Before it ran:* **Written, not run; refreshed 2026-10-25.** Re-pointed from the A3 baseline store to A3 v2 (`0e32f24a…`) on the Mac Studio, read-only and Ray-free (`sqlite3` and `json` only, in the pattern of 11's `seam_census.py`), at `HEAD`: B1 changed no `GkSourcePolicyData` or `GkSource` file (`git diff 98e68d3 HEAD --stat`). Reconciled with 11, which already counted the policy table (363 / 71 + 1 / 725, no `fail`, no `incomplete`): A3 does not re-take that count but checks it, compares its shape with verification §5.5 (now restricted to $10^5$–$10^7$/Mpc as well, since the full-store shape has shifted), tabulates the one `minimal` row, and answers README §2 (q)'s $z_{\rm response}$-versus-$k$ question by a stated proxy, because there is no `incomplete` row to cluster. Opens all three README §2 (o)/(p) issues. Records under `var/handover-03/`, committed on the laptop. |
| A4 | **MEASUREMENT** | Decide whether A2's fixture, whose $G$ phase is the raw `phase_spline` production abandoned at `GkTk-remedial` 09, changes A2's representation term or its Levin cost growth materially. Six of A2's cells with a `PrimitivePhase` $G$, against a threshold the prompt fixes. A decision test, not a re-take. | 04 | ⬜ **Written 2026-09-21, not run; refreshed 2026-10-25.** Runs on the laptop in a worktree at `98e68d3`: `realistic_large_x.py` raises at `HEAD` since B1 (`[12-realistic-large-x-reads-the-retired-clamp-records]`), and A4 judges A2's pre-B1 numbers. It is now the only planned answer to `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`, since `levin-phase-floor` 02 was superseded and its N2 not tested. Every cell records `max_depth`, regions per group and phase-limited regions, as that issue's narrowings ask, beside the verdict and not in it; two reproduction cells measure the venv's drift since A2. The threshold for "material" is unchanged. |
| B1 | **REMEDY** | Remove the $T_k$ seam gap by one of README §7 **D1**'s three recorded constructions. Not a tuning decision: holding a phase constant across up to 0.44 rad is wrong under any objective. | 12 | ✅ **Done, 2026-10-25: the code `049c7c8`, R8 in B1's records commit.** **R8** (`handover-12-r8-reference-20261025T210551`, on the Mac Studio at `8a02f0c`) ended `done`, exit 0, in about 3.5 min. Its fingerprint **`1b332ffa…`** differs from R3's `39809dca…` in `QuadSource` only: each of its six rows gained 3 nodes (371, 371, 371, 428, 428, 478 against R4's 368, 368, 368, 425, 425, 475), with the values at R3's nodes bit-identical. All 864 `QuadSourceIntegral` rows carry `partition.handover`, and none a clamp key. The continuity residuals, recorded and not judged: at the switch node $|\Delta T_{\rm rel}|$ 1.7e-7–3.2e-7 and $|\Delta T'_{\rm rel}|$ 4.2e-6–5.7e-6, growing to 1.5e-5–1.6e-5 and 3.0e-5–3.1e-5 at the second overlap node. The six readers exit 0 on a byte-identical store; the lock was taken and released. **Check 3, on the user's ruling:** a `QuadSource`-only difference meets prompt §6.4 and §7.3, because `QuadSourceIntegral`'s `inventory_spec` records identity and tags only. The prompt and the orchestrator note expected both classes to differ, which was their misreading. The ruling opened `[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]` (§3). **R8's fingerprint is the rehearsal reference from here on**, kept by its run directory on the Mac Studio; its store was retired on 2026-10-25 after the records commit (`docs/handover/b1/r8/out/retire_r8.txt`). On the `source-remediation` board B1 closed `[08-handover-clamp-error]` and `[12-handover-clamp-error-in-production]`, and narrowed `[05-numeric-region-is-now-the-accuracy-floor]` (its $T_k$ hand-over part) and `[06-source-spline-residual-vs-handover]`. Record: [`HANDOVER-B1.md`](../../docs/handover/HANDOVER-B1.md) §5, [`docs/handover/b1/r8/`](../../docs/handover/b1/r8/). **The code** landed 2026-10-25 as `049c7c8`, rebased onto `handover-remedial` from `handover-12-b1` once `levin-phase-floor` 02 was superseded; its review passed all eight checks of `orchestrator/prompt-12.md` §A.3. `ComputeTargets/handover.py` holds `HANDOVER_OVERLAP_NODES = 2` and `handover_switch_node`; `TkSourceFunctions.crossover_z` is the switch node, `z_init` is new, `handover_continuity` records both representations at the switch and overlap nodes; `QuadSource` stores the overlap; `QuadSourceIntegral` has no clamp and checks every representation strictly, and its `partition` metadata carries `handover` instead of the clamp gaps. Tests (a)–(g) pass and both breakages are caught; `ComputeTargets` 737 → 742. On the exact fixture the continuity residuals are 3e-14 in $T$ and the LG truncation (≤ 6.5e-5) in $T'$; end to end, T-first at $x_{\rm resp} = 980$ is 6.0e-05 from the reference, against A2's 3.4e-04 closed and 3.6e-03 clamped ([`HANDOVER-B1.md`](../../docs/handover/HANDOVER-B1.md)). ⬜ **Written 2026-10-25 as prompt 12**, on the user's decision **D1 = (b), the Liouville–Green initialisation kept at the extremum**. Writing it found that every A3 v2 `TkNumericIntegration` row already stores 118–126 numeric samples below $z_{\rm init}$; the gap is made by `TkSourceFunctions` and `QuadSource` discarding them (README §2 (a), corrected). So B1 changes consumers only: switch at the first node below $z_{\rm init}$, keep two numeric overlap nodes, remove the clamp, record continuity at the shared nodes. No producer, no schema. Rehearsal R8 must differ from R3 in `QuadSource` and `QuadSourceIntegral` only. |
| B2 | **MEASUREMENT** | Re-run A2 at the same configurations with the gap gone and attribute what is left, by term. | — | ⬜ **Grouping only.** Blocked on B1 and A2 by README §4 item 2's masking argument. |
| B3 | **REMEDY** | Make `quality = "incomplete"` and `type = "fail"` load-bearing; add the missing `_classify_crossover` tests; fix the ungated dump at `main.py:2929`. No prerequisites. | — | ⬜ **Grouping only.** |
| B4 | **REMEDY** | Size the $G$ overlap by construction with L1/L2 so that `_classify_crossover` cannot fail on the grid that ships. Needs A3 (the unit is grid intervals) and README §7 **D7**. | — | ⬜ **Grouping only.** |
| C1 | **REMEDY** | Tighten `find_phase_extremum`'s `root_scalar` to a floor justified by the method; key the two WKB row types on the numeric row rather than on an absolute-`1e-7` match of a $z\sim10^{12}$ float; filter the stored initial values. | — | ⬜ **Grouping only.** |
| C2 | **REMEDY** | One spelling of `numeric_policy` across both metadata objects, and the maximise-numeric branch either reachable or deleted; fix the two warning messages that lose the offending value. Changes a datastore key, which is free. | — | ⬜ **Grouping only.** Owns the three issues README §2 (o) and (p) record, which **prompt 03 opens formally**. |
| D1 | **REMEDY** | The $G_k$ consumer spline at the hand-over: `GkSourcePolicyData.py:638-752` splines the numeric $G$ over source-grid nodes carrying up to 1.24 rad of oscillation per interval, at ×631–×37,700 the solver's error. | — | ⬜ **Grouping only.** Needs A3 first (README §4 item 1). |
| D2 | **SCOPE QUESTION** | The phase as carried producer→consumer, $\delta\theta\simeq h^4x/384$ growing linearly in $x$, plus the anchoring floor and the storage granularity beneath it. The largest scope question in the campaign — README §7 **D4**. | — | ⬜ **Grouping only.** |
| E1 | **MEASUREMENT** | Re-take `[06-source-spline-residual-vs-handover]`'s stale trade-off table on the curvature-criterion grid, at $b = 0$ **and** $b \ne 0$ — which A1 has now made scoreable. | — | ⬜ **Grouping only.** |
| E2 | **DECISION** | Scan $x_T$ against both oracles at large response $x$, at $b = 0$ and $b = 0.2$, with the gap gone and the residue attributed. Move `TkNumericIntegration.py:148-149`'s window, or record on measurement that it stays. | — | ⬜ **Grouping only.** Last, by README §4 item 3. |
| 05 | **REMEDY** | Make `main.py` and the six `extract_*.py` start their own Ray instance unless told otherwise: `--ray-address` defaults to `local`, not `auto`, which attaches to any Ray instance on the machine. The flag stays optional, because the A3 v2 launcher passes none. Assigned here by the user on 2026-10-12, to land before the A3 v2 launch or with it. | 05 | ✅ **Done, 2026-10-16.** `DEFAULT_RAY_ADDRESS` and the six readers' defaults read `local`, with their help text and the comment above each `ray.init` saying so; `ComputeTargets/tests/test_ray_address_default.py` (7 tests) reads the seven scripts with `ast` and pins Ray's own resolution of `local`. `quadsource_atol_sweep.py` unchanged (SHA-256 `2a985303…`). `ComputeTargets` 693 → **700**; the other six roots unchanged. |
| 06 | **SET-UP** | Bring the Mac Studio's checkout to this branch and give it a Python 3.12 venv equal to the laptop's; run R3's four launches there (R4) at the current launcher and `datastorekit` 0.5.0, which no rehearsal build has run. Runs on the Mac Studio; the records are committed on the laptop. | 06 | ✅ **Done, 2026-10-22. R4 holds on the Mac Studio:** both stores `39809dca…`, equal to each other, their sidecars and R3's recorded `reference` and `resumed-c` (`compare_with_recorded.py` verdict `equal`). The venv is Python 3.12.15 with a `pip freeze` identical to the laptop's (101 lines); the three packages at their pins, none editable; the seven roots pass (`ObjectFactories` 280, the tree's count, where the prompt quoted the pre-`v0.5.0` 279; `RunDrivers` 115 with one skip, a test that reads a laptop-only gitignored manifest). Four manifests: `git_dirty` `false`, `HEAD` `3152af9`, host `d120d1cf03551e6f`, launcher `b2fa4ef1…`. The Mac Studio is 2–13 % faster per stage than the laptop at `--cpus 6`. One stop-and-ask before any launch (the venv not ignored), resolved by `.git/info/exclude`; ten deviations logged, four of them shell slips under zsh. Record: [`A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §1. **R4's two stores were retired on 2026-10-25** (user's decision) with `store retire` at `b90752d`, each after a dry run in which a fresh fingerprint matched `39809dca…` and only R4's own runs named it: ten files went, the two sidecars stay as `retired` tombstones and R4's four run directories as the record; the 06 archive `var/handover-06.tar.gz` was deleted. |
| 07 | **LAUNCH** | Launch A3 v2 on the Mac Studio by the launcher's production scope at (1e-32, 1e-7), `--cpus 6`, into `var/datastores/handover-A3-v2-lambdacdm.sqlite`; one check; end. | 07 | ✅ **Launched 2026-10-22 21:11:54 BST** as `handover-07-a3-v2-20261022T211155` (driver pid 84436, the lock's owner), at `b90752d`, clean, on host `d120d1cf03551e6f`. Scope `full build: main.py's own 9280-item QuadSourceIntegral work list at atol=1e-32 rtol=1e-07, 6 cpus, LambdaCDM, 8 log-spaced k over 100000-3e+08 /Mpc`; four `replaced 1 occurrence`, `--zend 0.1`, no `--no-prune-unvalidated`, local Ray. Estimate **39–44 h** (R4's QSI stage ×0.97 of R3's). The session's permission classifier refused the first launch attempt; the user approved it. **The build is running; the Mac Studio's checkout stays at `b90752d` until 08.** Record: [`A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §2. |
| 08 | **MEASUREMENT** | After the build: the sidecar and fingerprint, the after-build checklist, the readers, the depth-cap census against the A3 baseline's, the archive. | 08 | ✅ **Done, 2026-10-25.** The sidecar's fingerprint is `0e32f24a…`, the run's; every store file byte-identical from before `store create` to after the last reader. Census `no divergence`, soundness `clean` (integrity `ok`, no foreign-key row, inventory without problem), one `version` row per shard, no journal or in-flight row. Readers: five of six and `extract_Gk_data` without `--models` exit 0; **`extract_GkWKB_data` exits 1** in a plot (`[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`), a §8 stop the user resolved: "Release the lock, close 08". **Depth census** (the baseline's method, reproduced as `depth_census.py`, tested on R4's 864 rows; re-counted by the orchestrator with `sqlite3` `mode=ro`, equal in every bin): 9280 rows at (1e-32, 1e-7); NULL 2489, depth 0 99, 1–19 4427, **20: 2265 at mean 454 s, 1179 unconverged (52.1 %)**; **286.3 CPU-hr, 285.8 at depth 20**, 249 of them in the low band; up to 966,508 regions. **`rtol = 1e-7` did not remove the depth-20 cliff.** The run took 48.74 h wall; the serial tail was ≤ 2.52 h (5.2 %); 15 `IntegrationWarning`s, none tied to a row. The lock was released 00:51:03; the Mac Studio's checkout is free. Record: [`A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §3. |
| 09 | **REMEDY** | `compute_analytic_G`/`Gprime` return `None` where $1+3w\le0$ (spec 02 R2's domain) or on overflow; one reader helper plots only finite points within three decades of $|G|$; `safe_fabs` in the GkWKB reader; `extract_Gk_data`'s CSV columns. A laptop rehearsal must stay `39809dca…`. | 09 | ✅ **Done, 2026-10-25.** `constant_w_form_applies(w)` (`1 + 3w > 0`, no band) gates both functions before any Bessel call, and a non-finite value is `None`; inside the domain every value is **bit-identical** to the old expression (tested to the bit at $w = 1/3, 0, 0.9, -1/3+10^{-3}$). `extract_common.sensible_log_points` (`ANALYTIC_PLOT_DECADES = 3`, a `signed` form for `extract_Gk_data`'s linear-$y$ plot) filters every `analytic_G_w` plot series in the four readers against that plot's $|G|$; `extract_GkWKB_data` uses `safe_fabs` for every analytic column; `extract_Gk_data`'s CSV `analytic_G_w`/`analytic_Gprime_w` hold the $w(z)$ form. 32 new tests in three modules; all six deliberate breakages caught by named tests. `ComputeTargets` 700 → **732**, the other six roots unchanged (the known wall-clock flake failed before and after). **Rehearsal R5** (`handover-09-r5-reference-20261025T094215`, `git_dirty` true at `1739e0d`): `done` in 18 m 55 s, fingerprint **`39809dca…`**, `compare_with_recorded.py` `equal` against R3's `reference`; no `NULL` in any `analytic_G_w`/`analytic_Gprime_w` row; the six readers exit 0 with the store unchanged. **R5's store was retired on 2026-10-25** (user's decision) with `store retire` at `bacbab6`, after a dry run in which a fresh fingerprint matched `39809dca…` and only R5's own run named it: the five store files went, the sidecar stays as a `retired` tombstone, and R5's run directory, its launch log and the text records in `var/handover-09/` stay as the record; the readers' output, `var/handover-09/extract/` (183 MB), was deleted. Log: [`logs/09-…`](logs/09-analytic-g-w-outside-its-domain.md). |
| 10 | **MEASUREMENT** | Every reader on A3 v2 at 09's tree, on the Mac Studio; the GkWKB read-back complete; the store unchanged. | 10 | ✅ **Done, 2026-10-25**, at `57f67a9` on the Mac Studio, whose checkout moved from `b90752d` for the first time since the launch; `raypoolkit` moved to `v0.3.0` (`67a75b97…`), the one line of `pip freeze` that differs from `pip_06.txt`. **All seven readers exit 0** (six with `--models LambdaCDM`, `extract_Gk_data` without), 10:50–10:57 GMT, 879 MB; every store file identical after each, and `store fingerprint` afterwards matches the sidecar's `0e32f24a…`. **The GkWKB read-back is complete:** 1392 full-range PDFs, 1392 PNGs and 1392 CSVs, equal to each other and to the reader's work list (8 k × 174 sampled `z_source`, its log's `Queue summary: 1392 lookup`), counted by the agent, the remote orchestrator and the laptop. The pair 08 died on, $k = 10^5$, $z_{\rm source} = 1.3633\times10^{16}$, has a 30,366 B PDF (08: a 1,039 B fragment), its PNG and its CSV; `sensible_log_points` drops 35 of its 154 `analytic_G_w` points, the largest 7.57e+243, and 5162 of 109,594 over this run's 1392 pairs. **That pair was in the sample by chance** (about 1 in 10): the readers draw unseeded (`[10-the-readers-draw-their-work-lists-unseeded]`). The lock (`v0.3.0`, at `~/.raypoolkit/rehearsal-lock`) was taken and released, `ray-processes` empty. Review check 9 failed on `verification.md`'s links (`reread/…`, copied from §3's equally broken `after/…`); corrected to `a3-v2/reread/…` on committing, by the user's decision. The note's expectation that `status` exits 3 while held with no pid was the note's misreading of RayPoolKit's layout table, not a package defect. Records: [`A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §4, [`docs/handover/a3-v2/reread/`](../../docs/handover/a3-v2/reread/). |
| 11 | **MEASUREMENT** | Findings §3.3 items 2–3 on A3 v2, read-only and Ray-free on the Mac Studio. First, the `GkSourcePolicyData` summary block from the build's `stdout.log`, checked against a count of the table. Then the measured clamp gaps (`partition.subintervals[].clamp_gaps_log1pz`; findings §3.3 named the allowance `max_clamp_gap_log1pz` instead) for `Tq`, `Tr` and the source spline on all 9280 rows, set beside the 2026-09 one-run figures. Last, $|{\rm total}/{\tt analytic\_rad}-1|$ by clamped-factor count in the seam band. The input to A3 (03) and to **D1**. | 11 | ✅ **Done, 2026-10-25**, at `757ef01` on the Mac Studio, in parallel with `levin-phase-floor` 01. Read-only and Ray-free; the store, its sidecar and `stdout.log` are byte-identical, and the fingerprint is `0e32f24a…`. **The policy:** the block agrees with the table in every cell, re-counted on the Mac Studio and again from the laptop: `numeric` 363, `mixed` 71 + 1 `minimal`, `WKB` 725, and no `fail` or `incomplete`. The `minimal` row is $k = 3.09\times10^6$, $z_{\rm response} = 3.96\times10^{10}$. **The gaps**, on 9280 rows and 19,289 sub-intervals: the clamp is on 6791 rows (every row with a Levin sub-interval). `Tq` is non-zero on 12,880 sub-intervals and `Tr` on 13,582, with median 1.15e-02 and max 2.26e-02 in $\log(1+z)$ (0.98 mean grid step). The source spline is clamped on 423. None is near the allowance. **The oracle**, over 6080 seam-band rows: unclamped median 4.98e-06, one clamp 6.6e-02, two 5.5e-03, maxima to 96.9. The strata are tied to $z_{\rm response}$, so this does not separate the clamp's error. It narrows `[12-handover-clamp-error-in-production]` and answers README §2 (p)'s shipping-grid question, and it opened one issue. Record: [`A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §5, [`docs/handover/a3-v2/seam/`](../../docs/handover/a3-v2/seam/). |

---

## 3. Active and unresolved issues

- **[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]** *(opened 2026-09-23
  by the A3 baseline run, not by a prompt)* — on production's **squeezed** triples the source
  integral's tolerance pair cannot be met, the Levin bisection runs to
  `DEFAULT_LEVIN_MAX_DEPTH = 20`, and the row is stored unconverged at **220× the cost of a row
  that stops short of the cap**. Measured over all **7600** `QuadSourceIntegral` rows in
  `var/datastores/handover-A3-baseline-lambdacdm-shard*.sqlite`, on the tree at `704a12e` (clean):

  | `WKB_Levin_max_depth` | rows | mean `compute_time` |
  |---|---|---|
  | NULL (no Levin call) | 2489 | 0.062 s |
  | 1–19 | 3054 | 0.776 s |
  | **20** | **2057** | **170.6 s** |

  The cap is a cliff, not a scaling: `max_depth` on the expensive rows is **20 exactly, with zero
  variance**. Depth-20 rows carry **97.5 of the store's 98.2 CPU-hours**, and **43.2 %** of them
  (889) are stored with `total_converged = 0`. Of the 195 rows costing more than 60 s, **74.4 %**
  are unconverged. The cost is **entirely Levin**: `WKB_Levin_elapsed` sums to **351,296 s** of the
  **353,489 s** of `compute_time`, against **9 s** of `numeric_quad_compute_time` across all 7600
  rows.

  **The mechanism.** `_adaptive_levin` accepts a region on
  `resolved = abserr < local_atol or relerr < rtol` (`AdaptiveLevin/levin_quadrature.py:1991`, and
  again at `:2179`), and `local_atol` is `atol` distributed across subregions **by length share**
  (`_local_atol`, `:1672`). These rows reach `WKB_Levin_num_regions` of up to **1,530,190**, so a
  region's share of `atol = 1e-32` is around **1e-38** and the first branch is dead. Everything
  must then clear `rtol = 1e-8` against a phase representation that delivers about six digits: the
  median `total_abserr / |total|` on the expensive rows is **2.15e-06**. `levin_quadrature.py:2327`
  describes this state in its own words — *"the cost can be two orders of magnitude higher than
  necessary"* — and the measured ratio is 220.

  **Which constant, and why the change that caused it was reasonable.** `5255ac0` (2026-09-10)
  tightened `DEFAULT_QUADRATURE_ATOL` from `1e-25` to `1e-32`, on `prompts/source-remediation`
  log 12's finding that at `1e-25` ~58 % of production work items "converged" before doing any
  work. That finding is not in dispute and the direction was right. What the change also did,
  unmeasured, is remove the only reachable branch for **this** geometry: the expensive rows have
  median `|total| = 3.31e-24`, so at `1e-25` `atol` was met immediately and at `1e-32` it cannot be
  met at any subdivision. The regime moved from `atol`-bound to `rtol`-bound, and `rtol = 1e-8` is
  unattainable here. **The defect is the pair, not the tightening**: nothing sized `rtol` for a
  phase that cannot deliver eight digits, because until `5255ac0` `atol` was hiding it.

  **Why the sweep that already exists did not see it.** `prompts/tolerance-convergence` prompt 06
  swept exactly this axis and reported `atol` **inert over twenty-eight decades**
  ([`QUADSOURCE-READONLY.md`](../../docs/tolerance-convergence/QUADSOURCE-READONLY.md) §0
  finding 1, §4). **That measurement stands and is not superseded.** It measured the *residual*, on
  the eighteen offline cases of `test_quadsource_integral.py`, and on those cases `atol` is inert.
  It could not see this for two reasons, both structural: its instrument has **no `(k, q, r)`
  triangle** — the squeezed geometry below does not occur in a single constant-$w$ configuration —
  and the statistic that moves is the **cost**, which a residual sweep does not record. Its own
  finding 4 recorded `DEFAULT_LEVIN_MAX_DEPTH = 20` and `limit = 100` as **never chosen**; this
  entry is what that costs. A different question on a different instrument, not a contradiction.

  **The geometry, exactly — three conditions, all necessary.** (i) **`r == k ≫ q`**, the hard leg
  on the response and a soft `q`: every worst triple has this shape, and the mirror `k ≪ q ≈ r`
  costs 0.3 s. (ii) **mid-range `k`**: the blow-up peaks at `k = 9.7e6`/Mpc and is **absent at both
  ends** — `k = 1e5` has zero rows over 60 s, and so does `k = 3e8`. (iii) **low `z_response`**,
  i.e. large `eta_response`: within one triple the cost grows roughly as $\eta_R^3$ and then
  flattens as the cap binds. For `(k, q, r) = (9.7e6, 9.85e5, 9.7e6)`: **15.5 s** at $z = 8343$
  ($\eta_R = 50.8$), **737 s** at $z = 1589$, **6245 s** at $z = 174$, **12,714 s** at $z = 11$.
  Binned by $k\eta_R$ the **median** cost is flat at ~1.3 s across **twelve decades** — Levin is
  doing what Levin is for — and only the mean moves. Any statistic that does not separate the tail
  will miss this.

  **Impact.** Cost, and what a stored `total` is worth. At the settings that shipped, finishing the
  baseline costs **8–10 wall-days** on six CPUs (the issue below), and 74 % of what it would add is
  unconverged. It also reaches **E1** and **B2**, which re-score against this store, and it
  narrows `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` below.

  **Measured 2026-09-23** by
  [`docs/handover/quadsource_atol_sweep.py`](../../docs/handover/quadsource_atol_sweep.py) — nine
  fixed production work items (four severe, three moderate, two controls) over eight tolerance
  pairs, through main.py's own pipeline by five exact substitutions. The zero point is the
  baseline's own rows, read back as lookups: 7237.2 s over the nine, 7 at depth 20, 5 unconverged.
  Each cell is the worst relative difference of `total` against that zero point over the nine:

  | `atol` | `rtol` | worst $\lvert\Delta\rvert/\lvert total\rvert$ | time, 9 cases | speed-up |
  |---|---|---|---|---|
  | 1e-22 | 1e-8 | 1.8e+00 | 3.0 s | — |
  | 1e-25 | 1e-8 | 1.8e+00 | 2.5 s | — |
  | 1e-28 | 1e-8 | 3.2e-01 | 5.6 s | — |
  | 1e-30 | 1e-8 | 1.1e-02 | 100.5 s | — |
  | 1e-32 | 1e-5 | 1.1e-02 | 4.9 s | ×1477 |
  | 1e-32 | 1e-6 | 1.1e-02 | 31.1 s | **×233** |
  | **1e-32** | **1e-7** | **1.9e-06** | **1060.5 s** | **×6.8** |
  | 1e-32 | 1e-8 | — (reference) | 7237.2 s | 1 |

  **`rtol` is the lever and `atol` is not**, which is the opposite of what the cost alone
  suggested. Relaxing `atol` collapses the cost and **destroys the answer**: at 1e-22 and 1e-25
  several cases return with the *wrong sign*, and 1e-28 is still 32 % out. So `5255ac0` was right
  in a stronger sense than log 12 claimed for it — 1e-25 is too loose on *this* geometry, not only
  on the grounds log 12 gave — and `atol` is emphatically **not** inert here, which is a second
  way prompt 06's finding does not transfer. At production `atol`, `rtol = 1e-7` reproduces every
  one of the nine to **1.9e-06** at **1/6.8** of the cost, and `rtol = 1e-6` gives seven digits on
  eight of nine at **1/233**, with one case (the $z = 1589$ moderate) pinned at 1.1e-02 — that
  same case pins the 1e-30 and 1e-5 rows too, so it is a property of the case and not of the
  tolerance.

  **Two qualifications ship with this table, and the first is the binding one.** (i) The
  `(1e-32, 1e-8)` reference is itself stored `total_converged = 0` on **5 of the 9** cases, so
  "agrees to 1.9e-06" means *agrees with a number the code declined to certify*, not with truth.
  This sweep has no independent oracle and `analytic_rad` cannot be one — prompt 06's finding 3
  established that it is computed at the caller's own pair and moves with it. (ii) All nine cases
  sit at $z_{\rm response} \ge 43.7$; the 1024 items at $z \le 6.32$ that
  `[a3-baseline-quadsource-integrals-are-1680-short]` records as never attempted are **unmeasured
  at every tolerance**, and their $\eta_R$ reaches 2.8× the largest sampled here. The speed-ups
  above must not be extrapolated into that block, which is where most of the cost is.

  Incidentally confirming the issue below: at `rtol = 1e-6` seven of the nine rows are still
  flagged unconverged while agreeing to seven digits. At `atol = 1e-32` the `atol` branch of
  `resolved` can never fire, so `total_converged` carries almost no information.

  **The full record, with every table and the structural correction of §6, is
  [`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md`](../../docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md)**,
  written against `050a7e3` on a clean tree. It is transcribed rather than regenerable: the sweep
  store it was measured on is disposable and is being deleted, so the numbers live in the document
  (`prompts/run-registry` README §0 item 4).

  **Phases 2, 3a and 3b, measured 2026-09-23/24**, extend that table into the two bands phase 1
  could not see. Phase 1's nine cases all sit at $z_{\rm response}$ between 43.7 and 1589, which
  is one band of three.

  | band | cases | `rtol` 1e-6 | `rtol` 1e-7 | cost at 1e-7 | what limits it |
  |---|---|---|---|---|---|
  | seam, $z_{\rm response} > z_{\rm min}$ | 9 | 1.01e-06 | 9.97e-07 | 7.7 s | floor ~1e-06 |
  | mid, $z_{\rm response}\sim10^3$ (c5) | 9 | **1.1e-02** | 1.9e-06 | 1060 s | **quadrature** |
  | low, $z_{\rm response} \le 6.32$ | 4 | 1.08e-05 | 2.84e-06 | 2598 s | floor ~1e-05 |

  (Each figure is the worst relative difference over that band's cases against its own tightest
  rung, which is 1e-8 for the seam and low bands and 1e-8 for the mid band.)

  **Corrected 2026-10-02** (`a3-v2-readiness`, the orchestrator's review of prompt 01): the seam
  row's `rtol` 1e-7 cell, 9.97e-07, is not against the tightest rung. The sweep's §4 gives
  9.97e-07 for **1e-6 against 1e-7**, and **8.26e-07** for 1e-7 against 1e-8, which is the figure
  this column means. The conclusion is unchanged. The seam row's 1e-6 cell, 1.01e-06, is not in
  §4, and was not re-derived.

  **`rtol = 1e-7` is the loosest rung converged in all three bands, and 1e-6 is ruled out by the
  mid band alone.** c5 — $(k, q, r) = (3.09\text{e}6, 1\text{e}5, 3.09\text{e}6)$ at
  $z_{\rm response} = 1589$ — is the case that decides it: there `rtol` 1e-5 and 1e-6 agree with
  *each other* to 3.6e-04 and are both **1.1e-02** from the converged value, which 1e-7 reaches in
  245 regions and 1e-8 confirms to 1.0e-06 after 24,015. Two loose rungs agreeing is not
  convergence, and c5 is the counterexample inside this sweep.

  **Phase 3b settles the low band, and the plateau there is real.** Four rungs across three
  decades agree to ~1e-05; on $k = q = r = 3.05\text{e}7$, $z = 0.1$ the 1e-7 and 1e-8 answers are
  **bit-identical** (same value, same 50,043 regions); and the 1e-7 → 1e-8 step is *smaller* than
  the 1e-6 → 1e-8 step, which is the opposite of the c5 signature. So low $z$ is limited by the
  representation and not by the quadrature, and no tolerance reaches beneath that floor — the
  lever there is **D1**/**D2**, not this constant. The region counts say the same thing: 52 → 698
  → 152,634 → 1,238,448 across the four rungs on one case, for a change of 1e-05 in the answer.

  **For science outputs, someone should re-run at `rtol = 1e-9`** (user, 2026-09-24). There is no
  oracle that could substitute: `domenech.py` and `kohri_terada.py` are both constant-$w$ and
  production is LambdaCDM with the Saikawa–Shirai QCD equation of state, so self-convergence is
  the only instrument and one rung tighter is the only way to use it. The qualification to carry
  with that recommendation is the band structure above — at low $z$ a tighter rung will buy cost
  and no accuracy, because the floor is not quadrature; the band where 1e-9 could still bite is
  the mid one, where 1e-7 was still moving.

  **Clarified by the user, 2026-10-02** (`prompts/a3-v2-readiness` README §6.1, D1): `1e-9` meant
  **one rung tighter than the pair a store is built at**. It was said when `1e-8` was the shipped
  `rtol`. A3 v2 is built at `(1e-32, 1e-7)`, so its science check is `1e-8`. The same day the user
  decided that `DEFAULT_QUADRATURE_RTOL` itself becomes `1e-7` (D2, that campaign's prompt 01), so
  that `main.py`, `--build` and `extract_QuadSourceIntegral_data.py` share the pair. That is a change
  to a constant this board says is not its to take; the decision is the user's, recorded there.

  **Shipped 2026-10-02** (`prompts/a3-v2-readiness` prompt 01, `2a44f7e`):
  `DEFAULT_QUADRATURE_RTOL` is now `1e-7`. **This issue stays open.** The dead `local_atol` branch
  and the depth cap `DEFAULT_LEVIN_MAX_DEPTH = 20` described above are unchanged, and
  `DEFAULT_QUADRATURE_ATOL` is still `1e-32`; what changed is that the shipped pair is the one the
  sweep found converged in all three bands.

  **Next step:** regenerate the store at `(atol, rtol) = (1e-32, 1e-7)` (see that issue for the
  decision and the scope). **The remedy for the constant itself is not this campaign's to take.** `DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL` belong to
  `prompts/levin-refactor` and `prompts/qsi-phase-groups` (`tolerance-convergence` README §0.4),
  and `DEFAULT_LEVIN_MAX_DEPTH` is owned by nobody. This board measures and hands over. Indexed at
  `docs/OPEN_ISSUES.md` §1.1.

  **Narrowed 2026-10-25 by prompt 08 (the user's decision the same day): the cliff survives at
  `rtol = 1e-7`.** A3 v2, the store regenerated at `(1e-32, 1e-7)`, holds all 9280 rows, and prompt
  08's depth census (the method of the table above, reproduced as
  `docs/handover/a3-v2/after/scripts/depth_census.py` and re-counted by the orchestrator with
  `sqlite3` `mode=ro`) gives: NULL 2489 rows, depth 0 99, 1–19 4427, **depth 20: 2265 rows at a mean
  454 s, 1179 of them unconverged (52.1 %)**; **286.3 CPU-hours, 285.8 of them at depth 20**, 249 in
  the low band ($z < 9$) the baseline never reached; up to 966,508 regions. The most expensive rows
  have this entry's `r == k ≫ q` geometry. So `rtol = 1e-8` was not the cause: the length-share
  `local_atol` mechanism above, on squeezed triples, is, and it persists one decade looser. **What
  stands:** the sweep's finding that `rtol = 1e-7` is converged *in value* in every band; it does not
  relieve the *cost*, and the low band is representation-limited (D1/D2). **What remains:** the
  remedy, owned as before by `levin-refactor` / `qsi-phase-groups`, with `DEFAULT_LEVIN_MAX_DEPTH`
  owned by nobody. The comparison is not like-for-like (the pair differs, the baseline lacked 1680
  items, no per-band baseline figure exists): `docs/handover/A3-V2-VERIFICATION.md` §3.

  **Narrowed 2026-10-25 by an analysis of A3 v2, not a prompt — a hypothesis, not yet a
  measurement.** [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.
  - **The finding.** On the depth-20 rows, the declared `total_abserr/|total|` tracks
    $\varepsilon k\eta(z_{\rm response})$, the double-precision floor of the phase the driver is
    handed: log–log slope 0.84 over 1666 mid- and low-band rows, median ratio 2.2
    (`docs/handover/a3-v2/findings/depth20_floor.txt`). `gktk-remedial` Layer 1 measured the phase
    delivered at exactly that floor.
  - **The proposed mechanism.** On `r == k`, the slow phase group is a sum of separately reduced
    remainders whose large $\tau_s$ terms should cancel. What is left is non-smooth jitter at the
    floor, which no subdivision resolves. Only `depth_max` can end the bisection: the
    length-share `local_atol` above is dead, and `phase_limited`'s eq. (151) floor has no term for
    error in the input phase.
  - **The refinement.** The length-share mechanism is the reason no exit exists; the floor is what
    drives the bisection to the cap.
  - **What would decide it.** Findings §2.6 is the test, read-only or offline, and §2.7 is a
    three-layer remedy.
  - **Owner: still none.** The findings document is written to plan a campaign from.

  **Assigned (2026-10-25):** to [`prompts/levin-phase-floor`](../levin-phase-floor/README.md), opened
  that day from findings §2 on the user's decision. It owns the cliff's cause and remedy (§2.7
  layers 1–3); this entry stays here and is narrowed or closed here. Its planning reconnaissance
  (that README §0.2) refines "the slow group" above. Over the 629 depth-20 rows of more than
  100 s:
  - the slow `G+r` on `r == k` carries 52.7 % of the per-group Levin time;
  - the fast `G-r` carries 33.5 %, and the lone `G` 6.3 %.

  So the campaign carries a second hypothesis beside this one: $G$'s own remainder carries the floor
  into every group containing $G$.

  **Narrowed 2026-10-25 by `levin-phase-floor` prompt 01**, a census of every per-group record in
  A3 v2:
  - **The floor is not confined to cancelling groups.** 43.9 % of the depth-20 Levin time is on
    groups with no cancelling pair ($S = 0.4385$).
  - **Both classes track $\varepsilon k\eta_R$.** Cancelling groups have a slope of 0.968 and a
    median $e_g/F$ of 2.12; the rest have 0.822 and 1.49.
  - **So H-cancel fails as the whole account, and H-anchor holds** (P1–P2).
  - **The "slow group" above does not cancel exactly** (that campaign's
    `[01-slow-group-leading-terms-do-not-cancel]`). $G$'s phase runs at $k/H$ and $T$'s at
    $c_s k/H$, so on `r == k` the slow group `G+r` is about 3.7 times slower than `G-r`, not
    stationary. The "cancel identically" of the second bullet above is overstated.

  The cost still sits where the floor exceeds `rtol`. [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §1.

  **Narrowed (2026-10-25, levin-phase-floor 03):** seven depth-20 rows re-run bit-identically and
  instrumented.
  - **Every depth-20 region is a Clenshaw–Curtis fallback region**, 189,196 of 189,196, so the
    bisection to the cap is in `_adaptive_levin`'s `is_direct` branch (that campaign's
    `[03-depth20-regions-are-all-clenshaw-curtis]`).
  - **The discrepancy that drives it does not fall with refinement:** flat in $h$ from depth 10 to
    20 (slopes 0.027–0.089), at $D/\sigma_\theta$ 0.078–0.113.
  - **Removing the phase jitter alone lowers it by 3.1e6–2.4e8**, to round-off; and `phase_err`
    reaches `abserr` in only 0.85 % of the depth-20 regions.
  - Prompt 03's M3 fails on its slope clause as written, so the mechanism verdict as written does
    not hold; that campaign's gate G1 decides how it is read.

  [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §3.

  **Narrowed (2026-10-25, `levin-phase-floor` 04):** on 03's seven depth-20 rows the cap is the
  phase floor's.
  - **With it removed,** whether by exact remainders or by taking $k\tau_R$ out of $G$'s phase
    alone, the rows need 66–167 regions instead of 22,147–52,909, and none reaches depth 20 at
    `rtol = 1e-7`. The `1e-7` pair is then reachable on them.
  - **At `1e-8`** a single short chain per group still reaches the cap
    (`[04-x-at-1e-8-still-reaches-depth-20]`). The jitter-free totals at `1e-7` and `1e-8` differ by
    at most 6.7e-6 relative.

  [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §4.

  **Narrowed (2026-10-26, `levin-phase-floor` 05): the cap goes in production code, where it is
  still there.** Layer 1(a) (`120fdfe`) takes $k\tau_R$ out of $G$'s phase.
  - **The capped rows.** On the 11 sampled A3 v2 depth-20 rows still capped under B1's code, no
    region reaches depth 18 at `rtol = 1e-7`. The maximum depth is 5–11 at `1e-7` and at `1e-8`,
    and the time is 644 s → 4.4 s.
  - **The declared error.** It holds against `1e-8` on all 11.
  - **On two uncapped mid rows** the `1e-7` error misses the change to `1e-8`: by 1.29× on
    `mid-G-r-3` (2.7e-6 relative) and 1.70× on `mid-G-3` (3.0e-7). That is an ordinary-row
    tolerance effect, not the floor (its W5).
  - **20 of the 31 sampled stored depth-20 rows** are not capped under B1's code at all
    (`[05-b1-uncaps-most-cheap-depth-20-rows]`).

  [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.2.

- **[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]** *(opened 2026-09-23 by the A3
  baseline run, not by a prompt)* — `QuadSourceIntegral` records `total_converged` and
  `total_phase_limited`, and **nothing reads either**. A row whose bisection ran to the depth cap
  without meeting a tolerance is written, keyed, and served to every later consumer exactly like a
  row that converged. In the A3 baseline store that is **889 rows** (43.2 % of the 2057 at depth
  20, 11.7 % of all 7600). The flag is there, so the data is recoverable by anyone who thinks to
  select on it; nothing in the pipeline does, and no `QuadSourceIntegral` consumer joins on it.

  **Impact.** Every downstream use of this store, and of any store built at a tolerance pair that
  reaches the cap — so it is the same population as the issue above and disappears with it if that
  pair changes. Today it is latent rather than wrong: the stored `total` is the integral the code
  computed, and `levin_quadrature.py:2328` says the estimate is "usually still good". What is
  missing is any way for a consumer to *decline* it.

  **Next step: subsumed by the regeneration — user decision, 2026-09-23.** The store is to be
  rebuilt from scratch at the chosen tolerance pair, so these rows are not cleaned, they cease to
  exist. What survives the rebuild is the *design* question, and only if the adopted pair still
  reaches the depth cap: recorded so that it is not lost, not so that it is acted on now. Two things are true about it. First, there is **no defined
  route to remove such rows**: the only mechanism in the tree is `main.py --prune-unvalidated`,
  which drops rows whose validation flag is clear, so removing these would mean clearing that flag
  by hand on a selection the schema was not designed to express — a hand edit against a datastore,
  which is the kind of thing this project does not do. Second, it may not need doing: if the sweep
  above moves the tolerance pair, **every one of these rows is regenerated anyway**, because the
  pair is part of the datastore key. The decision therefore waits on the sweep. If the pair does
  *not* move, the right shape is probably a filter at the consumer and a `--require-converged`
  gate, not a deletion. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Narrowed 2026-10-25 by prompt 08 (the user's decision the same day): live again.** The pair
  moved, and the regeneration made the rows above cease to exist, as the subsumption said. But its
  condition holds — the adopted pair, `(1e-32, 1e-7)`, **still reaches the depth cap** — so the
  design question survives on the new store: A3 v2 stores **2262** rows with `total_converged = 0`,
  **1179** of them at depth 20, and `total_phase_limited = 1` on 525, each keyed and served like any
  converged row, with nothing reading either flag and no defined route to remove one. The shape
  suggested above — a filter at the consumer and a `--require-converged` gate, not a deletion — is
  now the live question. Counts: `docs/handover/a3-v2/after/out/depth_census_08.txt`.

  **Narrowed 2026-10-25 by an analysis of A3 v2, not a prompt: the flag is the wrong thing to gate
  on.** `total_converged` ANDs per-group flags, and each Levin call checks `rtol` against its
  **own** group's value (`ComputeTargets/QuadSourceIntegral.py:1080`), while the groups cancel in
  `total`. So the flag says little about `total_abserr/|total|`:
  - of the 1086 depth-20 rows flagged converged, 257 declare above 1e-6 and 132 above 1e-4;
  - the largest is **5.0e-2**.

  A `--require-converged` gate on the flag would admit those rows, so any gate belongs on
  `total_abserr/|total|`. Five depth-20 rows store a declared error of +inf (see `levin-refactor`'s
  `[04-roundoff-floor-can-be-infinite]`). Measurement:
  [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.5.

  **Assigned (2026-10-25):** to [`prompts/levin-phase-floor`](../levin-phase-floor/README.md). What
  `total_converged` should mean, and whether a consumer gate reads `total_abserr/|total|`, is that
  campaign's layer 3, decided by the user at its gate G2 (its README §6.3 Q4). Its prompt 01
  re-counts §2.5's figures on the store, at group level too. This entry stays here.

  **Narrowed 2026-10-25 by `levin-phase-floor` prompt 01.** §2.5's counts reproduce exactly:
  1086 / 257 / 132, largest 5.03e-2. The census also found:
  - **At group level**, 2541 of the depth-20 rows' converged group records declare $e_g > 10^{-6}$.
  - **558 rows store `total_abserr = +inf`**, not five, and all are unconverged. A gate on
    `total_abserr/|total|` would reject all of them, which may or may not be right (`levin-refactor`
    `[04-roundoff-floor-can-be-infinite]`).

  [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §1.

  **Narrowed (2026-10-25, `levin-phase-floor` 04, N4): the stored error can also be understated.**
  - **On 3 of 7 depth-20 rows** (R3, R6 and R7), $|{\rm total}_{\rm stored} - {\rm total}_X|$ is
    1.16–1.94× the stored `total_abserr`. That is 2.7e-4 relative on R3's cancelling total and
    6.6–7.3e-7 on R6 and R7.
  - **On R6 and R7** the stored row is unconverged, while the jitter-free re-run at `1e-8` converges
    with a declared error 30–41× smaller.
  - **The caveat.** $X$ keeps $\varphi$'s sample noise and the `TWO_PI` modulus, so it is the
    jitter-free limit, not the truth.

  [`docs/levin-phase-floor/04/issues.md`](../../docs/levin-phase-floor/04/issues.md) I8.

  **Narrowed (2026-10-26, `levin-phase-floor` 05): with the floor gone, most capped rows still
  store `total_converged` False, and the stored error is not always honest.**
  - **Convergence.** Under layer 1(a), 21 of 31 sampled depth-20 rows are unconverged at `1e-7`,
    and 19 of 31 at `1e-8`. On the 11 rows still capped under B1's code, 9 stay unconverged,
    although their depth falls to 5–11. The unconverged pieces are Levin groups with 1–17 regions,
    often `Gqr` groups. 3 rows have an infinite `total_abserr` under both codes (its W7).
  - **The error.** On `low-G-3` the change moves the total by 2.48× the error declared before it.
    At B1's code, R3, R6 and R7 are now within it (0.41, 0.40, 0.17; W6).
  - **In the rehearsal pair** R9/R10, one of 864 rows, $(10^6, 3.162\times10^5, 10^6, 329.64)$,
    moves by 28× its declared error; the next largest moves by 0.60×.
  - **A3 v2 itself** is pre-B1. B1 moves its stored totals by up to 1.5e5× their error
    (`[05-b1-moves-stored-a3v2-totals-beyond-their-error]`).

  [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.

- **[01-recon-off-cut-closed-form-is-ill-conditioned]** *(opened 2026-09-20 by prompt 01)* —
  [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) §9's implementation
  brief prescribes §5.2's second boxed form for the **off-cut** coefficient $C(\tilde y)$, a
  difference of two ${}_2F_1$ at argument $\tfrac{1-\tilde y}{2}$. The form is correct — it
  reproduces `mpmath`'s `legenq(..., type=3)` to $10^{-48}$ term by term at 50 digits — but it is
  **catastrophically ill-conditioned away from $\tilde y = 1$**. At $b = 0.2$, $\tilde y = 150$ its
  two terms are each $2.2426\times10^{5}$ and their difference is $1.02\times10^{-8}$: a 13-order
  cancellation, which amplifies the $1.1\times10^{-17}$ by which a double's `b` differs from the
  number meant into a per-cent-level error in the result. Measured against the raw Olver call, and
  **identical at `mp.dps` 30, 60 and 120** — raising the working precision does not help, because
  the loss is in the inputs: **6.2e-15 at $\tilde y = 1.5$, 9.5e-10 at $\tilde y = 40$, 8.6e-08 at
  $\tilde y = 150$.** The `q-smooth` fixture at $b = 0.2$ sits at $\tilde y = 149.5$, so this is a
  fixture shape and not a corner. Recon §10 item 5 left exactly this untested and this is its
  answer. **Impact:** documentary and prospective. Nothing in the tree is wrong — prompt 01 used the
  2020 paper's own $1/\tilde y^2$ form instead (`1912.tex:743-747`, DLMF 14.3.7), which has no
  cancellation anywhere, is regular at $b = 0$, and agrees with the raw call to rounding at every
  $\tilde y$ tried — but a later agent re-implementing from §9 would inherit the defect, and §5.2's
  box is still the right form **on-cut**, where `domenech._R_nu` uses it. **Next step:** an
  additive subsection or an inline caveat in recon §5.2 and §9 saying which form to use where, by
  whoever next owns `docs/handover/DOMENECH-KERNEL-RECON.md` (CLAUDE.md invariant 6 — the document
  was correct for what it measured, so this is an addition, not a rewrite). Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Deviations from
  the prompt" §1. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[01-recon-section-7-N-symbol-drops-a-pi]** *(opened 2026-09-20 by prompt 01)* —
  [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) §2.2 defines
  $\mathcal N \equiv \pi4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}$, which at $b = 0$ is
  $\pi\cdot\tfrac\pi4\cdot\tfrac32 = \tfrac{3\pi^2}{8}$. §7's $b\to0$ reduction prints
  "$\mathcal N = \tfrac{3\pi}{8}$" — the same constant **without** the $\pi$, i.e. the object
  `domenech.kernel_constant(b)` and not $\mathcal N$. **Impact:** documentary only. §7's displayed
  reduction of $I_{\rm target}(b=0)$ to $\tfrac98\times$ Kohri & Terada's eq. (25) is correct as
  written and was reproduced here to rounding (prompt 01's test 5 holds at $\le4.7\times10^{-15}$
  away from $u = 0.01$), so no arithmetic depends on the symbol. But a reader hand-checking §7
  against §2.2 loses a factor of $\pi$ and will conclude one of the two is wrong. **Next step:**
  one symbol, by whoever next owns that document — naturally together with the issue above, which
  touches the same file. Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Observations not
  acted on" item 2. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[02-realistic-fixture-Gk-phase-is-the-superseded-construction]** *(opened 2026-09-20 by prompt
  02)* — `ComputeTargets/tests/test_phase_groups.py:299` documents `BesselPhaseGk` as built *"the
  way `GkSourcePolicyData._create_functions` builds the real one: the phase is a `phase_spline`
  over $\log(1+z')$ through (div 2pi, mod 2pi) samples"*. **That has not been true since
  `prompts/GkTk-remedial` prompt 09.** Production's `GkSourcePolicyData._build_phase`
  (`:145-190`, consumed at `:726`) returns a `PrimitivePhase` — the leading $-k\,\Delta\tau$ from
  the background table with only a small residual splined — which is precisely how that campaign
  discharged `[12-phase-spline-error-grows-with-x]`'s $\delta\theta \simeq h^4x/384$ for $G_k$
  (board M14: 4.189e-08 rad against 7.286e-03 for the same samples, a ratio of 1.739e+05).
  `TkSourceFunctions` *is* current — prompt 10 gave it a `PrimitivePhase` too — so the realistic
  flavour's $T$ half is production's and its $G$ half is the construction production abandoned.
  Consequences, in order of how much they matter: the module docstring of
  `test_quadsource_integral.py` claims the realistic flavour "is the accuracy production can
  expect", and that is **stale for $G$**; prompt 02's representation term (4.08e-05 – 2.68e-03,
  `REALISTIC-LARGE-X.md` §4) is therefore an **upper bound** on production's rather than an
  estimate of it, because it carries a term production has already removed; and the Levin cost
  growth of the issue below may be the same staleness seen from the cost side.
  **Impact:** **B2**, which re-runs this harness once the gap is gone and attributes what is left —
  it should fix the fixture before it scores the residue, or its numbers will be pessimistic by an
  unknown factor for $G$. Also any later reader of the test module's docstring. Nothing in
  production is wrong; this is an instrument that has fallen behind the thing it models.
  **Next step:** rebuild `BesselPhaseGk`/`OffsetBesselPhaseGk` on `PrimitivePhase` as
  `_build_phase` does, and correct both docstrings. Not done here: prompt 02 §5 forbids editing
  `test_quadsource_integral.py`, and `test_phase_groups.py` is not in its scope either.
  Measurement: [`logs/02-realistic-flavour-large-x-harness.md`](logs/02-realistic-flavour-large-x-harness.md),
  "Observations not acted on" item 1. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]** *(opened 2026-09-20 by prompt 02)* —
  the Levin driver's region count is **flat in $x$ on the exact flavour and grows steeply on the
  realistic one**, and the growth may be a property of the fixture rather than of production.
  Measured over the $x_{\rm resp}$ ladder (`REALISTIC-LARGE-X.md` Table 8): exact **7→9**
  (`together`, six rungs), **7→13** (`T-first`), **17→30** (`q-smooth`) — flat, which is what
  Levin quadrature is for. Realistic **1,304→50,463** (`together`, peaking at $10^7$ and falling to
  15,827 at $10^8$), **720→75,570** (`T-first`), **855→31,358** (`q-smooth` over 980 to $10^5$).
  Wall time follows: 33× to 3,239× the exact flavour, growing 1.7× / 3.6× / **8.6×** per decade of
  $x_{\rm resp}$ by shape, which is what forced `q-smooth` realistic to stop at $10^5$.

  **The level is understood; the growth is not.** At the base rung the integration range spans
  5.40 decades of $(1+z)$, i.e. ~540 cells of the 100-per-$\log_{10}z$ grid, against 720–1,304
  regions — the driver bisects to about the spline knot scale, which a piecewise-cubic phase
  forces. But the knot count grows only **540 → 1,041** across the ladder (the range widens by one
  decade per rung, because `z_source_max` tracks $k$ while $z_{\rm response}$ is fixed), a factor of
  1.9, while the region count grows by 12× (`together`), 105× (`T-first`) and 37× (`q-smooth`).
  Regions per knot therefore rise from **1.3–2.4** at the base rung to **15–80** at the top: four to
  six extra levels of bisection the knot count does not account for.

  **Why it may be an artefact.** The fixture's $G$ half is the superseded raw `phase_spline` of the
  issue above, whose representation floor is $\delta\theta \simeq h^4x/384$ — **linear in $x$**.
  `_adaptive_levin` accepts a region on `resolved or phase_limited or depth_max`, and
  `phase_limited` requires `abserr <= phase_err`, so a region is forgiven only once its quadrature
  error has fallen *to* the representation floor. A floor that grows linearly in $x$ forces
  bisection that grows with $x$. If that is the mechanism, **production does not have it** — its
  $G$ phase is a `PrimitivePhase` — and this harness's whole cost curve, including the `q-smooth`
  ceiling, is a statement about the instrument rather than about the pipeline.
  **Impact:** bounded and cost-only. It does **not** touch any $N$ in `REALISTIC-LARGE-X.md`: every
  term in its Table 3 is a difference taken at fixed flavour, so a region count affects both sides
  identically, and the measured additivity (interaction $\le$ 2.82e-05 of the clamp term) would not
  survive if it did. What it touches is how far **B2** and any later re-run of this harness can
  reach, and whether `q-smooth` above $10^5$ is genuinely out of range or merely out of range for a
  stale fixture.
  **Next step:** re-run **one** cell — `q-smooth`, realistic, closed seam, $x_{\rm resp} = 10^5$ —
  with the $G$ phase built as a `PrimitivePhase`, and compare `Levin_regions` (31,358) and
  `integral_time` (514.8 s) against what is recorded here. A collapse confirms the mechanism; no
  change refutes it and points instead at the weakly-oscillatory gate (`Levin_fraction` swings
  0.09–29.1 across the factorial) or at the reference tolerance pair itself. **Deliberately not run
  by prompt 02** — out of its scope, and it needs the fixture change the issue above owns.
  Measurement: [`logs/02-realistic-flavour-large-x-harness.md`](logs/02-realistic-flavour-large-x-harness.md),
  "Observations not acted on" item 2. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Narrowed 2026-09-23 by the A3 baseline run.** The strong form of the hypothesis above —
  *"If that is the mechanism, **production does not have it**"* — is **refuted by measurement**.
  Production's $G$ phase **is** a `PrimitivePhase`, and production has a Levin blow-up anyway:
  2057 of 7600 stored `QuadSourceIntegral` rows sit at `max_depth = 20` with up to 1,530,190
  regions, at 220× the mean cost of a row that stops short of the cap
  (`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` above). So a stale
  `phase_spline` floor cannot be the whole explanation of unbounded subdivision in this driver.
  What the new measurement supplies is a **second, sufficient mechanism** that the harness shares:
  `local_atol` is `atol` divided by a region's length share, so at 1e6 regions the `atol` branch
  of `resolved` is dead whatever the phase is, and `rtol` alone must be met. That is present in
  the harness too. **This issue is not closed**: its own question — whether the *fixture's* 12–105×
  growth in $x$ is the fixture's or production's — is still open, and the one-cell experiment
  above is still the way to answer it. What has changed is that a null result there no longer
  means the cost curve is safe, and that the experiment should record `max_depth` and the region
  count against `local_atol` rather than against `phase_err` alone.

  **Narrowed 2026-10-25 by an analysis of A3 v2, not a prompt.** A3 v2 reaches 966,508 regions
  with production's `PrimitivePhase` $G$. The proposed mechanism in
  [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2 is a phase floor
  $\varepsilon k\eta$ that `phase_limited` cannot see, and a candidate third explanation for the
  realistic flavour's growth in $x$. The harness's $T$ half carries that floor too, so it does not
  need the stale $G$ phase. The harness runs at `REF_RTOL = 1e-12`
  (`realistic_large_x.py:122`), and $\varepsilon x_{\rm resp}$ exceeds that from
  $x_{\rm resp} \approx 4.5\times10^3$, so on nearly every rung. This does not explain why the exact
  flavour stays flat, unless its composed phases are exact where the realistic flavour's are not;
  that is not checked. The one-cell experiment above stands, and it should also
  record which group carries the deepest regions (findings §2.6 item 1).

  **Assigned (2026-10-25):** to [`prompts/levin-phase-floor`](../levin-phase-floor/README.md). Its
  prompt 02 injects deterministic phase noise into A2's fixtures, the exact flavour included, at
  production's `rtol`. Its prediction N2 asks whether noise alone makes the exact flavour grow as
  the realistic one does. If it does, that answers this entry by sufficiency, and the entry is
  narrowed rather than closed. It stays here.

  **Planning note (2026-10-25, the refresh of prompt 04):** `levin-phase-floor` 02 was stopped and
  superseded by its 03 the same day, and N2's original form was not tested (that campaign's
  README §2). So nothing on that campaign's plan answers this entry any longer. **A4
  ([`04-gk-phase-decision-test.md`](04-gk-phase-decision-test.md)) is now the only planned answer:**
  it runs the one-cell experiment above, on both `q-smooth` seams, with the old phase reproduced
  beside the new, and records `max_depth`, the regions per group and the phase-limited regions,
  as the two narrowings above ask. It runs at `98e68d3`, because the harness raises at `HEAD`.
  The assignment to `levin-phase-floor` is unchanged: that campaign still owns the depth-20
  mechanism this entry's narrowings point at, and lists the entry under its C5.

- **[09-two-readers-plot-the-radiation-form-under-the-w-label]** *(opened 2026-10-25 by prompt 09,
  on reading the readers; not acted on, by `CLAUDE.md` rule 4)* — `extract_GkSource_data.py` (its
  log–log $G_k$ plot, `plot_Gk`) and `extract_QuadSourceIntegral_data.py` (`plot_Gk`, its `GkSource.pdf`)
  each draw a curve labelled "Analytic $G_k$ [$w=w(z)$]" from `abs_analytic_G_rad_x`/`_y`, the
  **radiation** form, plotted twice under two labels. Their `abs_analytic_G_w_points` are built (and,
  since prompt 09, filtered through `sensible_log_points`) but used only in the plot's `if`. In
  any case the `GkSource` value table stores no `analytic_G_w` column (`ObjectFactories/GkSource.py`
  holds the two `_rad` columns only), so a `GkSource` read back from a store carries
  `analytic_G_w = None`. **Impact:** a reader of either plot sees the radiation form twice and takes
  one of them for the $w(z)$ form; no computation reads it. The same kind of mislabel as
  `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]` (closed, §4). **Next step:**
  decide whether these plots should show the $w(z)$ form at all. If they should, `GkSource` must
  store it (a schema change) and each plot must draw `abs_analytic_G_w_x`/`_y`, through the helper.
  If not, the second curve and its label go. Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[09-extract-gk-data-writes-redshift-objects-into-its-z-response-column]** *(opened 2026-10-25 by
  prompt 09, seen while checking the `analytic_G_w` CSV columns on the rehearsal store; not acted
  on, by `CLAUDE.md` rule 4)* — `extract_Gk_data.py:294` builds `z_response_column` as
  `[value.z for value in values]`, and `GkNumericValue.z` is a `CosmologyConcepts.redshift` object,
  so every `z_response` cell of every `extract_Gk_data` CSV reads
  `<CosmologyConcepts.redshift.redshift object at 0x…>`. Seen in all 52 CSVs that reader wrote on
  `handover-09-reference` (`var/handover-09/gk-csv-check.txt`). The `sort_values(by="z_response")`
  then orders by the objects' own comparison. **Impact:** the CSVs cannot be read back against
  response redshift; their other columns are right, and the plots are unaffected. The other readers
  write `value.z.z`. **Next step:** `value.z.z`, as the other readers do, with a test like
  `test_analytic_g_w_readers`'s CSV test. Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[10-the-readers-draw-their-work-lists-unseeded]** *(opened 2026-10-25 by prompt 10's review,
  from the agent's `issues.md` item 1; not acted on, by `CLAUDE.md` rule 4)* — every reader draws
  its `k` and `z_source` subsamples with an unseeded `random.sample`: `extract_GkWKB_data.py:472`
  takes 174 of A3 v2's ~1740 source redshifts, and `extract_Gk_data.py:416` and `:432`,
  `extract_GkSource_data`, `extract_TkWKB_data`, `extract_tensor_source_data` and
  `extract_QuadSourceIntegral_data` draw the same way. **Impact:** a reader's output is a different
  sample on every run, so a re-run cannot be compared count for count with an earlier one
  (`extract_Gk_data` wrote 1251 files with `--models` and 1980 without, from the same 522 lookups,
  against 08's 1593 and 1593); a check that names a pair passes only when the draw includes it
  (prompt 10's acceptance 2 held at odds of about 1 in 10); and a "complete" read-back is complete
  for the sample, not the store. **Next step:** a `--seed`, or a mode that plots a named pair or the
  full grid. Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[11-seam-band-rows-deviate-from-the-oracle-by-up-to-two-orders]** *(opened 2026-10-25 by prompt
  11's review)* — In A3 v2's seam band ($z_{\rm response} > 10^5$, 6080 rows),
  $|{\rm total}/{\tt analytic\_rad} - 1|$ has a median of 5.3e-04 but a maximum of **96.9**. The
  tail:
  - 20.5 at the $10^8$ decade;
  - 17.8 and 96.9 at $10^{10}$;
  - every stratum that has a clamp has maxima above 1.

  A2 bounded the clamp term at 0.91 of `total`. So rows two orders out are not obviously the clamp
  alone. They may be rows where `analytic_rad` is near a zero, or rows where something else fails.
  Prompt 11 tabulated them and did not look further (its `issues.md` item 5). **Impact:** unknown
  until the rows are identified. If they are near-zero oracles, it is none; otherwise they are the
  worst stored `total`s on the grid. **Next step:** list the top rows from
  `var/handover-11/out/subintervals_11.csv` on the Mac Studio (not committed; `*.csv` is excluded) with $|{\tt analytic\_rad}|$ against its
  neighbours in $(k, q, r)$, read-only. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[12-realistic-large-x-reads-the-retired-clamp-records]** *(opened 2026-10-25 by prompt 12, B1;
  not acted on, by the prompt's own §4.1)* — B1 removed `metadata["partition"]["max_clamp_gap_log1pz"]`
  and every `subintervals[i]["clamp_gaps_log1pz"]`. `docs/handover/realistic_large_x.py`, A2's
  harness, reads `item["clamp_gaps_log1pz"]` (`:332`, `:666`) and `partition["max_clamp_gap_log1pz"]`
  (`:356`), so at `HEAD` every cell raises `KeyError`; its docstring also describes `_ClampedTk`
  (`:45`, `:194`). The prompt forbids editing it (its checkpoint is gated on its own SHA-256) and
  makes a re-run after B1 B2's. Three related records of the same removal:
  - `docs/levin-phase-floor/noise_injection.py` (another session's, untracked when B1 ran) imports
    that module and `test_quadsource_integral`'s `Case`, and is now a record of its own SHA;
  - `ComputeTargets/primitive_phase.py:73`, `:102` and `ComputeTargets/tests/test_primitive_phase.py:416`
    name `QuadSourceIntegral._ClampedPhase`, and `main.py:1887-1893` says `QuadSource`'s values end at
    the larger hand-over;
  - a pre-B1 store's `QuadSource` rows (A3 v2's) stop at the last node above $z_{\rm init}$, and
    B1's strict source check refuses them, so a resume of an old store at B1's tree must recompute
    `QuadSource` first.

  `seam_census.py` is a byte-identical record and stays; `analyse_quadsource_integral.py` reads the
  keys with `.get` and does not raise. **Impact:** A2's tables cannot be regenerated, or extended
  to B2's configurations, until the two reads are replaced; the stale comments mislead a reader.
  **Owner: B2**, which re-runs A2 after B1 (prompt 12 §4.1). **Next step:** in B2, read
  `partition["handover"]` where the gaps were read, and correct the three comments. Indexed at
  `docs/OPEN_ISSUES.md` §1.1.

- **[03-gksource-policy-accepts-a-value-that-raises]** *(opened 2026-10-25 by prompt 03 (A3), from
  README §2 (o) item 1; indexed since 2026-09-19 as "none yet")* — `GkSourcePolicy` accepts
  `numeric_policy = "maximize-numeric"` (`MetadataConcepts/GkSourcePolicy.py:6`, checked at
  `:25-30`). The value persists and joins `GkSourcePolicyData`'s key, and `_classify_crossover`
  raises on it (`ComputeTargets/GkSourcePolicyData.py:472-475`): it branches only on
  `"maximize-WKB"` (`:457`) and `"minimize-WKB"` (`:465`), and the latter fails the constructor's
  check. So the branch at `:465-470` cannot be reached by any constructible value. The raise fires
  only on a `mixed` row with a non-empty band, inside a Ray remote. **Impact:** latent; every caller
  passes `"maximize-WKB"` (`main.py:4321-4344`, `extract_QuadSourceIntegral_data.py:1189-1197`,
  `config/defaults.py:190-202`), and only the fixture store writes `"maximize-numeric"`
  (`StoreFixtures/real_store_fixtures.py:464`, `:480`). **Owner: C2.** Measurement:
  [`a3-v2/policy-geometry/issues.md`](../../docs/handover/a3-v2/policy-geometry/issues.md) item 1.
  Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[03-quadsource-policy-vocabulary-differs]** *(opened 2026-10-25 by prompt 03 (A3), from README
  §2 (o) item 2)* — `QuadSourcePolicy` spells the value `"maximize_numeric"`
  (`MetadataConcepts/QuadSourcePolicy.py:6`), a third spelling beside `"maximize-numeric"` and
  `"minimize-WKB"`. Nothing reads the field, though it keys `QuadSourceIntegral`. Both constructors
  reassign the offending value to the default before formatting their warning
  (`GkSourcePolicy.py:25-30`, `QuadSourcePolicy.py:60-65`), so the warning prints the default.
  **Impact:** none on a run today. **Owner: C2.** Measurement: the same `issues.md`, item 2.
  Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[03-gk-classification-is-diagnosed-and-ignored]** *(opened 2026-10-25 by prompt 03 (A3), from
  README §2 (p); narrowed in the index the same day by prompt 11)* — `_classify_crossover`
  (`GkSourcePolicyData.py:287-504`) can return `quality = "incomplete"` (`:358`, `:374`, `:390`) or
  `type = "fail"` (`:342-344`), and nothing acts on either. The readers of `quality` are a plot
  label, the summary block, an `incomplete` list used only for a dump, and `_classify_Levin`'s
  early return. `main.py:385` takes the policy for a work item with no test of either field, and an
  `incomplete` route then raises in `build_partition` (`QuadSourceIntegral.py:325-345`). The dump at
  `main.py:3224-3225` is not gated on `--dump-incomplete`, so with the flag unset it raises a
  `TypeError` in a Ray task whose reference is discarded (`:2801-2804`, `:2812-2814`).
  `_classify_crossover` has no test. **On A3 v2, none of these routes was taken:** no `fail` and no
  `incomplete` row among 1160 (11, reproduced by A3). **Impact:** a fail-late contract,
  unenforced and unexercised on the grid that ships. **Owner: B3** (the contract), with B4 (the
  invariant). Measurement: the same `issues.md`, item 3. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[03-the-g-crossover-is-one-super-horizon-node-on-every-complete-row]** *(opened 2026-10-25 by
  prompt 03 (A3), from its census; not acted on)* — On all 71 `mixed`/`complete` rows of A3 v2,
  `crossover_z` is the same grid node, $3.121867\times10^{15}$, at every $k$ and every
  $z_{\rm response}$, 82 intervals below the grid top.
  - `primary_WKB_largest_z` is the grid top ($2.0636\times10^{16}$) on those rows. So
    `maximize-WKB` takes the first node whose WKB clearance exceeds `CLEARANCE_GOOD` = 0.05, and the
    82 is that clearance in intervals at the top of this grid (ratio 1.0055 on every row).
  - That node is super-horizon at every $k$: 3.11 e-folds outside at $k = 3\times10^8$/Mpc, 4.26 at
    $9.56\times10^7$, more than 5 at every smaller $k$. Numeric $G$ is consumed above it and WKB $G$
    below it.
  - The overlap runs from $z_{e4}$ to the grid top, 8.8–17.0 e-folds, not the half e-fold README
    §2 (q) describes. The half-e-fold band itself carries both representations on all 72 rows, so
    the lever map is not falsified (prompt 03 §7).

  **Impact:** unmeasured, and possibly large for the plan. The seam on the grid that ships is not
  where README §2 (q), B4 and D1 assume it sits. `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`
  (tolerance-convergence) measures the numeric $G$ spline three to four e-folds inside the horizon,
  where these rows do not consume it. Whether the WKB $G$ is accurate where it is consumed, from a
  super-horizon node down, is not measured here. **Next step:** the user rules whether README
  §2 (q), B4's lever map and D1's scope are re-read on this, before B4 or D1 is written.
  Measurement: [`POLICY-GEOMETRY.md`](../../docs/handover/POLICY-GEOMETRY.md) §3, §5, §6;
  `issues.md` items 4 and 5. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[03-gksource-wkb-holes-above-the-primary-run]** *(opened 2026-10-25 by prompt 03 (A3), from its
  census; not acted on)* — 12 of 1160 `GkSource` rows have a hole in their WKB values with
  $z_{\rm response} \ge z_{e4}$: 11 `numeric` and the `minimal` row. No row has one with
  $z_{\rm response} < z_{e4}$, which is the case `GkSource.py:283-305` warns about. On the `minimal`
  row the hole is what cuts the overlap to 7 intervals: the WKB run ends at $N = 3.507$, the hole is
  at 3.484, and values resume at 3.438. 17 rows have WKB nodes above their primary run.
  **Impact:** small. It explains A3 v2's one `minimal` row, and it says nothing about the
  `complete` rows. Whether a hole is the producer's or the grid's is not measured. **Next step:**
  none until B4. Measurement: `POLICY-GEOMETRY.md` §8; `issues.md` item 6. Indexed at
  `docs/OPEN_ISSUES.md` §1.1.

**Assigned here, held on another board:**

- **[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]** — *opened 2026-10-19 by
  `rrk-v020-adoption` prompt 03, narrowed by its prompt 04, **held by that board**
  ([`../rrk-v020-adoption/IMPLEMENTATION_STATE.md`](../rrk-v020-adoption/IMPLEMENTATION_STATE.md)
  §3), and assigned to this campaign by the user on 2026-10-19, since it owns the A3 v2 launch.* After
  a `SIGKILL` of a build's driver, Ray's GCS server can be left reparented to pid 1, stuck in its own
  shutdown, where a `SIGTERM` cannot end it; until it is gone the rehearsal lock cannot be released.
  Seen at R3's `hardkill-a` (Ray 2.43.0), and again on 2026-10-25 at `py313-venv` 01's (Ray 2.59.0,
  the same stack, alive about 1 h 56 m until a person's `SIGKILL`); the other board holds both
  measurements. `CLAUDE.md` rule 5 has the manual step (after `end`, a person
  checks `ray-processes` and ends a GCS server still listed with `kill -KILL <pid>`); A3 v2's
  inheritance is `docs/rrk-v020-adoption-verification.md` §3.5. **What is left here** is the cause
  and any fix: for example a parent-death watch on the GCS server in
  `docs/handover/quadsource_atol_sweep.py`, or a report to Ray. Not yet scheduled; the user may not
  take it up at once. The entry and its measurements stay on the other board, which is right where
  the two differ; indexed at `docs/OPEN_ISSUES.md` §1.1.

---

## 4. Resolved issues

Closed at `levin-phase-floor`'s gate G1, 2026-10-26:

- **[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]** *(opened 2026-10-25 by prompt 12's
  rehearsal R8, on the Mac Studio; opened on this board by the user's ruling on R8's review)* — B1
  changed the stored integrals, and the store fingerprint shows no difference in
  `QuadSourceIntegral`.
  - **What was found.** `compare_with_recorded.py`, R8 against R3, reports one differing class,
    `QuadSource`: its rows' value count, 3 more `QuadSourceValue` rows each. `QuadSourceIntegral`
    has R3's digest (`8913eda5ce7b` in both). Yet on the one triple both readers' extracts hold,
    63 of 96 `total`s differ from R4's (R3-equal) values, by up to 1.8e-03 of the row set's largest
    $|{\rm total}|$, and all 864 rows carry B1's `partition.handover`.
  - **Why.** Format 6's `inventory_spec()` for `QuadSourceIntegral`
    (`ObjectFactories/QuadSourceIntegral.py:965`) is its identity only, parents and tags, with "no
    validated flag and no value table", so it does not see `total`, `metadata` or the partition. Its
    cross-shard `source` parent resolves to the `QuadSource` row's `reference_digest(key, tags)`,
    which does not include that row's value count.
  - **What it did here.** Prompt 12 §6.4 and §7.3 and the orchestrator note expected R8 to differ
    in `QuadSourceIntegral` too. The user ruled that a `QuadSource`-only difference meets them
    (log 12 §6, "The user's rulings").

  **Impact:** a rehearsal that checks "equal to the reference fingerprint" cannot catch a
  regression in `QuadSourceIntegral` values. A change to how integrals are computed, with
  `QuadSource` unchanged, leaves the fingerprint equal; every R-rehearsal's "equal to `39809dca`"
  certified the integrals' identities and counts, not their values. **Next step:** the likely fix
  adds a value table to `QuadSourceIntegral`'s `inventory_spec`; a fingerprint change costs nothing
  to make. **Assigned (2026-10-26):** to a prompt of this campaign, by the user at `levin-phase-floor`'s
  gate G1. It must land, with a new rehearsal reference, before that campaign's layer 1 prompt
  (05), which changes `QuadSourceIntegral` values and must be seen to. Measurement: [`HANDOVER-B1.md`](../../docs/handover/HANDOVER-B1.md) §5.3
  and §5.4; [`docs/handover/b1/r8/issues.md`](../../docs/handover/b1/r8/issues.md). Indexed at
  `docs/OPEN_ISSUES.md` §1.1.

  **Narrowed (2026-10-26), and the next step above corrected: the gap is in `datastorekit`, and it
  covers every class.**
  - **What the fingerprint holds.** `datastorekit.store_inventory` (pinned `v0.5.0`) builds each
    record from its identity (leaves and parent digests), its tags, its validated flag and a
    **count** of its value-table rows. Its module docstring says the value tables are "counted …
    and never listed", and that "a computed-values field can be added to `Record` later".
  - **Why the next step was wrong.** A value table in `QuadSourceIntegral`'s `inventory_spec` would
    add a row count, not a value.
  - **The reach.** No class's computed values are in any fingerprint. Every rehearsal certified
    "equal to the reference" for identities, tags and counts only. Where a value was certified, a
    separate comparison did it, as B1's did.

  **Assigned (2026-10-26), by the user at `levin-phase-floor`'s G1:**
  - **The fix is DatastoreKit's.** A computed-values field on `Record`, declared per class, with
    floats canonicalised bit-exactly. It is released there, and the pin is moved here in a commit of
    its own (`CLAUDE.md`), followed by this repository's specs declaring their value columns and a
    new rehearsal reference.
  - **It is not on 05's path.** It must land **before the next production build**.
    `levin-phase-floor` 05 goes ahead in parallel and certifies its own `QuadSourceIntegral` values
    by a direct comparison.

  **Closed 2026-10-26 as a misreading, by the user's decision.** The fingerprint never claimed
  to hold computed values. `datastorekit.store_inventory` says so in its module docstring. What a
  fingerprint certifies is what a store holds, which is what resume and integrity checks need:
  its work items, their parents and tags, and how many value rows each has. So:
  - **No DatastoreKit change is made.** The first assignment of the same day, to a computed-values
    feature, is withdrawn as out of proportion.
  - **What earlier records mean.** Wherever a record says a rehearsal reproduced or equalled a
    reference fingerprint, it means identities, tags and value-row counts. Where values were
    certified, a separate comparison did it, as B1's did. The records are left as written
    (`CLAUDE.md`, campaign conventions rule 6).
  - **The need is met by a standing rule and a tool.** A prompt that changes how values are
    computed certifies them directly: a rehearsal pair, compared by physical identity (`CLAUDE.md`,
    campaign conventions rule 7). The tool is `tools/store_value_diff.py`, which `levin-phase-floor` 05 adds and is the
    first to use.

Closed by `py313-venv` prompt 01, 2026-10-25:

- **[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]** *(opened 2026-10-22 by the
  planning of prompts 06 and 07, from a read of the Mac Studio's checkout; not by a prompt)* —
  `origin/main` carries `9678f28` (2026-10-06, "Bump scipy and ray pins for macOS update and Python
  3.13"), committed on the Mac Studio, which this branch does not: it moves `ray` from 2.43.0 to
  2.45.0, the earliest release with a Python 3.13 wheel, and `scipy` to 1.16.3. This branch took the
  `scipy` half independently (`8abb3da`) and stays on Python 3.12 (MacPorts 3.12.15) with
  `ray==2.43.0` (`requirements.txt:75`), which every rehearsal R0–R3 ran on. The two branches
  therefore disagree on the interpreter and on `ray`, and a merge of this branch into `main`
  conflicts at `requirements.txt:75`. **Impact:** none on A3 v2 — the user decided on 2026-10-22
  that the Mac Studio's venv is Python 3.12 from this branch's pins (prompt 06 §0.2). It is a
  `ray` pin move with its own rehearsal, which `CLAUDE.md` keeps as this repository's to make.
  **Next step:** the user's — "we should move this checkout onto it at some point soon"
  (2026-10-22); not before A3 v2 has ended. Indexed at `docs/OPEN_ISSUES.md` §1.23.

  **Assigned (2026-10-25):** to [`py313-venv`](../py313-venv/README.md), planned the same day
  after A3 v2 and prompt 10, on the user's decision to move both hosts before the queued analysis
  jobs. It moves both venvs to Python 3.13.16 with `ray` 2.59.0 and `SQLAlchemy` 2.0.55, rehearsed
  on each host (R6, R7). Its prompt 01 closes this entry.

  **Closed 2026-10-25 by py313-venv 01:** this branch is now on Python 3.13.16 (MacPorts), with
  `ray` **2.59.0**, later than `main`'s 2.45.0, and `SQLAlchemy` 2.0.55. The laptop's venv was
  rebuilt from `requirements.txt` and its freeze differs from 3.12's in those two lines only. R6
  reproduced `39809dca…` on all three of its stores, through the four launches and the hard-kill
  pair, equal to R3's recorded fingerprints. A merge of this branch into `main` resolves
  `requirements.txt:75` to this branch's line, `ray==2.59.0`. The Mac Studio's venv follows in
  py313-venv 02. Record:
  [`docs/py313-venv-verification.md`](../../docs/py313-venv-verification.md) §1;
  [`../py313-venv/logs/01-move-the-laptop-venv-to-python-3.13.md`](../py313-venv/logs/01-move-the-laptop-venv-to-python-3.13.md).
  Its row left `docs/OPEN_ISSUES.md` §1.23.

Closed by a direct commit, 2026-10-25:

- **[10-extract-gkwkb-data-never-closes-its-theta-figure]** *(opened 2026-10-25 by prompt 10's
  review, from the agent's `issues.md` item 3; not acted on, by `CLAUDE.md` rule 4)* — `plot_Gk` in
  `extract_GkWKB_data.py` closes its main figure (`plt.close()`, `:319`) and then opens a second for
  the theta plot (`:322`), which is never closed. Each plotting worker keeps one figure per pair, and
  matplotlib's "More than 20 figures have been opened" `RuntimeWarning` appears twice in prompt 10's
  GkWKB log; 08's run died before any worker reached 20. **Impact:** memory in the plotting workers
  only; all 1392 pairs were written. Pre-existing, not 09's. **Next step:** close the theta figure.
  Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Closed 2026-10-25 by a direct commit, outside the campaign's prompts (user's decision):**
  `plot_Gk` now calls `plt.close()` after saving the theta figure. `extract_TkWKB_data.py` had the
  same defect twice, its theta and friction figures, and both are closed the same way.
  `ComputeTargets/tests/test_readers_close_their_figures.py` reads every `extract_*_data.py` with
  `ast` and requires each `fig = plt.figure()` to be followed by a `plt.close()` in its block or an
  enclosing one, not past a loop. On the old readers it fails at exactly the three lines
  (`extract_GkWKB_data.py:322`, `extract_TkWKB_data.py:308` and `:330`). `ComputeTargets` 732 → 737;
  the known wall-clock flake failed with and without the change. Not re-run on A3 v2.

Closed by prompt 10, 2026-10-25:

- **[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]** *(opened
  2026-10-25 by prompt 08, on the Mac Studio; it stopped the prompt at §3, under §8)*.
  `extract_GkWKB_data.py --database var/datastores/handover-A3-v2-lambdacdm.sqlite --ray-address local
  --models LambdaCDM` **exits 1** after 24 s and 344 files.
  - **The error.** A `ray::plot_Gk()` task raises `OverflowError: cannot convert float infinity to
    integer`, from `extract_GkWKB_data.py:285` → `fig.savefig` (PDF backend) → matplotlib's tick
    formatter (`ticker.py:2327`, `_is_close_to_int`). Full traceback:
    `var/handover-08/extract/extract_GkWKB_data.log:32-101`.
  - **The failing pair**, identified by what it left behind.
    - `plots/full-range/k=1e+05-k-serial=1/zsource=1.3633e+16-z-serial=19.pdf` is a 1,039-byte
      fragment with no `.png` beside it and no CSV. It is the only full-range PDF without its PNG:
      50 PDFs against 49 PNGs and 49 CSVs.
    - `extract_GkWKB_data.py:285` is exactly that plot's `fig.savefig(fig_path)`, the full-range PDF.
    - So the failure is on `k = 1e5`/Mpc at `z_source = 1.3633e16` (serial 19), near the top of the
      source grid (whose largest is 2.0636e16), at the full-range log–log plot of `|G_WKB|`, `|G|`
      and the analytic forms.
    - A tick at infinity means an axis limit at infinity, so a plotted value there is presumably
      `0`, `inf` or `nan` on a log axis. **The data were not read to confirm this** (§8: stop and
      report).
  - **Corrected on the way:** this session first suspected the single-point Green's function at
    `z_source = 0.1259` (serial 1730, the first source redshift above `--zend 0.1`, whose CSV has
    one row). That pair's full-range and zoom plots were written, so it is **not** the failing one.
  - **Why no rehearsal saw it** (inferred). R0–R4 read rehearsal stores of 3 k over 1e5–1e6 at
    `--zend 250`, whose source grid differs. A3 v2 is the first store read at production scope.
  - **The store is not implicated.** Every store file had the same SHA-256, size and mtime before and
    after this reader and every other. `store fingerprint` afterwards matches the sidecar
    (`0e32f24a…`). The five other readers, and `extract_Gk_data` without `--models`, exit 0.
  - **Impact.** A3 v2's GkWKB read-back (CSVs and plots) is incomplete: it stops at `(k = 1e5, z_source = 1.3633e16)`,
    after 8 lookups and 50 of its plots. The other readers are unaffected. Nothing was recomputed or changed.
  - **Resolution of the stop.** The user ruled on 2026-10-25: **"Release the lock, close 08"**. 08
    closed with this issue open, and the lock was released.
  - **Next step.** Fix the reader, as its own work, and re-run `extract_GkWKB_data.py` on the A3 v2
    store. First establish what `(k = 1e5, z_source = 1.3633e16)` plots, read-only. **Not
    assigned.**
  **Diagnosed 2026-10-25 by the orchestrator, read-only on the Mac Studio** (`sqlite3`/`mode=ro` over
  the four shards; nothing written). **The data are finite; one stored diagnostic column is
  astronomically large.** For the failing pair (`GkWKBIntegration` serial 106 on shard 3, 86
  `GkWKBValue` rows, $z$ 0.1–1.6e9), `G_WKB` and `analytic_G_rad` agree and lie in 9.6e18–4.0e21,
  but **`analytic_G_w` reaches 7.57e+243**; it exceeds 1e30 on the 12 rows at $z \le 0.4$ (1.4e34 at
  $z = 0.1$, 2.1e57 at $z = 0.398$). The numeric branch (serial 346, 68 rows) is finite throughout.
  - **Why the column is that large.** `compute_analytic_G` (`ComputeTargets/analytic_Gk.py:5-24`)
    is the constant-$w$ Bessel form, with $b = (1-3w)/(1+3w)$ and order $\tfrac12 + b$, and
    `GkWKBIntegration` (`:544`) and `GkNumericIntegration` (`:475`) evaluate it at each response
    point with that point's `wBackground`. In LambdaCDM $w(z)$ crosses $-1/3$ at late times, where
    $1 + 3w \to 0$, the order diverges, and $Y_\nu$ of a large order is astronomically large. The
    form is outside its domain there: it is a comparison aid, valid for constant $w$.
  - **Why the reader dies.** It plots $|$`analytic_G_w`$|$ beside $|G|$ on log–log axes
    (`extract_GkWKB_data.py:205-266`, `set_loglog_axes` in `extract_common.py:426`), so the $y$-axis
    spans ~230 decades. Matplotlib's log locator then proposes tick locations past the largest
    double, which are `inf`, and `LogFormatter` calls `round(log10(inf))`: the traceback's
    `OverflowError` (`ticker.py:2327`). That last step is read from the traceback and matplotlib's
    behaviour, not reproduced.
  - **How widespread.** Over all four shards, `GkWKBValue` rows with $|$`analytic_G_w`$|$ in
    [1e30, 1e100): 11,432 rows on 3770 pairs; [1e100, 1e200): 1913 on 1913; [1e200, 1e308): **293 rows
    on 293 pairs**. The reader stopped at its first such pair. Every pair whose response grid reaches
    the $w = -1/3$ crossing is presumably exposed. No rehearsal could see it: their `--zend 250`
    never reaches $z < 1$.
  - **What it touches.** `analytic_G_w` feeds no computation: it is stored, copied into `GkSource`
    values (`GkSource.py:498-506`) and plotted or written to CSV by four readers. `G_WKB`, `G` and
    `QuadSourceIntegral` do not read it. **Not a defect of the store's science content.**
  - **Seen in passing, not part of this issue:** opened as
    `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`, below.
  - **Next step**, narrowed. Two decisions, either sufficient for the reader: (a) **reader-side**, plot
    `analytic_G_w` only where it is finite and within some decades of $|G|$, or where $|1+3w|$ is not
    small (no store change); or (b) **compute-side**, store `analytic_G_w` (nullable) as `NULL`
    outside the constant-$w$ form's domain, which changes new stores' payloads and their
    fingerprints, and leaves A3 v2 as it is. (a) is the fix for A3 v2's read-back in any case.
  **The user decided both (2026-10-25):** "plot only where the value is sensible, but also store
  `NULL` in future runs outside the valid range. Changing new store's fingerprints can never be an
  argument not to make a useful fix." **Assigned to prompt 09** (the code) **and prompt 10** (the
  re-read of A3 v2 on the Mac Studio, which closes it).
  (`var/handover-08/extract/` is on the Mac Studio; it was not copied.)
  **Narrowed 2026-10-25 by prompt 09: fixed in code at `1c19998`; closes when prompt 10 has
  re-read A3 v2.** `compute_analytic_G`/`Gprime` now return `None` outside $1+3w>0$ and on a value
  that is not finite, so new stores hold `NULL` there; every plotted `analytic_G_w` series in the four
  readers goes through `extract_common.sensible_log_points`, which drops `None`, non-finite and zero
  points and any more than three decades above the plot's $|G|$; `extract_GkWKB_data.py` applies
  `safe_fabs`, not `fabs`, to every analytic column. A3 v2 keeps what it stores; the reader fix is
  what reads it, and has not yet been run on it. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Closed 2026-10-25 by prompt 10**, on the Mac Studio at `57f67a9`: `extract_GkWKB_data.py` on
  the A3 v2 store exits 0 after 171 s and 9744 files, and its read-back is complete: 1392
  full-range PDFs, PNGs and CSVs, equal to its work list. The pair above has a 30,366 B PDF, its PNG
  and its CSV, with 35 of 154 `analytic_G_w` points dropped by `sensible_log_points`. The store's
  files are unchanged and its fingerprint is the sidecar's. Whether the pair is drawn at all is a
  matter of chance, which is `[10-the-readers-draw-their-work-lists-unseeded]` (§3). Record:
  `docs/handover/A3-V2-VERIFICATION.md` §4; log: `logs/10-…`.

Closed by prompt 09, 2026-10-25:

- **[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]** *(opened 2026-10-25 by the
  orchestrator's diagnosis of the issue above, read-only; not by a prompt)* —
  `extract_Gk_data.py:282-283` builds its CSV columns `analytic_G_w` and `analytic_Gprime_w` from
  `value.analytic_G_rad` and `value.analytic_Gprime_rad`, so every `extract_Gk_data` CSV labels the
  radiation form as the $w(z)$ form, and the $w(z)$ form is never written. The plots in the same
  reader (`:142-240`) use the right attribute. **Impact:** any reading of those CSVs that compares
  the two forms sees them identical. **Assigned to prompt 09** (2026-10-25), which closes it.
  Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Closed 2026-10-25 by prompt 09:** `extract_Gk_data.py:299-300` now builds `analytic_G_w_column`
  and `analytic_Gprime_w_column` from `value.analytic_G_w` and `value.analytic_Gprime_w`, so its CSVs
  carry the $w(z)$ form under its own name; a `None` (new stores, outside $1+3w>0$) is written as an
  empty cell. Guarded by `ComputeTargets/tests/test_analytic_g_w_readers.py`
  `test_the_csv_maps_each_w_column_to_its_own_attribute`, which caught the mislabel restored as
  deliberate breakage (e). Log: `logs/09-…`.

Closed on the user's decision, 2026-10-25 (a `.gitignore` line):

- **[06-nothing-tracked-ignores-venv]** *(opened 2026-10-22 by prompt 06, on the Mac Studio)* —
  **nothing this repository tracks ignores `venv/`.** `handover-remedial`'s `.gitignore` holds only
  `var/` (`main`'s is empty), so a checkout with a venv in it is clean only if something untracked
  ignores it: on the laptop `.git/info/exclude:40` is `venv`; on the Mac Studio the Python 3.13 venv
  carried its own `venv/.gitignore` (`*`), which `python3.13 -m venv` writes and `python3.12 -m venv`
  does not. The new 3.12 venv therefore showed `?? venv/`. **Impact:** `runregistrykit` 0.2.0 stamps
  `git_dirty = bool(git status --porcelain)`, which counts untracked files, so every manifest of a run
  launched from such a checkout records `git_dirty: true` — a stop for every rehearsal, and a dirty
  provenance for A3 v2. No R4 run was affected: 06 stopped before any launch, and on the
  orchestrator's ruling appended `venv` to the Mac Studio checkout's `.git/info/exclude` (line 28),
  untracked; all four manifests record `git_dirty: false`. A fresh clone meets it again.
  **Next step:** add `venv/` to the tracked `.gitignore` in a commit of its own, or record that every
  checkout must carry the exclude line; not in 06, and not during A3 v2 (the Mac Studio's checkout is
  frozen while it runs, and a `.gitignore` commit would not be on it). Measurement: `logs/06-…`,
  deviation 2. Not assigned. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Closed 2026-10-25, on the user's decision the same day:** `.gitignore` now names `venv/`, with a
  comment saying why, so a checkout with a venv in it is clean whatever wrote the venv and whatever
  its `.git/info/exclude` holds. The laptop's and the Mac Studio's exclude lines are harmless and
  stay. Checked: `git check-ignore -v venv` resolves through `.gitignore` before
  `.git/info/exclude`.

Closed by prompt 08's review, 2026-10-25:

- **[a3-baseline-quadsource-integrals-are-1680-short]** *(opened 2026-09-23 by the A3 baseline run,
  not by a prompt)* — the LambdaCDM baseline store holds **7600 of the 9280**
  `QuadSourceIntegral` objects its own geometry defines, and the run that was filling it was
  **stopped by the user on 2026-09-23** rather than allowed to finish. The count is exact and is
  set arithmetic on store serials, not on floats: 145 response redshifts × 64 triangle-closing
  `(k, q, r)` triples = 9280; the store holds 7600; **1680 are missing**. (A first pass over
  round-tripped CSV floats gave 2581 and was wrong — `redshift.z` does not survive a CSV round
  trip, and 901 rows appeared to be off-grid when they were not.)

  **What is missing is the expensive end, entirely.** All 1680 are at `z_response ≤ 174.1`, and
  **1024 of them at `z_response ≤ 6.32` have no row at all** — that block has never been attempted.
  Its $\eta_R$ runs out to **13,728** at $z = 0.1$ (main.py's own "latest tau" line) against a
  largest *measured* $\eta_R$ of **4936** at $z = 8.33$, so the untouched block is a factor 2.8
  beyond anything the cost model above is fitted on. Two independent projections of the remainder
  at the shipped settings: a per-triple power law in $\eta_R$, extrapolated and capped at 4× each
  triple's largest measured cost, gives **1021 CPU-hr ≈ 8.2 wall-days** at the 5.2 effective CPUs
  the run achieved; the flat rate from the run's own last session (48 items in 6 h 55 m at a mean
  of 2710 s) gives **~10.1 wall-days**, and that is a *lower* bound because that session only
  reached $z \ge 8.3$. About **300 of the 1680** — the squeezed triples at
  $k = 3.1\times10^6$, $9.7\times10^6$ and $3.0\times10^7$ — carry roughly **90 %** of it.

  **Impact.** **E1** and **B2**, which need this store as the comparator for the hand-over change,
  and **A3** itself, whose policy rows are complete but whose integral outcomes are not. The store
  is *usable* — 7600 rows over the full $k$ grid and the whole high-$z$ range — but it is **not a
  complete response-redshift grid** and nothing that assumes one may be run against it.

  **Superseded 2026-09-23 by a user decision: the store is to be regenerated from scratch** at
  the tolerance pair the sweep above settles, rather than finished at the shipped one. That closes
  three things at once — the 1680, the 889 unconverged rows of the issue above, and **54 rows the
  first run of the sweep wrote into this store in error** (a copied `ShardedPool` primary still
  names the *source* store's shard files, fixed in `2ebb7b6`; the rows are at six non-production
  pairs and the production population was left at exactly 7600, so nothing was lost). Rebuilding
  costs **~16 minutes** for everything upstream of the integral — measured from this store's own
  timestamps, 21:02:30 to the first `QuadSourceIntegral` row at 21:18:24, covering the background
  model, both $T_k$ sectors, both $G_k$ sectors, `GkSource` and the policies — so there is no case
  for reusing any of it.

  **Note (2026-09-26): the sweep store has been retired.** `var/datastores/handover-atol-sweep`,
  the working copy whose first run wrote the 54 rows above, was retired by the user with
  `store retire` on 2026-09-25 (completed `23:35:57`), under
  [`store-retirement`](../store-retirement/IMPLEMENTATION_STATE.md) prompt 05. Its four shards and
  its primary are gone. Its sidecar stays as the tombstone, recording when, why and the
  fingerprint of what it held. The four sweep runs still name its path, and following them now
  lands on that tombstone. So `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md:15-17`, which says the
  store was "**deleted after this document was written**", is now true. It was written in
  anticipation, and it was not edited. The 54 rows in the live A3 store are unaffected, and so is
  that store. Record:
  [`../store-retirement/logs/05-retire-the-two-stores.md`](../store-retirement/logs/05-retire-the-two-stores.md).

  **Note (2026-10-04): the A3 baseline has been retired too.** `var/datastores/handover-A3-baseline-lambdacdm`
  was retired by the user with `store retire` on 2026-10-04 (completed `21:31:59`). The registry
  ran at `81c0a9a`, because at `HEAD` its format-1 fingerprint could not be compared and its rows
  could no longer be read; at `81c0a9a` a fresh fingerprint matched, digest `433b7fc3…`. Its four
  shards and its primary are gone; its sidecar is the tombstone. The measurements in the three
  `a3-baseline-*` entries were taken from it and stand; nothing can re-take them, and the
  regeneration (A3 v2) is what they point to. Recorded in
  [`../datastore-generic/IMPLEMENTATION_STATE.md`](../datastore-generic/IMPLEMENTATION_STATE.md)
  §3, the fallbacks issue.

  **Next step:** do **not** restart it at the shipped tolerance pair. The sweep named in the first
  issue above decides the pair; if the pair moves, the whole store is regenerated anyway and the
  1680 are moot, and if it does not, the 8–10 days is the honest price and is a decision for
  whoever owns E1. The run itself is recorded at
  `var/runs/handover-03-a3-baseline-resume-20260923T024847` (state `killed`, stage
  "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31%" — that percentage counts **batches of 750
  dispatched**, not integrals completed, which is what made the remaining work look small). All
  four shards pass `PRAGMA quick_check` after the stop. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Closed 2026-10-25 by prompt 08's review, on the user's decision the same day.** The
  regeneration this entry's next step called for was made: A3 v2
  (`var/datastores/handover-A3-v2-lambdacdm.sqlite` on the Mac Studio, built by
  `handover-07-a3-v2-20261022T211155` at `b90752d`, fingerprint `0e32f24a…`) holds **all 9280**
  `QuadSourceIntegral` rows at `(1e-32, 1e-7)`, 2320 on each shard, among them every one of the 1088
  low-band items ($z_{\rm response} < 9$), the 1024 at $z \le 6.32$ included. The price was 48.74
  wall-hours and 286.3 CPU-hours, against this entry's 8–10 wall-days at the shipped pair.
  Record: `docs/handover/A3-V2-VERIFICATION.md` §3; `logs/08-after-the-a3-v2-build.md`.

Closed by `rrk-v020-adoption` prompt 03, 2026-10-19:

- **[a3-v2-a-hard-killed-run-blocks-its-store]** *(opened 2026-10-16 by a measurement ahead of the
  A3 v2 launch, not by a prompt; a `runregistrykit` defect at `v0.1.0`, drafted for RunRegistryKit
  in [`runregistrykit-stale-run-brief.md`](runregistrykit-stale-run-brief.md))* — after a hard
  kill, the registry blocks the store's fingerprint, copy, move, retire and amend, and no registry
  command clears the block. A run killed by `SIGKILL`, power loss, a reboot or the OOM killer never
  reaches `Run.finish`, so its `status.json` says `state: running` for ever. `list` shows it stale
  (`!!`), correctly. But every store operation that checks for a running run counts it "alive
  **or stale**": `_running_runs_naming` (`runregistrykit/stores.py:998`, the filter at `:1017`),
  called by copy and move (`_refuse_if_in_use`, `:1029`), fingerprint (`:1338`), retire (`:1664`)
  and amend (`:2017`). A later run on the same store finishes with `fingerprint=True`. It excludes
  only itself (`taken_by=self`, `__init__.py:476`), so it records `fingerprint_error` and never
  writes the sidecar. `begin` (`__init__.py:604`) refuses a tombstone (`:686-698`) and nothing else,
  so it does not refuse a new run on a store that a `running` run names.

  **This is by design, and the remedy is incomplete.** Each refusal names its own remedy: "a
  stale run is ended by a person, with Run.finish" (`stores.py:1038`, `:1344`, `:1667`, `:2020`).
  RunRegistryKit tests that message (`tests/test_store_retire.py:436`, against a stand-in dead pid).
  But `Run.finish` is the Python API only. The command line has `list` and `store
  show|create|adopt|copy|move|fingerprint|retire|amend` (`__main__.py:418-522`), and none of them
  writes a `status.json`. `Run.finish` itself checks neither the pid nor the heartbeat
  (`__init__.py:424-459`), so it would end a live run just as readily.

  **Measured 2026-10-16** at SGK `f1a7de3` (clean), with `runregistrykit` 0.1.0 (commit `9f8ebe4`;
  the installed `__init__.py`, `stores.py` and `__main__.py` are byte-identical to the tag). There
  was no Ray and no build. The store was a throwaway `StoreFixtures.real_store_fixtures` store,
  `var/datastores/handover-stale-run-probe.sqlite` (two shards, `store_id` `dab353cd…`), and the
  runs went in the main checkout's `var/runs/`. The sequence:
  1. `store create`, then `store fingerprint --write`: digest `d9328538…`.
  2. A child process ran `runregistrykit.begin(..., results=<store>)` and one heartbeat, and was
     sent `SIGKILL` (returncode −9). Its `status.json` was unchanged by the kill: `state: running`,
     `pid 88637`, `exit_code: null`.
  3. `list` showed `!! handover-stale-run-probe-killed-20261016T110844  running`, then "44 run(s)
     … 41 finished, 1 stale, 2 unknown".
  4. A second `begin` on the same store **was not refused** (`results_store_id` `dab353cd…`).
     `finish("done", exit_code=0, fingerprint=True)` returned `state: done` with
     `fingerprint_error: "RuntimeError: Cannot fingerprint …: run
     handover-stale-run-probe-killed-20261016T110844 is running (stale, pid 88637) and names this
     store by its results and results_store_id. … Nothing was read or written"`. There was no
     `fingerprint_sidecar` field.
  5. `store fingerprint --write`, `store retire --dry-run`, `store retire --without-fingerprint
     --dry-run` and `store copy` each exited 1, with the same "is running (stale, pid 88637)"
     refusal. The sidecar's SHA-256 was unchanged by all four.
  6. Then `kill -0 88637` gave "no such process", and a person's
     `runregistrykit.Run("<run dir>").finish("killed", exit_code=137)` ended the run. After that,
     `list` showed it `killed` with no `!!`, `store fingerprint` matched `d9328538…`, `store retire
     --dry-run` passed, and `store retire` deleted the primary and its two shards. The sidecar is
     the tombstone, `completed 2026-10-16T11:11:25+01:00`. The two probe run directories stay in
     `var/runs/` as `killed` and `done`, and nothing under `var/datastores/` is live. The full
     transcript is not committed; the outputs quoted here are exact.

  **Impact on A3 v2.** A `SIGTERM` is unaffected: the driver's handler ends the run `killed`/143,
  as rehearsed (`docs/runregistry-extraction-verification.md` §3.2–3.3). After a **hard** kill,
  the store **can be resumed**. Neither `begin` nor the driver's `--resume` pre-check
  (`docs/handover/quadsource_atol_sweep.py:535-545`, which reads the store and not the registry)
  refuses. The resumed run ends `done`. But it **cannot be fingerprinted, copied, moved, retired
  or amended through any registry command**. The resumed run's own
  `finish(..., fingerprint=True)` (`docs/gktk-remedial/scoped_pipeline_run.py:476`, through
  `quadsource_atol_sweep.py:654-656`) records `fingerprint_error`, and the sidecar keeps whatever
  fingerprint it had before: none, for a new A3 v2 store. The only route out is the one the
  refusal names, step 6 above: a person calls the package's Python API on the dead run's
  directory, after checking `kill -0` by hand, and then runs `store fingerprint --write`. That is
  the package's documented API and not a patch. Whether a person uses it is the user's call; SGK
  adds no code for it. The block does not stop the launch or a resume, and it costs no rows. The
  related inherited issue `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`
  (now RunRegistryKit's) can leave a `SIGTERM`'d run reading `running` by another route, and would
  meet the same block.

  **Next step:** the brief above is for the user to hand to RunRegistryKit. It names options, and
  RunRegistryKit's choice decides the fix. Until RunRegistryKit's index lists the issue, it stays
  open here. Then it moves to §4 as transferred, and its row leaves `docs/OPEN_ISSUES.md` §1.1.
  A fix reaches SGK by a pin move, in a commit of its own. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Assigned (2026-10-18): [`prompts/rrk-v020-adoption`](../rrk-v020-adoption/README.md).** The
  user handed the brief over, and RunRegistryKit's `interrupted-runs` campaign answered it: `end`
  (its prompt 04) and `begin`'s refusal of a store an unfinished run names (05), released as
  `runregistrykit` `v0.2.0` on 2026-10-18. Its board indexed and closed the issue, so it is not
  open there to be transferred. The fix reaches SGK by that campaign's pin, its prompt 01. Its
  prompt 03 closes this entry, once the route of `end` and then the resume has been shown on A3
  v2's own launcher, a build killed by `SIGKILL` (its README §2.2, 03, acceptance 3). The index
  row has moved to `docs/OPEN_ISSUES.md` §1.22.

  **Resolution (2026-10-19).** Closed by `rrk-v020-adoption` prompt 03, at `17916cb`: the route
  of `end` and then the resume was shown on A3 v2's own launcher, `docs/handover/quadsource_atol_sweep.py`,
  with `runregistrykit` `v0.2.0` (`0bae3a3c…`) installed by that campaign's prompt 01. The build
  `rrk-v020-adoption-03-hardkill-a-20261018T234619` was killed by `SIGKILL` in `CALCULATE QUADRATIC
  SOURCE INTEGRALS`; its status stayed `running` and `list` marked it stale, `!!`. A `--resume` was
  then refused by `v0.2.0`'s `begin` (`runregistrykit/__init__.py:1119` at the tag), naming the dead
  run, `stale` and `python -m runregistrykit end`, with no run directory created and every store
  file byte-identical. `v0.2.0`'s `end` recorded the dead run `lost`, verified (evidence `host`
  `this`, `boot` `this`, `pid_answers` `false`), and refused a second `end`. The resume,
  `rrk-v020-adoption-03-hardkill-b-20261019T001145`, then ended `done` with its fingerprint
  `39809dca…` recorded and `fingerprint_sidecar` `written`: step 4 above, which `v0.1.0` refused.
  The store equals the reference store and R2's recorded fingerprints and passes every read-only
  check. `Run.finish` was called by hand on no run. Record:
  [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §2 and
  [`../rrk-v020-adoption/logs/03-rehearse-through-v0.2.0.md`](../rrk-v020-adoption/logs/03-rehearse-through-v0.2.0.md).
  The row leaves `docs/OPEN_ISSUES.md` §1.22. The same rehearsal found that Ray's GCS server can
  outlive a `SIGKILL`ed driver; that is a new issue on the `rrk-v020-adoption` board,
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`, and not part of this one.

Closed by prompt 05, 2026-10-16:

- **[00-main-and-the-readers-default-to-ray-address-auto]** — *opened 2026-10-12 by the
  `runregistry-extraction` plan, **held by that board**, and assigned to this campaign by the user
  on 2026-10-12, since it owns the A3 v2 launch.* `main.py` and the six readers defaulted
  `--ray-address` to `auto`, which attaches to any Ray instance on the machine.

  **Resolution.** The seven defaults are `local`, which starts a fresh instance and ignores
  `RAY_ADDRESS`; the flag stays optional, because `quadsource_atol_sweep._main_py_argv` passes
  none. `ComputeTargets/tests/test_ray_address_default.py` pins the defaults, the single
  `ray.init(address=args.ray_address)` in each script, and Ray's resolution of `local`. **Closed on
  the `runregistry-extraction` board's §4**, which holds the resolution in full; the row is deleted
  from `docs/OPEN_ISSUES.md` §1.1. Record:
  [`logs/05-ray-address-defaults-to-local.md`](logs/05-ray-address-defaults-to-local.md).

Closed by prompt 01, 2026-09-20:

- **[01-general-w-normalisation-is-predicted-not-measured]** — *opened 2026-09-18 with the
  Kohri–Terada audit, and **owned by the `radiation-oracle` board**, where it is closed.* That
  campaign's oracle covers $b = 0$ only, so $N = -9/8$ was measured there and the general-$w$
  $N(b) = -(3+2b)^2/(2(2+b)^2)$ was **predicted and unchecked**, with the nine $b = 0.2$ fixture
  cases unscored.

  **Resolution.** Prompt 01 measured it. `total_from_I_rev` scores
  `evaluate_QuadSource_integral`'s `total` against a quadrature of Domènech (4.11)
  `eq:Isimpledef` inside the corrected (4.10), integrated between the code's own limits, on all
  nine $b = 0.2$ cases of `test_quadsource_integral` in the exact flavour at the reference pair
  $(10^{-45}, 10^{-12})$. **$N = -1.194214876033$, constant: min and max agree to 5.285e-13
  (4.425e-13 relative), over $x$ from 5.3 to $1.9\times10^3$, $u$ from 0.01 to 10, and both
  Gervois–Navelet branches.** The worst deviation from the derived
  $-\frac{(3+2b)^2}{2(2+b)^2} = -1.194214876033$ is **4.130e-13**, and against $I_{\rm rev}$ —
  where the prediction is $-1$ exactly — **3.461e-13**. Every per-case deviation is inside the sum
  of the two sides' own declared errors (the quadrature's 1.7e-14 to 6.9e-12, the pipeline's
  1.0e-12 to 1.1e-11). The constancy is the load-bearing statistic
  (`KOHRI-TERADA-ORACLE.md` §0) and it is what the test asserts first; the value is asserted
  separately, and prompt 01's breakage 1 shows the two failing independently. This also removes
  that issue's stated obstacle — its "next step" asked whether a general-$w$ quadrature oracle
  built from the code's own $\Phi$ was worth having "given that it shares machinery with the thing
  it checks". It does **not** share machinery: `domenech.py` is a transcription of two published
  papers, with the outer Bessels from `scipy.special` and the Legendre sector from `mpmath`, and
  it is bridged at $b = 0$ to `kohri_terada.py`, which a different campaign transcribed from a
  third paper. **Closed on the `radiation-oracle` board's §4**, per that campaign's ownership and
  campaign README §5 rule 4; the row is deleted from `docs/OPEN_ISSUES.md` §1.9, which is now
  empty. Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Verification
  performed", acceptance item 4.
