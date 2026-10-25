# Campaign index

One line per campaign under `prompts/`. It says which campaigns are live, when each opened and
was last touched, what it owns, and where its open issues are indexed. It is an index: the
campaign's `README.md` holds the plan and its `IMPLEMENTATION_STATE.md` the status; if this file
and a board disagree, the board is right. Add a line when a campaign folder is created and
change its status when the board closes it. The open-issue column is a snapshot from
[`docs/OPEN_ISSUES.md`](../docs/OPEN_ISSUES.md) on the date in the header; the section pointer
is the durable part.

**Last updated:** 2026-10-25 · 31 campaigns: 3 live, 1 seeded, 27 closed or complete.
Open-issue counts were last taken on 2026-09-29, except `a3-v2-readiness`'s and
`datastore-integrity`'s, taken on 2026-10-02, `datastore-generic`'s, taken on 2026-10-06, and
`datastore-generic-followup`'s, taken on 2026-10-07, `dk-and-rpk-migration`'s, taken on
2026-10-12, `runregistry-extraction`'s, taken on 2026-10-16, `rrk-v020-adoption`'s, taken on
2026-10-19, `dk-v050-adoption`'s, taken on 2026-10-22, and `rpk-v030-adoption`'s, `py313-venv`'s and
`levin-phase-floor`'s, taken on 2026-10-25.

| Campaign | Status | Opened → last board update | Owns | Open issues (§ of `OPEN_ISSUES.md`; rows naming its board) |
|---|---|---|---|---|
| [`levin-phase-floor`](levin-phase-floor/IMPLEMENTATION_STATE.md) | **live** — 1 of 4 measurement prompts landed: 01 (A3 v2 phase-group census, Mac Studio, 2026-10-25): H-cancel fails, H-anchor holds; 02 and 03 written; 04 held; remedy after gate G1 | 2026-10-25 | A3 v2's depth-20 Levin cost cliff and the honesty of a `QuadSourceIntegral` row's stored error and convergence; findings §2.6's test as two hypotheses (H-cancel, H-anchor), then §2.7's remedy | §1.24; 2 of its own (and 4 assigned from `handover` and `GkTk-remedial`, held there) |
| [`py313-venv`](py313-venv/IMPLEMENTATION_STATE.md) | closed 2 / 2, 2026-10-25 | 2026-10-25 | the venv on the laptop and the Mac Studio moved from Python 3.12.15 to 3.13.16, `ray` 2.43.0 → 2.59.0, `SQLAlchemy` 2.0.39 → 2.0.55; rehearsals R6 (with the hard-kill pair) and R7; A3 v2 read under the new venv | §1.23; 0 (the one assigned from `handover` closed by 01; the three opened closed) |
| [`rpk-v030-adoption`](rpk-v030-adoption/IMPLEMENTATION_STATE.md) | closed 1 / 1, 2026-10-25 | 2026-10-25 | SGK's move from `raypoolkit` `v0.2.0` to `v0.3.0`: the package's `rehearsal_lock` replaces `tools/rehearsal_lock.py`; every Ray start given `/tmp/ray-sgk` | —; 0 |
| [`dk-v050-adoption`](dk-v050-adoption/IMPLEMENTATION_STATE.md) | closed 1 / 1, 2026-10-22 | 2026-10-22 | SGK's move from `datastorekit` `v0.4.0` to `v0.5.0`; the spec guard's condition for an empty `extra_values` | —; 0 |
| [`rrk-v020-adoption`](rrk-v020-adoption/IMPLEMENTATION_STATE.md) | closed 5 / 5, 2026-10-19 (closed at 04, reopened for 05: C3 reversed, clean refusals) | 2026-10-18 → 2026-10-19 | SGK's move from `runregistrykit` `v0.1.0` to `v0.2.0`; the route after a hard kill (`end`, then the resume) in `CLAUDE.md`; rehearsal R3 and the hard-kill pair on the A3 v2 launcher | §1.1, §1.22; 1 (opened by 03, narrowed by 04, assigned to `handover` 2026-10-19; the one assigned from `handover` closed by 03) |
| [`runregistry-extraction`](runregistry-extraction/IMPLEMENTATION_STATE.md) | closed 10 / 10, 2026-10-16 | 2026-10-12 → 2026-10-16 | `RunRegistry/` frozen, then replaced by the pinned `runregistrykit`; the fingerprint's encoding versioned in DatastoreKit; SGK's `config/fingerprint.py`; rehearsals R0–R2; the two briefs; `tools/rehearsal_lock.py` | §1.1, §1.21; 1 |
| [`dk-and-rpk-migration`](dk-and-rpk-migration/IMPLEMENTATION_STATE.md) | closed 9 / 9, 2026-10-12 | 2026-10-11 → 2026-10-12 | SGK's adoption of `datastorekit` `v0.2.2` and `raypoolkit` `v0.2.0`, DatastoreKit first; the fingerprint rehearsals R0–R2; each package's gate G2 | §1.20; 2 |
| [`perturbation-sound-speed`](perturbation-sound-speed/AUDIT.md) | **seeded** — audit and script only, no README or board | 2026-09-29 | the $c_s^2$ the transfer-function sector reads: $dp/d\rho$, not $p/\rho$, on the QCD background | §1.16; 1 |
| [`datastore-generic-followup`](datastore-generic-followup/IMPLEMENTATION_STATE.md) | closed 3 / 3, 2026-10-07 | 2026-10-06 → 2026-10-07 | the user's decisions on `datastore-generic`'s three open issues: a `--drop` that leaves references refused; `resolve()`'s absent parent as `None`; the closed-store prefix stated once | §1.19; 0 |
| [`datastore-generic`](datastore-generic/IMPLEMENTATION_STATE.md) | closed 9 / 9, 2026-10-06 | 2026-10-04 → 2026-10-07 | the datastore layer made generic again (no project table named in the layer, guarded by `test_layer_is_generic`); every old-store fallback deleted | §1.18, §1.19; 0 |
| [`a3-v2-readiness`](a3-v2-readiness/IMPLEMENTATION_STATE.md) | closed 10 / 10, 2026-10-04 | 2026-10-02 → 2026-10-04 | the seven defects a build or a reader of the A3 v2 store would meet; `DEFAULT_QUADRATURE_RTOL = 1e-7` | §1.17; 2 |
| [`datastore-integrity`](datastore-integrity/IMPLEMENTATION_STATE.md) | closed 12 / 12, 2026-10-02 | 2026-09-25 → 2026-10-02 | every known datastore-integrity defect, before the A3 v2 store is built | §1.15, §1.17; 12 |
| [`handover`](handover/IMPLEMENTATION_STATE.md) | **live** — 3 of 10 groupings landed, and 05; 06 (A3 v2 host, R4) landed 2026-10-22, R4 holds on the Mac Studio; 07 launched A3 v2, which ended `done` 2026-10-24; 08 landed 2026-10-25: A3 v2 kept and verified on the Mac Studio; 09–10 landed 2026-10-25; 11 (A3 v2 seam record) landed 2026-10-25; D1 decided (b) and B1 written as prompt 12; three §3 issues assigned to `levin-phase-floor` | 2026-09-20 → 2026-10-25 | the numeric → Liouville–Green seam of $T_k$ and $G_k$; the Domènech kernel reconnaissance; the general-$b$ oracle; the A3 v2 launch on the Mac Studio | §1.1; 10 |
| [`datastore-readback`](datastore-readback/IMPLEMENTATION_STATE.md) | **live** — 1 of 1 written landed, resumable A3 run is the user's | 2026-09-22 → 2026-09-26 | the read-back guard and the audit of every object factory | §1.10; 0 |
| [`store-fingerprint`](store-fingerprint/IMPLEMENTATION_STATE.md) | complete 5 / 5, closed 2026-09-25 | 2026-09-24 → 2026-09-29 | read-only store reader, structured inventory, content fingerprint in the sidecar | §1.13; 1 |
| [`store-retirement`](store-retirement/IMPLEMENTATION_STATE.md) | complete 5 / 5 | 2026-09-25 → 2026-09-26 | `store retire`: a store's own files go, its sidecar stays | §1.14; 4 |
| [`datastore-portability`](datastore-portability/IMPLEMENTATION_STATE.md) | complete 3 / 3 | 2026-09-24 → 2026-09-25 | a moved store fails closed instead of recreating empty shards | §1.12; 0 |
| [`run-registry`](run-registry/IMPLEMENTATION_STATE.md) | complete 3 / 3 | 2026-09-22 → 2026-09-25 | `RunRegistry/`, `var/runs/`, the liveness rule, the six rules in `CLAUDE.md` | §1.11; 3 |
| [`radiation-oracle`](radiation-oracle/IMPLEMENTATION_STATE.md) | complete 1 / 1 | 2026-09-18 → 2026-09-20 | the Kohri–Terada radiation oracle for the source integral | §1.9; 0 |
| [`test-suite-runtime`](test-suite-runtime/IMPLEMENTATION_STATE.md) | complete 1 / 1 | 2026-09-19 | the suite's wall-clock; the 3-Bessel plot cost | —; 0 |
| [`tolerance-convergence`](tolerance-convergence/IMPLEMENTATION_STATE.md) | closed 6 / 6 plus insertions | 2026-09-12 → 2026-09-18 | every accuracy knob in the pipeline, calibrated; `TOLERANCE-CONVERGENCE.md` and `TOLERANCE-PROVENANCE.md` | §1.5; 14 |
| [`background-solver-robustness`](background-solver-robustness/IMPLEMENTATION_STATE.md) | complete 9 / 9 | 2026-09-16 | the root solves and evaluation path of `LambdaCDM_GenericEOS` | §1.8; 3 |
| [`qcd-background-audit`](qcd-background-audit/IMPLEMENTATION_STATE.md) | closed 16 / 16 | 2026-09-13 → 2026-09-16 | `T(z)` on `QCD_Cosmology`, the break-point set, the source grid | §1.7; 9 |
| [`phase-representation`](phase-representation/IMPLEMENTATION_STATE.md) | closed at 1 / 2 | 2026-09-13 | the primitive-phase representation; prompt 02 blocked on the knot vector | §1.6; 2 |
| [`GkTk-remedial`](GkTk-remedial/IMPLEMENTATION_STATE.md) | closed, 20 prompts | 2026-09-09 → 2026-09-29 | the shared conformal-time primitive, the $\tau$ tables, the phase residual; the $G_k$/$T_k$ WKB sectors | §1.4; 17 |
| [`transfer-remedial`](transfer-remedial/IMPLEMENTATION_STATE.md) | complete | 2026-09 → 2026-09-19 | the Bessel amplitude and phase construction, `LiouvilleGreen/bessel_phase.py` | —; 8 |
| [`qsi-phase-groups`](qsi-phase-groups/IMPLEMENTATION_STATE.md) | complete 1 / 1 | 2026-09-11 | the phase-group Levin integration in `QuadSourceIntegral` | §1.2; 2 |
| [`source-remediation`](source-remediation/IMPLEMENTATION_STATE.md) | complete 13 / 13 | 2026-09 → 2026-09-14 | sound speed, source term, source time integral; `ComputeTargets/phase_groups.py` | —; 10 |
| [`spec-transcription`](spec-transcription/README.md) | complete; all spec files signed off 2026-09-07 | 2026-09-07 | `docs/spec/`, the transcribed derivation | —; 0 |
| [`levin-refactor`](levin-refactor/IMPLEMENTATION_STATE.md) | complete 10 / 10 | 2026-09 → 2026-09-04 | `AdaptiveLevin/levin_quadrature.py` | §1.3; 7 |
| [`backport-modules`](backport-modules/IMPLEMENTATION_STATE.md) | complete 10 / 10 | 2026-09 → 2026-09-04 | the shared Ray and Datastore infrastructure modules | —; 2 |

Planning and fact-finding material that belongs to no campaign yet lives under
[`docs/one-loop-planning/`](../docs/one-loop-planning/), for the `OneLoopIntegral` campaign to
come.
