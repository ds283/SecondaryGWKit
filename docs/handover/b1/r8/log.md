## §6 — rehearsal R8 on the Mac Studio

**Prompt:** [`prompts/handover/12-remove-the-tk-gap.md`](../12-remove-the-tk-gap.md) §6, under
the orchestrator's [`prompt-12.md`](../orchestrator/prompt-12.md) §B.1 and §B.5.
**Host:** mps026298 (Mac Studio), checkout at the dispatch SHA `8a02f0c` (B1's code `049c7c8`).
**Model:** Claude Opus 5.5, as a subagent of the Mac Studio orchestrator, in two turns: §6.1–§6.3
and launch; then §6.4–§6.7 after the orchestrator reported the run's end.
**Date:** 2026-10-25, 21:02–21:20 UTC.
**Commit:** none. The records are committed on the laptop in B1's records commit (§B.4).
**Result:** COMPLETE WITH DEVIATIONS.
- R8 ran to `done`, exit 0.
- Its fingerprint, `1b332ffa…`, differs from R3's `39809dca…` in `QuadSource` alone.
- All 864 `QuadSourceIntegral` rows carry `partition.handover`, and none records a clamp.
- Each `QuadSource` row gained 3 nodes, and its values at R3's nodes are bit-identical.
- The six readers exited 0 and left the store byte-identical.
- The lock was released.
- No §8 stop condition fired.

The dispatch expected `QuadSourceIntegral` to differ in the fingerprint too, and it does not. That
is explained by the fingerprint's design, not by the build (deviation 7). It is put to the
orchestrator as a question.

The measurements are in `var/handover-12/verification.md`, for `docs/handover/HANDOVER-B1.md` §5.

### What was done

1. **§6.1.** Re-checked §B.5's facts read-only. All matched; the laptop had already moved the
   checkout to `8a02f0c`. Then ran the `ComputeTargets` and `ObjectFactories` roots: 742 OK and
   280 OK, equal to the laptop's counts.
2. **§6.2.** `acquire`, exit 0.
3. **§6.3.** Launched §B.1's command verbatim (`sh` 21715). Found the driver as the child of the
   `sh` (`pgrep -P 21715`): 21718. Ran `set-pid 21718`, exit 0. One check: the registry showed the
   run `running` with pid 21718, and the launch log had no traceback. The turn then ended. The
   orchestrator waited on the pid. The driver exited 0 at about 21:09:24 UTC; the run took about
   3.5 minutes.
4. **§6.4.** Hashed the store's five files. Ran `compare_with_recorded.py` against R3's recorded
   reference: exit 1, one difference, `QuadSource`'s tag set ("6 recorded, 6 now", the value
   count). Recorded R8's digest as the rehearsal reference.
5. **§6.5.**
   - `scripts/r8_rows.py`, `mode=ro`: 864/864 rows have `partition.handover`, and 0 rows have any
     key containing `clamp`. The continuity table is by $k$ and node. `z_switch` is the first
     source-grid node below `z_init` for all three $k$. Each `QuadSource` row gained 3 nodes against
     R4's (R3-equal) extract, 0 were removed, and the values at common nodes are identical.
   - Supplement, `scripts/compare_qsi_extract.py`: R8's and R4's `QuadSourceIntegral` values on
     the one common extracted triple.
6. **§6.6.** `run_readers.sh`, six readers, every exit 0. Every reader left the store identical.
   The store's SHA-256 after all reads equals the hash taken before §6.4.
7. **§6.7.** `ray-processes` empty (exit 0), `release` exit 0, `status`: free, no Ray processes.

### Deviations from the prompt

#### 1. The checkout was not fast-forwarded by this agent — STRUCTURALLY REQUIRED

§6.1 says "Fast-forward to the code commit". The dispatch forbids any git write. §B.5 says the
laptop fast-forwards the checkout before the orchestrator starts. It was at `8a02f0c` with a clean
status, so this agent checked it and did not move it.

#### 2. "R3's extent" is R4's extract — STRUCTURALLY REQUIRED

R3's store is retired; only its run directory and fingerprint remain. §6.5's `QuadSource`
comparison uses the CSVs `extract_tensor_source_data` wrote from handover 06's resumed store, R4,
whose fingerprint equalled R3's. That fingerprint covers identities and value counts, not values.
So "values unchanged at R3's nodes" means unchanged against R4's values, which were written as
`repr` floats and so compare bit-for-bit.

#### 3. A supplementary integral comparison — IMPLEMENTATION CHOICE

§6.5 does not ask for `QuadSourceIntegral` values. They were compared, from the readers' CSVs only,
because §6.4's result left open whether the integrals had changed at all. Only one triple is common
to both extracts, because the readers' triples are unseeded. 63 of 96 `total`s differ, by at most
1.8e-03 of max|total|. Recorded, not judged.

#### 4. A `ray-processes` look made through a pipe — UNINTENDED DRIFT

After the readers, `ray-processes --ray-temp-dir /tmp/ray-sgk` was first run inside a compound
command piped to `tail`. `CLAUDE.md` forbids that. It listed that shell's own two processes (27267,
27336), not Ray. The look was repeated correctly, on its own and by `>>` into `state/lock.log`,
where it listed nothing. No lock state was changed by the first one; it is not in the transcript.

#### 5. A mistyped `status` in the lock transcript — UNINTENDED DRIFT

`status --project SecondaryGWKit …` exited 2 with a usage error: `status` takes no `--project`. It
was re-run as `status --ray-temp-dir /tmp/ray-sgk`, exit 0. Both are in `state/lock.log`, lines
8–14. It was a look after `release`, not a lock operation.

#### 6. One scratch file written to `/tmp` — UNINTENDED DRIFT

A listing command redirected to `/tmp/x`, outside the dispatch's writable paths. It held directory
names of the two extracts, nothing else. It was deleted in the next command. No other path outside
the dispatch's list was written. `find -newer` after §6.7 lists only the store's five files, the
run directory's four files, the launch log and `var/handover-12/`. A3 v2's files keep their mtimes
of 24–25 October.

#### 7. `QuadSourceIntegral` does not differ in the fingerprint — IMPLEMENTATION CHOICE (a question)

The launch's `--purpose`, §B.1, Part A's "State handed to the next step" and the prompt's §6.4 all
expected R8 to differ from R3 in `QuadSource` **and** `QuadSourceIntegral`. It differs in
`QuadSource` only. This agent judged it not a §8 stop, because §8 stops on a difference in "any
table other than" those two, and none differs.

The cause is in the code, read only:
- `QuadSourceIntegral`'s format-6 `inventory_spec()` has no value table, so it does not see
  `total`, `metadata` or the partition.
- Its `source` parent resolves to the `QuadSource` row's `reference_digest(key, tags)`, which
  excludes the value count.
- The supplement (deviation 3) shows the integrals did change.

Having that explanation, the agent went on with the remaining read-only steps (§6.5, §6.6) and the
lock's release (§6.7), rather than stopping with the lock held. **The orchestrator should confirm
that reading.** If it does not, nothing done since is irreversible except the lock's release, and a
re-run would re-acquire it. Proposed as `[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]`
in `issues.md`.

### Verification performed

| check | result | where |
|---|---|---|
| roots, ComputeTargets / ObjectFactories | 742 OK / 280 OK | `out/roots_*.txt` |
| R8 run | `done`, exit 0, pid 21718, `git_dirty` false, `8a02f0c` | `var/runs/handover-12-r8-reference-20261025T210551/` |
| fingerprint vs R3 | exit 1; differs in `QuadSource` only | `out/fingerprint_against_R3.txt` |
| `partition.handover` / clamp keys | 864/864 / 0 | `out/r8_rows.txt` |
| `QuadSource` nodes | +3 per row, 18 total, 0 removed, common values identical | `out/r8_rows.txt` |
| readers | 6 × exit 0; store identical | `out/readers.txt`, `extract/` |
| store hash before §6.4 vs after §6.6 | identical | `state/store_sha256_*.txt` |
| lock transcript | acquire 0, set-pid 0, ray-processes 0, release 0, status 2 (usage), status 0 | `state/lock.log` |

### Observations not acted on

- The fingerprint is blind to `QuadSourceIntegral` values (deviation 7; proposed issue).
- The readers' file counts follow their unseeded subsample (`[10-the-readers-draw-their-work-lists-unseeded]`).
- At $z_{\rm resp}$ = 6.985e9, above both $z_{\rm init}$, `numeric_quad` moved by 3.0e-13 of
  max|value|. That is consistent with the source spline's coefficients changing a little above the
  added nodes; not traced.
- R8's store has no sidecar. `quadsource_atol_sweep.py --build` does not create one; the
  fingerprint is in `status.json`. The store is the user's to keep or retire (§B.4).

### State handed to the next step

- R8's fingerprint `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` is the
  rehearsal reference from here on, recorded by `handover-12-r8-reference-20261025T210551`.
- The lock is free, no Ray process of `/tmp/ray-sgk` is left, and the registry shows 12 runs, all
  finished.
- `var/handover-12/` holds `out/`, `state/`, `scripts/`, `extract/` (CSVs: not to be copied back,
  per §B.4) and these three records.
- Open question for the orchestrator: deviation 7.
