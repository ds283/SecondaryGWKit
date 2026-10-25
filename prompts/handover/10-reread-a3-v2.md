# Prompt 10 — re-read A3 v2 with the fixed readers

**Campaign:** [`README.md`](README.md) · **Board item:** **10** (outside workstreams A–E) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with 09. **Held** until 09 has landed and been reviewed.
**Where it runs:** on the Mac Studio, in a Claude Code session there, as 06–08. **It commits
nothing there**; everything it writes is under `var/handover-10/`.
**Closes:** `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`,
when acceptance 2 holds.
**Recommended model:** **Opus**.

**Gate:**
- 09 landed and reviewed; its rehearsal store `39809dca…`;
- the dispatch commit is 09's records commit or later, pushed; **no commit since `b90752d` touches
  `ObjectFactories/`, `config/` or `main.py`**: `git diff --stat b90752d <dispatch SHA> --
  ObjectFactories config main.py` is empty. A3 v2 was built at `b90752d`, and an old store is read
  at its own SHA unless the schema it was written with is unchanged (`CLAUDE.md`; this condition is
  what makes reading it at the dispatch SHA sound);
- on the Mac Studio: `runregistrykit list` shows nothing `running` or stale; the rehearsal lock is
  free; the checkout's `git status --short` is empty.

---

## 0. What this prompt is

08 read A3 v2 with every reader, and `extract_GkWKB_data.py` died on a stored `analytic_G_w` of
7.57e+243. 09 fixed the readers (and, for new stores, the column). This prompt moves the Mac
Studio's checkout to the fix and reads A3 v2 again, so the store has a complete read-back.

---

## 1. Steps

Make `var/handover-10/{out,state}`; save every command and its output; run commands with variable
arguments through `bash`.

1. `git fetch origin && git merge --ff-only origin/handover-remedial`; `HEAD` is the dispatch SHA;
   `git status --short` empty; the Gate's `git diff --stat` empty. **The checkout moves for the first
   time since A3 v2's launch**: this is allowed now, since nothing runs from it.
2. `./venv/bin/pip freeze` equals `docs/handover/a3-v2/rehearsal/out/pip_06.txt`'s (09 changes no
   pin).
3. `./venv/bin/python -m runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite`:
   live, fingerprint `0e32f24a…`. The SHA-256, size and mtime of every store file:
   `state/store_files_before.txt`.
4. **The lock.** `acquire --purpose "SecondaryGWKit handover 10: re-read A3 v2"`; Ray runs here. Exit
   3 means stop and ask.
5. **The readers**: `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
   var/datastores/handover-A3-v2-lambdacdm.sqlite var/handover-10/extract`, all six, then
   `extract_Gk_data` with `nomodels`, as 08 ran them. Record each one's exit, wall time, files written
   and size (`out/readers_10.txt`).
6. **The GkWKB read-back is complete**: its log shows every `(k, z_source)` pair it was given was
   plotted. Count the full-range PDFs, PNGs and CSVs: they are equal to each other and to the number of
   pairs the reader's own work list holds. The pair 08 died on, `k=1e+05-k-serial=1/zsource=1.3633e+16-z-serial=19`,
   has its PDF, PNG and CSV. Report how many plotted `analytic_G_w` points `sensible_log_points`
   dropped, store-wide if the reader reports it, otherwise for that pair from its CSV.
7. Every store file unchanged against step 3 (`state/store_files_after.txt`); `store fingerprint`
   (without `--write`) equal to the sidecar's; no Ray process left.
8. `release` the lock; `status` free; `ray-processes` empty (`out/lock_10.txt`).

## 2. Acceptance

1. Every reader exits 0; the store is byte-identical and its fingerprint the sidecar's.
2. The GkWKB read-back is complete as step 6 says, the failing pair included.
3. The lock was taken and released.
4. Records under `var/handover-10/`: `out/`, `state/`, `log.md` (`qcd-background-audit` README §5.1's
   form), `verification.md` (**§4 "Re-read with the fixed readers"** of
   `docs/handover/A3-V2-VERIFICATION.md`), `issues.md`.

The orchestrator copies all but `extract/` to `docs/handover/a3-v2/reread/`, commits them, and moves
`[08-the-gkwkb-reader-fails-…]` to the board's §4, closed.

## 3. Stop conditions — stop and ask

- Any Gate item not as stated.
- A reader exits non-zero, or any store file changes.
- The GkWKB counts disagree, or the failing pair's files are missing.
- The lock is held, or will not release.
