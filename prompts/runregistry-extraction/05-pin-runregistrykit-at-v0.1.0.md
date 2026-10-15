# Prompt 05 — pin `runregistrykit` at `v0.1.0`

**Campaign:** [`README.md`](README.md) · **Board items:** **K2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-15, with prompts 06–08, at `14bcac0`, after E2 was recorded as holding (board
§2, §5.10).
**Gate:**
- E2 holds (board §2): RunRegistryKit's `v0.1.0`, tag object `fd5ce48…`, on commit `9f8ebe43…`;
- 04 landed and reviewed (board §5.7), and R0's and R1's stores retired (§5.8);
- `git status` is clean, and no commit after `14bcac0` touches code (anything outside `prompts/` and
  `docs/`);
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds `datastorekit` 0.4.0 at `c8570e84…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, neither
  editable, and no `runregistrykit`.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt`, one install, and the roots.

**Read first:**

1. `CLAUDE.md`, "The datastore layer is `datastorekit`; the work queue is `raypoolkit`" (a pin is a
   tag, in a commit of its own), and the freeze section (`RunRegistry/` does not change).
2. [`README.md`](README.md): §2.2 (05's charter), §4.2 (rollback), §5 (rules 7, 9, 11–14) and §8.
3. **RunRegistryKit's checklist for SGK**, item 1, "The client and its versions":
   `git -C /Users/ds283/Documents/Code/RunRegistryKit show aaab408:docs/adoption/secondarygwkit.md`.
   The checklist is at `aaab408`, a commit after the tag; nothing under `runregistrykit/` or in
   `pyproject.toml` changed after the tag (board §5.10).
4. RunRegistryKit's `README.md` at the tag, "Supported versions" and "Installing".
5. The board: §2 (E2's row) and §5.10.
6. [`03-pin-datastorekit-at-v0.4.0.md`](03-pin-datastorekit-at-v0.4.0.md) and its log: this prompt is
   03 again for a new package. Where this prompt is silent, 03 governs.

---

## 1. What is wanted

RunRegistryKit's `v0.1.0` is SGK's `RunRegistry/`, imported from `a45aefc` and made generic, with
the store layer as the extra `[stores]` (E2). 06 adopts it. **This prompt installs it and nothing
else**, so that the install and the adoption are two reverts, not one.

`v0.1.0` declares `requires-python = ">=3.12"`, no core dependency, and `[stores]` =
`datastorekit>=0.4.0`, which SGK's `requirements.txt:17` already satisfies. The orchestrator's
`pip install --dry-run -r requirements.txt` with the new line gave `Would install
runregistrykit-0.1.0` and nothing else (board §5.10).

**Nothing imports it after this prompt.** `RunRegistry/` stays the registry every driver calls
until 06. A manifest written between 05 and 06 still records only `datastorekit` and `raypoolkit`:
`RunRegistry.RECORDED_DISTRIBUTIONS` is frozen, and `runregistrykit` enters `packages` at 06.

## 2. What to do

**2.1 Before.** Save `./venv/bin/pip freeze`, and the registry's `package_provenance()` as 03's
2.1 printed it.

**2.2 The pin.** `requirements.txt` gains one line, in its alphabetical place, after `rpds-py` and
before `rsa`:

```text
runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0
```

With the extra: SGK uses the store layer. No other line of the file moves.

**2.3 The dry run, then the install.**
- `./venv/bin/pip install --dry-run -r requirements.txt`, saved whole. It must say it would install
  `runregistrykit-0.1.0` and nothing else.
- Then `./venv/bin/pip install -r requirements.txt`, saved whole.
- After it: `pip freeze` again, and its diff against 2.1's, which must be the one `runregistrykit`
  line, naming commit `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`; `pip check`; `package_provenance()`
  again, which must equal 2.1's.
- **What was installed**, read and not used:
  - `./venv/bin/pip show runregistrykit`, and the distribution's `direct_url.json` (its
    `vcs_info.commit_id` is `9f8ebe43…`, `requested_revision` `v0.1.0`; no `dir_info.editable`);
  - `./venv/bin/python -c "import importlib.metadata as m; print(m.version('runregistrykit'))"`,
    which prints `0.1.0`;
  - **the core imports nothing heavy:** `./venv/bin/python -c "import sys, runregistrykit,
    runregistrykit.fingerprint; print(sorted(n for n in ('ray', 'sqlalchemy', 'datastorekit') if n
    in sys.modules))"`, which prints `[]`.
  Run each from the repository root. Nothing here reads `runregistrykit.toml`, which does not exist
  yet (06 writes it), so do not run `python -m runregistrykit list` or any `store` command: each
  would refuse for want of the file.
- **Nothing in SGK imports it:** `git grep -n 'runregistrykit'` outside `prompts/` and `docs/` is
  empty before and after.

**2.4 The roots.** All seven, serially, in the foreground, by README §8's command. Expected, at 03's
counts: AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
ObjectFactories 271, RunRegistry 286, tools 91.

**2.5 The checklist's item 1.** Check its claims against the venv: Python 3.12.15, `ray` 2.43.0,
`SQLAlchemy` 2.0.39, `datastorekit` 0.4.0, `raypoolkit` 0.2.0. Record any that is false. Its other
items are 06's, 07's and 08's.

## 3. Acceptance

1. **The dry run adds only `runregistrykit`.** After install it is 0.1.0 at `9f8ebe43…`,
   `requested_revision` `v0.1.0`, not editable; `datastorekit` stays 0.4.0 at `c8570e84…`;
   `package_provenance()` equals 2.1's; `pip check` clean; `pip freeze`'s diff the one line.
2. **Nothing imports it**, and importing it loads none of `ray`, `sqlalchemy` or `datastorekit`.
3. **The seven roots pass at 03's counts** (§2.4). A failure among README §8's four known wall-clock
   tests is re-run alone and recorded; any other failure, or any count that differs, stops the
   prompt (§4).
4. `requirements.txt` differs from `HEAD` by the one added line, and nothing under `RunRegistry/` or
   `var/` is changed by the commit.
5. **No deliberate breakage**: this prompt adds no code and no test (README §5 rule 8).

## 4. Stop conditions — stop and ask

- The dry run would install, upgrade or remove anything but `runregistrykit`.
- After install `runregistrykit` is not at `9f8ebe43…`, or is editable, or `datastorekit` moved.
- Importing `runregistrykit` loads `ray`, `sqlalchemy` or `datastorekit`.
- A root fails other than by README §8's known tests, or a count differs from §2.4's.

**If you stop after the install**, say so: the venv then holds `runregistrykit` while
`requirements.txt` may not be committed. Do not uninstall it yourself; the orchestrator decides
(README §4.2).

## 5. What this prompt does not do

- It imports `runregistrykit` from no SGK code, test or `docs/` script (06). §2.3's reads are
  verification steps.
- It writes no `runregistrykit.toml` and no `config/fingerprint.py` (06).
- It changes nothing under `RunRegistry/`, no test, and no other pin.
- It writes nothing under `var/`, runs no build, starts no Ray, and does not take the rehearsal lock.

## 6. The log and the board

**The files you may change, and nothing else:** `requirements.txt` (the one line); under
`measure/out/`, `05-pip.txt` (2.1 and 2.3, each block headed by its command) and `05-roots.txt`
(2.4); the log; this campaign's board; and `docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The log**, at `logs/05-pin-runregistrykit-at-v0.1.0.md`, in README §5.1's form.

**The board:** the header, the §1 row for 05, item K2, and `### 5.11 After prompt 05 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
