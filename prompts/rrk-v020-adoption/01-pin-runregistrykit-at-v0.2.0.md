# Prompt 01 — pin `runregistrykit` at `v0.2.0`

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**, **B1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-18, with prompts 02–03, at `79434bf`, with gate E recorded as holding (board
§2).
**Gate:**
- the planning commit has landed, and `git status` is clean;
- no commit after the planning commit touches code (anything outside `prompts/` and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- the venv holds `runregistrykit` 0.1.0 at `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`,
  `datastorekit` 0.4.0 at `c8570e849334d09085bbd84876095fc9e22bd005` and `raypoolkit` 0.2.0 at
  `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, none editable;
- `git worktree list` shows the main checkout alone.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt`, two of `CLAUDE.md`, one install,
the baselines and the roots.

**Read first:**

1. `CLAUDE.md`, "The datastore layer is `datastorekit`; …" (a pin is a tag, in a commit of its own)
   and "Long-running jobs — the run registry".
2. [`README.md`](README.md): §0 and §0.2 (the start tree and its counts), §2.2 (01's charter), §4
   (01's line), §4.2 (rollback), §5 (rules 9–13) and §8.
3. **RunRegistryKit at the tag** (`/Users/ds283/Documents/Code/RunRegistryKit`), read with
   `git -C <RRK> show v0.2.0:<path>`:
   - `docs/adoption/v0.2.0/secondarygwkit.md`, §1 ("The pin") and §2 ("Its runs");
   - `docs/adoption/v0.2.0/README.md`, §1 and §2;
   - `docs/releases/v0.2.0.md`, §3 (what a `v0.1.0` record reads as) and §5 (what `list` prints).
4. `runregistry-extraction`'s
   [`05-pin-runregistrykit-at-v0.1.0.md`](../runregistry-extraction/05-pin-runregistrykit-at-v0.1.0.md)
   and its log. This prompt is that one again for the next tag. Where this prompt is silent, that
   one governs.

---

## 1. What is wanted

`v0.2.0` is RunRegistryKit's `interrupted-runs` release (README §0). **This prompt installs it and
corrects the two sentences of `CLAUDE.md` that name the pin, and nothing else.** The rules that
`v0.2.0` changes are 02's, so that the install and the adoption are two reverts and not one.

Unlike `runregistry-extraction` 05, the package is already imported everywhere, so the install
takes effect at once: the next `list`, the next `begin` and every store command run `v0.2.0`. That
is why §2.1 takes the baselines **before** the install, and §2.4 reads them again **after** it.

## 2. What to do

**2.1 The baselines, before.** Save each under `measure/out/`, every block headed by its command:
- `./venv/bin/pip freeze`, and
  `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))`, printed as
  JSON, into `01-pip.txt`;
- `./venv/bin/python -m runregistrykit list` (stdout and stderr, and its exit status) into
  `01-list.txt`. README §0.2 expects 45 runs: 43 finished, 2 `unknown`;
- `./venv/bin/python -m runregistrykit store show <primary>` for each of the 21 sidecars (README
  §0.2: 20 under `var/datastores/`, 1 under `var/datastores/backup-pre-resume-20260921T091011/`)
  into `01-store-show.txt`. Each must read as a tombstone with no problem. Take each primary's path
  from its sidecar's name, as `runregistry-extraction` 01 did;
- the seven roots, by README §8's command, serially, in the foreground, into `01-roots.txt`. Expect
  §0.2's counts.

**2.2 The pin.** `requirements.txt:81` changes from

```text
runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0
```

to

```text
runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.2.0
```

No other line of the file moves.

**2.3 The dry run, then the install.**
- `./venv/bin/pip install --dry-run -r requirements.txt`, saved whole in `01-pip.txt`. It must say
  `Would install runregistrykit-0.2.0` and nothing else. The orchestrator measured exactly that on
  2026-10-18 (README §0.2).
- Then `./venv/bin/pip install -r requirements.txt`, saved whole.
- After it:
  - `pip freeze` again, whose diff against 2.1's must be the one `runregistrykit` line, naming
    commit `0bae3a3c501a8264ee4be5bc24f06422af38720f`;
  - `pip check`;
  - `package_provenance(...)` again. `runregistrykit` must be 0.2.0 at `0bae3a3c…`,
    `requested_revision` `v0.2.0` and `editable` `false`, and the other two entries must equal
    2.1's.
- **What was installed**, read:
  - `./venv/bin/pip show runregistrykit`;
  - the distribution's `direct_url.json` (`vcs_info.commit_id` `0bae3a3c…`, no
    `dir_info.editable`);
  - `importlib.metadata.version('runregistrykit')`, which is `0.2.0`;
  - **the core still imports nothing heavy:**
    `./venv/bin/python -c "import sys, runregistrykit, runregistrykit.fingerprint; print(sorted(n for n in ('ray', 'sqlalchemy', 'datastorekit') if n in sys.modules))"`
    prints `[]`;
  - `./venv/bin/python -c "import runregistrykit; print(runregistrykit.TERMINAL_STATES); print(runregistrykit.this_host())"`.
    The states are four, ending `lost`. `this_host()`'s `id` and `boot` are not `null` on this
    machine, as the release notes' §2.6 measured. Record the `id` and `name`; the `boot` is
    recorded too, since 03 checks every launch against it.

**2.4 The baselines, after.**
- **`list` again**, into `01-list.txt` beside 2.1's. Compare the two row by row. **Each of the 45
  runs must have the same state and marker as before.** The summary line gains its new clause,
  "*N* with no host recorded". *N* must count every run directory that has a `status.json`:
  every one of them was written under `v0.1.0` (release notes §3). Two cases are expected and
  must be read, not assumed:
  - **The `a3-pilot` pair** have no manifest. Under `v0.2.0` a run whose status cannot be read,
    or does not say, is marked `!!` **only when it has a manifest** (release notes §5, `list`).
    Record whether either pair has a `status.json`, and how `v0.2.0` reads it.
  - **The stderr lines.** `v0.2.0` adds lines for stale runs, for runs whose status does not say,
    and for runs on another host. With nothing stale and nothing foreign, none should print.
    Record what does print.
- **`store show` again** on the 21, into `01-store-show.txt` beside 2.1's. Each must read as it
  did, with no new problem.
- **The roots again**, at the same counts. The release notes say no SGK test count changes, and
  the adoption note's §6–§8 say SGK's registry tests pass as they stand.

**2.5 `CLAUDE.md`.** Its two statements of the pin, which the commit makes false (README §6.2 C4):
- `:56`, "pinned at `v0.1.0` with the extra `[stores]`" → "pinned at `v0.2.0` with the extra
  `[stores]`". The rest of that sentence, which says when and in which campaign it was adopted,
  stays. Add one clause after it: "and moved to `v0.2.0` on <date> in the `rrk-v020-adoption`
  campaign".
- `:88`, "(RunRegistryKit, pinned at `v0.1.0`)" → "(RunRegistryKit, pinned at `v0.2.0`)".

Nothing else in `CLAUDE.md` changes here: its rules are 02's.

**2.6 The note's §1 and §2.** Check the adoption note's claims about the pin and SGK's runs
against what you measured: `requirements.txt:81`, `:17`, `:75` and `:88`, and the runs reading as
"no host recorded". Record any claim that is false. Its other sections are 02's and 03's.

## 3. Acceptance

1. **The dry run adds only `runregistrykit`.** After install it is 0.2.0 at `0bae3a3c…`,
   `requested_revision` `v0.2.0`, and not editable. `datastorekit` and `raypoolkit` are unchanged;
   `pip check` is clean; `pip freeze`'s diff is the one line.
2. **Importing it loads none of `ray`, `sqlalchemy` or `datastorekit`.**
3. **The seven roots pass at §0.2's counts, before and after.** A failure among README §8's four
   known wall-clock tests is re-run alone and recorded. Any other failure, or any count that
   differs, stops the prompt (§4).
4. **`list` and `store show` read every record as before** (§2.4). The one difference is
   `list`'s new summary clause, whose count is accounted for.
5. `requirements.txt` differs from `HEAD` by the one line, and `CLAUDE.md` by §2.5's two edits.
   Nothing under `var/` is changed, other than by the reads in §2.
6. **No deliberate breakage.** This prompt adds no code and no test (README §5 rule 8).

## 4. Stop conditions — stop and ask

- The dry run would install, upgrade or remove anything but `runregistrykit`.
- After install `runregistrykit` is not at `0bae3a3c…`, or is editable, or another package moved.
- A root fails other than by README §8's known tests, or a count differs from §0.2's.
- `list` gives a run a different state or marker than before the install, or exits non-zero; or a
  `store show` reads a sidecar differently or reports a problem.
- `this_host()` gives `null` for `id` or `boot`. 03's acceptance needs both.

**If you stop after the install**, say so: the venv then holds `v0.2.0` while `requirements.txt`
may not be committed. Do not reinstall `v0.1.0` yourself; the orchestrator decides (README §4.2).

## 5. What this prompt does not do

- It changes no code, no test, and no rule of `CLAUDE.md` (02).
- It runs no build, starts no Ray, and does not take the rehearsal lock.
- It does not call `begin`, `end`, `Run.finish` or any store command that writes. `list` and
  `store show` only read.
- It writes nothing under `var/`.

## 6. The log and the board

**The files you may change, and nothing else:**
- `requirements.txt` (the one line) and `CLAUDE.md` (§2.5);
- under `measure/out/`, `01-pip.txt`, `01-list.txt`, `01-store-show.txt` and `01-roots.txt`;
- the log, this campaign's board, and `docs/OPEN_ISSUES.md` only if §3 of the board gains an entry.

**The log**, at `logs/01-pin-runregistrykit-at-v0.2.0.md`, in README §5.1's form.

**The board:**
- the header;
- the §1 row for 01, and items K1 and B1;
- the RunRegistryKit G1 row of §2: "pin moved at <commit>; the record on RunRegistryKit's board is
  the user's";
- `### 5.1 After prompt 01 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
