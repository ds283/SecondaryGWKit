# Prompt 03 — pin `datastorekit` at `v0.4.0`

**Campaign:** [`README.md`](README.md) · **Board items:** **K1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-14, with prompt 04, at `a5a6f6d`, after E1 was recorded as holding (board §2).
**Gate:**
- E1 holds (board §2): DatastoreKit's `v0.4.0`, tag object `8e7c1d77…`, on commit `c8570e84…`;
- 02a landed and reviewed (board §5.4);
- `git status` is clean, and no commit after 02a's review (`a5a6f6d`) touches code (anything outside
  `prompts/` and `docs/`);
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, neither
  editable, and no `runregistrykit`.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt`, one install, the roots, and a
reading of three release notes.

**Read first:**

1. `CLAUDE.md`, "The datastore layer is `datastorekit`; the work queue is `raypoolkit`" (a pin is a
   tag, in a commit of its own) and the freeze section (`RunRegistry/` does not change).
2. [`README.md`](README.md): §0.1 (why R1 isolates this pin), §2.2 (03's charter), §4.2
   (rollback), §5 (rules 7, 9, 11–14) and §8.
3. **DatastoreKit's checklist for SGK, at the tag**, three addenda of it:
   `git -C /Users/ds283/Documents/Code/DatastoreKit show v0.4.0:docs/adoption/secondarygwkit.md`,
   the sections "Addendum: v0.2.3", "Addendum: v0.3.0" and **"Addendum: v0.4.0 (fingerprint prompt
   02, 2026-10-14)"**, the last with its closing table "For SGK's prompt 03".
4. The board: §2 (E1's row), §3 (`[00-recorded-fingerprints-lack-the-encoding-format]`) and §5.5.
5. 02a's log, "State handed to the next prompt".

---

## 1. What is wanted

DatastoreKit's `v0.4.0` carries the store fingerprint's algorithm, `datastorekit.store_fingerprint`,
with SGK's format as an argument and the package's own `ENCODING_FORMAT` beside the digest (E1).
SGK will call it at 06. **This prompt moves the pin and nothing else**, so that R1 (04) can show that
the pin alone leaves the rehearsal's fingerprint at `39809dca…`, and that the package's new function
agrees with SGK's on real stores, before anything here depends on it.

The pin moves three releases, `v0.2.2` → `v0.4.0`, through `v0.2.3` and `v0.3.0`. Under
`datastorekit/`, exactly three files differ between `v0.2.2` and `v0.4.0`: `SQL/factory_base.py`
(`v0.2.3`), `object.py` (`v0.3.0`) and the new `store_fingerprint.py`. `store_inventory.py`, which
SGK's own fingerprint code reads, is unchanged. `pyproject.toml` differs by its version line alone.
The orchestrator's `pip install --dry-run` against the edited line gave `Would install
datastorekit-0.4.0` and nothing else.

## 2. What to do

**2.1 Before.** Save `./venv/bin/pip freeze` and the registry's `package_provenance()` (as R0's
`prelaunch_checks_02.txt` printed it: `PYTHONPATH=. ./venv/bin/python -c "import json, RunRegistry;
print(json.dumps(RunRegistry.package_provenance(), sort_keys=True))"`, or the equivalent the module
offers; find it by reading `RunRegistry/__init__.py`, which you do not change).

**2.2 The pin.** `requirements.txt:17`, the one line, from
`datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2` to
`datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.4.0`. No other line of the file moves.

**2.3 The dry run, then the install.**
- `./venv/bin/pip install --dry-run -r requirements.txt`, saved whole. It must say it would install
  `datastorekit-0.4.0` and nothing else.
- Then `./venv/bin/pip install -r requirements.txt`, saved whole.
- After it: `pip freeze` again, and its diff against 2.1's, which must be the one `datastorekit`
  line; `pip check`; `package_provenance()` again, which must give `datastorekit` version `0.4.0`,
  commit `c8570e849334d09085bbd84876095fc9e22bd005`, `requested_revision` `v0.4.0`,
  `editable: false`, and `raypoolkit` exactly as before.
- **One read of the new module, to show what was installed:**
  `PYTHONPATH=. ./venv/bin/python -c "import datastorekit.store_fingerprint as m;
  print(m.ENCODING_FORMAT)"`, which prints `1`. Nothing in SGK imports it, and nothing here makes it
  do so: `git grep -n 'store_fingerprint'` outside `prompts/` and `docs/` is empty before and after.

**2.4 SGK's own fingerprint, through the new pin.** `store_inventory.py` is unchanged, so SGK's
`fingerprint_of` must give what it gave at R0. Read-only, with no `--write` and no `--listing`, on
each of R0's two stores:

```bash
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint var/datastores/runregistry-extraction-02-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint var/datastores/runregistry-extraction-02-resumed.sqlite
```

Each must print format 6, digest `39809dcac5c7b231…`, 21 classes, and `recorded: matches`, exit 0.
Take the SHA-256, size and mtime of every file of both stores (primary and shards) before and after,
and show them equal. This is a read through SGK's code, not R1: R1 rebuilds the stores (04).

**2.5 The roots.** All seven, serially, in the foreground, by README §8's command. Expected, at
02a's counts: AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39, LiouvilleGreen 148 (1
skipped), ObjectFactories 271, RunRegistry 286, tools 91. The `v0.4.0` addendum reads that no count
changes with the pin; this measures it.

**2.6 The three addenda.** For each of `v0.2.3`, `v0.3.0` and `v0.4.0`, list every item it asks of
SGK or says applies to SGK, and for each say, by name: **done here** (how), **does not apply at the
pin** (why, citing the addendum's own line or table row), or **06's** (the item, so that 06's
writer finds it). The `v0.4.0` addendum's closing table is the spine for its own items; check each
row's claim about SGK's code against SGK's tree at `HEAD`, by `git grep` or reading, and record
any you find false.

## 3. Acceptance

1. **The dry run moves only `datastorekit`.** After install, `package_provenance()` gives
   `datastorekit` 0.4.0 at `c8570e84…`, `requested_revision` `v0.4.0`, not editable; `raypoolkit`
   unchanged; `pip check` clean; `pip freeze`'s diff the one line.
2. **The seven roots pass at 02a's counts** (§2.5). A failure among README §8's four known
   wall-clock tests is re-run alone and recorded; any other failure, or any count that differs,
   stops the prompt (§4).
3. **The three addenda are read** and every item tabled as §2.6 says.
4. **SGK's fingerprint of R0's two stores** is `39809dca…`, matching each sidecar, and every store
   file is byte-identical around the reads.
5. `requirements.txt` differs from `HEAD` by line 17 alone, and nothing under `RunRegistry/` or
   `var/` is changed by the commit.
6. **No deliberate breakage**: this prompt adds no code and no test (README §5 rule 8).

## 4. Stop conditions — stop and ask

- The dry run would install, upgrade or remove anything but `datastorekit`.
- After install `datastorekit` is not at `c8570e84…`, or is editable.
- A root fails other than by README §8's known tests, or a count differs from §2.5's.
- An addendum asks something of SGK at the pin that this prompt does not cover.
- SGK's fingerprint of an R0 store differs from `39809dca…`, or a store file changes around the
  read. At this pin that would be a DatastoreKit defect (README §5 rule 11): record it, change
  nothing, stop.

**If you stop after the install**, say so: the venv then holds `v0.4.0` while `requirements.txt`
may not be committed. Do not reinstall `v0.2.2` yourself; the orchestrator decides (README §4.2).

## 5. What this prompt does not do

- It does not call, import or test `datastorekit.store_fingerprint` in SGK code (06). §2.3's one
  read is a verification step.
- It changes nothing under `RunRegistry/`, no test, no `docs/` script, and no other pin.
- It writes nothing under `var/`: no `store create`, `fingerprint --write`, `retire` or `begin()`.
- It runs no build and starts no Ray; it does not take the rehearsal lock (04 does).
- It does not retire R0's stores (the orchestrator does, after 04's review; board §5.5).

## 6. The log and the board

**The files you may change, and nothing else:** `requirements.txt` (line 17); under
`measure/out/`, `03-pip.txt` (2.1 and 2.3, each block headed by its command), `03-fingerprint.txt`
(2.4), `03-roots.txt` (2.5) and `03-addenda.txt` (2.6); the log; this campaign's board; and
`docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The log**, at `logs/03-pin-datastorekit-at-v0.4.0.md`, in README §5.1's form.

**The board:** the header, the §1 row for 03, item K1, and `### 5.6 After prompt 03 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
