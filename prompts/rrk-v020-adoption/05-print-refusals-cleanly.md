# Prompt 05 — print the drivers' refusals cleanly

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**–**P3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-19, after the campaign's close, at `b87a4df`, on the user's reversal of
README §6.2 **C3** (README §6.1 **U5**).
**Gate:**
- 04 landed (`34bf4f7`) and was reviewed (`454c7d8`); R3's stores are retired (`b87a4df`);
- `git status` is clean;
- nothing is `running` or stale in the registry, the rehearsal lock is free, and no Ray process of
  this repository is up.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One small exception class and one `__main__` block per driver,
a handful of `raise` sites, and one test module.

**Read first:**

1. [`README.md`](README.md): §6.2 **C3** (what was decided and why), §6.1 **U5** (its reversal),
   §2.2 (05's charter) and §5 (rules 7, 8 and 10).
2. `docs/handover/quadsource_atol_sweep.py`: `run_build` (`:512-600`),
   `assert_store_is_self_consistent` (`:669-736`), `prepare` (`:739-781`), `sweep` (`:862-880`),
   `main` and the `__main__` block (`:970-1055`).
3. `docs/gktk-remedial/scoped_pipeline_run.py`: `register` (`:517-560`), `main` and the `__main__`
   block (`:589-768`).
4. **RunRegistryKit's `begin` at the tag**, `git -C /Users/ds283/Documents/Code/RunRegistryKit show
   v0.2.0:runregistrykit/__init__.py`, from `def begin(` (`:987`): its two `RuntimeError`s, a
   retired store (`:1102`) and a store a run that has not finished names (`:1119`), each ending
   `No run directory was created`.
5. The tests that already pin these refusals, which must keep passing **unchanged** (README §5 rule
   7): `RunDrivers/tests/test_quadsource_atol_sweep_prepare.py` (its `assertRaises(RuntimeError)`
   and the message fragments it matches), `test_build_rehearsal_scope.py`,
   `test_hard_killed_predecessor.py` (the refusal at `register`: `RuntimeError`, the message, and
   that nothing is taken over), and `test_pipeline_adoption.py`.

---

## 1. What is wanted

**Today** every refusal of the two drivers ends the process with a Python traceback. A refusal is
a check that stops the driver before any work starts. The operator reads it in a detached launch's
log, and the one line that matters is at the bottom, under a traceback that adds nothing. That
holds for the drivers' own refusals, such as "Refusing to build into an existing store". It holds
equally for `runregistrykit.begin`'s, such as the refusal of a store a stale run names, which is the
route after a hard kill (`CLAUDE.md` run-registry rule 5).

**Wanted:** each refusal prints as one clean message on stderr, prefixed `!! `, and the process
exits **1**, the status a traceback gives today, so that nothing that reads the exit status moves.
**Everything else keeps its traceback**: a failure during the work, and an internal invariant that
does not hold, are not refusals, and their traceback is evidence.

## 2. What to do

**2.1 The class.** Each driver defines, near its top:

```python
class Refusal(RuntimeError):
    """<one paragraph: a check that stops the driver before any work starts; printed by the
    `__main__` block as one `!!` line, exit status 1, with no traceback; a `RuntimeError`, so that
    every caller and test that catches `RuntimeError` still does>"""
```

It is a subclass of `RuntimeError` **on purpose**: the existing tests assert `RuntimeError`, and
they are not modified (README §5 rule 7).

**2.2 The refusal sites**, and only these. Each `raise RuntimeError(` becomes `raise Refusal(`, its
message unchanged word for word:
- **`quadsource_atol_sweep.py`:**
  - `run_build`: the existing-store refusal (`:537`) and the not-a-directory one (`:547`);
  - `assert_store_is_self_consistent` (`:736`). It is called from `run_build`'s `--resume` branch,
    from `prepare` and from `run_child`, and is a refusal in each: it stops before a write;
  - `prepare`: the `--force` refusal (`:755`) and the re-raise of `copy_store`'s refusal (`:779`),
    keeping its `from e`.
- **`scoped_pipeline_run.py`:**
  - `main`: the existing-datastore refusal (`:657`) and the not-a-directory one (`:662`);
  - `register`: the missing-package message (`:531`), keeping its `from exc`.

**2.3 `begin`'s refusals.** At each of the three `runregistrykit.begin(` calls (`run_build`
`:556`, `sweep` `:869`, `register` `:544`), a `RuntimeError` from `begin` is re-raised as
`Refusal(str(e))` with `from e`, its message unchanged. Nothing else is caught. `begin`'s other
exceptions (a `ConfigError`, a `FileExistsError`, an `OSError`) keep their tracebacks. `begin`
creates nothing before it refuses, so there is no run to finish.

**2.4 The `__main__` block** of each driver:

```python
if __name__ == "__main__":
    try:
        main()
    except Refusal as exc:
        sys.stdout.flush()
        print(f"!! {exc}", file=sys.stderr)
        sys.exit(1)
```

- `main()` itself still raises, so a test that calls `main()` or a function still sees the
  exception.
- A multi-line message (`assert_store_is_self_consistent`'s) prints as it is, with `!! ` before its
  first line only.
- No refusal is raised after `register` or `begin` has taken over `sys.stdout` and `sys.stderr`.
  Confirm this from the code and say so in the log; it is why the print needs no care about a
  `RegisteredStream`.

**2.5 What does not change:**
- every other `raise RuntimeError(`, which is a failure during the work or an invariant:
  - in `quadsource_atol_sweep.py`, the case selector (`:361`), `substitute_main_py` (`:399`), the
    two model filters (`:486`, `:627`) and the sweep's child exit (`:948`);
  - in `scoped_pipeline_run.py`, the model filter (`:713`) and the `main.py` literal count (`:729`);
- `parser.error(…)`, which already prints a clean usage error and exits 2;
- every message's wording;
- the order of the checks, and where `import ray` stands in `run_build` (before the pre-checks). It
  imports Ray and starts nothing; moving it is not this prompt's.

**2.6 The test**, a new module `RunDrivers/tests/test_clean_refusals.py`. No Ray is started,
nothing under `var/` is read or written, and no runs root but a temporary one is used.
1. **The class.** In each driver, `Refusal` is a subclass of `RuntimeError`.
2. **The sites, pinned by `ast`.** Read each script with `ast`, as
   `ComputeTargets/tests/test_ray_address_default.py` reads scripts. Assert that every `raise
   Refusal(` lies in exactly the functions §2.2–§2.3 name, with the counts per function, and that
   the `raise RuntimeError(` sites of §2.5 remain, with theirs. A new refusal, or a failure turned
   into a refusal, then fails this test until the test is changed with it.
3. **The clean print, end to end.** One subprocess per driver, with `cwd` the repository root and
   `sys.executable`, on a refusal reached before any registration:
   - `quadsource_atol_sweep.py --build --database <an existing file in a temporary directory>
     --atol 1e-32 --rtol 1e-7`;
   - `scoped_pipeline_run.py --k-min 1e5 --k-max 1e6 --k-count 2 -- --database <an existing file>`.

   Each must exit **1**, with stderr beginning `!! ` and holding the refusal's text, no
   `Traceback`, and no Ray process started (the launcher imports `ray` and must not call
   `ray.init`: its stdout has no `bootstrapping` line).
4. **`begin`'s refusal becomes a `Refusal`.**
   - At `register`: with a temporary runs root, begin a run on a database path from a child process,
     as `test_hard_killed_predecessor` does, and leave it alive. `scoped_pipeline_run.register(…,
     runs_root=<root>)` on the same path raises `Refusal`, its message `begin`'s. Reuse that
     module's helpers by import, not by copy.
   - At `run_build` and at `sweep`, which have no runs-root seam (README §6.2 C5): patch
     `runregistrykit.begin` in the test, with `unittest.mock`, to raise `RuntimeError("Cannot begin
     run x … No run directory was created")`. Call `run_build(args)` on a database path that does
     not exist (parsed arguments with `--build --register x --purpose …`), and `sweep(args)` with
     `--register x --purpose …`. Each raises `Refusal`, carrying that message, with `__cause__` the
     patched error. Nothing past `begin` runs: neither Ray nor a child process starts.
5. **A failure keeps its type.** With `runregistrykit.begin` patched to raise `OSError`, `run_build`
   raises `OSError`, not `Refusal`.

## 3. Acceptance

1. **The roots pass**: `RunDrivers` at 103 plus this module's tests, every other root at 02's
   counts. A failure among README §8's four known wall-clock tests is re-run alone and recorded
   with the load (04's ruling 9).
2. **Every existing test passes unmodified**, `test_quadsource_atol_sweep_prepare`,
   `test_build_rehearsal_scope`, `test_hard_killed_predecessor` and `test_pipeline_adoption` among
   them.
3. **The diffs are §2's and no more.** In each driver, `git diff` touches only the class, the
   sites of §2.2–§2.3, and the `__main__` block. Every message is byte-identical.
4. **The deliberate breakages are caught** (README §5 rule 8). Each is applied, run, recorded in
   the log as a diff, and reverted:
   1. the launcher's `__main__` block calls `main()` bare: test 3 sees a `Traceback`;
   2. `Refusal` subclasses `Exception`, not `RuntimeError`: test 1 fails, and so do existing tests
      in `test_quadsource_atol_sweep_prepare`. Name them;
   3. `run_build` lets `begin`'s `RuntimeError` through: test 4 fails;
   4. `register` lets `begin`'s `RuntimeError` through: test 4 fails;
   5. the model filter's `raise RuntimeError` at `quadsource_atol_sweep.py:627` becomes `raise
      Refusal`: test 2 fails;
   6. the `__main__` block exits 2: test 3 fails.
5. **The launcher's new hash** (`shasum -a 256 docs/handover/quadsource_atol_sweep.py`) is
   recorded. It is no longer R3's `2a985303…`, which is expected. A3 v2's manifests will carry the
   new one.
6. `black --check` is clean on every changed `.py` file.

## 4. Stop conditions — stop and ask

- An existing test fails, or cannot pass without being modified.
- A message would have to change, or a check move, for the clean print to work.
- A refusal of §2.2 turns out to be reachable **after** `register` or `begin` has taken over the
  output streams.
- `begin` at `v0.2.0` raises `RuntimeError` for something that is not a refusal.
- Test 3 or 4 cannot be written without starting Ray, writing under `var/`, or adding a seam to
  either driver.

## 5. What this prompt does not do

- It changes no message, no check, no build path after `begin`, and no reader. It runs no build and
  starts no Ray. It takes no lock.
- It does not change `CLAUDE.md`. Rule 5's route is unchanged, and the refusal it describes still
  ends the launch with exit status 1. The launch log now ends with one `!!` line, where it ended
  with a traceback's last line.
- **No rehearsal.** The change is confined to what runs before `begin` returns. R3 showed the
  build path, which this prompt does not touch.

## 6. The log and the board

**The files you may change, and nothing else:**
- `docs/handover/quadsource_atol_sweep.py`, `docs/gktk-remedial/scoped_pipeline_run.py` (§2);
- `RunDrivers/tests/test_clean_refusals.py` (new);
- `docs/rrk-v020-adoption-verification.md`: a new **§4, "After the close: clean refusals (05)"**,
  appended. It gives the refusal sites, the output form, the exit status, and the launcher's new
  hash beside §3.5's, for `handover` to read. Nothing above it changes;
- under `measure/out/`, `05-roots.txt`;
- the log `logs/05-print-refusals-cleanly.md`, in README §5.1's form; this campaign's board; and
  `docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The board:** the header, the §1 row for 05, items P1–P3, and
`### 5.6 After prompt 05 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
