# Campaign — py313-venv

**Written:** 2026-10-25 on branch `handover-remedial` at `ebfae11`, by Claude Opus 5.5. It moves
the venv on both hosts, the laptop (`mps026088`) and the Mac Studio (`mps026298`), from Python
3.12.15 to **3.13.16**, and moves two pins: `ray` from 2.43.0 to **2.59.0** and `SQLAlchemy` from
2.0.39 to **2.0.55**. Every other pin stays as it is. The user asked for the move on 2026-10-25,
after `handover` prompt 10 and before the analysis jobs queued behind it (§6.1).

It has **no audit document of its own**. The facts it rests on are measured in §0.2, and 01
measures them again before acting. Where the two differ, 01's numbers govern.

## 0. What changes

**The interpreter.** MacPorts `/opt/local/bin/python3.13`, 3.13.16 on both hosts (§0.2). Every
venv this repository has used for a rehearsal or a build since the registry began was 3.12.15.

**Two pins, and nothing else in `requirements.txt`:**

| Line | Before | After | Why |
|---|---|---|---|
| `requirements.txt:75` | `ray==2.43.0` | `ray==2.59.0` | 2.43.0 has no `cp313` wheel; 2.59.0 (2026-10-02) is the latest release |
| `requirements.txt:88` | `SQLAlchemy==2.0.39` | `SQLAlchemy==2.0.55` | the latest 2.0 release (2026-10-16), the newest DatastoreKit's `<2.1` allows |

Neither should change a stored value. Ray schedules and transports work but computes nothing, and
SQLAlchemy writes what it is given. The interpreter's float arithmetic is IEEE either way, and
numpy and scipy, which do the numerical work, keep their pins. **So the rehearsal fingerprint
stays `39809dca…`, and that is the acceptance test.** If it moves, that is a finding to explain
before anything builds on the new venv (§4.3).

**The registry records more.** `runregistrykit.toml`'s `distributions` gains `ray`,
`SQLAlchemy`, `numpy` and `scipy` (§6.2 C3). A manifest already records the interpreter
(`host.python`). After this move it is the first time two venvs have built stores here, and a
manifest should say which one built each store.

### 0.1 What this is not

- **Not SQLAlchemy 2.1.** DatastoreKit's `pyproject.toml` pins `sqlalchemy>=2.0.39,<2.1`, and its
  README says 2.1 is excluded. Lifting that is DatastoreKit's work, in a release of its own, which
  then reaches this repository by a pin move (`CLAUDE.md`). 2.1.0 is a month old (2026-09-24) and
  already at 2.1.6.
- **Not numpy, scipy, pandas or the rest.** Moving these could change what is stored, and so the
  fingerprint, and pandas 3 has breaking changes. That needs a decision of its own and a campaign
  of its own (§6.1 U3).
- **Not Python 3.14.** It would force numpy and scipy upgrades: numpy 2.2.4 has no `cp314` wheel.
- **Not the packages' CI.** DatastoreKit, RayPoolKit and RunRegistryKit test a "high" end of
  Python 3.13.16, Ray 2.55.1 and SQLAlchemy 2.0.46. Ray 2.59.0 and SQLAlchemy 2.0.55 are within
  their bounds but not in their matrices. Raising those is each package's work. For this
  repository, R6 and R7 are the test that counts (§6.2 C5).
- **Not `main`.** It has `9678f28` (Python 3.13, `ray` 2.45.0). 01 closes the issue that records
  the disagreement (§1), and a later merge resolves `requirements.txt:75` to this branch's line.

### 0.2 Reconciliation against `ebfae11`

Measured by the orchestrator on 2026-10-25 while writing this plan.
- **The laptop.** `handover-remedial` at `ebfae11`, clean, equal to `origin/handover-remedial`.
  The registry holds 53 runs: 51 finished and 2 `unknown` (pre-registry), none `running` or
  stale. The venv is 3.12.15, 817 MB. The disk has 56 GB free (94 % used). MacPorts has 3.11,
  3.12, 3.13 (3.13.16) and 3.14.
- **The Mac Studio** (`ssh macstudio-tunnel`, read-only). `handover-remedial` at `57f67a9`, four
  commits behind, clean. Its venv is 3.12.15. The registry holds five runs, all finished, the A3
  v2 build `handover-07-a3-v2-20261022T211155` `done` among them. The lock is free. MacPorts has
  3.13.16, and 118 GiB is free. `~/Documents/Code/retired-venvs/` holds
  `SecondaryGWKit-py313-20261022`, the `ray` 2.45.0 venv that `handover` 06 retired.
  `var/handover-06/recorded/` holds R3's two recorded run directories.
- **The pins resolve.** A dry run of `requirements.txt` with the two lines moved, in a fresh
  Python 3.13.16 venv (`pip install --dry-run --ignore-installed`), resolves 101 packages with
  `ray` 2.59.0 and `SQLAlchemy` 2.0.55, and numpy 2.2.4 and scipy 1.16.3 unchanged. Its only
  warnings are the two yanked pins that 3.12 resolves too (`aiohttp` 3.11.14, `aiohttp-cors`
  0.8.0). Every compiled pin (numpy, scipy, pandas, pyarrow, matplotlib, grpcio, greenlet,
  pydantic_core, memray, py-spy) has a macOS arm64 wheel for Python 3.13.
- **Python 3.13's removed modules** (`cgi`, `crypt`, `imghdr`, `pipes`, `telnetlib`, …): no live
  `.py` file imports one.
- **What Ray exposes to this repository.** Outside `docs/` and `prompts/`, the private names in
  use are `ray._private.services`, in `ComputeTargets/tests/test_ray_address_default.py`, and the
  command-line string `ray.util.client.server`, in `raypoolkit.rehearsal_lock`. The first
  exists to fail if a Ray upgrade changes what `address="local"` means. The second is what the
  lock looks for among Ray's processes. R6 checks both against 2.59.0.
- **What a manifest records.** `host.python` is the interpreter. `packages` covers the three git
  packages only, because `distributions` is `["datastorekit", "raypoolkit"]`. For a PyPI
  distribution, `runregistrykit.package_provenance` gives `version`, with `url` and `commit` null.
  No test reads `distributions`.

## 1. Scope

**In scope:**
- **01, on the laptop.** The two pins. A new 3.13 venv, with the old one moved aside, not deleted.
  `runregistrykit.toml`'s `distributions`. `CLAUDE.md`. The seven roots and `black` under the new
  venv. **R6**: R3's four launches, then R3's hard-kill pair, under the lock. It closes
  `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` (`handover` board §3), which was
  assigned here on 2026-10-25.
- **02, on the Mac Studio.** The checkout fast-forwarded, the venv rebuilt the same way, its freeze
  equal to the laptop's. The seven roots. **R7**: R3's four launches. The A3 v2 store read under
  the new venv: its fingerprint and its six readers.

**Out of scope:** §0.1, and:
- **What the code computes or stores.** No `.py` file changes except by `black`, which should
  change none. No table, column or key changes.
- **The packages.** A defect found in one is recorded and reported, never patched or worked around
  here (`CLAUDE.md`).
- **ChamPBH and StochasticInstantons.** They keep their own venvs. The shared lock finds an
  instance by its temp dir, whatever Ray version started it.
- **Records.** `prompts/` and dated `docs/` stay as written.
- **The issue's cause on `main`.** This campaign does not touch `main`.

## 2. Prompts

| # | Prompt | Covers | Where | Model | Status |
|---|---|---|---|---|---|
| 01 | [`01-move-the-laptop-venv-to-python-3.13.md`](01-move-the-laptop-venv-to-python-3.13.md) | the pins; the laptop's venv; `distributions`; `CLAUDE.md`; the roots; R6 with the hard-kill pair | laptop | Opus | landed `d4e5504`, reviewed 2026-10-25 |
| 02 | [`02-move-the-mac-studio-venv-to-python-3.13.md`](02-move-the-mac-studio-venv-to-python-3.13.md) | the Mac Studio's checkout and venv; the roots; R7; A3 v2 read under the new venv | Mac Studio | Opus | landed and reviewed on the Mac Studio 2026-10-25; records in the closing commit |

Both are written now, because neither prompt's task depends on what the other finds. 02 is
**held** until 01 has landed and been reviewed, because 02's freeze must equal 01's, and 02's
expected counts are 01's. There is no close prompt. The orchestrator's review of 02, in its own
commit, closes the campaign.

**The queued analysis jobs follow 02.** They read A3 v2 or run offline, so they need no rebuild
and run on the new venv (§6.1 U4).

## 3. Datastores

- **01** builds R6's three throwaway stores, `var/datastores/py313-venv-01-{reference,resumed,hardkill}.sqlite`.
- **02** builds R7's two, `var/datastores/py313-venv-02-{reference,resumed}.sqlite`, on the Mac
  Studio. It **reads** A3 v2 (`var/datastores/handover-A3-v2-lambdacdm.sqlite`, `0e32f24a…`) and
  must leave it byte-identical.

Retiring the throwaway stores is the user's job, after each review.

## 4. The interfaces

### 4.1 Rollback

Each prompt is one commit, reverted by `git revert`. **The venv must be put back too**, because the
install is not in `git`. Each prompt moves the old venv aside and records where it went, so the
rollback is:

```bash
mv venv ~/Documents/Code/retired-venvs/SecondaryGWKit-py313-failed-<date>
mv ~/Documents/Code/retired-venvs/SecondaryGWKit-py312-<date> venv
```

A venv records its interpreter's absolute path in `pyvenv.cfg`, and that path does not change, so
moving the venv back restores it. A rollback during a held lock releases the lock first.

### 4.2 What 01 hands on to 02

The new `requirements.txt` and `runregistrykit.toml`, pushed. The laptop venv's `pip freeze`
(`measure/out/01-pip.txt`, its last freeze) and the seven roots' counts (`01-roots.txt`, after),
which 02 must equal. R6's fingerprints.

### 4.3 If a fingerprint moves

Stop, change nothing, and record both fingerprints and `compare_fingerprints`. The orchestrator
then asks the user whether to bisect. The bisection is:
1. Python 3.13 with `SQLAlchemy==2.0.39`. This separates SQLAlchemy.
2. Python 3.12 with `ray==2.59.0`, which has a `cp312` wheel. This separates the interpreter from
   Ray. They cannot be separated the other way, because `ray` 2.43.0 has no `cp313` wheel.

The bisection is not in either prompt, because it is only wanted if the fingerprint moves.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:
1. **No test is modified, and none is added.** The venv either passes the suite as it stands or it
   does not. A test that fails under 3.13 or Ray 2.59 is a finding: stop and ask.
2. **No test needs Ray.** The real-Ray check is the rehearsal.
3. **The pins are what the venv holds.** `pip freeze` after the install differs from the one before
   it only in `ray` and `SQLAlchemy`, whatever the interpreter. A pin is never moved by hand to make
   an install pass.
4. **The old venv is moved, never deleted.** Retiring it is the user's.
5. **A new line in a log is a finding.** That covers a `DeprecationWarning`, a `FutureWarning`, or
   a Ray message that R3's logs do not have at the same slug. Each is recorded and explained, and
   any that bears on correctness is opened in §3. A line is not a stop by itself.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE`
  or `UNINTENDED DRIFT`;
- **Verification performed**, with every count against the baselines taken before;
- **What is new in the logs** (§5 rule 5);
- **Observations not acted on**;
- **State handed on**.

## 6. Decisions

### 6.1 Made by the user (2026-10-25)

| | Decided |
|---|---|
| **U1** — Ray | `2.59.0`, the latest, over the packages' tested 2.55.1 |
| **U2** — SQLAlchemy | `2.0.55` now; 2.1 later, after a DatastoreKit release that allows it |
| **U3** — the other pins | Unchanged |
| **U4** — when | Before the queued analysis jobs, on both hosts |

### 6.2 Choices of method made in writing this plan

These are the orchestrator's choices, and the user can reverse any of them.
- **C1 — the name** `py313-venv`.
- **C2 — the laptop first.** The laptop is where the suite and the records live, and the Mac Studio
  holds A3 v2. A failure found on the laptop costs the Mac Studio nothing.
- **C3 — `distributions` gains `ray`, `SQLAlchemy`, `numpy` and `scipy`.** These are the
  distributions whose version could change what a store holds or how it was written. `host.python`
  already records the interpreter. It is one line in this repository's own config, and it changes
  what manifests record from 01 onwards. R6's manifests then differ from R3's in `packages` by
  those four keys, as expected (01 §3).
- **C4 — R6 runs R3's hard-kill pair as well** (`rrk-v020-adoption` 03 §2), which `handover` 06
  left out for R4. The route after a hard kill depends on how Ray shuts down, and Ray moves by 16
  minor releases. It also measures `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` on
  2.59.0, which narrows that issue whatever the answer. Only on the laptop. R7 does not repeat it,
  as R4 did not.
- **C5 — the packages' CI is not a gate.** Their bounds allow both versions, so R6 and R7 are the
  real-Ray test of 2.59.0 for this repository's use.
- **C6 — `CLAUDE.md` gains the interpreter.** "Repository mechanics" says the venv is Python 3.13
  on both hosts, built with MacPorts `python3.13`, and when it moved. Its sentence on the `ray` and
  `SQLAlchemy` pins stays.
- **C7 — R6 and R7 are named after R5** (`handover` 09's reference on the laptop, 2026-10-25), the
  last rehearsal.

## 7. Baselines

Re-measure every root at `HEAD`, clean, before the venv moves:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

**Under load, four tests are known to fail on wall clock or chance:**
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`; `test_background_tau.test_build_cost`;
`test_numeric_phase_cut.test_the_print_is_gated_both_ways`; and LiouvilleGreen's
`test_3bessel_analytic.test_JJJ`. Re-run a failure alone before calling it real. A wall-clock test
that fails under 3.13 and passes under 3.12 at the same load is a finding (3.13 changes the
interpreter's speed), not a stop.
