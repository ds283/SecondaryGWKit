# Prompt 07 — guard the fingerprint format against spec changes

**Campaign:** [`README.md`](README.md) · **Board items:** **V1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-15, with prompts 05, 06 and 08, at `14bcac0`, after E2 was recorded as holding
(board §2, §5.10).
**Gate:**
- 06 landed and reviewed: `RunRegistry/` is gone, `config/fingerprint.py` holds
  `FINGERPRINT_FORMAT = 6` and the fingerprinter, and the roots pass at 06's counts;
- `git status` is clean, and no commit after 06's review touches code (anything outside `prompts/`
  and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows nothing `running`.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One small test module, one pinned table, and five breakages.

**Read first:**

1. `CLAUDE.md`: "Repository mechanics" (tests, `black`).
2. [`README.md`](README.md): §2.2 (07's charter), §5 (rules 7–10), §6.1 (Q5).
3. **[`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md) §3.5**:
   why the guard exists, and why the golden fingerprint alone is not it.
4. **RunRegistryKit's recipe**: `git -C /Users/ds283/Documents/Code/RunRegistryKit show
   aaab408:docs/adoption/README.md`, §6, "The Q5 guard", the recipe, the sketch and its
   breakages; and the SGK checklist's item 8, at the same commit, `docs/adoption/secondarygwkit.md`.
   Nothing of the guard is shipped by the package; it is SGK's test.
5. `config/fingerprint.py` as 06 left it, with its format history; and
   `datastorekit.store_inventory` as installed (`inventory_specs`, `canonical_json`,
   `InventorySpec`, `Parent`, `ParentSet`).
6. 06's log, "State handed to the next prompt".

---

## 1. What is wanted

SGK's fingerprint format is a number the author bumps when what the inventory records changes
(`config/fingerprint.py`'s history: six bumps, each with a key change). **Nothing makes the bump
happen.** A change to a factory's `inventory_spec()` without a bump leaves format 6 describing two
different encodings, and every fingerprint taken after it reads as a content difference in that
class. The golden fingerprint would fail, but it says "the data changed", not "you forgot the
format".

**The guard** pins, for format 6, the SHA-256 of the canonical form of every `inventory_spec()` in
the registry, in registry order (Q5). A spec change then fails a test that says which class changed
and that the format needs a bump with a new pin beside it.

**The orchestrator ran the recipe's sketch on SGK's registry at `14bcac0`**, read-only: 21 specs
over the 38-name registry, in registry order `version`, `store_tag`, `redshift`, `wavenumber`,
`wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`,
`BackgroundModel`, `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`,
`GkWKBIntegration`, `GkSourcePolicy`, `GkSourcePolicyData`, `GkSource`, `QuadSourcePolicy`,
`QuadSource`, `QuadSourceIntegral`, `OneLoopIntegral`. Every spec went through `canonical_json`
with the recipe's key conversion; the `int`-keyed `Parent.types` of `BackgroundModel` and
`wavenumber` among them. Two pairs have equal digests (`version` and `store_tag`; `GkSourcePolicy`
and `QuadSourcePolicy`): their specs are equal in shape, and a spec does not hold its class's name.
That is why the pin is a list of `[class, digest]` pairs, not a set of digests.

## 2. What to do

**2.1 The pin**, in `config/fingerprint.py`, beside `FINGERPRINT_FORMAT`:

```python
SPEC_DIGESTS = {
    6: [
        ["version", "<64 hex>"],
        ...
    ],
}
```

one entry per format that has a pin, each a list of `[class, SHA-256]` in registry order, with a
comment saying what it is, how it is computed (pointing at the test), and the rule: **a change to
any `inventory_spec()` bumps `FINGERPRINT_FORMAT` and adds a new entry here, beside the old ones,
never over them**, with a new golden file, as the history above says. Take the values by running
the test's own function at 06's tree, not by typing them.

**2.2 The test**, `ObjectFactories/tests/test_fingerprint_format_guard.py`, with:
- `plain(value)` and `spec_digests(factories)`, the recipe's sketch, adapted to this repository's
  style. `plain` writes each mapping key as its own JSON (`json.dumps(k)`), so `0` and `"0"` stay
  apart. If a value in SGK's specs is of a type the sketch does not handle (a set, an enum), stop
  and say so: the orchestrator found none at `14bcac0`;
- **`TestFormatGuard`**, two tests:
  1. `test_every_spec_is_pinned_for_the_current_format`: `FINGERPRINT_FORMAT` is a key of
     `SPEC_DIGESTS` (failing with a message that says the format was bumped without a pin), and
     `spec_digests(factories)` equals `SPEC_DIGESTS[FINGERPRINT_FORMAT]`. On a difference the
     message names each class whose digest differs, each class added and each removed, and says:
     a spec changed, so bump `FINGERPRINT_FORMAT` and add a new pin and golden file beside the old
     ones;
  2. `test_a_key_keeps_its_type`: `plain({0: "a"}) != plain({"0": "a"})`, and the canonical JSON of
     the two differs. This pins the conversion the guard depends on.
- It imports `factories` from `config.datastore` and the two names from `config.fingerprint`. It
  opens no store and nothing under `var/`, and needs no Ray (README §5 rule 10).

**2.3 The roots.** All seven, by README §8's command. Expected: 06's counts, with ObjectFactories
**279** (06's 277 and these 2).

## 3. Deliberate breakage (README §5 rule 8)

Each applied to the working tree, run, shown failing, and reverted. Never committed. The log
records each as a diff, exactly as applied, with the failing test and the message's line that
names what failed.
1. **A leaf added to one spec, with no bump**: one field added to `QuadSourceIntegral`'s
   `inventory_spec()` (`ObjectFactories/QuadSourceIntegral.py`, a key or value column) → test 1
   fails, naming `QuadSourceIntegral` and no other class.
2. **The format bumped with no new pin**: `FINGERPRINT_FORMAT = 7` → test 1 fails, saying there is
   no pin for format 7. (06's `TestConfigFingerprint` and `TestGolden` fail too; record which.)
3. **A class added to the registry with a spec**: a stand-in factory with an `inventory_spec()`
   appended to a copy of `factories` that the test reads (by patching `config.datastore.factories`
   for the run, or by the least intrusive way you find; say which) → test 1 fails, naming the
   added class.
4. **A polymorphic key's type changed**: `COSMOLOGY_TYPES`'s key `0` written as `"0"` (in
   `ObjectFactories/cosmology_types.py`) → test 1 fails, naming `BackgroundModel` and `wavenumber`.
5. **The key conversion weakened**: `plain` writing `str(k)` in place of `json.dumps(k)` → test 2
   fails, and with breakage 4 also applied, test 1 passes: the guard would miss it. Record both
   runs.

If a breakage does not fail as stated, stop: the guard is not testing what it claims.

## 4. Acceptance

1. **The test passes at 06's tree**, and `SPEC_DIGESTS[6]` has 21 entries in registry order, the
   class names as §1 lists them.
2. **The breakages of §3** each fail as stated.
3. **The roots pass** at §2.3's counts; a failure among README §8's four known wall-clock tests is
   re-run alone and recorded.
4. **Nothing stored changes**: no spec, factory or registry entry is changed by the commit;
   `config/fingerprint.py` gains the pin and its comment only.

## 5. Stop conditions — stop and ask

- A spec holds a value the sketch cannot make plain without a choice that could make two different
  specs alike.
- `inventory_specs(factories)` gives other than 21 classes, or another order than §1's.
- A breakage does not fail as stated.
- A root fails other than by README §8's known tests, or a count differs from §2.3's.

## 6. What this prompt does not do

- It changes no spec, no factory, no registry entry, and no format number.
- It changes no golden file and no test 06 wrote.
- It runs no build, writes nothing under `var/`, and does not take the rehearsal lock.
- It does not offer the guard to ChamPBH or StochasticInstantons: RunRegistryKit's checklists do.

## 7. The log and the board

**The files you may change, and nothing else:** `config/fingerprint.py` (the pin and its comment);
`ObjectFactories/tests/test_fingerprint_format_guard.py`, new; `measure/out/07-roots.txt`; the log;
this campaign's board; and `docs/OPEN_ISSUES.md` only if §3 gains an entry. Both `.py` files
`black`-clean.

**The log**, at `logs/07-guard-the-fingerprint-format.md`, in README §5.1's form, with §3's five
breakages as diffs.

**The board:** the header, the §1 row for 07, item V1, and `### 5.13 After prompt 07 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
