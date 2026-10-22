# Orchestrator — prompt 01, pin `datastorekit` at `v0.5.0`

Read [`../README.md`](../README.md) first: §0–§0.2, §2.2 (01's charter and the close), §4.1, §5,
§6.2 and §8. This is the campaign's only note. Its conventions are `rrk-v020-adoption`'s
([`../../rrk-v020-adoption/orchestrator/prompt-01.md`](../../rrk-v020-adoption/orchestrator/prompt-01.md)),
carried over; 01 is that campaign's 01 again for another package. What differs is said here.

**You do not write code.** You may:
- run the suites, `pip` read-only commands, and `python -m runregistrykit list`;
- read DatastoreKit's and RunRegistryKit's repositories with `git`, never changing them;
- run a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§3).

**The prompt:** [`01-pin-datastorekit-at-v0.5.0.md`](../01-pin-datastorekit-at-v0.5.0.md)
**Board items:** K1, G1, B1 · **Closes:** nothing · **Opens:** nothing expected ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- The campaign is planned (`99199ca`) and gate E holds. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
- Nothing is `running` or stale in the registry.
- The venv holds `datastorekit` 0.4.0 at `c8570e84…`, not editable.
- `git worktree list` shows the main checkout alone.

## 0. What makes this prompt unusual

**The install takes effect before the commit, and the commit does not undo it.** From the install
on, every import of `datastorekit` in this checkout is `v0.5.0`'s, and `git revert` of 01 leaves
the venv there (README §4.1 gives the reinstall). So the review checks the venv directly, not only
the diff.

**The prompt edits a test module.** README §5 rule 7 allows it because 01 names the module and the
reason. Check that the edit is the helper's condition, its docstrings, one import and one new test,
and that neither existing assertion moved.

**What moves on purpose:** `requirements.txt:17`; the venv's `datastorekit`; the guard's `plain`
and its new test; `CLAUDE.md:51`'s sentence.

**What must not move:** `config/fingerprint.py` whole (`FINGERPRINT_FORMAT` 6, `SPEC_DIGESTS[6]`'s
21 pairs); every golden file under `ObjectFactories/tests/data/`; `raypoolkit` and `runregistrykit`
in the venv; anything under `var/`.

## 1. Dispatch

Dispatch 01 to a fresh Sonnet session in the main checkout, with the prompt file as its brief. It
runs about five minutes of suites twice; nothing in it is long enough for the run registry.

## 2. Review

On its report, before the close:
1. `git show --stat HEAD`: the five paths of the prompt's §6, and the campaign's own records.
2. `git diff HEAD~1 -- config/ ObjectFactories/tests/data/` is empty.
3. `package_provenance` of the venv: `datastorekit` 0.5.0 at `f47b9a25…`, not editable; the other
   two unchanged.
4. Re-run ObjectFactories at `HEAD`: 280, OK.
5. The log's mutation record: M1 and M2 each applied, each failing where the prompt says.
6. The log's check of the addendum's claims about SGK.

## 3. The close

If the review passes, one commit of the orchestrator's: the board's header (**CLOSED — 1 of 1
landed and reviewed**), its §1 row, gate FP, `### 5.2 Review and close (<date>)`, and the
campaign's line in `prompts/INDEX.md` ("closed 1 / 1, <date>"), with the index's header count.
Residue the review finds is fixed first, in a follow-up commit of its own.
