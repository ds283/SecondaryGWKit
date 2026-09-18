# Orchestrator — prompt 06a, the close-out

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../06a-close-out-and-provenance.md`](../06a-close-out-and-provenance.md)
**Model:** Opus. **Production code changed: no.** Two new documents under `docs/`, this campaign's
log and board, and `docs/OPEN_ISSUES.md`. `config/defaults.py` must be **byte-identical**, and
**nothing under `docs/tolerance-convergence/`** may appear in the diff at all — the five campaign
documents and `inventory.py` are read, never edited.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**The dispatch template is used exactly as written.** §5 rule 8's parameter freeze applies and 06a
is inside it. If you find yourself editing the template for this prompt, you have misread something.

**Why it runs now.** 06 (`2033cfc`) landed the last measurement and regenerated the inventory. §7
**D11** (2026-09-18) split §3.6 into 06 and 06a for §5 rule 1's reason: the campaign's named
deliverable should not be reverted by a failure in a sweep. 06a was **held until 06 landed** so that
it is written against what 06 actually leaves — the same principle that held 06 itself from
2026-09-16. It was written on 2026-09-18, after 06.

**Precondition:** 06 has landed. **Do not dispatch 06a if `ComputeTargets` is not 521, if
`config/defaults.py` is not at blob `76bab78…`, or if `inventory.py --check` does not pass** — 06
left all three and this prompt moves none of them.

**One ordering note that is not this campaign's.** `prompts/radiation-oracle` adds tests and will
raise `ComputeTargets` above 521. **It must not land before 06a**, or 06a's baseline moves for a
reason no prompt in this campaign authorised and check 6 below becomes a false stop. The user
settled this on 2026-09-18: 06a first, then the oracle.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the baselines. At `2033cfc` they are **521**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5) and **39** (`CosmologyModels`).

```bash
git rev-parse HEAD:config/defaults.py          # must be 76bab78d9e9374263a91da87d86b509b2ed019d0
PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check; echo "exit=$?"
```

`--check` must report **up to date, exit 0** — the reverse of what it did before 06. If it is out of
date, something has moved `TOLERANCE-INVENTORY.md` or a factory since 06 and that is a question, not
a thing to regenerate.

**Record the row count, because the prompt's §1 turns on it:**

```bash
grep -c "^| \`" docs/tolerance-convergence/TOLERANCE-INVENTORY.md   # sanity only
grep -n "^\*\*[0-9]* rows\.\*\*" docs/tolerance-convergence/TOLERANCE-INVENTORY.md
```

It must say **44 rows**. The campaign README §3.6a says 39 plus six; that is the stale forecast the
prompt corrects, and an orchestrator who "fixes" the prompt to match the README has reintroduced the
double count.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `06a-close-out-and-provenance`, model
**Opus**, **unmodified** — including the standing parameter-freeze sentence, which applies.

Add this sentence, and no more:

> **You are the last prompt in this campaign.** There is no next prompt to hand work to, so
> "Observations not acted on" and your §3 issues are the only places a finding survives. Use them
> rather than widening your scope.

**Do not tell the agent what it will fail to establish.** Prompt §3 names three groups — the two it
knows cannot be established, the one that must not be mis-filed, and the five at risk. That is
already the maximum help that does not do the work: the agent still has to walk all 44 rows and
decide each. An orchestrator who enumerates the answer has removed the only acceptance condition
this prompt has.

**Do not accept a provenance you cannot trace.** This is the review's whole substance. Check 4 is
not a spot check — it is the prompt's §3, and a plausible sentence is exactly what failure looks
like.

## 3. The review — six checks

1. **Nothing outside the two new documents and the bookkeeping moved.**

   ```bash
   git diff --name-only HEAD~1 HEAD
   git diff --stat HEAD~1 HEAD -- docs/tolerance-convergence/ docs/spec/ config/defaults.py main.py ComputeTargets/ CosmologyModels/ Datastore/ Quadrature/ AdaptiveLevin/ LiouvilleGreen/
   ```

   The second must be **empty** — a single line is a **stop**, including in a campaign document the
   agent thought was stale. The first must contain only `docs/TOLERANCE-CONVERGENCE.md`,
   `docs/TOLERANCE-PROVENANCE.md`, `docs/OPEN_ISSUES.md` and
   `prompts/tolerance-convergence/`. Confirm the `config/defaults.py` blob is unchanged from §1.

2. **All 44 rows are covered.** Read `docs/TOLERANCE-PROVENANCE.md` against inventory §5.4 and count.
   A parameter present in §5.4 and absent from the note is a gap, and the note's own coverage claim
   is not evidence — count them.

3. **Every entry carries README §1.2's five fields.** Value and what it keys; the choosing
   measurement **with its grid generation** (§2 (b)) and the reference's drift (§5 rule 5); the
   competing floor; the cost times the object count (§2 (c)); campaign, prompt, log and date. An
   entry missing the floor or the object count is not a §1.2 entry — those two are what make the note
   usable rather than decorative.

4. **The unestablished ones say so in §1.2's own words, and nothing is invented.** This is the check
   the prompt exists for.

   - `DEFAULT_LEVIN_MAX_DEPTH` and `limit = 100` **must** be recorded as unestablished. If either
     carries a justification, read it: if it is the geometric reading from the code comment
     ("1/2^20 is roughly 1E-6"), that is the fabrication §3 forbids, dressed as a citation.
   - `BESSEL_ORDER_CHECK_TOL` **must not** be in the unestablished list — it is argued at
     `QuadSourceIntegral.py:79-92` and that is a provenance.
   - For the five §5.4 marks "never chosen", check each entry distinguishes a *use* from a *choice*.

   **Trace three entries at random back to the log or document they cite** and confirm the numbers
   are there. An entry that cites a log which does not contain its number is the failure mode, and
   it does not announce itself.

5. **§4's two supersessions are reflected.** Standing note 26 — log 12's `rtol` bit-identity does not
   transfer at `atol = 1e-32` — and note 27, `analytic_rad` is not a fixed oracle. Neither document
   may state the superseded version as current. README §6.2's last row still does; the prompt may
   not rewrite it, so check the agent recorded the staleness rather than either copying it forward
   or editing the charter.

6. **The suites and the bookkeeping.** `ComputeTargets` **must not fall below 521** and should not
   rise — this prompt adds no test; `CosmologyModels` **39**. `inventory.py --check` still passes.
   The three grid digests unmoved. Board row 06a, item **T12**, §3/§4, and `docs/OPEN_ISSUES.md` with
   its count and date corrected, all in the **same commit**. `black --check` clean.

## 4. What a good outcome looks like

- A provenance note in which **the admissions are as visible as the justifications**. A note where
  every one of 44 rows has a confident paragraph is the failure mode, not the success: the record
  demonstrably does not contain a choice for at least seven of them.
- The campaign's three `unchanged` results stated as **results**, with their dominating factors, and
  in proportion to the two parameters that actually moved.
- A close-out that a reader who has never opened a campaign log can use to decide whether a given
  constant is worth revisiting — which is §1.2's stated purpose.

**What a good outcome does *not* look like:** complete coverage, fluent prose, and a provenance for
`DEFAULT_LEVIN_MAX_DEPTH`. That note would be worse than no note, because a later reader would trust
it.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **The agent reports it cannot establish a provenance the campaign itself was supposed to have
  created** — that is, for a parameter *this* campaign set. The inherited and never-chosen ones are
  expected; one of the campaign's own would mean a prompt settled a value without recording why, and
  §3.6a says that prompt "has not finished".
- **The agent proposes to edit `config/defaults.py`'s comment at `:163`**, or any campaign document,
  or README §6.2. All are stops even though all three are genuinely stale.
- **Inventory §5.4 does not read 44 rows**, or `--check` fails at the parent.
- **`ComputeTargets` moves in either direction.**
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and passes roughly one run in three. Confirm by
re-running that module alone before attributing anything to the commit.

## 6. After it lands

**This closes the campaign.** Report to the user:

1. Which parameters' provenance **could not be established**, and whether any of them is one this
   campaign set.
2. The coverage count against §5.4's 44 rows.
3. What the close-out says about the three `unchanged` results and the two that moved.
4. Which issues closed, opened and stayed open, and the counts.
5. That README §6.2's last row and `config/defaults.py:163` remain stale by design, with the issue
   that now carries them.

Then **`prompts/radiation-oracle` prompt 01 is next**, per the user's decision of 2026-09-18. Its
orchestration is that campaign's, not this one's; note for whoever picks it up that it **raises**
`ComputeTargets` above 521, which is expected there and authorised by its own prompt §3.
