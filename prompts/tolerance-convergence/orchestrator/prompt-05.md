# Orchestrator — prompt 05, the schema half of the decoupling

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../05-replace-the-vestigial-key-columns.md`](../05-replace-the-vestigial-key-columns.md)
**Model:** Opus. **Production code changed: yes, and this is the campaign's first large one** —
four factories, four compute classes, `main.py` at four sites, six `extract_*.py` scripts and the
test modules that name the columns. **`config/defaults.py` is not among them and must be
byte-identical.**

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This is the first of two prompts §3.5 was split into** (campaign README §7 **D9**, 2026-09-18,
the user's decision). 05 is the **schema** half — board item **T9**, the columns — and changes no
number. 05a is the **tolerance** half — **T8** and **T10**, the per-target constants, the `main.py`
tolerance objects and the widened `ast` guard. The order is deliberate: after 05, four of the eight
targets carry no tolerance at all, so 05a's guard enumerates a smaller and truer set instead of
wiring per-target tolerances through four targets that 05a would then delete.

**Precondition:** prompt 04b has landed (`cc1f035`) and D3 is settled (2026-09-18). **Do not
dispatch 05 if `config/defaults.py` is not byte-identical to its state at `bc6dc97`** — three
campaigns and two rebases have left it untouched and prompt 05 must leave it untouched too; if
something has already changed it, find out what before starting.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the baselines. At `cc1f035` they are **491**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5) and **39** (`CosmologyModels`). `ComputeTargets` must not fall below
**452**, the campaign floor, and **should rise**: this prompt adds test modules.

**Record the schema as it stands, because this time four tables must move:**

```bash
./venv/bin/python -c "
import re,pathlib
for f in ('BackgroundModel','GkWKBIntegration','TkWKBIntegration','GkSource','GkNumericIntegration','TkNumericIntegration'):
    t=pathlib.Path('Datastore/SQL/ObjectFactories/%s.py'%f).read_text()
    cols=re.findall(r'sqla\.Column\(\s*\"([a-z_A-Z0-9]+)\"',t)
    print('%-22s %s'%(f,[c for c in cols if 'tol' in c or 'gauss' in c or c=='break_point_kind']))"
```

At `cc1f035` the first four read `['atol_serial', 'rtol_serial']` and the two numeric factories read
`['atol_serial', 'rtol_serial', 'break_point_kind']`. **The last two must be unchanged afterwards.**

**And record the two things that must not move:**

```bash
git rev-parse HEAD:config/defaults.py
grep -n "^TAU_GAUSS_ORDER\|^CS_TAU_GAUSS_ORDER\|^FRICTION_F_GAUSS_ORDER" ComputeTargets/BackgroundModel.py
grep -n "^RHO_GAUSS_ORDER" ComputeTargets/phase_residual.py
```

All four orders read **4**. Note `config/defaults.py`'s blob hash; §3 check 4 compares against it.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_run_identity ComputeTargets.tests.test_numeric_break_point_key 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `05-replace-the-vestigial-key-columns`,
model **Opus**. **The standing parameter-freeze sentence must be replaced**, because §5 rule 8 names
prompt 05 as the one prompt that may change a parameter and this half specifically may not. Use this
in its place, verbatim:

> **README §5 rule 8 names prompt 05 as the prompt that may change a parameter. This half may not,
> and that is the point of the split (README §7 D9).** `config/defaults.py` is byte-identical when
> you finish, and all four `*_GAUSS_ORDER` constants are still 4. If you find yourself needing to
> change a number to make your prompt pass, stop and say so rather than changing it — the numbers
> are prompt 05a's.

Add this sentence, and no more:

> **Four object types lose their tolerance pair and four keep it.** `GkNumericIntegration`,
> `TkNumericIntegration`, `wavenumber_exit_time` and `QuadSourceIntegral` keep theirs because
> theirs reach a solver; they and `ComputeTargets/tests/test_numeric_break_point_key.py` must be
> untouched. Do not widen the grant yourself.

**Do not tell the agent which mechanism to use** for prompt §3's second invariant — whether the
order is passed in from `main.py` or read by the compute class from its own module constant. §3
gives it the invariant and asks it to justify its choice; check 3 below turns on that justification.
An orchestrator who picks the mechanism has removed the only design question in the prompt.

**Do not reassure it about `RESIDUAL_WKB_REGION_MARGIN`.** Prompt §2 tells it what prompt 04
measured and then tells it to verify that reading itself. If it comes back saying the margin should
be keyed, that is §5's second stop and not something to talk it out of.

## 3. The review — seven checks

1. **The four tables moved, and the other two did not.** Re-run §1's schema reader. The first four
   must show the pair **gone**; `BackgroundModel` must show `tau_gauss_order`, `cs_tau_gauss_order`
   and `friction_F_gauss_order`; both WKB factories `rho_gauss_order`; `GkSource` **nothing new**.
   `GkNumericIntegration` and `TkNumericIntegration` must read exactly as they did, `break_point_kind`
   included. **A fifth table in the diff is a stop.**

2. **The order is filtered on, not merely selected.** This is the whole prompt and it is the check
   most easily passed by accident:

   ```bash
   grep -n -A 14 "\.filter(" Datastore/SQL/ObjectFactories/BackgroundModel.py | grep -n "gauss_order\|atol_serial"
   ```

   Read the build query of each of the three factories that gain a column and confirm the order
   appears in the `.filter(...)` and not only in the `select_from` join or the selected columns.
   **A column that is written and read back but never compared reproduces the exact defect this
   prompt exists to close**, one identifier later, and every other check here would pass.

3. **The key moves when the constant moves, and the agent proved it.** Find the test prompt §3 and
   acceptance row 3 require, and **run it**. Then read the log's account of the mechanism: what
   stops a row recording an order it was not computed at? A mechanism that lets `main.py` pass
   `tau_gauss_order=6` while `BackgroundModel` computes at 4 is a defect of the same family as the
   one being fixed, however well the column is keyed. The agent was told to choose and justify;
   **the justification is what you are reviewing, not the choice.**

4. **No number moved.**

   ```bash
   git diff HEAD~1 HEAD -- config/defaults.py
   ```

   Must be **empty**, and the blob hash must equal §1's. The four orders must still read 4.
   **Anything else is a stop** — it is prompt 05a's work done early, and it breaks the split.

5. **`main.py` stayed inside four sites.**

   ```bash
   git diff HEAD~1 HEAD -- main.py
   ```

   Only the `BackgroundModel`, `GkWKBIntegration` (**two** sites, at ~2165 and ~2215) and `GkSource`
   `object_get` calls, plus the `TkWKBIntegration` batch dict. The `ray.get` tolerance block at
   ~3559-3568, the `atol`/`rtol` tolerance objects and anything to do with `Tk_numeric_atol` are
   **prompt 05a's** and must be untouched here. The second `GkWKBIntegration` site is the one an
   agent misses; check for it by name rather than trusting the count.

6. **The readers followed, and the right ones.** All six scripts of prompt §5. Then the trap:

   ```bash
   git diff HEAD~1 HEAD -- extract_Gk_data.py
   ```

   `extract_Gk_data.py:407`'s batch dict feeds **`GkNumericIntegration`**, which keeps its pair, so
   that dict must be **unchanged** while the `BackgroundModel` lookup at :305 changes. An agent that
   stripped both has broken a live lookup; an agent that changed neither has left a reader querying
   a dropped column. Check every script compiles:

   ```bash
   ./venv/bin/python -m py_compile extract_Gk_data.py extract_GkWKB_data.py extract_TkWKB_data.py extract_GkSource_data.py extract_tensor_source_data.py extract_QuadSourceIntegral_data.py
   ```

7. **The suites, the diff and the bookkeeping.** Re-run both suites.
   `ComputeTargets` **must not fall below 452**, should read **491 or more**, and a *fall* is a stop
   even above the floor; `CosmologyModels` **39**. `test_numeric_break_point_key.py` shows **zero**
   lines changed:

   ```bash
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/test_numeric_break_point_key.py
   ```

   Board row 05, item **T9**, §3/§4, and `docs/OPEN_ISSUES.md` in the **same commit** with its count
   and date corrected — `[20-wkb-gauss-orders-not-in-lookup-key]` closes, so the count falls by one
   unless the prompt opened something. `black --check` clean. And the digests:

   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid 2>&1 | grep -E "^(OK|FAILED|Ran )"
   ```

## 4. What a good outcome looks like

- Four tables whose key says what actually determines the row, and two numeric tables untouched
  beside them.
- A test that fails if someone adds a column without keying it — which is the failure mode this
  prompt is most likely to ship, and the only one the suite would not otherwise see.
- `CS_TAU_GAUSS_ORDER` and `FRICTION_F_GAUSS_ORDER` recorded somewhere in the store for the first
  time; `TOLERANCE-INVENTORY.md` §2.4 measured that they reach no key, no label and no tag today.
- `GkSource` with no accuracy column at all, and a line where they used to be saying why: it
  assembles, it does not integrate.
- `config/defaults.py` byte-identical across four campaigns and two rebases.

**What a good outcome does *not* look like:** a green suite with three new columns that `store()`
writes, `build()` selects and nothing filters on. That leaves the tree green, the schema bigger and
the defect exactly where it was.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **The agent proposes to key `RESIDUAL_WKB_REGION_MARGIN`**, or to add any column D3's table does
  not list, or to leave any of the four with its pair. D3 is the user's and its table is exhaustive.
- **The agent proposes to change `config/defaults.py`**, any `*_GAUSS_ORDER` value, or anything in
  `main.py` beyond the four sites. Relay it; it is a D9 question.
- **The agent reports that the order cannot be keyed without changing what the computation does.**
  Relay the argument verbatim (rule 5).
- **The agent asks you to choose the mechanism** of prompt §3, or a column name, or a type. Relay
  verbatim; do not pick one.
- **`ComputeTargets` falls at all**, even above 452 — this prompt adds tests and removes none, so a
  fall means something else broke.
- **A moved source-grid digest.**
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and sits on it — observed at 0.0601–0.0612 across
consecutive runs at an unchanged commit, passing roughly one run in three. It is unrelated to this
prompt's diff. Confirm by re-running that module alone before attributing anything to the commit.

## 6. After it lands

**Hand back to the user briefly, then continue.** This prompt settles nothing that is the user's to
settle — D3 was settled 2026-09-18 and D9 with it — so the hand-back is a report, not a decision.
Report:

1. The four tables' schemas before and after, and confirmation that the two numeric ones are
   unchanged.
2. The mechanism the agent chose for §3's invariant, and its argument that a row cannot claim an
   order it was not computed at.
3. What it found about `RESIDUAL_WKB_REGION_MARGIN`.
4. The reader set it actually changed, against D3's count of three.
5. `[20-wkb-gauss-orders-not-in-lookup-key]` closed, and the counts on the board and the index.

Then **prompt 05a is next and is not written yet.** Its charter is campaign README §3.5a and its
board items are **T8** and **T10**; D1 closed 2026-09-17 so nothing gates it but this commit. Write
it after the user has read this, and **write it against what 05 actually left** — in particular the
log's "State handed to the next prompt" list of which `object_get` sites still pass a tolerance,
which is the input 05a's guard is built from. Do not dispatch it in the same turn.
