# Prompt 02 — The accuracy-parameter inventory

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board item **T3**; README §2 **(a)**, **(c)**, **(g)**
**Scores:** `RECONCILIATION.md` §2.1 — is its table complete and right?
**Depends on:** prompt 01, for the grid generations only. It measures no accuracy, so it does not
need the convergence facility; but it cites grid generations by the names prompt 01 gave them.
**Recommended model:** **Opus** — this is the prompt with no predecessor, and the reason it exists
is that the 2026-09-12 plan allocated targets to prompts from a count nobody had verified and got
it wrong. Prompts 03 and 04 take their scope from your table. A missed parameter is a parameter
that never gets measured and never appears in `docs/TOLERANCE-PROVENANCE.md`.

**Files you may create or touch:**
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` (**new**),
`docs/tolerance-convergence/inventory.py` (**new** — the script that regenerates it),
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** any production file, any test file, and `config/defaults.py` above all — **this
prompt reads and writes nothing but its own two documents** (README §5 rule 8). You will find
things that want fixing. Record them in the log's "Observations not acted on" and open §3 issues;
do not fix them (README §5 rule 4).

**Read first:** README §1.2 (the five fields — your table is what prompt 06 assembles from), §2
(a), (c), (g), §3.2 **including its 2026-09-16 amendment**, §5; `RECONCILIATION.md` §2.1 and
**§7.2**; `prompts/background-solver-robustness/PROVENANCE.md` **in full** — it is the shape your
entries should take, and three of your entries are already written in it; `config/defaults.py` in
full; board §3's "Recorded by the rebase, not owned here" bullets.

---

## 1. Why this prompt exists, and what went wrong without it

The 2026-09-12 plan said "four of the five targets share two tolerance constants". Measured at the
rebase: **eight** object types are keyed on an accuracy parameter, the shared pair keys **six** of
them, `rtol` alone keys a seventh, and of the six **only one uses the value it is given**. Two
targets the plan never mentioned — `wavenumber_exit_time` and `BackgroundModel` — turned out to be
in scope, and `BackgroundModel` is the largest instance of the campaign's central case.

That miscount was not careless. It came from reading the *documents* rather than the tree. **Read
the tree.** Where a document and the code disagree, the code is the finding and the document is an
issue.

## 2. The search, and why it is not a grep

A parameter belongs in this inventory if it **changes the accuracy of a computed quantity, or sits
in a lookup key claiming to**. That is not the same as a parameter spelled `atol`.

Two worked examples, both real, both in scope, and both of which a keyword grep gets wrong:

- **`LiouvilleGreen/bessel_phase.py:870` `bessel_phase`** takes `atol`, `rtol`, `phase_atol` **and**
  `amplitude_rtol`. The first two are marked *deprecated and ignored*; the live ones are the second
  two, which `main.py:1119` and `:1127` pass as `1e-12`. A grep for `atol=` finds a parameter that
  does nothing and misses one that does.
- **`ComputeTargets/BackgroundModel.py:35-44`** — `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`,
  `FRICTION_F_GAUSS_ORDER`, all `4`. No tolerance anywhere, and they are the only thing that sets
  that object's accuracy. Meanwhile the `atol`/`rtol` columns that *are* in its lookup key
  (`BackgroundModel.py:409`) describe nothing.

So: sweep by **call site of a numerical method** and by **lookup-key column**, and use keyword
search only to cross-check that sweep. At minimum the inventory must reach:

1. the five `config/defaults.py` accuracy constants, and `DEFAULT_LEVIN_THRESHOLD`;
2. the four `*_GAUSS_ORDER` constants and `RESIDUAL_WKB_REGION_MARGIN`;
3. every `root_scalar` / `brentq` / `solve_ivp` / quadrature call in production code that carries a
   tolerance, wherever it is — `LiouvilleGreen/integration_tools.py:95`,
   `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:636` and `:1137`,
   `CosmologyConcepts/wavenumber.py:979-984`, `main.py:1119` and `:1127`, and `AdaptiveLevin/`;
4. every `atol`/`rtol` **column** on a `Datastore/SQL/ObjectFactories/` table, whether or not any
   code reads it.

**Where you stop.** Numerical-method parameters that are not accuracy parameters —
`DEFAULT_STRING_LENGTH`, batch sizes, `DEFAULT_FLOAT_PRECISION` and
`DEFAULT_REDSHIFT_RELATIVE_PRECISION` where they are used for identity rather than accuracy — are
out. But **say in the document that you excluded them, and why**, one line each. An inventory whose
boundary is undocumented cannot be checked. If one of those turns out to key an object's identity
in a way that behaves like an accuracy knob, that is a finding; report it, do not silently include
or exclude it.

## 3. What each entry records

One row per parameter, and the columns are chosen so that prompt 06 can assemble
`docs/TOLERANCE-PROVENANCE.md` from your table plus prompts 03 and 04's measurements — README §1.2:

| Column | Notes |
|---|---|
| **Parameter** | name and defining `file:line` |
| **Value** | as at your `HEAD` |
| **Keys which object types** | the datastore types whose lookup key contains it, or "none" |
| **Reaches** | **solver** / **lookup key** / **both** / **neither** — this is the column the campaign turns on, and §2 (a) says four of eight are "lookup key only" |
| **Method it feeds** | adaptive ODE (which integrator), root solve (which method, bracketed or not), fixed-order quadrature (which order), or none |
| **The real knob** | for a Liouville–Green-type representation this is an integer order, not this parameter (§2 (a)) |
| **Object count of the sector** | §2 (c): 50 per model, ~65,000 per model, once per cosmology, once per run. **A parameter without an object count has no cost and cannot be traded** |
| **Provenance** | the campaign, prompt and log that chose it — **or the words "never chosen"** |
| **Owned by** | this campaign's prompt 03, 04 or 05; another campaign; or nobody |

**The provenance column is the point of the document.** README §1.2 requires that where the
provenance of a constant cannot be established from the record, the note says so *in those words*
rather than inventing one. Apply that here. "Never chosen" is the expected and correct entry for
several of these, and writing it is the finding.

**Lift, do not re-derive.** The three `LambdaCDM_GenericEOS.py` solves are settled by the campaign
that owns that file, and `prompts/background-solver-robustness/PROVENANCE.md` already records them
in exactly this shape — including, for the crossing probe, that it is a tolerance **nobody ever
chose**, which is the phrasing §1.2 asks for. Cite it. Note that the three sites README §3.2
originally listed have moved: `:579` is now `:636`, `:864` left production code for
`CosmologyModels/tests/T_z_reference.py:285`, and `:1008` is now `:1137`
(`RECONCILIATION.md` §7.2).

## 4. The question the prompt must answer

**Is README §2 (a)'s table complete and right?** Answer it explicitly, in those terms, in the
document and in the log. Specifically:

1. **Are there eight keyed object types, or more?** Enumerate from
   `Datastore/SQL/ObjectFactories/`, not from the README.
2. **Which parameters are live, which are vestigial, and which are vestigial in the computation but
   load-bearing in the key?** That three-way split is the campaign's subject and §7 D3 is the
   decision it feeds. `GkSource` is the fourth vestigial case and is different in kind — it
   assembles rather than integrating, so there is no order to put in its key. Say so.
3. **Does any parameter key an object type that §2 (a) does not list?**
4. **Does any object type carry an accuracy parameter that is in no key at all?**
   `[20-wkb-gauss-orders-not-in-lookup-key]` says five do. Confirm or correct the count.

If §2 (a) is wrong, **say so plainly and give the corrected table**. That is a successful outcome
for this prompt, not a problem: it is the second time this campaign's own count has been checked
against the tree, and the first time it was checked it was wrong.

## 5. The script

`docs/tolerance-convergence/inventory.py` regenerates `TOLERANCE-INVENTORY.md`, so that a later
reader re-runs it rather than re-reading `main.py`. It must:

- derive the **key columns** by reading the `ObjectFactories/` table definitions programmatically,
  not by transcription — a hand-typed key list goes stale silently, which is how this campaign's
  subject came to be miscounted;
- derive the **values** of the `config/defaults.py` and `*_GAUSS_ORDER` constants by importing
  them, not by parsing;
- read `main.py` with `ComputeTargets/tests/test_main_plumbing.load_main_py_functions` or with
  `ast`, **never by importing it** — `main.py` parses `sys.argv`, opens a Ray connection and a
  `ShardedPool` at module scope (`CLAUDE.md`);
- need **no Ray and no datastore** to run;
- print the object counts it uses as **inputs it was given**, with their source, rather than
  pretending to derive them — 50 and ~65,000 come from `main.py:1215` and `GkTk-remedial` README
  §6, and a script that recomputed them would be measuring the pipeline, not inventorying it.

The prose findings — §4's four answers, the exclusion boundary, the "never chosen" verdicts — are
**written by you into the document**, not generated. The script regenerates the table; the document
carries the argument. Say in the document which parts are which, so a later re-run does not
overwrite the reasoning.

## 6. What this prompt must not do

- **No measurement of accuracy.** Not one convergence run, not one drift figure. This prompt
  establishes what there is to measure; 03 and 04 measure it. If you find yourself wanting to know
  whether a tolerance is right, that is the finding "nobody has measured this" — record it and
  move on.
- **No recommendation.** Not "this should be tightened", not "this should be dropped". §7 D1 and D3
  are the user's, informed by 03 and 04.
- **No edits to `config/defaults.py`**, including comment-only ones, however stale a comment looks.
- **Do not renumber, re-sort or rewrite README §2 (a).** Report the corrected table in *your*
  document; prompt 06 reconciles the README.

## 7. Acceptance

| Check | Threshold |
|---|---|
| Every parameter of §2's list 1–4 appears in the table | and the document says how the sweep was made exhaustive |
| Every row has all nine columns filled | "unknown" is not a value; "never chosen" is |
| §4's four questions | answered explicitly, in the document and in the log |
| The exclusion boundary | stated, with one line per excluded class |
| `inventory.py` re-run from a clean checkout | reproduces the table section of the document byte-for-byte; quote the command |
| The three `LambdaCDM_GenericEOS.py` entries | cite `background-solver-robustness/PROVENANCE.md`, at the re-anchored line numbers, and are not re-derived |
| `ComputeTargets` suite | 452 → 452, OK |
| `CosmologyModels` suite | 39 → 39, OK |
| Production and test files in the diff | **zero** |
| `black --check` on `inventory.py` | clean |

## 8. Stop conditions

- **A parameter reaches a solver that §2 (a) says is not keyed on one, or vice versa, in a way that
  changes which prompt owns a target.** Report it and stop — the allocation of 03 and 04 depends on
  it, and getting this wrong is the failure this prompt exists to prevent.
- **You find a ninth keyed object type**, or an accuracy parameter on a target neither 03 nor 04
  covers. Stop and report; the user decides where it goes.
- **`inventory.py` cannot derive the key columns programmatically** without importing `main.py` or
  standing up a datastore. Say what blocks it and stop rather than transcribing the list by hand.
- **You need to change a production file, a test file or `config/defaults.py`** for any reason.

## 9. Deliverables

1. `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` and `docs/tolerance-convergence/inventory.py`.
2. `logs/02-accuracy-parameter-inventory.md` on `GkTk-remedial` §5.1's template, with:
   - **Verification performed** carrying §7's checks, and the §4 answers in full;
   - **Observations not acted on** — expected to be long. Every stale comment, every parameter
     with no provenance, every naming inconsistency you found and left alone;
   - **State handed to the next prompt** giving **prompt 03 its target list and prompt 04 its
     target list, explicitly and by name**, with each target's object count and the current value
     of the parameter that keys it. That hand-off is this prompt's primary output: §4 of README
     says 02 is what tells 03 and 04 what they own.
3. Board row 02, item row **T3**, and any §3 issues you opened — with `docs/OPEN_ISSUES.md` updated
   in the **same commit**, count and date corrected (`CLAUDE.md`).
4. One commit, README §5 rule 1.

**This prompt ends in a natural stopping point** (README §4.1). Prompts 03 and 04 do not start until
the user has read your table, because if it is wrong they will measure the wrong things.
