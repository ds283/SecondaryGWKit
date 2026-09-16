# Orchestrator — prompt 02, the accuracy-parameter inventory

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../02-accuracy-parameter-inventory.md`](../02-accuracy-parameter-inventory.md)
**Model:** Opus. **Production code changed:** none. **Test code changed:** none.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt ends the batch.** There is no `prompt-03.md`, by decision (README §3 table). When 02
lands, you hand its table to the user and stop.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and both suite counts (**452** and **39**). Nothing else: this
prompt has no numerical baseline because it takes no measurement.

**Precondition:** prompt 01 has landed. Prompt 02 cites grid generations by the names prompt 01
gave them, so if 01 is not in, the inventory's provenance column cannot say which generation a
figure was taken on.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `02-accuracy-parameter-inventory`, model
**Opus**.

**Rule 4 matters more here than anywhere else in the campaign.** Prompt 02's entire value is that
it reads the *tree* rather than the plan — the 2026-09-12 plan miscounted its own subject because
it read the documents. An agent that has seen prompt 03 will inventory what prompt 03 expects to
find. Do not let it read another prompt in this campaign, and do not summarise them for it.

## 3. The review — five checks

1. **Is the diff really empty of code?**

   ```bash
   git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs/tolerance-convergence'
   ```

   must be **empty**. Not "only a comment", not "only a test". README §5 rule 8 puts every
   parameter change in prompt 05, and this prompt reads.

2. **Does the script regenerate the table?** Run it yourself from the repository root and diff its
   output against the document's table section. The log must quote the command; run that command,
   not one you invent.

   Check *how* the key columns were derived. The prompt requires them read programmatically from
   `Datastore/SQL/ObjectFactories/`, because a hand-typed key list goes stale silently — and a
   stale hand-typed list is precisely how this campaign's subject came to be miscounted the first
   time. **A transcribed list is a stop**, even if it is correct today.

   Confirm the script needs no Ray and no datastore, and that it does not import `main.py`.

3. **Are §4's four questions answered?** Explicitly, in the document *and* the log:
   - are there eight keyed object types, or more?
   - which parameters are live, which vestigial, which vestigial-in-computation but load-bearing-in-key?
   - does any parameter key a type §2 (a) does not list?
   - does any object type carry an accuracy parameter that is in no key at all — five, per
     `[20-wkb-gauss-orders-not-in-lookup-key]`?

   **"§2 (a) is wrong" is a successful answer**, not a problem. Read it on its merits; do not send
   the agent back to make the table agree with the README.

4. **Does the provenance column say "never chosen" where nothing chose the value?** README §1.2
   requires those words rather than an invented justification, and several entries should carry
   them — `find_phase_extremum`'s `xtol=1e-6, rtol=1e-4`, `main.py`'s Bessel `phase_atol`/
   `amplitude_rtol`, `DEFAULT_LEVIN_THRESHOLD`. **An inventory in which every parameter turns out
   to have a provenance has invented some.** Spot-check two entries against the documents they
   cite.

   Check the three `LambdaCDM_GenericEOS.py` entries **lift**
   `prompts/background-solver-robustness/PROVENANCE.md` rather than re-deriving it, and use the
   re-anchored line numbers (`:636`, `:1137`, and `T_z_reference.py:285`) rather than README
   §3.2's original `:579` / `:864` / `:1008`.

5. **The suites and the bookkeeping.** Both unchanged — **452** and **39**. Board row 02, item row
   **T3**, any §3 issues opened, and `docs/OPEN_ISSUES.md` in the **same commit** with count and
   date corrected. `black --check` clean on `inventory.py`.

   Expect the log's **Observations not acted on** to be long. A short one from this prompt is a
   warning sign: it swept every accuracy parameter in the pipeline and found nothing worth
   recording?

## 4. What a good outcome looks like

- A table whose rows a later reader can act on without re-reading `main.py`, and a script that
  regenerates it.
- An explicit verdict on README §2 (a), with a corrected table if it is wrong.
- An exclusion boundary stated in one line per excluded class.
- "State handed to the next prompt" giving **prompt 03's target list and prompt 04's target list by
  name**, each with its object count and current parameter value. That hand-off is the prompt's
  primary output — README §4 says 02 is what tells 03 and 04 what they own, and **03 and 04 are
  written from it.**

## 5. Stop and ask the user — and after this prompt, always

Beyond [`README.md`](README.md)'s standing list:

- **Always, when 02 lands.** This is a natural stopping point (README §4.1) and prompts 03 and 04
  do not exist yet. Hand the user:
  - the inventory table;
  - §4's four answers, and in particular whether README §2 (a) is complete and right;
  - the two target lists, which are what 03 and 04 will be written against;
  - any §3 issue the prompt opened.

  Ask whether §2 (a) is to be corrected before 03 and 04 are written, and whether the target
  allocation is the one the user wants.

- **A ninth keyed object type, or a parameter on a target neither 03 nor 04 covers.** The user
  decides where it goes; do not assign it.
- **The agent recommends a value, or says a parameter "should be" tightened or dropped.** That is
  §7 D1 and D3, and it is the user's after 03 and 04 report. Out of scope here even when the
  argument is good.
- **The agent edited `config/defaults.py`**, including a comment-only change to a stale figure.
  There is a known stale comment in that neighbourhood — `[06-node-solve-comment-quotes-a-
  superseded-node-count]` — and it belongs to whichever prompt next has that file in scope, not to
  this one.
