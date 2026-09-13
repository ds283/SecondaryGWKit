# Prompt 01 — Make `WKB_mod_2pi`'s cycle count consistent with its remainder

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[13-wkb-mod-2pi-cycle-count-inconsistent]` on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3
**Measurements:** [`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md)
§3.7 (the defect, the worked example, the rate) and §3.5 (its cost at the consumer)
**Design facts:** README §2 (a), (b), (c)
**Depends on:** nothing. This is the first prompt of the campaign.
**Recommended model:** Opus — the change is a few lines, but it moves stored data and the
reasoning about *which* few lines is the whole prompt.

**Files you may touch:** `LiouvilleGreen/WKBtools.py`,
`LiouvilleGreen/range_reduce_mod_2pi.py`, `LiouvilleGreen/tests/test_range_reduce.py`,
`LiouvilleGreen/tests/test_wkbtools.py` (create it if it does not exist),
`ComputeTargets/tests/test_tk_source_functions.py` **only** for the stale comment named in §3 item
4, plus this campaign's log and board, the `GkTk-remedial` board entry, and
`docs/OPEN_ISSUES.md`.

**Do not touch:** `Quadrature/integrators/WKB_phase_function.py` (the call site needs no edit — see
§2 item 4), `ComputeTargets/GkWKBIntegration.py`, `TkWKBIntegration.py`, `primitive_phase.py`,
`GkSource.py`, `GkSourcePolicyData.py`, `main.py`, any `Datastore/` factory, and everything README
§4 lists (`AdaptiveLevin/`, `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
`thirdparty/`, any `extract_*.py`, `transfer-remedial`'s files).

Read first: `LiouvilleGreen/WKBtools.py:1-95` (all four functions and their comments);
the **module docstring** of `LiouvilleGreen/range_reduce_mod_2pi.py` (the house rule on
re-reduction, and why the prime-factorisation scheme was removed); `simple_mod_2pi` in that file;
`LiouvilleGreen/tests/test_range_reduce.py`; `docs/gktk-remedial-verification.md` §3.7.

---

## 1. What is wrong

`LiouvilleGreen/WKBtools.py:15-27`:

```python
def WKB_mod_2pi(theta: float):
    theta_mod_2pi = fmod(theta, TWO_PI)                   # exact
    theta_div_2pi = int(floor(fabs(theta) / TWO_PI))      # a ROUNDED division, then floor
    ...
```

`fmod` is exact — that is its contract. `fabs(theta) / TWO_PI` is a correctly-rounded double
division. When the exact quotient lies within half an ulp **below** an integer, the division rounds
up across it, `floor` returns one cycle too many, and the pair no longer reconstructs its own
phase: `div * 2π + mod == theta - 2π`. The two halves of the representation are derived
independently, and only one of them is exact.

The worked example (§3.7), from the LambdaCDM $k=3\times10^8$ consumer set:

```
theta            = -3832989103139.361
|theta| / TWO_PI = 610039162582.0        (rounded)      -> floor = 610039162582
exact quotient   = 610039162581.99994    (frac 0.99994) -> floor = 610039162581
fmod(theta, TWO_PI) = -6.282807898877543
div * TWO_PI + mod - theta  =  -6.283203125     (= -2 pi)
```

**Measured rate.** 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ on LambdaCDM; 0
elsewhere on the production geometry. Uniform controls of 400,000 draws each: 0 at
$|\theta|\sim10^9$, 0 at $10^{11}$, **25 at $|\theta|\sim4\times10^{12}$** — where half an ulp of
$|\theta|/2\pi$ is $6.1\times10^{-5}$ cycles, so the rate *is* the half-ulp width and scales
linearly with $|\theta|$, i.e. with $k$.

**Cost when it fires.** 6.17 rad of consumer phase, against the $9.15\times10^{-4}$ rad floor
everything else on that geometry sits at (§3.5). The `GkSource` rectifier does **not** repair it:
its trigger is a cycle count jumping *up* as the source redshift rises, and this defect makes the
stored phase one cycle *more negative*, so its condition is false at the offending sample.

**`simple_mod_2pi` has the identical construction** (`range_reduce_mod_2pi.py`) and therefore the
identical defect. Its only caller is `WKB_product_mod_2pi`, which the comment at `WKBtools.py:33-37`
records as not used by production — retained for `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py`.
It is in scope here anyway: it is exported, its docstring promises `num == div_2pi * TWO_PI +
mod_2pi`, and `LiouvilleGreen/tests/test_range_reduce.py:13` asserts that promise. Leaving a known
-broken twin next to a fixed one is how this comes back.

## 2. The change

1. **Derive the cycle count from the exact remainder, not from a second rounded division.**
   `theta - theta_mod_2pi` is an exact multiple of $2\pi$ mathematically; forming it in floating
   point and dividing by `TWO_PI` gives a value whose distance from the true integer is many orders
   below $1/2$ at every $|\theta|$ this code sees, so rounding it to nearest recovers the integer
   exactly. State the bound you are relying on in a comment, with the $|\theta|$ at which it would
   stop holding.
   **Do not** simply replace `floor` with `round` on the *old* quotient — that moves the failure to
   quotients just above an integer rather than removing it.
   You may choose a different construction if you can show it is exact; the acceptance test in §3
   is the arbiter, not the particular expression.

2. **Preserve the negative-remainder convention exactly.** $\theta_{\rm mod}\in(-2\pi,0]$, the
   `theta_mod_2pi > 0` branch that increments the cycle count, and the sign handling for
   $\theta<0$. This is README §2 (a) and `GkTk-remedial` README §2 (e); it is a **stop condition**
   if it moves.

3. **Fix `simple_mod_2pi` the same way**, keeping *its* convention — a remainder carrying the sign
   of `num`, which is **not** the same convention as `WKB_mod_2pi`'s. Do not unify them; the
   comment at `WKBtools.py:9-14` exists because they deliberately differ.

4. **Nothing else changes.** `Quadrature/integrators/WKB_phase_function.py:299` calls
   `WKB_mod_2pi(theta)` and keeps calling it, unedited: the fix is inside the function. Likewise
   `apply_phase_offset`, `wrap_theta` and both producers' `store()`. If you find yourself editing a
   caller, stop — either the fix is in the wrong place or the prompt is wrong, and both are worth a
   question.

## 3. Tests

New or extended, in `LiouvilleGreen/tests/`:

1. **Self-consistency is the contract.** For both functions, over a large set of $\theta$ spanning
   $10^0$ to $4\times10^{12}$ in magnitude, both signs: `div * TWO_PI + mod` reconstructs `theta`
   to within one ulp of `theta`, and **never** off by a whole cycle. Include the §1 worked example
   `theta = -3832989103139.361` as an explicit regression case, by value.
2. **Adversarial abscissae, not random ones.** Random draws found 25 in 400,000; construct
   $\theta$ deliberately just below an integer number of cycles — e.g. `nextafter` walks from
   `N * TWO_PI` downwards for a spread of large `N` — so the test fails loudly on the old code
   rather than 1 time in 40,000. **Confirm it fails on the old code** and quote what it printed.
3. **The remainder does not move.** For the same set, `mod` is bit-identical to what the old
   implementation returned. This is the campaign's §2 (b) invariant and it is what guarantees no
   stored $G$ or $T$ changes.
4. `ComputeTargets/tests/test_tk_source_functions.py:542` carries a comment saying "`WKB_mod_2pi`
   uses `fmod`, which is exact". That is true of the remainder and was never true of the pair.
   Correct **that comment only** — no assertion, no tolerance, no other line in that file, which is
   otherwise `transfer-remedial`'s (`[10-transfer-remedial-tolerance-comments-stale]`).

Both suites must pass, and the `ComputeTargets` count must not fall:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
```

## 4. Verification and acceptance

- **The production measurement, re-run.** `docs/gktk-remedial/verify_production_path.py` §6 is the
  table that found this. Re-run the script and quote its section 6: the LambdaCDM $k=3\times10^8$
  row must read **0 inconsistent of 77,975**, every other row must stay 0, and the uniform control
  at $|\theta|\sim4\times10^{12}$ must go **25 → 0**. Do not edit the script.
- **Nothing else in that script's output moves.** Diff your run against a run at `HEAD~1`. The
  phase tables of §1, the consumer tables of §5 and the primitives of §0 must be unchanged, except
  the LambdaCDM $k=3\times10^8$ consumer row, which carries the 6.17 rad sample and should fall to
  the floor. **Quote its before and after.**
- **Cost.** The reduction runs once per stored sample. Measure it — a microbenchmark over $10^6$
  reductions, before and after — and record the per-call figure. There is no threshold; there is a
  requirement to know the number.

## 5. The datastore consequence — report it, do not paper over it

`theta_div_2pi` is a stored `nullable=False` column on `GkWKBValue` and `TkWKBValue`, and it is in
**no lookup key**. So a datastore written before this commit will be *served*, silently, with the
old cycle count at the affected samples — there is no schema change to raise a `RuntimeError` on,
as prompts 03/04 and 20 of `GkTk-remedial` could.

Do **not** invent a migration, and do **not** add a column or a key: both are out of scope and the
second would invalidate every row in every existing datastore for a defect that affects one sample
in 78,000. What this prompt owes is an honest record:

1. State the consequence in the log's "State handed to the next prompt", in the form
   `GkTk-remedial` log 13's regeneration list uses.
2. Add the row to that list by appending a dated line to the **same** §8 the campaign's close-out
   will write — or, if you are the first prompt to need it, say so in your log and leave the
   document alone; the close-out writes §8.
3. Say plainly which production quantities are affected (the reconstructed unwrapped phase in
   `build_phi_samples`, `PrimitivePhase.raw_theta`, and `QuadSourceIntegral`'s `_ClampedPhase`
   through them) and which are **not** (`theta_mod_2pi`, and therefore every stored $G$ and $T$).

If you believe a louder mechanism is warranted, **say so and stop** — that is the user's call, and
README §7 D3 records it as theirs.

## 6. Log and commit

Log to `logs/01-wkb-mod-2pi-cycle-count.md` per README §5.1. Update this campaign's board, move the
`GkTk-remedial` §3 entry `[13-wkb-mod-2pi-cycle-count-inconsistent]` to that board's §4 with the
measured result, correct the clause of `[10-wrap-theta-loop-at-large-phase]` that refers to it, and
update `docs/OPEN_ISSUES.md` — all in this commit.

Commit subject, or something equally specific:
`Derive the WKB cycle count from the exact remainder`.
