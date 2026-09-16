# Workstream E — make the model authoritative (prompt 09)

**Read [`README.md`](README.md) in this directory first.** The orchestrator rules, the three checks
and the dispatch template are there and are not repeated here.

**Prompt:** 09 (Opus).
**Precondition:** workstream B complete, with its completion criterion met — in particular README
§7 D2 answered and recorded. It is.

---

## 0. What this workstream is, and how it differs from A–D

Workstream B measured what the equality redshifts feed and put README §7 **D2** to the user. The
user chose **option (iii)** — the solve is authoritative at `main.py` — on the argument that
$\Omega_m/\Omega_r - 1$ is only accidentally right, because $\Omega_r$ is a present-day quantity and
`QCD_Cosmology`'s $g_*$ structure happens to sit twelve orders above $z_{\rm eq}$. A cosmology with
late entropy injection breaks it by much more than ulps, and silently.

**So this workstream moves a stored identity on purpose**, which no other workstream in this
campaign does. README §0.2's *"no prompt moves a stored number"* is scoped to A–C. Do not review
prompt 09 against it, and do not let the subagent treat the move as a deviation.

The mechanism is **not** D2 option (iii)'s wording. `main.py` does not import
`_find_rho_equality`; `BaseCosmology` declares two properties, `LambdaCDM` implements them with the
closed form (exact for that model), `LambdaCDM_GenericEOS` implements them from the solve its
constructor already runs and discards, and `main.py` asks. Prompt 09 §1 explains why, and the
review criteria below are written against that mechanism, not against D2's wording.

## 1. Baselines — take these before dispatching

```bash
git rev-parse HEAD
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .     # 447, OK
```

And **both grid digests**, by the method workstream B §1 used —
`ComputeTargets/tests/test_source_grid._production_grid`, which drives `main.py`'s own functions
through `load_main_py_functions`. At `db052ed` they were `a2c32f67` (QCD, 1,996) and `60a3205a`
(LambdaCDM, 1,778). **Re-take them rather than copying these**: workstream C's prompts land in
between and the point of a baseline is the tree it was taken on.

## 2. Prompt 09 — review criteria

Dispatch with the template, **amended** — see `README.md` §"The one amendment, for workstream E
only". The template's "no computed quantity moves ... stop" sentence is false here and would make
prompt 09 halt on its own acceptance; the replacement paragraph is in that section. Model:
**Opus**.

| # | Check | How |
|---|---|---|
| 1 | **The QCD digest is `4849552b`** | Against your own §1 baseline. Not "it moved" — **that exact value**, which prompt 03 and the workstream B orchestrator measured independently before the change existed |
| 2 | **`LambdaCDM`'s digest did not move** | `60a3205a`, 1,778. Its closed form is exact and it never reaches the feature path; a move here is a finding |
| 3 | **Exactly one sample moved** | Diff the two grids element-wise yourself. matter–$\Lambda$ is the same double either way (`0x1.36b4870e4a718p-2`), so a second moved sample means something else changed |
| 4 | **No fallback to the closed form** | Read `cosmology_feature_redshifts` in full. A `getattr(..., None)` followed by an arithmetic expression is the failure mode; prompt 09 §2.5 calls it a stop condition. **This is the most important check in the workstream** — it is the one mistake that looks like care |
| 5 | The initial guesses survive | `git show HEAD~1:CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py \| sed -n '502p;507p'` against the new file. Character-identical. The user decided they stay |
| 6 | `_find_rho_equality` untouched | Its body, bracket policy and both tolerances character-identical |
| 7 | §4.2 test 1 really fails on `HEAD~1` | **Run it yourself.** README §8 of the orchestrator: this is the campaign's single most important review step, and it matters more here than anywhere, because a test asserting `feature_z[0]` *is* the property passes trivially if the property returns the closed form |
| 8 | The guard test was reworked, not deleted | `test_the_three_closed_form_sites_agree` still exists and still asserts something. If it is gone, that is a stop |
| 9 | `test_source_grid.py:318`'s literal was re-pointed | It passes either way at `places=6`; check it was changed deliberately, not left green |
| 10 | The three checks | `ComputeTargets` **447**, `CosmologyModels` risen, `black` clean, `T_Z_REPRESENTATION_VERSION` **6** |

### Stop and report if

- The QCD digest is not `4849552b`.
- `LambdaCDM`'s digest moved.
- More than one sample moved.
- `main.py` contains any fallback to the closed form.
- The agent changed `_find_rho_equality` or either initial guess.
- §4.2 test 1 passes on `HEAD~1`.

## 3. Completion criterion for workstream E

- 09 is ✅ with its SHA and log.
- QCD digest `a2c32f67` → `4849552b` at 1,996 samples; LambdaCDM `60a3205a` at 1,778, unmoved;
  exactly one sample different.
- `main.py` computes no equality redshift and has no fallback.
- `[00-equality-redshift-closed-form-is-duplicated-three-times]` is on this board's §4 and out of
  `docs/OPEN_ISSUES.md`, count corrected.

Report to the user: the two digests before and after, the one sample that moved with its `hex()`
either side, and the confirmation that `HEAD~1` fails §4.2 test 1. **Tell them the regeneration is
theirs to run** — prompt 09 does not attempt it and adds no migration.
