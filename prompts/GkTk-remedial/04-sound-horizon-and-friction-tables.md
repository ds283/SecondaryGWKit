# Prompt 04 — The sound-horizon and friction tables

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §12.2 (the sound-horizon leading term; $F$ is $k$-independent), §12.7 bullet 1
("two more $k$-independent tables"), §13.2(a) ("`functions.cs_tau` … and the friction table $F$ a
third")
**Design facts:** README §2 (a), (b), (c); decision §7 D1.
**Depends on:** 03 (`CumulativeTable`, the `tau` accessor pattern, the schema pattern), 02
($N_{\tau_s}$, $N_F$).
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/BackgroundModel.py`, `Datastore/SQL/ObjectFactories/BackgroundModel.py`,
new `ComputeTargets/tests/test_background_cs_tau_friction.py`, plus the log and the status board.
**Do not touch:** `TkWKBIntegration.py`, `TkSourceFunctions.py` (their switch to these tables is
prompts 07 and 10), `main.py`.

Read first: README §2 (a)–(c), §5 rule 7 (**stand-ins must keep constructing**), §7 D1;
`RECONCILIATION.md` §2 item 1; review §12.2, §12.4 (the radiation closed forms), §12.7;
`logs/03-tau-primitive.md` "State handed to the next prompt".

---

## 1. Character of this commit

Two siblings of prompt 03's table, by the same machinery:

- $\tau_s(z)=\int c_s\,dz/H$ with $c_s=\sqrt{\,\texttt{wPerturbations}(z)\,}$ — the leading
  primitive for the transfer-function phase (review §12.2). Needs the (hi, lo) treatment: the
  phase $k\,\Delta\tau_s$ reaches $1.85\times10^{11}$ rad.
- $F(z)=\tfrac32\int(1+c_s^2)\,dz/(1+z)$ — the Liouville–Green friction integral, the integrand of
  `TkWKBIntegration.friction_RHS` (`:25-49`) and of spec 01 R23. A single double suffices:
  $F\le60$ and it enters as $e^{F(z)-F(z_i)}$, so $10^{-14}$ absolute error in $F$ is $10^{-14}$
  relative in the amplitude. State this in the docstring.

Convention, as for $\tau$: tables accumulate from the top of the grid; consumers use `delta`.
`F` as `TkWKBIntegration` stores it today is $F(z)-F(z_{\rm init})$ with $F(z_{\rm init})=0$
(`integrate_friction_function` starts from `[0.0]`); prompt 07 will produce exactly that from
`friction_F.delta(z_init, z)`.

## 2. What to build

1. **`compute_background`**: two more `CumulativeTable`s with $f=c_s/H$ (order $N_{\tau_s}$) and
   $f=\tfrac32(1+c_s^2)/(1+z)$ (order $N_F$). Guard `wPerturbations(z) < 0` with a clear
   `ValueError` naming the model and $z$ — a negative $c_s^2$ has no sound horizon. Payload keys
   `"cs_tau_hi_sample"`, `"cs_tau_lo_sample"`, `"friction_F_sample"`, plus the orders. Add the
   integrand evaluations to the returned `IntegrationData`.
2. **`BackgroundModelValue`**: `cs_tau`, `cs_tau_lo`, `friction_F` as keyword fields with `None`
   defaults (the `build()` path and stand-ins). Properties.
3. **`ModelFunctions`**: append `cs_tau` and `friction_F` **with namedtuple `defaults=(None, None)`**
   so that every existing thirteen-argument constructor — `test_tk_source_functions.FakeModel`,
   `test_phase_groups`, `docs/…/realbg.py` — keeps working unchanged (`RECONCILIATION.md` §2 item 1).
   Assert this in a test by constructing a `ModelFunctions` with thirteen positional arguments.
4. **`_create_functions`**: two more accessor objects of prompt 03's kind. `cs_tau(z)`,
   `cs_tau.delta(a, b)`; `friction_F(z)`, `friction_F.delta(a, b)`, the latter reconstructed from a
   single-limb table (`lo` all zeros) with `f = 1.5*(1+wPerturbations(z))/(1+z)` for partials.
5. **Datastore**: columns `cs_tau_Mpc`, `cs_tau_lo_Mpc` (`Float(64)`, `nullable=False`) and
   `friction_F` (`Float(64)`, `nullable=False`, dimensionless — no unit conversion); write, read and
   select them in both paths; update the regeneration comment from prompt 03 to name all four new
   columns.

## 3. Tests (`test_background_cs_tau_friction.py`)

1. `RadiationModel`: $\tau_s=\tau/\sqrt3$ to $\le2\times10^{-15}$ relative at the nodes and in
   `delta`; $F(z_a)-F(z_b)$ — as `friction_F.delta` — equals $2\ln\frac{1+z_b}{1+z_a}$ to
   $10^{-14}$ absolute (review §12.4: $F=2\ln(s/s_i)$).
2. `LambdaCDMModel`: $\tau_s$ and $F$ at the JSON checkpoints to $\le2\times10^{-14}$ relative and
   $\le10^{-13}$ relative respectively; `cs_tau.delta` over the short baselines to $\le10^{-13}$.
3. `QCDModel`: same against the adaptive references, to 3× their recorded floor; the $c_s^2$
   transition intervals (from prompt 02's log) are explicitly among the checked baselines.
4. `ModelFunctions(*thirteen_args)` constructs; its `cs_tau is None`.
5. Persisted round trip exact in `Mpc_units`; `friction_F` stored without unit scaling.
6. The old friction ODE agrees with the table: integrate `friction_RHS` with `solve_ivp` (DOP853,
   `atol=1e-10, rtol=1e-8`, as production does) on `LambdaCDMModel` from $z_{e3}(k=10^5)$ to $0.1$
   and show it differs from `friction_F.delta` by $2$–$4\times10^{-7}$ relative (review §12.3) —
   i.e. the test documents that the *ODE* was the inaccurate one.

## 4. Verification and acceptance

- New tests pass; `discover -s ComputeTargets/tests -t .` passes — in particular
  `test_tk_source_functions.py` and `test_phase_groups.py` **unchanged and passing**, which is the
  proof of item 3's defaults.
- README §6 row for $\tau_s$, $F$ met on LambdaCDM.
- `black --check` clean.

## 5. Log and commit

"State handed to the next prompt", verbatim: accessor names and interfaces; payload keys; column
names; orders; the measured $\tau_s$/$F$ accuracy and build cost on both models; the exact
value of `friction_F.delta(z_init, z)` convention prompt 07 must reproduce.

Commit subject, or something equally specific: `Tabulate the sound horizon and the LG friction integral per model`.
