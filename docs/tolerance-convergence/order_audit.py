"""
What do the four Gauss orders and ``RESIDUAL_WKB_REGION_MARGIN`` buy?
(``prompts/tolerance-convergence/`` prompt 04, board item **T7**.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/order_audit.py

It emits every table of ``docs/tolerance-convergence/ORDER-AUDIT.md`` on stdout and a progress log
on stderr. **It changes nothing.** The four orders and the margin live in
``ComputeTargets/BackgroundModel.py`` and ``ComputeTargets/phase_residual.py``, which README §5
rule 8 keeps closed until prompt 05; this prompt recommends and the user decides (README §7 **D3**).

**Four of the eight keyed object types have no tolerance to converge** (README §2 (a)).
``BackgroundModel``, ``GkWKBIntegration`` and ``TkWKBIntegration`` carry ``atol``/``rtol`` columns
that are part of the datastore lookup key and reach no solver; what sets their accuracy is four
integers -- $N_\\tau$, $N_{c_s\\tau}$, $N_F$, $N_\\rho$, all 4 -- and one float,
``RESIDUAL_WKB_REGION_MARGIN = 0.5``, and none of the five is in any key, label or tag
(``[20-wkb-gauss-orders-not-in-lookup-key]``). This script asks the campaign's question of them:
*what does this parameter buy, measured?*

**What is measured**, for ``RadiationModel``, ``LambdaCDMModel`` and ``QCDModel`` on the
**version-2** source grid at each cosmology's **own** production anchor (board standing note 18):

1. the three ``BackgroundModel`` primitives -- $\\tau$, $c_s\\tau$, $F$ -- at orders 2, 4, 6, 8,
   12, 16, scored as **difference error relative to the interval** over every production interval
   and over the cumulative from the top of the grid. They are $k$-independent, so this is one
   sweep per model and not fifty identical ones;
2. the two phase residuals -- $\\rho_G$, $\\rho_T$ -- at the same orders, at **every one of the
   fifty production wavenumbers** on all three models and in both sectors, scored as **phase
   error in radians** of $\\rho.\\mathrm{delta}(z_{\\rm anchor}, z)$ from the production anchor
   three e-folds inside the horizon. ``GkTk-remedial`` prompt 02 measured three wavenumbers, and
   the question this prompt exists to answer is whether it was right at those three and lucky at
   the other forty-seven;
3. the **cost** of each, in integrand evaluations counted rather than derived, at the recommended
   order and one order either side, times the object count of the sector the knob governs;
4. what ``RESIDUAL_WKB_REGION_MARGIN`` cuts, over a spread of margins either side of 0.5: nodes
   retained and discarded per ``(model, k, sector)``, the range over which the answer does not
   depend on it, and the value at which ``residual_node_range`` starts refusing;
5. how much of the band ``residual_node_range`` returns lies **outside the Liouville-Green region
   the source grid's density criterion is protecting**, and what the grid would be if the band
   were that region alone -- the measurement
   ``[01-density-criterion-imposed-outside-the-wkb-region]`` has been waiting for since it was
   opened, prompt 02a's guarded-node census having been withdrawn as evidence for it;
6. what the re-run of ``docs/gktk-remedial/residual_convergence.py`` changed in the fixture's
   ``convergence`` block, figure by figure against the 2026-09-10 one, read out of git rather
   than transcribed.

**The radiation column is an oracle, not a self-convergence.** README §3.1's anchor table gives
every one of the five integrands a closed form on ``RadiationModel``: ``tau_delta``,
``cs_tau_delta`` and ``friction_F_delta`` are exact interval quantities, ``rho_T`` is exact in
closed form, and ``rho_G`` is **identically zero** -- $C = 0$ in exact radiation -- which makes
the Green's-function residual a pure quadrature-error measurement with no reference to build at
all. It is the sharpest measurement in this document and §3 treats it as such.

On the two spline models there is no oracle, so the reference is a converged run of the same rule
at order 32, scored through :func:`reference_drift` against order 33, and README §5 rule 5 binds:
no number travels without its reference's drift beside it, and a cell that does not stand
``CRITERION_RATIO`` clear of that drift is **unresolved** and carries no conclusion.

**Everything goes through ``ComputeTargets/tests/convergence_reference.py``** (board standing note
14), for which one step of a :class:`GaussOrder` is one order and not a factor. No line of that
module is changed by this prompt.

No Ray and no datastore (``CLAUDE.md``).
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from math import exp, fabs, log, log1p, sqrt

import numpy as np
import scipy

from ComputeTargets.BackgroundModel import (
    CS_TAU_GAUSS_ORDER,
    FRICTION_F_GAUSS_ORDER,
    TAU_GAUSS_ORDER,
    _cosmology_break_points,
)
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq_leading
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq_leading
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.phase_residual import (
    RESIDUAL_WKB_REGION_MARGIN,
    RHO_GAUSS_ORDER,
    build_phase_residual,
    phase_residual_integrand,
    residual_node_range,
)
from ComputeTargets.tests.convergence_reference import (
    CRITERION_RATIO,
    GaussOrder,
    SourceGridSpec,
    UNITS,
    anchor_error,
    grid_cosmology,
    radiation_anchors,
    reference_drift,
    summarise,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    PRODUCTION_K_GRID_INV_MPC,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_END,
    PRODUCTION_Z_INIT_LAMBDACDM,
    PRODUCTION_Z_INIT_QCD,
    QCDModel,
    RadiationModel,
    REFERENCE_DATA_PATH,
    REFERENCE_K_VALUES,
    SOURCE_GRID_V0,
    SOURCE_GRID_V2,
    _cosmology_break_points as _wkb_break_points,
    difference_error,
    horizon_exit_z,
    phase_error,
    source_grid,
)
from CosmologyConcepts import redshift_grid_digest
from CosmologyConcepts.wavenumber import build_z_sample
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

#: the order ladder ``GkTk-remedial`` prompt 02 used, kept so that the two measurements are
#: comparable figure for figure
ORDERS = (2, 4, 6, 8, 12, 16)

#: the converged reference on the two spline models, and the one-order step that scores it.
#: ``GaussOrder`` is the campaign's knob for an integer parameter and steps by **one order**
#: (``convergence_reference.py``); 32 is two doublings above the top of the ladder.
REFERENCE_ORDER = 32

#: how far above the floor an order may sit and still be said to clear it. It is
#: ``residual_convergence.FLOOR_FACTOR``'s value and its meaning -- "within this factor of the
#: best any order in the ladder achieves" -- kept so that this re-take and the 2026-09-10 block
#: answer the same question. See :func:`choose_order` for why a literal "at or below the floor"
#: is not available for a knob whose floor is only visible through the ladder.
FLOOR_FACTOR = 3.0

#: where every residual is anchored: three e-folds inside the horizon, which is where the
#: producers anchor (``phase_residual.py``'s ``RESIDUAL_WKB_REGION_MARGIN`` comment) and what the
#: fixture's ``rho_anchor_efolds_subh`` records
RHO_ANCHOR_EFOLDS_SUBH = 3

#: the production orders, read and not changed (README §5 rule 8)
PRODUCTION_ORDERS = {
    "tau": TAU_GAUSS_ORDER,
    "cs_tau": CS_TAU_GAUSS_ORDER,
    "friction_F": FRICTION_F_GAUSS_ORDER,
    "rho": RHO_GAUSS_ORDER,
}

#: the margins §5 sweeps, production's 0.5 among them. The upper end is where the test
#: ``omega^2 >= margin * omega_0^2`` becomes ``correction >= 0``, which is a different question
#: from "is the mode inside the horizon" and is what the top of the ladder probes.
MARGINS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999)

#: the ``k`` indices at which the margin sweep also rebuilds the residual table. The node census
#: is taken at all fifty; the table build is 0.1-0.3 s on QCD and 2,400 of them would dominate
#: this script's runtime for a quantity §5.2 shows to be bit-identical wherever the anchor is
#: reachable at all.
MARGIN_TABLE_K_INDICES = (0, 12, 24, 37, 49)

#: how many nodes of the band each residual is scored at. The band is 800-2,300 nodes; the error
#: is a cumulative and is monotone in the number of panels crossed, so a log-spaced subsample
#: reports the same maximum for a thirtieth of the ``delta`` calls.
RESIDUAL_CHECKPOINTS = 30

#: the fixture, and the revision the **superseded** ``convergence`` block is read out of. Reading
#: it from git rather than transcribing it is what makes §1's comparison regenerable after this
#: prompt has overwritten the block.
PREVIOUS_BLOCK_REV = "74ddb39"

#: the three anchors and the version-2 grid at each, from prompt 03 (the radiation control) and
#: prompt 02a (the two production cosmologies, board standing note 18). A mismatch means the grid
#: construction has moved under this campaign's figures and is a stop, not a thing to re-baseline
#: (prompt 04 §11).
EXPECTED_GRIDS = {
    "RadiationModel": (2306, "3bef2c06"),
    "LambdaCDMModel": (1778, "60a3205a"),
    "QCDModel": (2034, "21ffc126"),
}
EXPECTED_RADIATION_ANCHOR = 44523947729.772957

#: the sector object counts §4.3 multiplies a per-object cost by. ``BackgroundModel`` is one
#: object per model (``main.py:1062``, outside every loop); the residual tables are one per
#: ``(model, k, sector)`` and are memoised per worker process
#: (``phase_residual.cached_phase_residual``); their consumers are the two WKB targets, whose
#: count on the version-2 grid is prompt 03's measurement of the identically-shaped
#: ``GkNumericIntegration`` sector.
OBJECT_COUNTS = {
    "BackgroundModel": {
        "RadiationModel": 1,
        "LambdaCDMModel": 1,
        "QCDModel": 1,
    },
    "residual_tables": {
        "RadiationModel": 100,
        "LambdaCDMModel": 100,
        "QCDModel": 100,
    },
    "GkWKBIntegration": {
        "RadiationModel": 29290,
        "LambdaCDMModel": 38105,
        "QCDModel": 58350,
    },
    "TkWKBIntegration": {
        "RadiationModel": 50,
        "LambdaCDMModel": 50,
        "QCDModel": 50,
    },
}

SECTOR_LEADING = {"Gk": Gk_omegaEff_sq_leading, "Tk": Tk_omegaEff_sq_leading}
SECTORS = ("Gk", "Tk")

_started = time.perf_counter()


def log_line(message: str = "") -> None:
    print(
        f"[{time.perf_counter() - _started:7.1f}s] {message}",
        file=sys.stderr,
        flush=True,
    )


def emit(text: str = "") -> None:
    print(text)


def fmt(x, spec=".2e") -> str:
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    value = float(x)
    if value == 0.0:
        return "0"
    return f"{value:{spec}}"


# ---------------------------------------------------------------------------------------------
# the subjects: model, cosmology, anchor, version-2 grid
# ---------------------------------------------------------------------------------------------


class Subject:
    """One model on the version-2 source grid at its **own** production anchor."""

    def __init__(self, name, model, cosmology, z_init, is_radiation=False):
        self.name = name
        self.model = model
        self.cosmology = cosmology
        self.z_init = float(z_init)
        self.is_radiation = is_radiation
        self.spec = SourceGridSpec(
            generation=SOURCE_GRID_V2, universal=True, z_init=self.z_init
        )
        self.grid = self.spec.build(cosmology)
        self.z_values = np.asarray(self.grid.z_values, dtype=float)
        self.breaks_u = _cosmology_break_points(
            cosmology, float(self.z_values[-1]), float(self.z_values[0])
        )
        self.anchors = radiation_anchors(model) if is_radiation else None
        # the cosmology as `compute_background` sees it. `grid_cosmology` forwards
        # `wPerturbations` for a stand-in that keeps it on `.functions` and supplies nothing
        # the wrapped object does not already have, so the three primitives below are built
        # from the **cosmology's own** H(z) and c_s^2(z) exactly as production builds them --
        # not from the background model's splines of them
        self.background_cosmology = grid_cosmology(cosmology)

    @property
    def label(self) -> str:
        return f"{self.grid.label}, anchor z_init = {self.z_init!r}"

    def horizon_ratio(self, k_inv_Mpc: float, z: float, sector: str) -> float:
        """``omega_0 (1+z)``: the sector's own leading frequency in units of the horizon."""
        leading = SECTOR_LEADING[sector](self.model, float(k_inv_Mpc), float(z))
        return sqrt(max(leading, 0.0)) * (1.0 + float(z))

    def efolds_outside_horizon(self, k_inv_Mpc: float, z: float, sector: str) -> float:
        ratio = self.horizon_ratio(k_inv_Mpc, z, sector)
        return -log(ratio) if ratio > 0.0 else float("inf")


def build_subjects() -> list:
    log_line("** building stand-in models and their version-2 source grids")

    radiation = RadiationModel()
    radiation_z_init = horizon_exit_z(
        radiation, PRODUCTION_LARGEST_K_INV_MPC, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
    )
    if fabs(radiation_z_init / EXPECTED_RADIATION_ANCHOR - 1.0) > 1e-13:
        raise RuntimeError(
            f"the radiation control's anchor is {radiation_z_init:.17g}, not prompt 03's "
            f"{EXPECTED_RADIATION_ANCHOR:.17g}: the anchor solve has moved and this prompt's "
            "figures would not be comparable with prompts 03 and 03a's"
        )
    subjects = [Subject("RadiationModel", radiation, radiation, radiation_z_init, True)]
    log_line(f"   RadiationModel: {subjects[-1].label}")

    lambda_cdm = LambdaCDMModel()
    subjects.append(
        Subject(
            "LambdaCDMModel",
            lambda_cdm,
            lambda_cdm.cosmology,
            PRODUCTION_Z_INIT_LAMBDACDM,
        )
    )
    log_line(f"   LambdaCDMModel: {subjects[-1].label}")

    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    qcd_grid = SourceGridSpec(
        generation=SOURCE_GRID_V2, universal=True, z_init=PRODUCTION_Z_INIT_QCD
    ).build(qcd_cosmology)
    # the QCD stand-in's background is splined on the grid it will be measured on, at its own
    # anchor (prompt 03, prompt 03a): a background cut from LambdaCDM's anchor leaves the top of
    # this grid outside the splines' range, QCD's anchor being 60 % higher
    qcd = QCDModel(qcd_grid.grid, cosmology=qcd_cosmology)
    subjects.append(Subject("QCDModel", qcd, qcd_cosmology, PRODUCTION_Z_INIT_QCD))
    log_line(f"   QCDModel: {subjects[-1].label}")

    for subject in subjects:
        samples, digest = EXPECTED_GRIDS[subject.name]
        if subject.grid.samples != samples or subject.grid.digest != digest:
            raise RuntimeError(
                f"{subject.name}: the version-2 grid at this anchor is "
                f"{subject.grid.samples} samples / {subject.grid.digest}, not the published "
                f"{samples} / {digest}. A moved grid digest is a stop (prompt 04 §11)."
            )
    return subjects


# ---------------------------------------------------------------------------------------------
# §2: the three BackgroundModel primitives, which are k-independent
# ---------------------------------------------------------------------------------------------


class CountingIntegrand:
    """Wrap an integrand and count the calls: one background evaluation each (README §2 (i))."""

    def __init__(self, f):
        self._f = f
        self.calls = 0

    def __call__(self, z):
        self.calls += 1
        return self._f(z)


def primitive_integrands(cosmology):
    """
    The three integrands ``compute_background`` hands to ``CumulativeTable``, rebuilt here in the
    same form rather than imported, because two of the three are module-private closures.

    Signs are production's and matter: ``tau`` and ``cs_tau`` accumulate ``+1/H`` and ``+c_s/H``,
    ``friction_F`` accumulates ``-(3/2)(1 + c_s^2)/(1+z)`` so that the table holds $F$ itself and
    ``delta(z_a, z_b) = F(z_b) - F(z_a)`` in the campaign's convention (README §2 (c)).
    """
    return {
        "tau": lambda z: 1.0 / cosmology.Hubble(z),
        "cs_tau": lambda z: sqrt(cosmology.wPerturbations(z)) / cosmology.Hubble(z),
        "friction_F": lambda z: -1.5 * (1.0 + cosmology.wPerturbations(z)) / (1.0 + z),
    }


def primitive_exact(subject, key):
    """
    The closed form on the exact-radiation control, in ``table.delta`` argument order.

    ``tau_delta(z_a, z_b)`` and ``cs_tau_delta(z_a, z_b)`` are already in it;
    ``friction_F_delta(z, z_ref) = F(z) - F(z_ref)`` is the other way round, so it is called
    reversed. README §3.1 consequence 2: the **interval** accessors are used and never the
    difference of two primitives, which loses a digit per decade of baseline ratio.
    """
    anchors = subject.anchors
    if key == "friction_F":
        return lambda a, b: anchors["friction_F"][1](b, a)
    return lambda a, b: anchors[key][1](a, b)


def primitive_intervals(z_values):
    """
    Where the difference error is scored: **every** production interval, and the cumulative from
    the top of the grid to a log-spaced set of nodes.

    The two are different questions. The increment says whether the rule resolves the integrand
    locally; the cumulative says what the consumer -- which always reads ``delta`` over a long
    baseline -- actually receives.
    """
    n = len(z_values)
    increments = [(float(z_values[i]), float(z_values[i + 1])) for i in range(n - 1)]
    idx = sorted(set(int(round(v)) for v in np.geomspace(1, n - 1, 24)) | {n - 1})
    cumulative = [(float(z_values[0]), float(z_values[j])) for j in idx]
    return increments, cumulative


def sweep_primitives(subject):
    """
    $N_\\tau$, $N_{c_s\\tau}$ and $N_F$ at every order, for one model.

    :return: ``{key: {...}}`` with the per-order maxima, the reference's drift (or the oracle's
        name on the control), and the counted integrand evaluations
    """
    z = subject.z_values
    increments, cumulative = primitive_intervals(z)
    integrands = primitive_integrands(subject.background_cosmology)
    out = {}

    for key, raw in integrands.items():
        log_line(f"   {subject.name} {key}: orders {ORDERS} + reference")

        counters = {}

        def build(knob, raw=raw, key=key):
            counted = CountingIntegrand(raw)
            table = CumulativeTable(
                z,
                counted,
                int(knob.order),
                break_points=subject.breaks_u,
                label=key,
            )
            counters[int(knob.order)] = counted.calls
            return table

        tables = {order: build(GaussOrder(order, key)) for order in ORDERS}

        if subject.is_radiation:
            exact = primitive_exact(subject, key)
            reference = None
            drift = None
        else:
            reference = build(GaussOrder(REFERENCE_ORDER, key))
            exact = None

        def score(table, pairs):
            errors = []
            for z_a, z_b in pairs:
                got = table.delta(z_a, z_b)
                want = (
                    exact(z_a, z_b) if exact is not None else reference.delta(z_a, z_b)
                )
                if want == 0.0:
                    continue
                errors.append((difference_error(got, want), z_b, z_a))
            return summarise(errors)

        rows = {}
        for order in ORDERS:
            rows[order] = {
                "increment": score(tables[order], increments),
                "cumulative": score(tables[order], cumulative),
                "evaluations": counters[order],
            }

        if not subject.is_radiation:
            smallest = min(
                (
                    rows[o][which]["max"]
                    for o in ORDERS
                    for which in ("increment", "cumulative")
                    if rows[o][which]["max"] > 0.0
                ),
                default=np.finfo(float).eps,
            )

            def error_measure(candidate, ref):
                errors = []
                for z_a, z_b in increments + cumulative:
                    want = ref.delta(z_a, z_b)
                    if want == 0.0:
                        continue
                    errors.append(
                        (difference_error(candidate.delta(z_a, z_b), want), z_b, z_a)
                    )
                return errors

            drift = reference_drift(
                build,
                GaussOrder(REFERENCE_ORDER, key),
                error_measure=error_measure,
                smallest_reported_difference=smallest,
                reference=reference,
            )

        out[key] = {
            "rows": rows,
            "drift": drift,
            "oracle": subject.is_radiation,
            "intervals": len(increments),
            "checkpoints": len(cumulative),
            "panels": len(increments) + int(subject.breaks_u.size),
        }
        best = min(rows[o]["cumulative"]["max"] for o in ORDERS)
        log_line(
            f"      cumulative max: "
            + "  ".join(f"N={o}:{rows[o]['cumulative']['max']:.2e}" for o in ORDERS)
            + (
                f"   (oracle)"
                if subject.is_radiation
                else f"   drift {drift.max:.2e} ({'ok' if drift.passed else 'NOT CONVERGED'})"
            )
            + f"   floor {best:.2e}"
        )
    return out


# ---------------------------------------------------------------------------------------------
# §3: the two phase residuals, at every production wavenumber
# ---------------------------------------------------------------------------------------------


def residual_case(subject, k_inv_Mpc, sector, margin=RESIDUAL_WKB_REGION_MARGIN):
    """
    The node range, the production anchor and the checkpoints one ``(k, sector)`` is scored on.

    The anchor is three e-folds inside the horizon, where the producers anchor. It is **not** a
    grid node, so ``CumulativeTable.delta``'s off-grid partial is what reaches it -- which is the
    production path (``phase_residual.py``'s "One table per wavenumber" note), not a convenience.
    """
    nodes = residual_node_range(
        subject.model, float(k_inv_Mpc), subject.z_values, sector, margin
    )
    z_anchor = horizon_exit_z(
        subject.cosmology, float(k_inv_Mpc), float(RHO_ANCHOR_EFOLDS_SUBH)
    )
    clamped = False
    if z_anchor > nodes[0]:
        # the band does not reach the production anchor: the anchor is clamped to the top node and
        # the case is flagged, because a residual table that cannot serve its own producer is a
        # finding about the margin and not a wavenumber to drop
        z_anchor, clamped = float(nodes[0]), True
    below = nodes[nodes <= z_anchor]
    if below.size < 2:
        return None
    idx = sorted(
        set(
            int(round(v)) for v in np.geomspace(1, below.size - 1, RESIDUAL_CHECKPOINTS)
        )
        | {below.size - 1}
    )
    return {
        "nodes": nodes,
        "z_anchor": float(z_anchor),
        "anchor_clamped": clamped,
        "checkpoints": [float(below[j]) for j in idx],
    }


def sweep_residuals(subject):
    """
    $N_\\rho$ at every production wavenumber in both sectors, for one model.

    On the control the reference is the closed form -- ``rho_G`` is identically zero and
    ``rho_T`` is exact -- so what the rule returns *is* the quadrature error, with nothing of a
    reference's own in it. On the two spline models it is an order-32 table scored against order
    33 (README §5 rule 5).
    """
    results = {sector: [] for sector in SECTORS}

    for sector in SECTORS:
        log_line(f"   {subject.name} rho_{sector}: 50 wavenumbers")
        for k_index, k in enumerate(PRODUCTION_K_GRID_INV_MPC):
            k = float(k)
            case = residual_case(subject, k, sector)
            if case is None:
                results[sector].append(
                    {"k": k, "skipped": "fewer than two nodes below the anchor"}
                )
                continue

            counters = {}

            def build(knob, case=case, k=k, sector=sector):
                integrand = phase_residual_integrand(subject.model, k, sector)
                counted = CountingIntegrand(integrand)
                table = CumulativeTable(
                    case["nodes"],
                    counted,
                    int(knob.order),
                    break_points=(
                        _wkb_break_points(
                            getattr(subject.model, "cosmology", None),
                            float(case["nodes"][-1]),
                            float(case["nodes"][0]),
                        )
                        if getattr(subject.model, "cosmology", None) is not None
                        else np.empty(0, dtype=float)
                    ),
                    label=f"rho_{sector}@k={k:.6e}",
                )
                counters[int(knob.order)] = counted.calls
                return table

            tables = {order: build(GaussOrder(order, "N_rho")) for order in ORDERS}

            if subject.is_radiation:
                anchor_fn = subject.anchors["rho_G" if sector == "Gk" else "rho_T"][1]
                exact = {
                    z: anchor_fn(k, z, case["z_anchor"]) for z in case["checkpoints"]
                }
                reference, drift = None, None
            else:
                reference = build(GaussOrder(REFERENCE_ORDER, "N_rho"))
                exact = None

            def score(table):
                errors = []
                for z in case["checkpoints"]:
                    got = table.delta(case["z_anchor"], z)
                    want = (
                        exact[z]
                        if exact is not None
                        else reference.delta(case["z_anchor"], z)
                    )
                    errors.append((phase_error(got, want), z, got))
                return summarise(errors)

            rows = {
                order: dict(score(tables[order]), evaluations=counters[order])
                for order in ORDERS
            }

            if not subject.is_radiation:
                smallest = min(
                    (rows[o]["max"] for o in ORDERS if rows[o]["max"] > 0.0),
                    default=np.finfo(float).eps,
                )

                def error_measure(candidate, ref):
                    return [
                        (
                            phase_error(
                                candidate.delta(case["z_anchor"], z),
                                ref.delta(case["z_anchor"], z),
                            ),
                            z,
                            0.0,
                        )
                        for z in case["checkpoints"]
                    ]

                drift = reference_drift(
                    build,
                    GaussOrder(REFERENCE_ORDER, "N_rho"),
                    error_measure=error_measure,
                    smallest_reported_difference=smallest,
                    reference=reference,
                )

            magnitude = fabs(
                (tables[max(ORDERS)] if reference is None else reference).delta(
                    case["z_anchor"], case["checkpoints"][-1]
                )
            )
            results[sector].append(
                {
                    "k": k,
                    "k_index": k_index,
                    "nodes": int(case["nodes"].size),
                    "band_top_z": float(case["nodes"][0]),
                    "z_anchor": case["z_anchor"],
                    "anchor_clamped": case["anchor_clamped"],
                    "rho_magnitude_rad": magnitude,
                    "rows": rows,
                    "drift": drift,
                    "oracle": subject.is_radiation,
                }
            )
        worst = {
            o: max(row["rows"][o]["max"] for row in results[sector] if "rows" in row)
            for o in ORDERS
        }
        log_line(
            f"      worst over k: " + "  ".join(f"N={o}:{worst[o]:.2e}" for o in ORDERS)
        )
    return results


# ---------------------------------------------------------------------------------------------
# §4: the target rule for an integer knob (README §6.1, as prompt 04 §7 adapts it)
# ---------------------------------------------------------------------------------------------


def choose_order(errors_by_order, floor, factor=FLOOR_FACTOR):
    """
    README §6.1 rules 2 and 3 for an integer: **loosest means lowest**, so sweep upwards and take
    the *first* order that clears the floor.

    "Clears" is ``<= factor * floor`` and not ``<= floor``, and the reason is that the floor of
    these knobs is not an independently known number. §6.2's row gives it as "double-precision
    accumulation over the grid", and the only way to see that level is the ladder itself -- so a
    literal ``<= floor`` would select whichever order happened to reach the minimum, which is an
    artefact of one order's rounding and not a statement about convergence. ``factor`` is
    :data:`FLOOR_FACTOR`, the same operational reading ``residual_convergence.py`` uses
    (``smallest_within_factor``), kept so that this re-take and the 2026-09-10 block answer the
    same question.

    Returns ``(order, monotone)``. ``monotone`` is false when a higher order in the ladder fails
    the same test -- a Gauss rule can get worse as the order rises, and the old block shows it
    (``plain`` ``tau`` runs 3.44e-07 at order 2, 4.46e-08 at 12 and back up to 1.03e-07 at 16), so
    "the first order that clears" is not automatically a statement that every higher order also
    clears (prompt 04 §7).
    """
    threshold = factor * floor
    chosen = None
    for order in sorted(errors_by_order):
        if errors_by_order[order] <= threshold:
            chosen = order
            break
    if chosen is None:
        return None, True
    monotone = all(
        errors_by_order[o] <= threshold for o in sorted(errors_by_order) if o >= chosen
    )
    return chosen, monotone


def cell(value, drift):
    """
    One table cell, marked with a dagger when the figure is **at or below the reference's own
    drift** and therefore carries no conclusion (README §5 rule 5).

    A dagger is not a defect in the measurement. On the two spline models orders 4 and above all
    sit at the double-precision floor, so the reference at order 32 and the candidate at order 4
    are indistinguishable; what survives the drift is the *step* from order 2, which is four to
    five orders larger. The exact-radiation control has no drift at all -- it has a closed form --
    and is what calibrates the rest.
    """
    text = fmt(value)
    if drift is not None and float(value) <= float(drift):
        text += "&dagger;"
    return text


def selection_resolved(errors_by_order, chosen, drift):
    """
    Whether the *choice* of order survives the reference's drift: the order below the chosen one
    must miss the floor by more than the drift, or the claim "the order below is not enough" is a
    claim about the reference.
    """
    if chosen is None:
        return False
    if drift is None:
        return True
    ladder = sorted(errors_by_order)
    i = ladder.index(chosen)
    previous = ladder[i - 1] if i > 0 else chosen
    return errors_by_order[previous] > float(drift)


def double_precision_floor(errors_by_order):
    """
    The floor these knobs compete against is **double-precision accumulation over the grid**
    (README §6.2's row), and the ladder measures it directly: it is the level at which raising
    the order stops buying anything.

    Taken as the smallest maximum any order in the ladder reaches. On the control that is a pure
    quadrature-plus-rounding figure with no reference in it at all, which is what makes the
    radiation column the calibration of the other two (README §5 rule 5).
    """
    return min(errors_by_order.values())


# ---------------------------------------------------------------------------------------------
# §5: RESIDUAL_WKB_REGION_MARGIN
# ---------------------------------------------------------------------------------------------


def margin_census(subject):
    """
    What the margin cuts, at every production wavenumber and both sectors, over :data:`MARGINS`.

    Three questions (prompt 04 §5): what it cuts, over what range the answer does not depend on
    it, and what happens as it approaches 1.
    """
    log_line(f"   {subject.name}: margin census over {len(MARGINS)} margins")
    rows = []
    for margin in MARGINS:
        for sector in SECTORS:
            retained, refused, reaches_anchor, tops = [], 0, 0, []
            for k in PRODUCTION_K_GRID_INV_MPC:
                k = float(k)
                try:
                    nodes = residual_node_range(
                        subject.model, k, subject.z_values, sector, margin
                    )
                except ValueError:
                    refused += 1
                    continue
                retained.append(int(nodes.size))
                z_anchor = horizon_exit_z(
                    subject.cosmology, k, float(RHO_ANCHOR_EFOLDS_SUBH)
                )
                if nodes[0] >= z_anchor:
                    reaches_anchor += 1
                tops.append(subject.efolds_outside_horizon(k, float(nodes[0]), sector))
            rows.append(
                {
                    "margin": margin,
                    "sector": sector,
                    "refused": refused,
                    "cases": len(PRODUCTION_K_GRID_INV_MPC),
                    "min_retained": min(retained) if retained else None,
                    "max_retained": max(retained) if retained else None,
                    "median_retained": (
                        float(np.median(retained)) if retained else None
                    ),
                    "grid_nodes": int(subject.z_values.size),
                    "reaches_production_anchor": reaches_anchor,
                    "band_top_efolds_outside_min": min(tops) if tops else None,
                    "band_top_efolds_outside_max": max(tops) if tops else None,
                }
            )
    return rows


def margin_sensitivity(subject):
    """
    Does the answer move when the margin does?

    The residual a producer reads is ``rho.delta(z_anchor, z)`` between two fixed redshifts, and
    the panels between them are the grid's own, so the margin can only change it by changing
    whether the anchor is reachable at all. That is a prediction, and this measures it: the same
    ``delta`` is rebuilt at every margin and compared **bit for bit** with the production one.
    """
    log_line(
        f"   {subject.name}: margin sensitivity at {len(MARGIN_TABLE_K_INDICES)} wavenumbers"
    )
    rows = []
    for sector in SECTORS:
        for k_index in MARGIN_TABLE_K_INDICES:
            k = float(PRODUCTION_K_GRID_INV_MPC[k_index])
            baseline = None
            entry = {"k": k, "sector": sector, "margins": {}}
            for margin in MARGINS:
                try:
                    case = residual_case(subject, k, sector, margin)
                except ValueError:
                    entry["margins"][margin] = {"refused": True}
                    continue
                if case is None:
                    entry["margins"][margin] = {"refused": True}
                    continue
                table = build_phase_residual(
                    subject.model,
                    k,
                    case["nodes"],
                    sector,
                    order=PRODUCTION_ORDERS["rho"],
                )
                z_bottom = float(case["nodes"][-1])
                value = table.delta(case["z_anchor"], z_bottom)
                if margin == RESIDUAL_WKB_REGION_MARGIN:
                    baseline = (value, case["z_anchor"], z_bottom)
                entry["margins"][margin] = {
                    "refused": False,
                    "nodes": int(case["nodes"].size),
                    "anchor_clamped": case["anchor_clamped"],
                    "rho": value,
                    "z_anchor": case["z_anchor"],
                    "z_bottom": z_bottom,
                }
            if baseline is not None:
                for margin, record in entry["margins"].items():
                    if record.get("refused"):
                        continue
                    same_interval = (
                        record["z_anchor"] == baseline[1]
                        and record["z_bottom"] == baseline[2]
                    )
                    record["bit_identical"] = same_interval and (
                        record["rho"] == baseline[0]
                    )
                    record["same_interval"] = same_interval
            rows.append(entry)
    return rows


def margin_refusal(subject):
    """
    The margin at which ``residual_node_range`` starts refusing, bisected per
    ``(sector, extreme k)``.

    The comment above ``RESIDUAL_WKB_REGION_MARGIN`` says a margin anywhere below ~1 would do and
    does not say what fails above. This is that measurement.
    """
    rows = []
    for sector in SECTORS:
        for k in (PRODUCTION_K_GRID_INV_MPC[0], PRODUCTION_K_GRID_INV_MPC[-1]):
            k = float(k)

            def ok(margin):
                try:
                    residual_node_range(
                        subject.model, k, subject.z_values, sector, margin
                    )
                    return True
                except ValueError:
                    return False

            lo, hi = 0.0, 1.0 + 1e-12
            if ok(hi):
                rows.append({"k": k, "sector": sector, "refuses_at": None})
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    lo = mid
                else:
                    hi = mid
            nodes_at_lo = residual_node_range(
                subject.model, k, subject.z_values, sector, lo
            )
            rows.append(
                {
                    "k": k,
                    "sector": sector,
                    "refuses_at": hi,
                    "nodes_just_below": int(nodes_at_lo.size),
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------
# §6: how much of the band lies outside the region the density criterion is protecting
# ---------------------------------------------------------------------------------------------


def band_census(subject):
    """
    Per ``(k, sector)``: how many of the band's nodes are **super-horizon** in that sector's own
    leading frequency, ``omega_0 (1+z) < 1``.

    This is the measurement ``[01-density-criterion-imposed-outside-the-wkb-region]`` asserts and
    has never had: prompt 01's 69-%-of-added-samples figure counts samples the criterion *added*
    above crossing, and prompt 02a's guarded-node census was withdrawn as evidence for it
    (board §3, the 2026-09-17 correction).
    """
    log_line(f"   {subject.name}: band census at 50 wavenumbers, both sectors")
    rows = []
    for sector in SECTORS:
        for k in PRODUCTION_K_GRID_INV_MPC:
            k = float(k)
            try:
                nodes = residual_node_range(
                    subject.model,
                    k,
                    subject.z_values,
                    sector,
                    RESIDUAL_WKB_REGION_MARGIN,
                )
            except ValueError:
                continue
            ratios = np.array(
                [subject.horizon_ratio(k, float(z), sector) for z in nodes], dtype=float
            )
            outside = int(np.sum(ratios < 1.0))
            z_anchor = horizon_exit_z(
                subject.cosmology, k, float(RHO_ANCHOR_EFOLDS_SUBH)
            )
            above_anchor = int(np.sum(nodes > z_anchor))
            rows.append(
                {
                    "k": k,
                    "sector": sector,
                    "nodes": int(nodes.size),
                    "outside_horizon": outside,
                    "fraction_outside": outside / float(nodes.size),
                    "band_top_z": float(nodes[0]),
                    "band_top_efolds_outside": subject.efolds_outside_horizon(
                        k, float(nodes[0]), sector
                    ),
                    "nodes_above_production_anchor": above_anchor,
                    "fraction_above_production_anchor": above_anchor
                    / float(nodes.size),
                    "z_anchor": float(z_anchor),
                }
            )
    return rows


def _horizon_limited_helpers():
    """
    ``main.source_grid_spacing_profile``, lifted with the **same** loader
    ``ComputeTargets/tests/wkb_reference.py`` uses but with ``residual_node_range`` replaced by
    one that also drops the super-horizon nodes.

    Nothing in ``main.py`` or in ``ComputeTargets/phase_residual.py`` is touched: the production
    function is executed unmodified, against a band it is handed. That is the only way to answer
    prompt 04 §5's "what would the grid look like if the band were the WKB region alone" without
    changing the thing being measured (README §0.5 holds the source grid fixed, and D6's
    carve-out was prompt 02a's and is spent).
    """
    from ComputeTargets.tests.test_main_plumbing import load_main_py_functions
    from CosmologyConcepts import (
        SOURCE_GRID_CONSUMER_TARGET_RAD,
        SOURCE_GRID_CROSSING_MASK_U,
        SOURCE_GRID_CUBIC_ERROR_CONST,
        SOURCE_GRID_CURVATURE_FD_STEP_U,
        SOURCE_GRID_CURVATURE_STEP_U,
        SOURCE_GRID_SPLINE_EDGE_FACTOR,
        SOURCE_GRID_SPLINE_EDGE_INTERVALS,
    )
    from CosmologyConcepts.wavenumber import SOURCE_GRID_MAX_GUARDED_FRACTION

    def horizon_limited_range(
        model, k, z_grid, sector, margin=RESIDUAL_WKB_REGION_MARGIN
    ):
        nodes = residual_node_range(model, k, z_grid, sector, margin)
        leading_fn = SECTOR_LEADING[sector]
        ratios = np.array(
            [
                sqrt(max(leading_fn(model, float(k), float(z)), 0.0)) * (1.0 + float(z))
                for z in nodes
            ],
            dtype=float,
        )
        inside = nodes[ratios >= 1.0]
        if inside.size < 2:
            raise ValueError(
                f"horizon_limited_range[{sector}]: fewer than two sub-horizon nodes for "
                f"k = {k:.8g}"
            )
        return inside

    return load_main_py_functions(
        [
            "cosmology_feature_redshifts",
            "pre_grid_background_proxy",
            "source_grid_spacing_profile",
        ],
        extra_globals={
            "np": np,
            "_cosmology_break_points": _wkb_break_points,
            "phase_residual_integrand": phase_residual_integrand,
            "residual_node_range": horizon_limited_range,
            "SOURCE_GRID_CONSUMER_TARGET_RAD": SOURCE_GRID_CONSUMER_TARGET_RAD,
            "SOURCE_GRID_CROSSING_MASK_U": SOURCE_GRID_CROSSING_MASK_U,
            "SOURCE_GRID_CUBIC_ERROR_CONST": SOURCE_GRID_CUBIC_ERROR_CONST,
            "SOURCE_GRID_CURVATURE_FD_STEP_U": SOURCE_GRID_CURVATURE_FD_STEP_U,
            "SOURCE_GRID_CURVATURE_STEP_U": SOURCE_GRID_CURVATURE_STEP_U,
            "SOURCE_GRID_MAX_GUARDED_FRACTION": SOURCE_GRID_MAX_GUARDED_FRACTION,
            "SOURCE_GRID_SPLINE_EDGE_FACTOR": SOURCE_GRID_SPLINE_EDGE_FACTOR,
            "SOURCE_GRID_SPLINE_EDGE_INTERVALS": SOURCE_GRID_SPLINE_EDGE_INTERVALS,
        },
    )


def horizon_limited_grid(subject, helpers):
    """
    The version-2 grid this cosmology would carry if the density criterion ran over the
    sub-horizon part of the band alone. **Measured, not shipped** -- prompt 04 recommends.
    """
    log_line(f"   {subject.name}: rebuilding the grid over a horizon-limited band")
    cosmology = grid_cosmology(subject.cosmology)
    break_z, feature_z = helpers["cosmology_feature_redshifts"](
        cosmology, PRODUCTION_Z_END, subject.z_init
    )
    base = build_z_sample(
        subject.z_init, PRODUCTION_Z_END, PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z
    ).z_values
    report = {}
    spacing = helpers["source_grid_spacing_profile"](
        cosmology,
        base,
        [float(k) / UNITS.Mpc for k in PRODUCTION_K_GRID_INV_MPC],
        report=report,
    )
    grid = build_z_sample(
        subject.z_init,
        PRODUCTION_Z_END,
        PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
        spacing=spacing,
    )
    return {
        "samples": int(len(grid.z_values)),
        "digest": redshift_grid_digest(grid.z_values),
        "production_samples": subject.grid.samples,
        "production_digest": subject.grid.digest,
        "base_samples": int(len(base)),
        "guarded": report.get("guarded"),
        "band": report.get("band"),
    }


# ---------------------------------------------------------------------------------------------
# §1: what the fixture re-run changed
# ---------------------------------------------------------------------------------------------


def previous_block(rev=PREVIOUS_BLOCK_REV):
    """
    The superseded ``convergence`` block, read out of git at :data:`PREVIOUS_BLOCK_REV`.

    Read rather than transcribed so that §1's comparison can be regenerated after this prompt's
    own commit has replaced the block in the working tree.
    """
    path = "ComputeTargets/tests/wkb_reference_data.json"
    text = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(text)["convergence"]


def current_block(path=None):
    """
    The block the measurement is compared against.

    With no ``--rerun-json`` it is the one in the tree, and §9 then reports that the re-run was
    **not written** (prompt 04 §2.4 / §11) and compares like with like. With ``--rerun-json`` it
    is the regenerated block from a dry run of ``residual_convergence.py --json-out``, so that
    §9's tables can be produced without the fixture having been overwritten. Both commands are in
    ``ORDER-AUDIT.md`` §9, so either way the document regenerates.
    """
    with open(REFERENCE_DATA_PATH if path is None else path, "r") as f:
        return json.load(f)["convergence"]


def block_comparison(old, new):
    """
    The regenerated block against the 2026-09-10 one, figure by figure for the four orders.

    Only quantities that exist in both are compared; a field the re-run adds (the reference's own
    drift, the grid generation) has no predecessor and is reported as new.
    """
    rows = []
    for field, label in (
        ("N_tau", "$N_\\tau$"),
        ("N_cs_tau", "$N_{c_s\\tau}$"),
        ("N_F", "$N_F$"),
        ("N_rho", "$N_\\rho$"),
    ):
        rows.append(
            {
                "what": label,
                "old": old["decision"][field],
                "new": new["decision"][field],
            }
        )
    rows.append(
        {
            "what": "`recommended_scheme`",
            "old": old["decision"]["recommended_scheme"],
            "new": new["decision"]["recommended_scheme"],
        }
    )
    rows.append(
        {
            "what": "`rho_adaptive_fallback_required`",
            "old": old["decision"]["rho_adaptive_fallback_required"],
            "new": new["decision"]["rho_adaptive_fallback_required"],
        }
    )

    detail = []
    for model in ("LambdaCDMModel", "QCDModel"):
        for scheme in old["models"][model]:
            if scheme not in new["models"][model]:
                continue
            for key in sorted(old["models"][model][scheme]):
                if key not in new["models"][model][scheme]:
                    continue
                a = old["models"][model][scheme][key]
                b = new["models"][model][scheme][key]
                detail.append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "key": key,
                        "json_vs_reference_old": a.get("json_vs_reference_max_rel"),
                        "json_vs_reference_new": b.get("json_vs_reference_max_rel"),
                        "orders_old": {
                            o: a["orders"][str(o)]["max_cumulative_rel_error"]
                            for o in old["orders"]
                        },
                        "orders_new": {
                            o: b["orders"][str(o)]["max_cumulative_rel_error"]
                            for o in new["orders"]
                        },
                        "drift_new": b.get("reference_error_bound"),
                        "drift_passed_new": b.get("reference_error_bound_passed"),
                    }
                )
    return rows, detail


def alignment_measurement(subject_qcd, block):
    """
    ``QCD_BREAK_POINT_ALIGNMENT_TOL``: how far the cosmology's declared break points sit from the
    branch boundaries the regenerated block records, per break.

    This is exactly the quantity ``ComputeTargets/tests/test_background_tau.test_qcd_break_points``
    asserts, re-measured here so that the constant is chosen from a number in this document and
    not from a number in a test's comment.
    """
    z_lo, z_hi = float(subject_qcd.z_values[-1]), float(subject_qcd.z_values[0])
    breaks = subject_qcd.cosmology.integration_break_points(z_lo, z_hi)
    rows = []
    for boundary in block["geometry"]["QCDModel"]["branch_boundaries"]:
        nearest = float(breaks[int(np.argmin(np.abs(breaks - boundary["u"])))])
        rows.append(
            {
                "name": boundary["name"],
                "block_u": boundary["u"],
                "declared_u": nearest,
                "offset": fabs(nearest - boundary["u"]),
                "z": boundary["z"],
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# §7: D3 -- what replaces the vestigial key columns
# ---------------------------------------------------------------------------------------------


def d3_tables():
    """
    The schema-churn cost of each D3 option, from the tree rather than from the plan.

    The three order-governed targets are the ones whose ``atol``/``rtol`` columns are in the
    lookup key and describe nothing (README §2 (a), confirmed by prompt 02's inventory §5.1).
    ``GkSource`` is a fourth vestigial case and is a different question -- it integrates nothing,
    so there is no order to put there.
    """
    from Datastore.SQL.Datastore import _factories

    rows = []
    for name in ("BackgroundModel", "GkWKBIntegration", "TkWKBIntegration", "GkSource"):
        factory = _factories[name]
        schema = factory.register()
        columns = [c.name for c in schema["columns"]]
        rows.append(
            {
                "table": name,
                "columns": len(columns) + 1,
                "has_atol": "atol_serial" in columns,
                "has_rtol": "rtol_serial" in columns,
            }
        )
    return rows


READER_SCRIPTS = (
    "extract_Gk_data.py",
    "extract_GkSource_data.py",
    "extract_GkWKB_data.py",
    "extract_QuadSourceIntegral_data.py",
    "extract_TkWKB_data.py",
    "extract_tensor_source_data.py",
)


def reader_sites():
    """
    Which of the six ``extract_*.py`` readers name each order-governed target, and how often.

    The counting is textual and deliberately so: prompt 05 has to visit each of these files, and
    what it needs is the list of sites, not a semantic model of them.
    """
    import re

    rows = []
    for script in READER_SCRIPTS:
        try:
            text = open(script, "r").read()
        except FileNotFoundError:
            continue
        rows.append(
            {
                "script": script,
                "BackgroundModel": len(re.findall(r"\bBackgroundModel\b", text)),
                "GkWKBIntegration": len(re.findall(r"\bGkWKBIntegration\b", text)),
                "TkWKBIntegration": len(re.findall(r"\bTkWKBIntegration\b", text)),
                "atol_uses": len(re.findall(r'"atol"\s*:', text)),
                "rtol_uses": len(re.findall(r'"rtol"\s*:', text)),
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def report_method(subjects, elapsed):
    emit("## 2. Method, and what every figure in this document carries")
    emit()
    emit(
        "Three models, each on the **version-2** source grid at **its own** production anchor "
        "(board standing note 18 -- the two production cosmologies do not share a `z_init`), "
        "with the four orders swept over `GkTk-remedial` prompt 02's own ladder so that the two "
        "measurements are comparable figure for figure."
    )
    emit()
    emit("| Model | Anchor `z_init` | Version-2 grid | Declared break points |")
    emit("|---|---|---|---|")
    for s in subjects:
        emit(
            f"| `{s.name}` | {s.z_init!r} | {s.grid.samples:,} samples, digest "
            f"`{s.grid.digest}` | {int(s.breaks_u.size)} |"
        )
    emit()
    emit(
        f"**Orders swept:** {', '.join(str(o) for o in ORDERS)}. **Reference on the two spline "
        f"models:** order {REFERENCE_ORDER}, scored against order {REFERENCE_ORDER + 1} through "
        f"`convergence_reference.reference_drift`, for which one step of a `GaussOrder` is **one "
        f"order** and not a factor. **Reference on `RadiationModel`:** none -- every one of the "
        f"five integrands has a closed form there (README §3.1), so the number the rule returns "
        f"*is* the error."
    )
    emit()
    emit(
        "**Error measures**, `GkTk-remedial` README §6's and unchanged: **difference error** "
        "relative to the interval for the three primitives, never relative to the absolute "
        "primitive; **phase error** in absolute radians for the two residuals."
    )
    emit()
    emit(
        f"**Criterion.** A figure is *resolved* when it exceeds its reference's own drift "
        f"(README §5 rule 5; `CRITERION_RATIO` = {CRITERION_RATIO:g} is the margin the facility "
        f"scores a reference by). Every table below carries the drift beside the number, and a "
        f"cell at or below it is marked **&dagger;** and carries no conclusion. On "
        f"`RadiationModel` there is no drift to clear, because there is no reference: the column "
        f"is the closed form."
    )
    emit()
    emit(
        '**The floor, and what "clears" it means for an integer.** README §6.2 gives the floor '
        "for these targets as double-precision accumulation over the grid, which is not a number "
        "anything in the tree states — the ladder is the only way to see it, and it is taken here "
        "as the best any order in the ladder reaches. A setting *clears* the floor when it is "
        f"within `FLOOR_FACTOR` = {FLOOR_FACTOR:g} of it. That is "
        "`residual_convergence.smallest_within_factor`'s definition, kept unchanged so that this "
        're-take and the 2026-09-10 block answer the same question; a literal "at or below the '
        "floor\" would select whichever order happened to reach the minimum, which is one order's "
        "rounding and not a statement about convergence."
    )
    emit()
    emit(
        f"Runtime {elapsed:.0f} s; Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}. Costs are **counted integrand evaluations**, never wall "
        f"time (README §2 (i))."
    )
    emit()


def report_primitives(subjects, primitive_results):
    emit(
        "## 3. $N_\\tau$, $N_{c_s\\tau}$ and $N_F$ — the three `BackgroundModel` primitives"
    )
    emit()
    emit(
        "They are **$k$-independent**: the integrands are $1/H$, $c_s/H$ and "
        "$-\\tfrac32(1+c_s^2)/(1+z)$, and no wavenumber enters. One sweep per model, therefore, "
        "and not fifty identical ones."
    )
    emit()
    for which, title in (
        ("cumulative", "3.1 Cumulative from the top of the grid"),
        ("increment", "3.2 Per production interval"),
    ):
        emit(f"### {title}")
        emit()
        emit(
            "| Model | Quantity | "
            + " | ".join(f"N = {o}" for o in ORDERS)
            + " | floor | reference drift | first order at the floor | selection resolved? |"
        )
        emit("|---|---|" + "---|" * (len(ORDERS) + 4))
        for s in subjects:
            for key in ("tau", "cs_tau", "friction_F"):
                block = primitive_results[s.name][key]
                errors = {o: block["rows"][o][which]["max"] for o in ORDERS}
                floor = double_precision_floor(errors)
                chosen, monotone = choose_order(errors, floor)
                drift = None if block["oracle"] else block["drift"].max
                emit(
                    f"| `{s.name}` | `{key}` | "
                    + " | ".join(cell(errors[o], drift) for o in ORDERS)
                    + f" | {cell(floor, drift)} | "
                    + ("oracle" if drift is None else fmt(drift))
                    + f" | {chosen}"
                    + ("" if monotone else " (not monotone above)")
                    + " | "
                    + fmt(selection_resolved(errors, chosen, drift))
                    + " |"
                )
        emit()


def report_residuals(subjects, residual_results):
    emit("## 4. $N_\\rho$ — the phase residual, at every production wavenumber")
    emit()
    emit(
        "`GkTk-remedial` prompt 02 measured three wavenumbers — $10^5$, $10^7$ and "
        "$3\\times10^8$/Mpc. This is all fifty, on all three models, in both sectors: "
        "**300 (model, $k$, sector) cases**. Every figure is the maximum over the band, in "
        "absolute radians, of $\\rho.\\mathrm{delta}(z_{\\rm anchor}, z)$ from the production "
        f"anchor {RHO_ANCHOR_EFOLDS_SUBH} e-folds inside the horizon."
    )
    emit()
    emit("### 4.1 The maximum over all fifty wavenumbers")
    emit()
    emit(
        "| Model | Sector | "
        + " | ".join(f"N = {o}" for o in ORDERS)
        + " | floor | worst reference drift | $|\\rho|$ range | first order at the floor | "
        "selection resolved? |"
    )
    emit("|---|---|" + "---|" * (len(ORDERS) + 5))
    summary = {}
    for s in subjects:
        for sector in SECTORS:
            rows = [r for r in residual_results[s.name][sector] if "rows" in r]
            errors = {o: max(r["rows"][o]["max"] for r in rows) for o in ORDERS}
            floor = double_precision_floor(errors)
            chosen, monotone = choose_order(errors, floor)
            summary[(s.name, sector)] = {
                "errors": errors,
                "floor": floor,
                "chosen": chosen,
                "monotone": monotone,
            }
            drift = None if rows[0]["oracle"] else max(r["drift"].max for r in rows)
            mags = [r["rho_magnitude_rad"] for r in rows]
            emit(
                f"| `{s.name}` | `{sector}` | "
                + " | ".join(cell(errors[o], drift) for o in ORDERS)
                + f" | {cell(floor, drift)} | "
                + ("oracle" if drift is None else fmt(drift))
                + f" | {fmt(min(mags))}–{fmt(max(mags))} | {chosen}"
                + ("" if monotone else " (not monotone above)")
                + " | "
                + fmt(selection_resolved(errors, chosen, drift))
                + " |"
            )
    emit()
    emit("### 4.2 Was prompt 02 lucky at the three wavenumbers it measured?")
    emit()
    emit(
        "The production order is 4. For each (model, sector), the maximum at order 4 over the "
        "three wavenumbers prompt 02 used, against the maximum over all fifty, and where the "
        "worst of the fifty falls."
    )
    emit()
    emit(
        "| Model | Sector | order-4 max over prompt 02's 3 $k$ | over all 50 $k$ | ratio | "
        "worst $k$ [1/Mpc] | $k$ above the floor, of 50 |"
    )
    emit("|---|---|---|---|---|---|---|")
    for s in subjects:
        for sector in SECTORS:
            rows = [r for r in residual_results[s.name][sector] if "rows" in r]
            three = [
                r
                for r in rows
                if any(fabs(r["k"] / kk - 1.0) < 1e-9 for kk in REFERENCE_K_VALUES)
            ]
            m3 = max(r["rows"][4]["max"] for r in three) if three else None
            m50 = max(r["rows"][4]["max"] for r in rows)
            worst_row = max(rows, key=lambda r: r["rows"][4]["max"])
            floor = summary[(s.name, sector)]["floor"]
            above = sum(1 for r in rows if r["rows"][4]["max"] > floor)
            emit(
                f"| `{s.name}` | `{sector}` | {fmt(m3)} | {fmt(m50)} | "
                f"{'--' if not m3 else f'{m50 / m3:.1f}x' if m3 > 0 else 'inf'} | "
                f"{worst_row['k']:.4g} | {above} |"
            )
    emit()
    return summary


def report_cost(
    subjects, primitive_results, residual_results, primitive_choice, rho_choice
):
    emit("## 6. Cost — counted evaluations times the object count of the sector")
    emit()
    emit(
        "README §2 (c): a per-object count without the object count beside it is not a cost, and "
        "the three knobs do not govern the same count. `BackgroundModel` is **one object per "
        "model** (`main.py:1062`, outside every loop), so its three orders are paid once. The "
        "residual tables are one per `(model, k, sector)` — **100 per model** — memoised inside "
        "the worker process (`phase_residual.cached_phase_residual`); their consumers are the "
        "two WKB targets, which pay `order` further evaluations each for the off-grid partial "
        "that reaches their own anchor."
    )
    emit()
    emit("### 6.1 `BackgroundModel`: the three primitives, one object per model")
    emit()
    emit(
        "| Model | Quantity | panels | N = "
        + f"{primitive_choice - 1} | N = {primitive_choice} | N = {primitive_choice + 1} |"
    )
    emit("|---|---|---|---|---|---|")
    for s in subjects:
        for key in ("tau", "cs_tau", "friction_F"):
            block = primitive_results[s.name][key]
            panels = block["panels"]
            emit(
                f"| `{s.name}` | `{key}` | {panels:,} | "
                + " | ".join(
                    f"{panels * o:,}"
                    for o in (
                        primitive_choice - 1,
                        primitive_choice,
                        primitive_choice + 1,
                    )
                )
                + " |"
            )
    emit()
    emit("### 6.2 The residual tables: 100 per model")
    emit()
    emit(
        f"| Model | Sector | panels per table (min–max) | 50 tables at N = {rho_choice - 1} | "
        f"N = {rho_choice} | N = {rho_choice + 1} |"
    )
    emit("|---|---|---|---|---|---|")
    for s in subjects:
        for sector in SECTORS:
            rows = [r for r in residual_results[s.name][sector] if "rows" in r]
            panels = [r["rows"][rho_choice]["evaluations"] // rho_choice for r in rows]
            total = sum(panels)
            emit(
                f"| `{s.name}` | `{sector}` | {min(panels):,}–{max(panels):,} | "
                + " | ".join(
                    f"{total * o:,}"
                    for o in (rho_choice - 1, rho_choice, rho_choice + 1)
                )
                + " |"
            )
    emit()
    emit(
        "The order ladder runs 2, 4, 6, 8, 12, 16, so orders 3 and 5 were not swept. They do not "
        "need to be: the cost is **exactly** `order * panels`, counted — the counter wrapped "
        "round the integrand returns `order` calls per panel and nothing else — so a column at "
        "any order is the panel count times it. The panel count is the band's intervals plus one "
        "for each declared break point inside the band, which is why the `Gk` and `Tk` rows of "
        "the same model differ."
    )
    emit()


def report_margin(subjects, margin_results, sensitivity_results, refusal_results):
    emit("## 7. `RESIDUAL_WKB_REGION_MARGIN` — what 0.5 buys")
    emit()
    emit(
        "It is `0.5`, it is in no key, and the comment above it "
        "(`ComputeTargets/phase_residual.py:225-238`) argues that it is *safe* rather than "
        "measuring what it *buys*. §7.1 is what it cuts, §7.2 is the range over which the answer "
        "does not depend on it, and §7.3 is what happens as it approaches 1."
    )
    emit()
    emit("### 7.1 What the margin cuts, at every production wavenumber")
    emit()
    emit(
        "The last column is the band's **top** node measured in e-folds of "
        "$\\omega_0(1+z)$ from horizon crossing, in the sector's own leading frequency: "
        "**positive is outside the horizon, negative is inside**. The `Tk` rows are negative at "
        "every margin on every model, which is §8's finding one section early."
    )
    emit()
    emit(
        "| Model | Margin | Sector | nodes retained, min / median / max | of grid | "
        "refuses | reaches the production anchor | band top, e-folds outside the horizon "
        "(min–max) |"
    )
    emit("|---|---|---|---|---|---|---|---|")
    for s in subjects:
        for row in margin_results[s.name]:
            emit(
                f"| `{s.name}` | {row['margin']:g} | `{row['sector']}` | "
                f"{fmt(row['min_retained'])} / {fmt(row['median_retained'], '.0f')} / "
                f"{fmt(row['max_retained'])} | {row['grid_nodes']:,} | "
                f"{row['refused']} of {row['cases']} | "
                f"{row['reaches_production_anchor']} of {row['cases']} | "
                f"{fmt(row['band_top_efolds_outside_min'], '.2f')}–"
                f"{fmt(row['band_top_efolds_outside_max'], '.2f')} |"
            )
    emit()
    emit("### 7.2 Where the answer stops depending on the margin")
    emit()
    emit(
        "The quantity a producer reads is $\\rho.\\mathrm{delta}(z_{\\rm anchor}, z)$ between "
        "two fixed redshifts, and the Gauss panels between them are the grid's own. So the margin "
        "can only change the answer by changing whether the anchor is reachable at all — a "
        "prediction, and this is the test of it: the same `delta` is rebuilt at every margin and "
        "compared **bit for bit** with the production one."
    )
    emit()
    emit(
        "| Model | Sector | $k$ [1/Mpc] | "
        + " | ".join(f"{m:g}" for m in MARGINS)
        + " |"
    )
    emit("|---|---|---|" + "---|" * len(MARGINS))
    for s in subjects:
        for entry in sensitivity_results[s.name]:
            cells = []
            for m in MARGINS:
                record = entry["margins"].get(m, {"refused": True})
                if record.get("refused"):
                    cells.append("refuses")
                elif record.get("bit_identical"):
                    cells.append(f"= ({record['nodes']})")
                elif not record.get("same_interval", True):
                    cells.append(f"anchor clamped ({record['nodes']})")
                else:
                    base = entry["margins"][RESIDUAL_WKB_REGION_MARGIN]["rho"]
                    cells.append(
                        f"{abs(record['rho'] - base) / abs(base):.0e} ({record['nodes']})"
                    )
            emit(
                f"| `{s.name}` | `{entry['sector']}` | {entry['k']:.4g} | "
                + " | ".join(cells)
                + " |"
            )
    emit()
    emit(
        "`=` means the residual the producer reads is **bit-identical** to production's at that "
        "margin, with the band's node count in brackets; a number is the relative departure "
        "where it is not. *Bit*-identical is the strong claim and it is what the prediction asks "
        "for: the panels between the two fixed redshifts are the same, so the only way the answer "
        "can move is the rounding of a cumulative whose **top** has moved, which is why the two "
        "cells that depart do so at the extreme margin and at the last bits. **`anchor clamped`** "
        "means the band no longer reaches the production anchor, which is the one way the margin "
        "can change what a producer gets — and the `Tk` rows show it happening between 0.9 and "
        "0.99."
    )
    emit()
    emit("### 7.3 Where `residual_node_range` starts refusing")
    emit()
    emit(
        "| Model | Sector | $k$ [1/Mpc] | refuses at margin | $1 -$ that | nodes just below |"
    )
    emit("|---|---|---|---|---|---|")
    for s in subjects:
        for row in refusal_results[s.name]:
            at = row["refuses_at"]
            emit(
                f"| `{s.name}` | `{row['sector']}` | {row['k']:.4g} | "
                + (f"{at:.15g}" if at is not None else "never below 1")
                + " | "
                + (fmt(1.0 - at) if at is not None else "--")
                + f" | {fmt(row.get('nodes_just_below'))} |"
            )
    emit()
    emit(
        "**Nothing refuses below 1, and the margin is not what makes it refuse.** At `margin = 1` "
        "the test `omega^2 >= margin * omega_0^2` becomes `correction >= 0`, so the walk stops at "
        "the first node where the Liouville-Green correction turns negative — which is a property "
        "of the background and not of the parameter. The `nodes just below` column is what is "
        "left there, and on the two production models it is a sixth to a half of the band."
    )
    emit()


def report_band(subjects, band_results, rebuild_results):
    emit(
        "## 8. How much of `residual_node_range`'s band lies outside the region the density "
        "criterion protects"
    )
    emit()
    emit(
        "`[01-density-criterion-imposed-outside-the-wkb-region]` has been carrying prompt 01's "
        "69-%-of-added-samples figure as its only evidence since it was opened, and prompt 02a's "
        "guarded-node census was explicitly **withdrawn** as a measurement of it (board §3, the "
        "2026-09-17 correction). This is the measurement it has been waiting for: at every "
        "production wavenumber and in both sectors, how many of the band's nodes are "
        "**super-horizon** in that sector's own leading frequency, $\\omega_0(1+z) < 1$."
    )
    emit()
    emit("### 8.1 The census")
    emit()
    emit(
        "| Model | Sector | band nodes (min–max) | super-horizon nodes (min–max) | fraction "
        "(min / median / max) | band top, e-folds outside (max) | nodes above the production "
        "anchor (median fraction) |"
    )
    emit("|---|---|---|---|---|---|---|")
    for s in subjects:
        for sector in SECTORS:
            rows = [r for r in band_results[s.name] if r["sector"] == sector]
            if not rows:
                continue
            fracs = [r["fraction_outside"] for r in rows]
            emit(
                f"| `{s.name}` | `{sector}` | "
                f"{min(r['nodes'] for r in rows):,}–{max(r['nodes'] for r in rows):,} | "
                f"{min(r['outside_horizon'] for r in rows):,}–"
                f"{max(r['outside_horizon'] for r in rows):,} | "
                f"{min(fracs):.3f} / {float(np.median(fracs)):.3f} / {max(fracs):.3f} | "
                f"{max(r['band_top_efolds_outside'] for r in rows):.2f} | "
                f"{float(np.median([r['fraction_above_production_anchor'] for r in rows])):.3f} |"
            )
    emit()
    emit("### 8.2 What the grid would be if the band were the WKB region alone")
    emit()
    emit(
        "`main.source_grid_spacing_profile` is executed **unmodified**, against a band it is "
        "handed: the lift of that function is given a `residual_node_range` that additionally "
        "drops the super-horizon nodes, and nothing in `main.py` or "
        "`ComputeTargets/phase_residual.py` is touched. Prompt 04 recommends; it does not change "
        "the band (README §0.5, and D6's carve-out was prompt 02a's and is spent)."
    )
    emit()
    emit(
        "| Model | production version-2 grid | over a horizon-limited band | difference | "
        "guarded nodes |"
    )
    emit("|---|---|---|---|---|")
    for s in subjects:
        r = rebuild_results[s.name]
        emit(
            f"| `{s.name}` | {r['production_samples']:,} / `{r['production_digest']}` | "
            f"{r['samples']:,} / `{r['digest']}` | "
            f"{r['samples'] - r['production_samples']:+,} samples | {fmt(r['guarded'])} |"
        )
    emit()


def report_fixture(rows, detail, alignment, old, new, rerun_json):
    emit("## 9. What the fixture re-run changed")
    emit()
    emit(
        f"The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` was generated "
        f"**{old.get('generated')}** by `{old.get('campaign')}` and is what every `*_GAUSS_ORDER` "
        f"comment in the tree cites. The re-run below is "
        f"**{new.get('generated')}**, by `{new.get('campaign')}`. The superseded block is read "
        f"out of git at `{PREVIOUS_BLOCK_REV}` rather than transcribed."
    )
    emit()
    if rerun_json is None:
        emit(
            "**The regenerated block is not in the tree**, so the two columns below are the same "
            "block and this section reports no change. Re-run it with `--rerun-json` pointing at "
            "a dry run to fill the comparison:"
        )
    else:
        emit(
            "**The regenerated block is not in the tree** (prompt 04 §11's first stop condition "
            "fired — see §0). What the comparison reads is a **dry run** of the generator, "
            "written to a scratch path and never to the fixture. Both commands, in order:"
        )
    emit()
    emit("```")
    emit("cp ComputeTargets/tests/wkb_reference_data.json /tmp/rerun.json")
    emit(
        "PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/residual_convergence.py "
        "--json-out /tmp/rerun.json"
    )
    emit(
        "PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/order_audit.py "
        "--rerun-json /tmp/rerun.json"
    )
    emit("```")
    emit()
    emit("### 9.1 The decision")
    emit()
    emit("| | 2026-09-10 | re-run |")
    emit("|---|---|---|")
    for row in rows:
        emit(f"| {row['what']} | `{row['old']}` | `{row['new']}` |")
    emit()
    emit("### 9.2 The reference floors the two test modules read")
    emit()
    emit(
        "`json_vs_reference_max_rel` is *how well the JSON's own reference values agree with a "
        "converged adaptive one*, and it is used as a **threshold**: "
        "`test_background_tau.test_qcd_nodes_against_adaptive_reference` and "
        "`test_background_cs_tau_friction.test_qcd_checkpoints` both assert *production error "
        "≤ 3 × this*. The second of those modules is outside prompt 04's carve-out."
    )
    emit()
    emit("| Model | Scheme | Quantity | 2026-09-10 | re-run | ratio |")
    emit("|---|---|---|---|---|---|")
    for row in detail:
        if row["key"] not in ("tau", "cs_tau", "friction"):
            continue
        a, b = row["json_vs_reference_old"], row["json_vs_reference_new"]
        emit(
            f"| `{row['model']}` | `{row['scheme']}` | `{row['key']}` | {fmt(a)} | {fmt(b)} | "
            + (f"{b / a:.3f}x" if a and b else "--")
            + " |"
        )
    emit()
    emit("### 9.3 The cumulative relative error, order by order")
    emit()
    emit(
        "**The last column is in two different units and the block says which per row.** The "
        "reference's drift is measured in the kind the case decides in: *relative* for the three "
        "primitives, whose decision reads `max_cumulative_rel_error`, and *absolute radians* for "
        "the residuals, whose decision reads `max_cumulative_abs_error` against a target in "
        "radians. A `rho` bound of 1e-21 is therefore 1e-21 **rad** and not a relative error. "
        "**NOT CONVERGED** means the reference moved by more than a tenth of the smallest "
        "difference that case goes on to report — which is the honest statement about a "
        "primitive whose smallest reported figure is 2e-16 and whose reference agrees with an "
        "independent order-40 rule only to 3e-15. It is not a defect in the re-run: the same "
        "cases carried the same disagreement in 2026-09-10's block, which simply did not record "
        "it."
    )
    emit()
    emit(
        "| Model | Scheme | Quantity | "
        + " | ".join(f"N = {o} (old → new)" for o in ORDERS)
        + " | reference error bound |"
    )
    emit("|---|---|---|" + "---|" * (len(ORDERS) + 1))
    for row in detail:
        cells = []
        for o in ORDERS:
            a = row["orders_old"].get(o)
            b = row["orders_new"].get(o)
            cells.append(f"{fmt(a)} → {fmt(b)}")
        bound = row["drift_new"]
        if bound is None:
            verdict = "not recorded"
        else:
            verdict = fmt(bound) + (
                "" if row["drift_passed_new"] else " **NOT CONVERGED**"
            )
        emit(
            f"| `{row['model']}` | `{row['scheme']}` | `{row['key']}` | "
            + " | ".join(cells)
            + f" | {verdict} |"
        )
    emit()
    emit("### 9.4 `QCD_BREAK_POINT_ALIGNMENT_TOL`")
    emit()
    emit(
        "How far each declared break point sits, in $u = \\log(1+z)$, from the branch boundary a "
        "block records. This is exactly what `test_background_tau.test_qcd_break_points` asserts, "
        "and it asserts it against **the block in the tree** — so the constant can only be taken "
        "back when the block is."
    )
    emit()
    emit("| Block | Break | $z$ | block's $u$ | declared $u$ | offset |")
    emit("|---|---|---|---|---|---|")
    for which, entries in alignment.items():
        for row in entries:
            emit(
                f"| {which} | `{row['name']}` | {row['z']:.6g} | {row['block_u']:.12g} | "
                f"{row['declared_u']:.12g} | {fmt(row['offset'])} |"
            )
    emit()
    tree_rows = alignment["in the tree (2026-09-10)"]
    rerun_rows = alignment["the re-run"]
    worst_tree = max(r["offset"] for r in tree_rows)
    binding_tree = max(tree_rows, key=lambda r: r["offset"])
    worst_rerun = max(r["offset"] for r in rerun_rows)
    binding_rerun = max(rerun_rows, key=lambda r: r["offset"])
    emit(
        f"**Against the block in the tree: {worst_tree:.6e} at `{binding_tree['name']}`** — which "
        f"is prompt 07's 1.418851e-04 reproduced, and is why the constant stands at 1.5e-04. "
        f"**Against the re-run: {worst_rerun:.6e} at `{binding_rerun['name']}`.**"
    )
    emit()
    alignment = rerun_rows
    worst = worst_rerun
    binding = binding_rerun
    emit(
        f"**Worst: {worst:.6e} at `{binding['name']}`.** The constant must exceed it; the "
        f"arithmetic that chooses it is in `ORDER-AUDIT.md` §9.4's prose and in "
        f"`ComputeTargets/tests/test_background_tau.py`'s comment."
    )
    emit()


def report_d3(tables, readers):
    emit("## 10. D3 — what replaces the vestigial key columns")
    emit()
    emit("### 10.1 The tables")
    emit()
    emit("| Table | Columns | `atol_serial` | `rtol_serial` |")
    emit("|---|---|---|---|")
    for row in tables:
        emit(
            f"| `{row['table']}` | {row['columns']} | {fmt(row['has_atol'])} | "
            f"{fmt(row['has_rtol'])} |"
        )
    emit()
    emit("### 10.2 The six `extract_*.py` readers")
    emit()
    emit(
        '| Script | `BackgroundModel` | `GkWKBIntegration` | `TkWKBIntegration` | `"atol":` | '
        '`"rtol":` |'
    )
    emit("|---|---|---|---|---|---|")
    for row in readers:
        emit(
            f"| `{row['script']}` | {row['BackgroundModel']} | {row['GkWKBIntegration']} | "
            f"{row['TkWKBIntegration']} | {row['atol_uses']} | {row['rtol_uses']} |"
        )
    emit()
    emit("### 10.3 The row counts a migration would touch")
    emit()
    emit(
        "No datastore is stood up here (`CLAUDE.md`), so what is given is the **object count of "
        "each sector** — the number of rows a full production run writes per model — and not a "
        "count read out of a store."
    )
    emit()
    emit(
        "| Target | Radiation | LambdaCDM | QCD | Shape, and where the count comes from |"
    )
    emit("|---|---|---|---|---|")
    sources = {
        "BackgroundModel": (
            "one per (cosmology, source grid) -- `main.py:1062`, outside every loop"
        ),
        "GkWKBIntegration": (
            "one per $(k, z_{\\rm source})$; the counts are prompt 03's measurement of the "
            "identically-shaped `GkNumericIntegration` sector, **on the version-2 grid at each "
            "cosmology's own anchor** -- not README §2 (c)'s ~65,000, which is a version-0 figure"
        ),
        "TkWKBIntegration": "one per $k$ -- 50 wavenumbers per model",
    }
    for name in ("BackgroundModel", "GkWKBIntegration", "TkWKBIntegration"):
        counts = OBJECT_COUNTS[name]
        emit(
            f"| `{name}` | "
            + " | ".join(
                f"{counts[m]:,}"
                for m in ("RadiationModel", "LambdaCDMModel", "QCDModel")
            )
            + f" | {sources[name]} |"
        )
    emit()


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "the order and margin audit of prompts/tolerance-convergence prompt 04; emits "
            "docs/tolerance-convergence/ORDER-AUDIT.md on stdout"
        )
    )
    parser.add_argument(
        "--rerun-json",
        default=None,
        metavar="PATH",
        help=(
            "a wkb_reference_data.json-shaped file holding the REGENERATED convergence block, as "
            "written by a dry run of `residual_convergence.py --json-out PATH`. Without it §9 "
            "compares the tree's block with itself and says the re-run was not written."
        ),
    )
    return parser.parse_args(argv)


def main(args):
    subjects = build_subjects()

    log_line("** stage 1: the three BackgroundModel primitives")
    primitive_results = {s.name: sweep_primitives(s) for s in subjects}

    log_line("** stage 2: the two phase residuals, at every production wavenumber")
    residual_results = {s.name: sweep_residuals(s) for s in subjects}

    log_line("** stage 3: RESIDUAL_WKB_REGION_MARGIN")
    margin_results = {s.name: margin_census(s) for s in subjects}
    sensitivity_results = {s.name: margin_sensitivity(s) for s in subjects}
    refusal_results = {s.name: margin_refusal(s) for s in subjects}

    log_line("** stage 4: the band against the WKB region")
    band_results = {s.name: band_census(s) for s in subjects}
    helpers = _horizon_limited_helpers()
    rebuild_results = {s.name: horizon_limited_grid(s, helpers) for s in subjects}

    log_line("** stage 5: the fixture re-run, against the superseded block")
    old, new = previous_block(), current_block(args.rerun_json)
    rows, detail = block_comparison(old, new)
    qcd = [s for s in subjects if s.name == "QCDModel"][0]
    alignment = {
        "in the tree (2026-09-10)": alignment_measurement(qcd, current_block()),
        "the re-run": alignment_measurement(qcd, new),
    }

    log_line("** stage 6: D3")
    tables, readers = d3_tables(), reader_sites()

    elapsed = time.perf_counter() - _started

    emit(
        f"<!-- generated {time.strftime('%Y-%m-%d')} by\n"
        f"     PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/order_audit.py\n"
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_method(subjects, elapsed)
    report_primitives(subjects, primitive_results)
    rho_summary = report_residuals(subjects, residual_results)

    emit("## 5. The four orders, under README §6.1 as prompt 04 §7 adapts it")
    emit()
    emit(
        "**Loosest means lowest.** The ladder is swept upwards and the answer is the *first* "
        "order that clears the floor at the **maximum** over the whole production grid — on all "
        "three models, not at a representative $k$. The floor is double-precision accumulation "
        "over the grid; the ladder measures it directly, as the level at which raising the order "
        f'stops buying anything, and "clears" is within `FLOOR_FACTOR` = {FLOOR_FACTOR:g} of '
        "it (§2)."
    )
    emit()
    emit(
        "| Knob | production | at the loosest order swept (N = 2) | floor (worst over models) | "
        "the floor dominates N = 2 by | first order at the floor | monotone above? | answer |"
    )
    emit("|---|---|---|---|---|---|---|---|")
    primitive_choice = 0
    for key, label in (
        ("tau", "$N_\\tau$"),
        ("cs_tau", "$N_{c_s\\tau}$"),
        ("friction_F", "$N_F$"),
    ):
        errors = {
            o: max(
                primitive_results[s.name][key]["rows"][o]["cumulative"]["max"]
                for s in subjects
            )
            for o in ORDERS
        }
        floor = double_precision_floor(errors)
        chosen, monotone = choose_order(errors, floor)
        primitive_choice = max(primitive_choice, chosen or 0)
        production = PRODUCTION_ORDERS[key]
        emit(
            f"| {label} | {production} | {fmt(errors[min(ORDERS)])} | {fmt(floor)} | "
            f"x{errors[min(ORDERS)] / floor:.3g} | {chosen} | {fmt(monotone)} | "
            + ("**`unchanged`**" if chosen == production else f"**{chosen}**")
            + " |"
        )
    rho_errors = {o: max(v["errors"][o] for v in rho_summary.values()) for o in ORDERS}
    rho_floor = double_precision_floor(rho_errors)
    rho_chosen, rho_monotone = choose_order(rho_errors, rho_floor)
    emit(
        f"| $N_\\rho$ | {PRODUCTION_ORDERS['rho']} | {fmt(rho_errors[min(ORDERS)])} | "
        f"{fmt(rho_floor)} | x{rho_errors[min(ORDERS)] / rho_floor:.3g} | {rho_chosen} | "
        f"{fmt(rho_monotone)} | "
        + (
            "**`unchanged`**"
            if rho_chosen == PRODUCTION_ORDERS["rho"]
            else f"**{rho_chosen}**"
        )
        + " |"
    )
    emit()
    emit(
        "README §6.1 rule 4 asks, where the answer is `unchanged`, for **the factor by which "
        "the floor dominates** beside it. The column above is that factor against the loosest "
        'order the ladder holds, which is the only "one step looser" available for an integer '
        "knob already at 4 with a ladder that skips 3; §§3 and 4 carry the per-model detail. The "
        "column is not a cost saving forgone: order 2 does not clear the floor anywhere, so it is "
        "not a candidate."
    )
    emit()

    report_cost(
        subjects,
        primitive_results,
        residual_results,
        primitive_choice or PRODUCTION_ORDERS["tau"],
        rho_chosen or PRODUCTION_ORDERS["rho"],
    )
    report_margin(subjects, margin_results, sensitivity_results, refusal_results)
    report_band(subjects, band_results, rebuild_results)
    report_fixture(rows, detail, alignment, old, new, args.rerun_json)
    report_d3(tables, readers)

    emit("## 11. Per-wavenumber appendix: $N_\\rho$, order by order")
    emit()
    emit(
        "The fifty wavenumbers behind §4.1, so that a reader can see the distribution rather "
        "than its maximum. `nodes` is the band `residual_node_range` returns at the production "
        "margin, of the grid's own count."
    )
    emit()
    for s in subjects:
        for sector in SECTORS:
            emit(f"**`{s.name}`, `{sector}`**")
            emit()
            emit(
                "| $k$ [1/Mpc] | nodes | $|\\rho|$ [rad] | N = 2 | N = 4 | N = 6 | N = 16 | "
                "reference drift |"
            )
            emit("|---|---|---|---|---|---|---|---|")
            for r in residual_results[s.name][sector]:
                if "rows" not in r:
                    emit(
                        f"| {r['k']:.4g} | -- | -- | -- | -- | -- | -- | {r['skipped']} |"
                    )
                    continue
                emit(
                    f"| {r['k']:.4g} | {r['nodes']:,} | {fmt(r['rho_magnitude_rad'])} | "
                    + " | ".join(fmt(r["rows"][o]["max"]) for o in (2, 4, 6, 16))
                    + " | "
                    + ("oracle" if r["oracle"] else fmt(r["drift"].max))
                    + " |"
                )
            emit()

    log_line(f"** done in {elapsed:.0f} s")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
