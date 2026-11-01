"""
levin-phase-floor prompt 06a (copied from prompt 06's scripts/rebuild.py,
docs/levin-phase-floor/06/wider/scripts/rebuild.py, a record, not edited). Changed here, each
change marked "prompt 06a", only for the census's producer cache (prompt 06a section 0.3 change 1,
section 2 item 3): ``rebuild_row`` takes ``cache`` (a ``CensusCache``); with it, after every
serial-only lookup, check and the existence query (unchanged, per row), the row's
``GkSourcePolicyData`` (with its ``GkSource``), ``TkWKBIntegration``s, ``TkNumericIntegration``s
and ``QuadSource`` come from the cache, each computed once (``_rebuild_row_census``), through the
same functions ``_rebuild_row_rerun`` calls. 06's text follows, unchanged.

levin-phase-floor prompt 06 section 6 (copied from prompt 05a's scripts/rebuild.py,
docs/levin-phase-floor/05a/wider/scripts/rebuild.py, itself 05's unchanged; a record, not edited).
Changed here, each change marked "prompt 06", only for the re-run of the producers (prompt 06
section 6; rerun.py):
- ``build_context`` also returns the run's solver map (``solvers``, which a producer's ``store()``
  reads its solver from) and the pool's redshift objects by serial (``redshift_by_id``, from the
  ``read_table("redshift")`` it already makes);
- ``rebuild_row`` takes ``rerun``. With ``rerun=False`` (``t5``) nothing changes: every producer
  value is read from A3 v2. With ``rerun=True`` (``s``, and U0's faithfulness clause at ``t5``) the
  stage's full lookups of the ``GkSource``, the ``GkSourcePolicyData`` and the two
  ``TkWKBIntegration`` are replaced by rerun.py's re-run chain, whose read-backs carry the stored
  rows' serials; every serial-only lookup, every check and the existence query are unchanged,
  and run first;
- every row records its ``GkSourcePolicyData``'s ``type``, ``quality``, ``crossover_z`` and
  ``Levin_z`` (prompt 06 section 6, descriptive: "``Levin_z`` under both trees");
- ``use_store_dir``, route (B): s reads the certified scratch copy of A3 v2 (prov.py).

(05's text follows, unchanged.)

levin-phase-floor prompt 05 section 6 (copied from prompt 04's scripts/rebuild.py,
docs/levin-phase-floor/04/scripts/rebuild.py, sha256 ee47bc50..., a record that is not edited).
Changed here, each change marked "prompt 05":
- ``STORE_DIR`` is the main checkout's ``var/datastores`` by absolute path (prov.py), since this
  copy lives one directory deeper than 04's;
- **``QuadSource`` is recomputed in-process** (prompt 05 section 6). At this tree ``handover`` B1's
  strict coverage check refuses A3 v2's stored ``QuadSource`` rows: they end at the last source-grid
  node at or above the earlier hand-over's ``z_init``, and the partition now needs the spline of f
  to reach the switch node and ``HANDOVER_OVERLAP_NODES`` nodes below it. So, after the stage's
  lookups, ``rebuild_row`` computes the row's ``QuadSource`` as ``main.py`` would compute it at
  this tree, with no production change: the object ``main.py``'s ``build_tensor_source_work``
  gets from ``pool.object_get("QuadSource", ...)`` for a pair it must compute (the factory's miss
  branch, ``ObjectFactories/QuadSource.py``, ``row_data is None``), with ``main.py``'s arguments
  (``z_sample=z_source_sample``, the two ``TkNumericIntegration`` serials, the seven ``Tk`` tags);
  ``compute(payload={"Tq": Tq, "Tr": Tr})`` with the stage's own full ``TkNumericIntegration``
  lookups (``compute_tensor_source_work``); then production's ``QuadSource.store()``, which
  populates the values (harness.py stands in ``compute_quad_source.remote`` and ``ray.wait``).
  ``main.py`` would then write the row and read it back in the ``QuadSourceIntegral`` stage. The
  store is never written here, so the read-back is formed as the factory's ``build()`` forms it
  from a stored row: a ``QuadSource`` built from a payload (store_id, compute_time, the values in
  descending ``z``, ``Tq_serial``, ``Tr_serial``), ``z_sample`` the values' redshifts, with
  ``_deserialized`` set and the crossover redshifts ``None`` (they are not persisted). Its
  ``store_id`` is the stored row's serial, because the ``QuadSourceIntegral`` row being recomputed
  is keyed on that serial (``source_serial``) and ``compute()`` checks it. Every value is a double
  that SQLite's REAL stores exactly, so the read-back holds the computed doubles bit for bit.
  The recomputed values are compared with the stored row's (``qs_check``): at every node the
  stored row holds, all nine value columns by ``==`` and by bit pattern; and the nodes added are
  listed against those B1 adds (the switch node of the earlier hand-over and
  ``HANDOVER_OVERLAP_NODES`` nodes below it). That is gate L0's first clause.

(Prompt 04's text follows, unchanged.)

levin-phase-floor prompt 04 (copied from prompt 03's scripts/rebuild.py at the dispatch SHA; the
only change is rebuild_row's optional ``rtol`` argument, which replaces the work object's
relative-tolerance object for the 1e-8 rung): prompt 03 step 4, rebuild a QuadSourceIntegral row's inputs as main.py does,
by the same keys and tags, on a read-only pool (scripts/harness.py), and build the unpopulated
QuadSourceIntegral whose compute() main.py would have called.

What is main.py's own code, and what is mirrored
------------------------------------------------
- **Called from main.py, unchanged** (extracted with
  ``ComputeTargets.tests.test_main_plumbing.load_main_py_functions``, which reads main.py with
  ``ast``; its globals are main.py's own top-level imports, executed from main.py's source):
  ``cosmology_feature_redshifts``, ``pre_grid_background_proxy``,
  ``source_grid_spacing_profile``, ``build_grid_tag_labels``, ``closes_triangle`` and
  ``build_QuadSourceIntegral_payload``.
- **Mirrored, statement for statement, with the main.py lines cited at each block**: the top
  level (tolerances, solvers, policies, the k sample) and ``run_pipeline``'s steps 0-2 (exit
  times, the source grid, the response grid, the tags, the background model, the Bessel phase
  objects), and the QuadSourceIntegral stage (``resolve_GkSource_parents``,
  ``resolve_QuadSourceIntegral_parents``, the existence query and the full lookups). These are
  closures inside ``run_pipeline`` and cannot be extracted.  Each RayWorkPool of main.py becomes
  the same ``pool.object_get`` / ``object_get_vectorized`` / ``object_read_batch`` call, made
  directly and in the same order of items.
- **A3 v2's arguments.** A3 v2 was built by ``RunDrivers/quadsource_atol_sweep.py --build``
  (run ``handover-07-a3-v2-20261022T211155``, manifest argv), which executes main.py with:
  the two k-grid literals replaced by ``np.logspace(np.log10(1e5), np.log10(3e8), 8).tolist()``
  (``k_sample()``); ``tol=DEFAULT_QUADRATURE_ATOL/RTOL`` replaced by ``1e-32`` / ``1e-7``;
  ``build_model_list`` filtered to LambdaCDM; argv ``--zend 0.1 --source-samples-log10z 100``
  and main.py's defaults otherwise (``--response-sparseness 12``, ``--run-label default``).

Every check the rebuild can make against the store is made and recorded in the returned
``checks`` list: the grid tags' labels equal the stored tag labels; every redshift object's z
equals the stored z; the background model, the six parents, the z_response and z_source_max
serials equal the stored row's; the existence query returns the stored row by its serial.
"""

import ast
import hashlib
import itertools
import sqlite3
import time  # prompt 06a
from pathlib import Path

import numpy as np
import ray

from ComputeTargets import GkSourceProxy, ModelProxy, QuadSourceIntegral
from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkSource import GkSource_parent_set_of_serials
from ComputeTargets.GkWKBIntegration import GkWKBIntegration
from ComputeTargets.QuadSourceIntegral import BesselPhaseProxy
from ComputeTargets.QuadSource import QuadSource, QuadSourceValue  # prompt 05
from ComputeTargets.handover import HANDOVER_OVERLAP_NODES, handover_switch_node  # prompt 05
from ComputeTargets.tests.test_main_plumbing import MAIN_PY, load_main_py_functions
from CosmologyConcepts import (
    build_z_sample,
    redshift_array,
    wavenumber_array,
    wavenumber_exit_time_array,
)
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from LiouvilleGreen.bessel_phase import bessel_phase
from Units import Mpc_units
from config.defaults import (
    DEFAULT_GK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_GK_NUMERIC_REL_TOLERANCE,
    DEFAULT_HEXIT_ABS_TOLERANCE,
    DEFAULT_HEXIT_REL_TOLERANCE,
    DEFAULT_QUADSOURCE_POLICY_1PT5,
    DEFAULT_TK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_TK_NUMERIC_REL_TOLERANCE,
)
from config.model_list import build_model_list
from extract_common import run_label_tag, source_grid_construction_tag

# --- A3 v2's arguments (see the module docstring) ---------------------------------------------
A3V2_K_SAMPLE = np.logspace(np.log10(1e5), np.log10(3e8), 8).tolist()
A3V2_QUAD_ATOL = 1e-32
A3V2_QUAD_RTOL = 1e-7
A3V2_ZEND = 0.1  # argparse type=float of "0.1"
A3V2_SOURCE_SAMPLES_PER_LOG10Z = 100
A3V2_RESPONSE_SPARSENESS = 12  # main.py DEFAULT_RESPONSE_SAMPLES_SPARSENESS
A3V2_RUN_LABEL = "default"  # main.py DEFAULT_RUN_LABEL

from prov import STORE_DIR  # noqa: E402  prompt 05: the main checkout's var/datastores
PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"


def use_store_dir(store_dir):
    """prompt 06, route (B): point PRIMARY and shard_file() at ``store_dir`` (A3 v2's directory for
    t5, the certified scratch copy's for s; prov.STORE_DIR_BY_TREE). The worker calls it once,
    before opening the pool."""
    global STORE_DIR, PRIMARY
    STORE_DIR = Path(store_dir)
    PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"
    return PRIMARY


def shard_file(shard: int) -> Path:
    return STORE_DIR / f"handover-A3-v2-lambdacdm-shard{shard:04d}.sqlite"


def _main_imports_namespace() -> dict:
    """main.py's own top-level import statements, executed from its source (nothing else)."""
    tree = ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = {}
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), str(MAIN_PY), "exec"),
        namespace,
    )
    return namespace


_MAIN = load_main_py_functions(
    [
        "cosmology_feature_redshifts",
        "pre_grid_background_proxy",
        "source_grid_spacing_profile",
        "build_grid_tag_labels",
        "closes_triangle",
        "build_QuadSourceIntegral_payload",
    ],
    extra_globals=_main_imports_namespace(),
)
cosmology_feature_redshifts = _MAIN["cosmology_feature_redshifts"]
source_grid_spacing_profile = _MAIN["source_grid_spacing_profile"]
build_grid_tag_labels = _MAIN["build_grid_tag_labels"]
closes_triangle = _MAIN["closes_triangle"]
build_QuadSourceIntegral_payload = _MAIN["build_QuadSourceIntegral_payload"]

MAIN_PY_SHA256 = hashlib.sha256(MAIN_PY.read_bytes()).hexdigest()


def read_row_ro(shard: int, serial: int) -> dict:
    """The stored row's columns (metadata excluded), read with sqlite3 mode=ro."""
    conn = sqlite3.connect(f"{shard_file(shard).as_uri()}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("QuadSourceIntegral")')]
        row = conn.execute(
            'SELECT * FROM "QuadSourceIntegral" WHERE serial = ?', (serial,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(
            f"rebuild: no QuadSourceIntegral serial {serial} on shard {shard}"
        )
    d = dict(zip(cols, row))
    d.pop("metadata")
    return d


def _check(checks, name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    if not ok:
        raise RuntimeError(f"rebuild check failed: {name}: {detail}")


def build_context(pool, log=print) -> dict:
    """main.py's top level and run_pipeline's steps 0-2 and the stage's set-up, for LambdaCDM."""
    checks = []
    units = Mpc_units()
    zend = A3V2_ZEND
    source_samples_per_log10z = A3V2_SOURCE_SAMPLES_PER_LOG10Z
    response_sparseness = A3V2_RESPONSE_SPARSENESS
    run_label = A3V2_RUN_LABEL

    # main.py:4200-4223 convert_to_wavenumbers / convert_to_redshifts
    def convert_to_wavenumbers(k_sample_set, is_source=False, is_response=False):
        return pool.object_get(
            "wavenumber",
            payload_data=[
                {
                    "k_inv_Mpc": k,
                    "units": units,
                    "is_source": is_source,
                    "is_response": is_response,
                }
                for k in k_sample_set
            ],
        )

    def convert_to_redshifts(z_sample_set, is_source=False, is_response=False):
        return pool.object_get(
            "redshift",
            payload_data=[
                {"z": z, "is_source": is_source, "is_response": is_response}
                for z in z_sample_set
            ],
        )

    # main.py:4243-4264 tolerances (the sweep driver substitutes the two quadrature sites)
    (
        hexit_atol,
        hexit_rtol,
        Gk_numeric_atol,
        Gk_numeric_rtol,
        Tk_numeric_atol,
        Tk_numeric_rtol,
        quad_atol,
        quad_rtol,
    ) = ray.get(
        [
            pool.object_get("tolerance", tol=DEFAULT_HEXIT_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_HEXIT_REL_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_GK_NUMERIC_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_GK_NUMERIC_REL_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_TK_NUMERIC_ABS_TOLERANCE),
            pool.object_get("tolerance", tol=DEFAULT_TK_NUMERIC_REL_TOLERANCE),
            pool.object_get("tolerance", tol=A3V2_QUAD_ATOL),
            pool.object_get("tolerance", tol=A3V2_QUAD_RTOL),
        ]
    )

    # main.py:4269-4310 solvers
    (
        solve_ivp_RK45,
        solve_ivp_DOP852,
        solve_ivp_Radau,
        solve_ivp_BDF,
        solve_icp_LSODA,
        cumulative_GL_tau,
        wkb_primitive_phase,
    ) = ray.get(
        [
            pool.object_get("IntegrationSolver", label="solve_ivp+RK45", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+DOP853", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+Radau", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+BDF", stepping=0),
            pool.object_get("IntegrationSolver", label="solve_ivp+LSODA", stepping=0),
            pool.object_get(
                "IntegrationSolver",
                label=BackgroundModel.TAU_SOLVER_LABEL_BASE,
                stepping=BackgroundModel.TAU_GAUSS_ORDER,
            ),
            pool.object_get(
                "IntegrationSolver",
                label=GkWKBIntegration.PHASE_SOLVER_LABEL_BASE,
                stepping=GkWKBIntegration.PHASE_SOLVER_STEPPING,
            ),
        ]
    )
    solvers = {
        "solve_ivp+RK45-stepping0": solve_ivp_RK45,
        "solve_ivp+DOP853-stepping0": solve_ivp_DOP852,
        "solve_ivp+Radau-stepping0": solve_ivp_Radau,
        "solve_ivp+BDF-stepping0": solve_ivp_BDF,
        "solve_ivp+LSODA-stepping0": solve_icp_LSODA,
        BackgroundModel.TAU_SOLVER_LABEL: cumulative_GL_tau,
        GkWKBIntegration.PHASE_SOLVER_LABEL: wkb_primitive_phase,
    }

    # main.py:4312-4346 policies
    GkSource_policy_1pt5, GkSource_policy_5pt0 = ray.get(
        [
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="1.5"',
                Levin_threshold=1.5,
                numeric_policy="maximize-WKB",
            ),
            pool.object_get(
                "GkSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize-WKB",
            ),
        ]
    )
    QuadSource_policy_1pt5, QuadSource_policy_5pt0 = ray.get(
        [
            pool.object_get("QuadSourcePolicy", **DEFAULT_QUADSOURCE_POLICY_1PT5),
            pool.object_get(
                "QuadSourcePolicy",
                label='policy="maximize-WKB"-Levin-threshold="5.0"',
                Levin_threshold=5.0,
                numeric_policy="maximize-WKB",
            ),
        ]
    )

    # main.py:4357-4376 the k sample (SWEEP_K_SAMPLE for both literals)
    source_k_array = ray.get(convert_to_wavenumbers(A3V2_K_SAMPLE, is_source=True))
    source_k_sample = wavenumber_array(k_array=source_k_array)
    response_k_array = ray.get(convert_to_wavenumbers(A3V2_K_SAMPLE, is_response=True))
    response_k_sample = wavenumber_array(k_array=response_k_array)

    # main.py:4378 with quadsource_atol_sweep.py's _filtered_build
    model_list = [m for m in build_model_list(pool, units) if m["label"] == "LambdaCDM"]
    _check(checks, "one LambdaCDM model", len(model_list) == 1, str(len(model_list)))
    model_cosmology = model_list[0]["cosmology"]

    # run_pipeline step 0, main.py:1042-1082 (RayWorkPool results come back in input order)
    def build_k_exit_work(k):
        return pool.object_get(
            "wavenumber_exit_time",
            k=k,
            cosmology=model_cosmology,
            atol=hexit_atol,
            rtol=hexit_rtol,
        )

    source_k_exit_times = wavenumber_exit_time_array(
        [ray.get(build_k_exit_work(k)) for k in source_k_sample]
    )
    response_k_exit_times = wavenumber_exit_time_array(
        [ray.get(build_k_exit_work(k)) for k in response_k_sample]
    )
    _check(
        checks,
        "exit times available",
        all(
            x.available for x in list(source_k_exit_times) + list(response_k_exit_times)
        ),
    )

    # run_pipeline step 2, main.py:1088-1153
    full_k_exit_times = source_k_exit_times + response_k_exit_times
    k_exit_earliest = full_k_exit_times.max
    z_init_grid = k_exit_earliest.z_exit_suph_e5
    break_z, feature_z = cosmology_feature_redshifts(model_cosmology, zend, z_init_grid)
    base_z_grid = build_z_sample(
        z_init=z_init_grid, z_end=zend, samples_per_log10z=source_samples_per_log10z
    ).z_values
    spacing = source_grid_spacing_profile(
        model_cosmology, base_z_grid, [float(k_exit.k) for k_exit in full_k_exit_times]
    )
    source_grid = k_exit_earliest.populate_source_grid(
        outside_horizon_efolds=5,
        samples_per_log10z=source_samples_per_log10z,
        z_end=zend,
        break_z=break_z,
        feature_z=feature_z,
        spacing=spacing,
    )
    source_z_grid = source_grid.z_values
    z_source_array = ray.get(convert_to_redshifts(source_z_grid, is_source=True))
    z_source_sample = redshift_array(z_array=z_source_array)
    z_protected_sample = ray.get(
        convert_to_redshifts(source_grid.protected_z, is_source=True)
    )
    _z_response_sample = z_source_sample.winnow(
        sparseness=response_sparseness, protect=z_protected_sample
    )
    z_response_array = ray.get(
        convert_to_redshifts([z.z for z in _z_response_sample], is_response=True)
    )
    z_response_sample = redshift_array(z_array=z_response_array)
    source_grid_label, response_grid_label = build_grid_tag_labels(
        z_source_sample.as_float_list(), z_response_sample.as_float_list()
    )
    log(
        f"   rebuild: source grid {source_grid_label}, response grid {response_grid_label}"
    )

    # every rebuilt redshift's z equals the stored z (read_table, mode=ro through the pool)
    redshift_objs = ray.get(pool.read_table("redshift"))  # prompt 06: kept, by serial
    stored_z = {z.store_id: z.z for z in redshift_objs}
    mismatched = [
        (z.store_id, z.z, stored_z.get(z.store_id))
        for z in list(z_source_sample) + list(z_response_sample)
        if stored_z.get(z.store_id) != z.z
    ]
    _check(
        checks,
        "rebuilt redshifts equal stored z exactly",
        len(mismatched) == 0,
        str(mismatched[:5]),
    )

    # main.py:1158-1191 tags
    (
        TkProductionTag,
        GkProductionTag,
        RunLabelTag,
        SourceGridConstructionTag,
        SourceZGridSizeTag,
        ResponseZGridSizeTag,
        OutsideHorizonEfoldsTag,
        LargestSourceZTag,
        SmallestSourceZTag,
        ResponseSparsenessZTag,
    ) = ray.get(
        [
            pool.object_get("store_tag", label="TkOneLoopDensity"),
            pool.object_get("store_tag", label="GkOneLoopDensity"),
            pool.object_get("store_tag", label=run_label_tag(run_label)),
            pool.object_get(
                "store_tag",
                label=source_grid_construction_tag(SOURCE_GRID_CONSTRUCTION_VERSION),
            ),
            pool.object_get("store_tag", label=source_grid_label),
            pool.object_get("store_tag", label=response_grid_label),
            pool.object_get("store_tag", label=f"OutsideHorizonEfolds_e3"),
            pool.object_get(
                "store_tag", label=f"LargestSourceRedshift_{z_source_sample.max.z:.5g}"
            ),
            pool.object_get(
                "store_tag", label=f"SmallestSourceRedshift_{z_source_sample.min.z:.5g}"
            ),
            pool.object_get(
                "store_tag", label=f"ResponseSparsenessZ_{response_sparseness}"
            ),
        ]
    )

    # main.py:1215-1233 background model
    bg_model = ray.get(
        pool.object_get(
            "BackgroundModel",
            solver_labels=solvers,
            cosmology=model_cosmology,
            z_sample=z_source_sample,
            tags=[
                RunLabelTag,
                SourceGridConstructionTag,
                LargestSourceZTag,
                SmallestSourceZTag,
            ],
        )
    )
    _check(
        checks, "background model available", bg_model.available, str(bg_model.store_id)
    )
    model_proxy = ModelProxy(bg_model)

    # main.py:1252-1294 Bessel phase objects
    largest_source_k = source_k_exit_times.max.k
    mod_f = bg_model.functions
    largest_tau = mod_f.tau(zend)
    largest_x = largest_source_k.k * largest_tau
    largest_x_with_clearance = 1.075 * largest_x
    b_value = 0.0
    log(
        f"   rebuild: largest source k = {largest_source_k.k_inv_Mpc:.5g}/Mpc, latest tau = "
        f"{largest_tau:.5g} (for z={zend:.5g}), largest x={largest_x:.5g}, largest x +7.5% "
        f"clearance={largest_x_with_clearance:.5g}"
    )
    Bessel_0pt5 = bessel_phase(
        0.5 + b_value, largest_x_with_clearance, phase_atol=1e-12, amplitude_rtol=1e-12
    )
    Bessel_0pt5_proxy = BesselPhaseProxy(Bessel_0pt5)
    Bessel_2pt5 = bessel_phase(
        2.5 + b_value, largest_x_with_clearance, phase_atol=1e-12, amplitude_rtol=1e-12
    )
    Bessel_2pt5_proxy = BesselPhaseProxy(Bessel_2pt5)

    # main.py:3293-3301 the stage's range
    z_source_integral_max_z = z_source_sample[10]
    z_source_integral_response_sample = z_response_sample.truncate(
        z_source_integral_max_z, keep="lower-strict"
    )

    return dict(
        checks=checks,
        units=units,
        quad_atol=quad_atol,
        quad_rtol=quad_rtol,
        Gk_numeric_atol=Gk_numeric_atol,
        Gk_numeric_rtol=Gk_numeric_rtol,
        Tk_numeric_atol=Tk_numeric_atol,
        Tk_numeric_rtol=Tk_numeric_rtol,
        GkSource_policy_1pt5=GkSource_policy_1pt5,
        QuadSource_policy_1pt5=QuadSource_policy_1pt5,
        source_k_exit_times=source_k_exit_times,
        response_k_exit_times=response_k_exit_times,
        z_source_sample=z_source_sample,
        z_response_sample=z_response_sample,
        source_grid_label=source_grid_label,
        response_grid_label=response_grid_label,
        tags=dict(
            TkProductionTag=TkProductionTag,
            GkProductionTag=GkProductionTag,
            RunLabelTag=RunLabelTag,
            SourceGridConstructionTag=SourceGridConstructionTag,
            SourceZGridSizeTag=SourceZGridSizeTag,
            ResponseZGridSizeTag=ResponseZGridSizeTag,
            OutsideHorizonEfoldsTag=OutsideHorizonEfoldsTag,
            LargestSourceZTag=LargestSourceZTag,
            SmallestSourceZTag=SmallestSourceZTag,
            ResponseSparsenessZTag=ResponseSparsenessZTag,
        ),
        bg_model=bg_model,
        model_proxy=model_proxy,
        b_value=b_value,
        largest_x_with_clearance=largest_x_with_clearance,
        Bessel_0pt5_proxy=Bessel_0pt5_proxy,
        Bessel_2pt5_proxy=Bessel_2pt5_proxy,
        z_source_integral_max_z=z_source_integral_max_z,
        z_source_integral_response_sample=z_source_integral_response_sample,
        solvers=solvers,  # prompt 06
        redshift_by_id={z.store_id: z for z in redshift_objs},  # prompt 06
    )


def rebuild_row(
    ctx: dict, pool, shard: int, serial: int, log=print, rtol=None, rerun=False, cache=None
) -> dict:
    """The QuadSourceIntegral stage for the one work item stored as (shard, serial).

    ``rtol``: a ``tolerance`` object to build the work object with in place of
    ``ctx["quad_rtol"]`` (prompt 04: the 1e-8 rung). Every lookup and check is unchanged; the
    existence query still uses ``ctx["quad_rtol"]``, the stored row's."""
    checks = []
    raw = read_row_ro(shard, serial)
    t = ctx["tags"]
    model_proxy = ctx["model_proxy"]
    z_source_sample = ctx["z_source_sample"]

    def by_id(arr, store_id, what):
        found = [x for x in arr if x.store_id == store_id]
        _check(
            checks,
            f"{what} store_id {store_id} found once",
            len(found) == 1,
            str(len(found)),
        )
        return found[0]

    # the work item, as main.py:4127-4140 builds the list: (z, k, q, r) with z from
    # z_source_integral_response_sample, (q, r) from combinations_with_replacement of the source
    # exit times, k from the response exit times
    k = by_id(ctx["response_k_exit_times"], raw["k_wavenumber_exit_serial"], "k")
    q = by_id(ctx["source_k_exit_times"], raw["q_wavenumber_exit_serial"], "q")
    r = by_id(ctx["source_k_exit_times"], raw["r_wavenumber_exit_serial"], "r")
    z_response = by_id(
        ctx["z_source_integral_response_sample"], raw["z_response_serial"], "z_response"
    )
    _check(
        checks,
        "z_source_max serial",
        ctx["z_source_integral_max_z"].store_id == raw["z_source_max_serial"],
        f"{ctx['z_source_integral_max_z'].store_id} vs {raw['z_source_max_serial']}",
    )
    source_list = list(ctx["source_k_exit_times"])
    in_work_list = (q, r) in list(
        itertools.combinations_with_replacement(source_list, 2)
    )
    _check(checks, "(q, r) is a combinations_with_replacement pair", in_work_list)
    _check(checks, "closes_triangle", closes_triangle(k, q, r))
    _check(checks, "model serial", ctx["bg_model"].store_id == raw["model_serial"])
    _check(checks, "atol serial", ctx["quad_atol"].store_id == raw["atol_serial"])
    _check(checks, "rtol serial", ctx["quad_rtol"].store_id == raw["rtol_serial"])
    _check(
        checks,
        "policy serial",
        ctx["GkSource_policy_1pt5"].store_id == raw["policy_serial"],
    )
    _check(
        checks,
        "quad_policy serial",
        ctx["QuadSource_policy_1pt5"].store_id == raw["quad_policy_serial"],
    )

    batch = [(z_response, k, q, r)]
    item_key = (z_response.store_id, k.store_id, q.store_id, r.store_id)

    # resolve_GkSource_parents, main.py:2464-2534, for [(k, [z_response])]
    Gk_tags = [
        t["GkProductionTag"],
        t["RunLabelTag"],
        t["SourceGridConstructionTag"],
        t["SourceZGridSizeTag"],
        t["ResponseZGridSizeTag"],
        t["LargestSourceZTag"],
        t["SmallestSourceZTag"],
        t["ResponseSparsenessZTag"],
    ]
    results = []
    for cls_name in ["GkNumericValue", "GkWKBValue"]:
        payload = {
            "model": model_proxy,
            "z": z_response,
            "parents_only": True,
            **(
                {"atol": ctx["Gk_numeric_atol"], "rtol": ctx["Gk_numeric_rtol"]}
                if cls_name == "GkNumericValue"
                else {}
            ),
            "tags": Gk_tags,
        }
        results.append(ray.get(pool.object_read_batch(cls_name, {"k": k}, **payload)))
    GkSource_parents = {
        (k.store_id, z_response.store_id): GkSource_parent_set_of_serials(
            z_source_sample.truncate(z_response, keep="higher"),
            dict(results[0]),
            dict(results[1]),
        )
    }

    # resolve_QuadSourceIntegral_parents, main.py:3303-3614, for this one item
    Tk_tags = [
        t["TkProductionTag"],
        t["RunLabelTag"],
        t["SourceGridConstructionTag"],
        t["SourceZGridSizeTag"],
        t["OutsideHorizonEfoldsTag"],
        t["LargestSourceZTag"],
        t["SmallestSourceZTag"],
    ]
    modes = {}
    modes.setdefault(q.store_id, q)
    modes.setdefault(r.store_id, r)
    modes = list(modes.values())
    Tk_numeric_serial_results = [
        ray.get(
            pool.object_get(
                "TkNumericIntegration",
                model=model_proxy,
                z_sample=None,
                k=k_exit,
                z_init=None,
                atol=ctx["Tk_numeric_atol"],
                rtol=ctx["Tk_numeric_rtol"],
                tags=Tk_tags,
                _do_not_populate=True,
            )
        )
        for k_exit in modes
    ]
    Tk_numeric_serials = {
        k_exit.store_id: obj.store_id if obj.available else None
        for k_exit, obj in zip(modes, Tk_numeric_serial_results)
    }
    WKB_modes = [
        k_exit for k_exit in modes if Tk_numeric_serials[k_exit.store_id] is not None
    ]
    Tk_WKB_serial_results = [
        ray.get(
            pool.object_get(
                "TkWKBIntegration",
                solver_labels=[],
                model=model_proxy,
                z_sample=None,
                k=k_exit,
                numeric_serial=Tk_numeric_serials[k_exit.store_id],
                tags=Tk_tags,
                _do_not_populate=True,
            )
        )
        for k_exit in WKB_modes
    ]
    Tk_WKB_serials = {k_exit.store_id: None for k_exit in modes}
    Tk_WKB_serials.update(
        {
            k_exit.store_id: obj.store_id if obj.available else None
            for k_exit, obj in zip(WKB_modes, Tk_WKB_serial_results)
        }
    )
    (QuadSource_serial_obj,) = ray.get(
        pool.object_get_vectorized(
            "QuadSource",
            {"q": q},
            payload_data=[
                {
                    "model": model_proxy,
                    "z_sample": None,
                    "r": r,
                    "Tq_serial": Tk_numeric_serials[q.store_id],
                    "Tr_serial": Tk_numeric_serials[r.store_id],
                    "tags": Tk_tags,
                    "_do_not_populate": True,
                }
            ],
        )
    )
    QuadSource_serials = {
        (q.store_id, r.store_id): (
            QuadSource_serial_obj.store_id if QuadSource_serial_obj.available else None
        )
    }
    (GkSource_serial_obj,) = ray.get(
        pool.object_get_vectorized(
            "GkSource",
            {"k": k},
            payload_data=[
                {
                    "model": model_proxy,
                    "z_response": z_response,
                    "z_sample": None,
                    "parents": GkSource_parents[(k.store_id, z_response.store_id)],
                    "tags": Gk_tags,
                    "_do_not_populate": True,
                }
            ],
        )
    )
    _check(checks, "GkSource (serial-only) available", GkSource_serial_obj.available)
    (GkSourcePolicy_serial_obj,) = ray.get(
        pool.object_get_vectorized(
            "GkSourcePolicyData",
            {"k": k},
            payload_data=[
                {
                    "source": GkSourceProxy(GkSource_serial_obj),
                    "policy": ctx["GkSource_policy_1pt5"],
                    "k": k,
                }
            ],
        )
    )
    serials = {
        "source_serial": QuadSource_serials.get((q.store_id, r.store_id)),
        "data_serial": (
            GkSourcePolicy_serial_obj.store_id
            if GkSourcePolicy_serial_obj.available
            else None
        ),
        "Tq_numeric_serial": Tk_numeric_serials[q.store_id],
        "Tq_WKB_serial": Tk_WKB_serials[q.store_id],
        "Tr_numeric_serial": Tk_numeric_serials[r.store_id],
        "Tr_WKB_serial": Tk_WKB_serials[r.store_id],
    }
    _check(
        checks,
        "all six parents found",
        all(v is not None for v in serials.values()),
        str(serials),
    )
    for name, value in serials.items():
        _check(
            checks,
            f"parent {name} equals the stored row's",
            value == raw[name],
            f"{value} vs {raw[name]}",
        )
    parents = serials

    # the existence query, main.py:3657-3691
    QSI_tags_query = [
        t["GkProductionTag"],
        t["TkProductionTag"],
        t["RunLabelTag"],
        t["SourceGridConstructionTag"],
        t["SourceZGridSizeTag"],
        t["ResponseZGridSizeTag"],
        t["LargestSourceZTag"],
        t["SmallestSourceZTag"],
        t["ResponseSparsenessZTag"],
    ]
    (stored,) = ray.get(
        pool.object_get_vectorized(
            "QuadSourceIntegral",
            {"k": k},
            payload_data=[
                {
                    "model": model_proxy,
                    "policy": ctx["GkSource_policy_1pt5"],
                    "quad_policy": ctx["QuadSource_policy_1pt5"],
                    "q": q,
                    "r": r,
                    "z_response": z_response,
                    "z_source_max": ctx["z_source_integral_max_z"],
                    "atol": ctx["quad_atol"],
                    "rtol": ctx["quad_rtol"],
                    **parents,
                    "tags": QSI_tags_query,
                }
            ],
        )
    )
    _check(
        checks,
        "existence query finds the stored row",
        stored.available and stored.store_id == serial,
        f"{stored.store_id}",
    )

    # the stage's full lookups, main.py:3733-4029, for this one item
    # prompt 06: with rerun, the GkSource, GkSourcePolicyData and TkWKBIntegration lookups are
    # replaced by the producers re-run in-process (rerun.py), after the TkNumericIntegration ones
    # prompt 06a: with ``cache`` (a CensusCache), the producers come from the census's producer
    # cache, each re-run once (rerun_k_G, rerun_T_mode) and assembled once per (k, z_response)
    # (assemble_Gk); every serial-only lookup, every check and the existence query above are
    # unchanged, and run first, per row
    if cache is not None:
        return _rebuild_row_census(
            ctx, pool, shard, serial, log, rtol, checks, raw, stored, k, q, r, z_response,
            GkSource_parents, GkSource_serial_obj, GkSourcePolicy_serial_obj, Gk_tags, Tk_tags,
            parents, cache,
        )
    if rerun:
        return _rebuild_row_rerun(
            ctx, pool, shard, serial, log, rtol, checks, raw, stored, k, q, r, z_response,
            GkSource_parents, GkSource_serial_obj, GkSourcePolicy_serial_obj, Gk_tags, Tk_tags,
            parents,
        )
    (GkSource_obj,) = ray.get(
        pool.object_get_vectorized(
            "GkSource",
            {"k": k},
            payload_data=[
                {
                    "model": model_proxy,
                    "z_response": z_response,
                    "z_sample": None,
                    "parents": GkSource_parents[(k.store_id, z_response.store_id)],
                    "tags": Gk_tags,
                }
            ],
        )
    )
    (GkPolicy_obj,) = ray.get(
        pool.object_get_vectorized(
            "GkSourcePolicyData",
            {"k": k},
            payload_data=[
                {
                    "source": GkSourceProxy(GkSource_obj),
                    "policy": ctx["GkSource_policy_1pt5"],
                    "k": k,
                }
            ],
        )
    )
    missing_Tk = list({q, r})
    Tk_numeric_results = [
        ray.get(
            pool.object_get(
                "TkNumericIntegration",
                model=model_proxy,
                z_sample=None,
                k=k_exit,
                z_init=None,
                atol=ctx["Tk_numeric_atol"],
                rtol=ctx["Tk_numeric_rtol"],
                tags=Tk_tags,
            )
        )
        for k_exit in missing_Tk
    ]
    Tk_numeric_serials_full = {
        k_exit.store_id: obj.store_id
        for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    (QuadSource_obj,) = ray.get(
        pool.object_get_vectorized(
            "QuadSource",
            {"q": q},
            payload_data=[
                {
                    "model": model_proxy,
                    "z_sample": None,
                    "r": r,
                    "Tq_serial": Tk_numeric_serials_full[q.store_id],
                    "Tr_serial": Tk_numeric_serials_full[r.store_id],
                    "tags": Tk_tags,
                }
            ],
        )
    )
    Tk_WKB_results = [
        ray.get(
            pool.object_get(
                "TkWKBIntegration",
                solver_labels=[],
                model=model_proxy,
                z_sample=None,
                k=k_exit,
                numeric_serial=Tk_numeric.store_id,
                tags=Tk_tags,
            )
        )
        for k_exit, Tk_numeric in zip(missing_Tk, Tk_numeric_results)
    ]
    # prompt 05: QuadSource recomputed in-process (module docstring), and compared with the stored
    Tk_numeric_full = {
        k_exit.store_id: obj for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    Tq_num = Tk_numeric_full[q.store_id]
    Tr_num = Tk_numeric_full[r.store_id]
    QuadSource_recomputed, qs_check = recompute_QuadSource(
        ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, QuadSource_obj
    )
    _check(
        checks,
        "QuadSource recomputed: keyed on the stored row's two TkNumericIntegrations",
        qs_check["keyed_as_stored"],
        str(qs_check["Tq_Tr_serials"]),
    )

    Gk_cache = {(k.store_id, z_response.store_id): GkPolicy_obj}
    source_cache = {(q.store_id, r.store_id): QuadSource_recomputed}  # prompt 05
    Tk_numeric_cache = {
        k_exit.store_id: obj for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    Tk_WKB_cache = {
        k_exit.store_id: obj for k_exit, obj in zip(missing_Tk, Tk_WKB_results)
    }

    # main.py:4058-4070, main.py's own function
    compute_payload = build_QuadSourceIntegral_payload(
        z_response,
        k,
        q,
        r,
        Gk_cache,
        source_cache,
        Tk_numeric_cache,
        Tk_WKB_cache,
        ctx["b_value"],
        ctx["Bessel_0pt5_proxy"],
        ctx["Bessel_2pt5_proxy"],
    )

    # the work object: what main.py:4087-4110's object_get returns for an item it must compute,
    # i.e. the factory's miss branch (ObjectFactories/QuadSourceIntegral.py, build(), row_data
    # is None), with main.py's arguments
    work = QuadSourceIntegral(
        payload=None,
        model=model_proxy,
        policy=ctx["GkSource_policy_1pt5"],
        quad_policy=ctx["QuadSource_policy_1pt5"],
        z_response=z_response,
        z_source_max=ctx["z_source_integral_max_z"],
        k=k,
        q=q,
        r=r,
        atol=ctx["quad_atol"],
        rtol=ctx["quad_rtol"] if rtol is None else rtol,
        label=None,
        tags=[
            t["TkProductionTag"],
            t["GkProductionTag"],
            t["RunLabelTag"],
            t["SourceGridConstructionTag"],
            t["SourceZGridSizeTag"],
            t["ResponseZGridSizeTag"],
            t["LargestSourceZTag"],
            t["SmallestSourceZTag"],
            t["ResponseSparsenessZTag"],
        ],
        **parents,
    )

    return dict(
        qs_check=qs_check,  # prompt 05
        checks=checks,
        raw=raw,
        stored=stored,
        work=work,
        compute_payload=compute_payload,
        k=k,
        q=q,
        r=r,
        z_response=z_response,
        parents=parents,
        GkPolicy_summary=policy_summary(GkPolicy_obj),  # prompt 06
        rerun=None,  # prompt 06
    )


# ---------------------------------------------------------------------------------------------
# prompt 06: the producers re-run in-process (rerun.py)
# ---------------------------------------------------------------------------------------------


def policy_summary(GkPolicy):
    """A GkSourcePolicyData's classification, exactly (floats as hex)."""

    def fz(x):
        return None if x is None else float(x).hex()

    return {
        "store_id": GkPolicy.store_id,
        "type": GkPolicy.type,
        "quality": GkPolicy.quality,
        "crossover_z": fz(GkPolicy.crossover_z),
        "Levin_z": fz(GkPolicy.Levin_z),
    }


def _rebuild_row_rerun(
    ctx, pool, shard, serial, log, rtol, checks, raw, stored, k, q, r, z_response,
    GkSource_parents, GkSource_serial_obj, GkSourcePolicy_serial_obj, Gk_tags, Tk_tags, parents,
):
    """rebuild_row's tail with the producers re-run: the TkNumericIntegration and QuadSource
    lookups and the QuadSource recomputation exactly as rebuild_row makes them; the GkSource,
    GkSourcePolicyData and TkWKBIntegration from rerun.rerun_row."""
    import rerun as RR

    t = ctx["tags"]
    model_proxy = ctx["model_proxy"]
    missing_Tk = list({q, r})
    Tk_numeric_results = [
        ray.get(
            pool.object_get(
                "TkNumericIntegration",
                model=model_proxy,
                z_sample=None,
                k=k_exit,
                z_init=None,
                atol=ctx["Tk_numeric_atol"],
                rtol=ctx["Tk_numeric_rtol"],
                tags=Tk_tags,
            )
        )
        for k_exit in missing_Tk
    ]
    Tk_numeric_serials_full = {
        k_exit.store_id: obj.store_id
        for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    (QuadSource_obj,) = ray.get(
        pool.object_get_vectorized(
            "QuadSource",
            {"q": q},
            payload_data=[
                {
                    "model": model_proxy,
                    "z_sample": None,
                    "r": r,
                    "Tq_serial": Tk_numeric_serials_full[q.store_id],
                    "Tr_serial": Tk_numeric_serials_full[r.store_id],
                    "tags": Tk_tags,
                }
            ],
        )
    )
    rr = RR.rerun_row(
        ctx,
        pool,
        shard_file(shard),
        k,
        q,
        r,
        z_response,
        GkSource_parents[(k.store_id, z_response.store_id)],
        GkSource_serial_obj,
        GkSourcePolicy_serial_obj,
        Gk_tags,
        Tk_tags,
        missing_Tk,
        Tk_numeric_results,
        log=log,
    )
    GkPolicy_obj = rr["GkPolicy"]
    Tk_numeric_full = {
        k_exit.store_id: obj for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    Tq_num = Tk_numeric_full[q.store_id]
    Tr_num = Tk_numeric_full[r.store_id]
    QuadSource_recomputed, qs_check = recompute_QuadSource(
        ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, QuadSource_obj
    )
    _check(
        checks,
        "QuadSource recomputed: keyed on the stored row's two TkNumericIntegrations",
        qs_check["keyed_as_stored"],
        str(qs_check["Tq_Tr_serials"]),
    )
    Gk_cache = {(k.store_id, z_response.store_id): GkPolicy_obj}
    source_cache = {(q.store_id, r.store_id): QuadSource_recomputed}
    Tk_numeric_cache = Tk_numeric_full
    Tk_WKB_cache = {k_exit.store_id: rr["Tk_WKB"][k_exit.store_id] for k_exit in missing_Tk}
    compute_payload = build_QuadSourceIntegral_payload(
        z_response,
        k,
        q,
        r,
        Gk_cache,
        source_cache,
        Tk_numeric_cache,
        Tk_WKB_cache,
        ctx["b_value"],
        ctx["Bessel_0pt5_proxy"],
        ctx["Bessel_2pt5_proxy"],
    )
    work = QuadSourceIntegral(
        payload=None,
        model=model_proxy,
        policy=ctx["GkSource_policy_1pt5"],
        quad_policy=ctx["QuadSource_policy_1pt5"],
        z_response=z_response,
        z_source_max=ctx["z_source_integral_max_z"],
        k=k,
        q=q,
        r=r,
        atol=ctx["quad_atol"],
        rtol=ctx["quad_rtol"] if rtol is None else rtol,
        label=None,
        tags=[
            t["TkProductionTag"],
            t["GkProductionTag"],
            t["RunLabelTag"],
            t["SourceGridConstructionTag"],
            t["SourceZGridSizeTag"],
            t["ResponseZGridSizeTag"],
            t["LargestSourceZTag"],
            t["SmallestSourceZTag"],
            t["ResponseSparsenessZTag"],
        ],
        **parents,
    )
    rerun_record = {
        key: value for key, value in rr.items() if key not in ("GkPolicy", "GkSource", "Tk_WKB")
    }
    return dict(
        qs_check=qs_check,
        checks=checks,
        raw=raw,
        stored=stored,
        work=work,
        compute_payload=compute_payload,
        k=k,
        q=q,
        r=r,
        z_response=z_response,
        parents=parents,
        GkPolicy_summary=policy_summary(GkPolicy_obj),
        rerun=rerun_record,
    )


# ---------------------------------------------------------------------------------------------
# prompt 06a: the census's producer cache (section 0.3 change 1, section 2 item 3)
# ---------------------------------------------------------------------------------------------


class CensusCache:
    """The producers of one worker, each computed once:
    - per k: every GkWKBIntegration re-run (rerun.rerun_k_G), its values at the response
      redshifts kept;
    - per (k, z_response): the GkSource and GkSourcePolicyData assembled from them
      (rerun.assemble_Gk), as ``Gk_cache`` is keyed. Only the current block is held (rows are run
      block by block); a block left and re-entered is assembled again, from the same values;
    - per mode: the full TkNumericIntegration lookup and the TkWKBIntegration re-run
      (rerun.rerun_T_mode);
    - per (q, r): the stored QuadSource's full lookup and the QuadSource recomputed
      (recompute_QuadSource).
    ``events`` collects what was built, with its timings and comparisons, for the worker to emit;
    the worker attributes them to the unit in which they were built."""

    def __init__(self, store_dir, warn_counter_factory=None):
        self.store_dir = Path(store_dir)
        self.G_values = {}  # k store_id -> {(wkb_serial, z_serial): (value serial, z_source serial, value)}
        self.Gk_key = None
        self.Gk = None
        self.Tk_numeric = {}
        self.T = {}
        self.QS = {}
        self.events = []
        self.warn_counter_factory = warn_counter_factory

    def G_for_k(self, ctx, pool, k, Gk_tags, log=print):
        import rerun as RR

        if k.store_id not in self.G_values:
            keep = [z.store_id for z in ctx["z_source_integral_response_sample"]]
            values, summary = RR.rerun_k_G(ctx, pool, self.store_dir, k, Gk_tags, keep, log=log)
            self.G_values[k.store_id] = values
            self.events.append({"kind": "producers_G", "k_serial": k.store_id, "summary": summary})
        return self.G_values[k.store_id]

    def Gk_for(
        self, ctx, pool, k, z_response, parents_set, GkSource_serial_obj,
        GkSourcePolicy_serial_obj, Gk_tags, log=print,
    ):
        import rerun as RR

        key = (k.store_id, z_response.store_id)
        if self.Gk_key != key:
            G_values = self.G_for_k(ctx, pool, k, Gk_tags, log=log)
            self.Gk_key = None
            self.Gk = None
            import sys as _sys

            counter = None
            if self.warn_counter_factory is not None:
                counter = self.warn_counter_factory(_sys.stdout)
                _sys.stdout = counter
            try:
                out = RR.assemble_Gk(
                    ctx, pool, self.store_dir, k, z_response, parents_set, GkSource_serial_obj,
                    GkSourcePolicy_serial_obj, Gk_tags, G_values,
                )
            finally:
                if counter is not None:
                    _sys.stdout = counter._stream
            self.Gk_key = key
            self.Gk = {
                "GkPolicy": out["GkPolicy"],
                "parents_set": parents_set,
                "GkSource_serial": GkSource_serial_obj.store_id,
                "GkSourcePolicy_serial": GkSourcePolicy_serial_obj.store_id,
            }
            ev = {k_: v for k_, v in out.items() if k_ not in ("GkPolicy", "GkSource")}
            ev["policy_summary"] = policy_summary(out["GkPolicy"])
            ev["policy_warnings"] = (
                None if counter is None else {"count": counter.count, "first": counter.lines}
            )
            self.events.append(
                {"kind": "Gk", "k_serial": k.store_id, "z_response_serial": z_response.store_id,
                 "summary": ev}
            )
        if (
            self.Gk["parents_set"] != parents_set
            or self.Gk["GkSource_serial"] != GkSource_serial_obj.store_id
            or self.Gk["GkSourcePolicy_serial"] != GkSourcePolicy_serial_obj.store_id
        ):
            raise RuntimeError("census cache: the row's GkSource keys are not the block's")
        return self.Gk["GkPolicy"]

    def Tk_numeric_for(self, ctx, pool, k_exit, Tk_tags):
        if k_exit.store_id not in self.Tk_numeric:
            self.Tk_numeric[k_exit.store_id] = ray.get(
                pool.object_get(
                    "TkNumericIntegration",
                    model=ctx["model_proxy"],
                    z_sample=None,
                    k=k_exit,
                    z_init=None,
                    atol=ctx["Tk_numeric_atol"],
                    rtol=ctx["Tk_numeric_rtol"],
                    tags=Tk_tags,
                )
            )
        return self.Tk_numeric[k_exit.store_id]

    def T_for(self, ctx, pool, k_exit, Tk_numeric, Tk_tags):
        import rerun as RR

        if k_exit.store_id not in self.T:
            rb, info, stats = RR.rerun_T_mode(ctx, pool, self.store_dir, k_exit, Tk_numeric, Tk_tags)
            self.T[k_exit.store_id] = (rb, Tk_numeric.store_id)
            self.events.append(
                {"kind": "producers_T", "mode_serial": k_exit.store_id, "info": info,
                 "T_values_vs_stored": stats}
            )
        rb, numeric_serial = self.T[k_exit.store_id]
        if numeric_serial != Tk_numeric.store_id:
            raise RuntimeError("census cache: a mode's TkNumericIntegration changed")
        return rb

    def QS_for(self, ctx, pool, model_proxy, q, r, Tq_num, Tr_num, Tk_tags):
        key = (q.store_id, r.store_id)
        if key not in self.QS:
            t0 = time.perf_counter()
            (QuadSource_obj,) = ray.get(
                pool.object_get_vectorized(
                    "QuadSource",
                    {"q": q},
                    payload_data=[
                        {
                            "model": model_proxy,
                            "z_sample": None,
                            "r": r,
                            "Tq_serial": Tq_num.store_id,
                            "Tr_serial": Tr_num.store_id,
                            "tags": Tk_tags,
                        }
                    ],
                )
            )
            read_back, qs_check = recompute_QuadSource(
                ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, QuadSource_obj
            )
            self.QS[key] = (read_back, qs_check, (Tq_num.store_id, Tr_num.store_id))
            self.events.append(
                {"kind": "QuadSource", "q_serial": q.store_id, "r_serial": r.store_id,
                 "seconds": time.perf_counter() - t0, "L0a_holds": qs_check["L0a_holds"]}
            )
        read_back, qs_check, serials = self.QS[key]
        if serials != (Tq_num.store_id, Tr_num.store_id):
            raise RuntimeError("census cache: a QuadSource's TkNumericIntegrations changed")
        return read_back, qs_check


def _rebuild_row_census(
    ctx, pool, shard, serial, log, rtol, checks, raw, stored, k, q, r, z_response,
    GkSource_parents, GkSource_serial_obj, GkSourcePolicy_serial_obj, Gk_tags, Tk_tags, parents,
    cache,
):
    """prompt 06a: _rebuild_row_rerun with the producers from the CensusCache: the
    TkNumericIntegration full lookups and the QuadSource (lookup and recomputation) once per mode
    and per pair; the GkSourcePolicyData from rerun.assemble_Gk once per (k, z_response); the
    TkWKBIntegration from rerun.rerun_T_mode once per mode. Then main.py's own
    build_QuadSourceIntegral_payload and the work object, per row, exactly as
    _rebuild_row_rerun builds them."""
    t = ctx["tags"]
    model_proxy = ctx["model_proxy"]
    missing_Tk = list({q, r})
    Tk_numeric_results = [
        cache.Tk_numeric_for(ctx, pool, k_exit, Tk_tags) for k_exit in missing_Tk
    ]
    Tk_numeric_full = {
        k_exit.store_id: obj for k_exit, obj in zip(missing_Tk, Tk_numeric_results)
    }
    Tq_num = Tk_numeric_full[q.store_id]
    Tr_num = Tk_numeric_full[r.store_id]
    GkPolicy_obj = cache.Gk_for(
        ctx, pool, k, z_response, GkSource_parents[(k.store_id, z_response.store_id)],
        GkSource_serial_obj, GkSourcePolicy_serial_obj, Gk_tags, log=log,
    )
    Tk_WKB = {
        k_exit.store_id: cache.T_for(ctx, pool, k_exit, Tk_numeric, Tk_tags)
        for k_exit, Tk_numeric in zip(missing_Tk, Tk_numeric_results)
    }
    QuadSource_recomputed, qs_check = cache.QS_for(
        ctx, pool, model_proxy, q, r, Tq_num, Tr_num, Tk_tags
    )
    _check(
        checks,
        "QuadSource recomputed: keyed on the stored row's two TkNumericIntegrations",
        qs_check["keyed_as_stored"],
        str(qs_check["Tq_Tr_serials"]),
    )
    Gk_cache = {(k.store_id, z_response.store_id): GkPolicy_obj}
    source_cache = {(q.store_id, r.store_id): QuadSource_recomputed}
    Tk_numeric_cache = Tk_numeric_full
    Tk_WKB_cache = {k_exit.store_id: Tk_WKB[k_exit.store_id] for k_exit in missing_Tk}
    compute_payload = build_QuadSourceIntegral_payload(
        z_response,
        k,
        q,
        r,
        Gk_cache,
        source_cache,
        Tk_numeric_cache,
        Tk_WKB_cache,
        ctx["b_value"],
        ctx["Bessel_0pt5_proxy"],
        ctx["Bessel_2pt5_proxy"],
    )
    work = QuadSourceIntegral(
        payload=None,
        model=model_proxy,
        policy=ctx["GkSource_policy_1pt5"],
        quad_policy=ctx["QuadSource_policy_1pt5"],
        z_response=z_response,
        z_source_max=ctx["z_source_integral_max_z"],
        k=k,
        q=q,
        r=r,
        atol=ctx["quad_atol"],
        rtol=ctx["quad_rtol"] if rtol is None else rtol,
        label=None,
        tags=[
            t["TkProductionTag"],
            t["GkProductionTag"],
            t["RunLabelTag"],
            t["SourceGridConstructionTag"],
            t["SourceZGridSizeTag"],
            t["ResponseZGridSizeTag"],
            t["LargestSourceZTag"],
            t["SmallestSourceZTag"],
            t["ResponseSparsenessZTag"],
        ],
        **parents,
    )
    return dict(
        qs_check=qs_check,
        checks=checks,
        raw=raw,
        stored=stored,
        work=work,
        compute_payload=compute_payload,
        k=k,
        q=q,
        r=r,
        z_response=z_response,
        parents=parents,
        GkPolicy_summary=policy_summary(GkPolicy_obj),
        rerun=None,
    )


# ---------------------------------------------------------------------------------------------
# prompt 05: QuadSource recomputed in-process, and compared with the stored row
# ---------------------------------------------------------------------------------------------

QS_VALUE_COLUMNS = (
    "source",
    "undiff",
    "diff",
    "analytic_source_rad",
    "analytic_undiff_rad",
    "analytic_diff_rad",
    "analytic_source_w",
    "analytic_undiff_w",
    "analytic_diff_w",
)


def _same_bits(a, b):
    if a is None or b is None:
        return a is b
    a = float(a)
    b = float(b)
    return a.hex() == b.hex() and (a == b or (a != a and b != b))


def recompute_QuadSource(ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, stored_qs):
    """The row's QuadSource as main.py computes it at this tree (module docstring), read back as
    the factory would read it, and its comparison with the stored row. Returns
    ``(read_back, qs_check)``."""
    # main.py:1897-1914 (build_tensor_source_work): the factory's miss branch, main.py's arguments
    work = QuadSource(
        payload=None,
        model=model_proxy,
        z_sample=ctx["z_source_sample"],
        q=q,
        r=r,
        Tq_serial=Tq_num.store_id,
        Tr_serial=Tr_num.store_id,
        label=None,
        tags=Tk_tags,
    )
    # main.py:1920-1923 (compute_tensor_source_work), then the queue's store step
    work.compute(payload={"Tq": Tq_num, "Tr": Tr_num})
    if work.store() is not True:
        raise RuntimeError("rebuild: QuadSource.store() did not resolve in-process")

    # the read-back, as ObjectFactories/QuadSource.py build() forms it from a stored row
    computed = sorted(work.values, key=lambda v: v.z.z, reverse=True)
    values = [
        QuadSourceValue(
            store_id=None,
            z=v.z,
            source=v.source,
            undiff=v.undiff,
            diff=v.diff,
            analytic_source_rad=v.analytic_source_rad,
            analytic_undiff_rad=v.analytic_undiff_rad,
            analytic_diff_rad=v.analytic_diff_rad,
            analytic_source_w=v.analytic_source_w,
            analytic_undiff_w=v.analytic_undiff_w,
            analytic_diff_w=v.analytic_diff_w,
        )
        for v in computed
    ]
    read_back = QuadSource(
        payload={
            "store_id": stored_qs.store_id,
            "compute_time": work.compute_time,
            "values": values,
            "Tq_serial": work.Tq_serial,
            "Tr_serial": work.Tr_serial,
        },
        model=model_proxy,
        z_sample=redshift_array([v.z for v in values]),
        q=q,
        r=r,
        label=stored_qs.label,
        tags=Tk_tags,
    )
    setattr(read_back, "_deserialized", True)

    # the comparison with the stored row (gate L0, first clause)
    stored_by_z = {v.z.store_id: v for v in stored_qs.values}
    comp_by_z = {v.z.store_id: v for v in work.values}
    common = [zid for zid in stored_by_z if zid in comp_by_z]
    removed = [zid for zid in stored_by_z if zid not in comp_by_z]
    added = [zid for zid in comp_by_z if zid not in stored_by_z]
    differing = []
    for zid in common:
        a, b = comp_by_z[zid], stored_by_z[zid]
        for col in QS_VALUE_COLUMNS:
            x, y = getattr(a, col), getattr(b, col)
            if not _same_bits(x, y):
                differing.append(
                    (zid, col, None if x is None else float(x).hex(), None if y is None else float(y).hex())
                )
    # B1's nodes: the switch node of the earlier hand-over and HANDOVER_OVERLAP_NODES below it
    grid = sorted(ctx["z_source_sample"], key=lambda z: z.z, reverse=True)
    z_inits = [z for z in (work.crossover_z_q, work.crossover_z_r) if z is not None]
    expected_added = None
    old_bottom_is_above_switch = None
    if len(z_inits) > 0:
        switch = handover_switch_node(max(z_inits), grid)
        i_sw = next(i for i, z in enumerate(grid) if z is switch)
        expected_added = [
            z.store_id for z in grid[i_sw : i_sw + HANDOVER_OVERLAP_NODES + 1]
        ]
        stored_bottom = min(stored_qs.values, key=lambda v: v.z.z)
        old_bottom_is_above_switch = (
            i_sw >= 1 and stored_bottom.z.store_id == grid[i_sw - 1].store_id
        )
    z_of = {z.store_id: z.z for z in grid}
    qs_check = {
        "stored_serial": stored_qs.store_id,
        "keyed_as_stored": (work.Tq_serial, work.Tr_serial)
        == (stored_qs.Tq_serial, stored_qs.Tr_serial),
        "Tq_Tr_serials": [work.Tq_serial, work.Tr_serial, stored_qs.Tq_serial, stored_qs.Tr_serial],
        "n_stored": len(stored_by_z),
        "n_recomputed": len(comp_by_z),
        "n_common": len(common),
        "n_common_differing_columns": len(differing),
        "differing_first": differing[:10],
        "removed": removed,
        "added": added,
        "added_z": [z_of.get(zid) for zid in added],
        "expected_added": expected_added,
        "added_is_B1s": (
            expected_added is not None and sorted(added) == sorted(expected_added)
        ),
        "old_bottom_is_node_above_switch": old_bottom_is_above_switch,
        "z_init_q": work.crossover_z_q,
        "z_init_r": work.crossover_z_r,
        "HANDOVER_OVERLAP_NODES": HANDOVER_OVERLAP_NODES,
        "read_back_store_id": read_back.store_id,
        "read_back_n_values": len(read_back.values),
    }
    qs_check["L0a_holds"] = bool(
        qs_check["keyed_as_stored"]
        and len(removed) == 0
        and len(differing) == 0
        and qs_check["added_is_B1s"]
        and old_bottom_is_above_switch
    )
    return read_back, qs_check
