"""
levin-phase-floor prompt 03, step 4: rebuild a QuadSourceIntegral row's inputs as main.py does,
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
from pathlib import Path

import numpy as np
import ray

from ComputeTargets import GkSourceProxy, ModelProxy, QuadSourceIntegral
from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkSource import GkSource_parent_set_of_serials
from ComputeTargets.GkWKBIntegration import GkWKBIntegration
from ComputeTargets.QuadSourceIntegral import BesselPhaseProxy
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

STORE_DIR = Path(__file__).resolve().parents[3] / "var" / "datastores"
PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"


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
    stored_z = {z.store_id: z.z for z in ray.get(pool.read_table("redshift"))}
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
    )


def rebuild_row(ctx: dict, pool, shard: int, serial: int, log=print) -> dict:
    """The QuadSourceIntegral stage for the one work item stored as (shard, serial)."""
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
    Gk_cache = {(k.store_id, z_response.store_id): GkPolicy_obj}
    source_cache = {(q.store_id, r.store_id): QuadSource_obj}
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
        rtol=ctx["quad_rtol"],
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
    )
