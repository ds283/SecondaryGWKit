"""
levin-phase-floor prompt 06 section 6: the producers re-run in-process. New in 06 (05a's harness
reads every producer value from A3 v2; prompt 06 section 6: "the re-run of the producers").

For one QuadSourceIntegral row, ``rerun_row`` re-runs, in this process and under the tree on
``PYTHONPATH``, every ``GkWKBIntegration`` whose value at the row's ``z_response`` its ``GkSource``
is assembled from, and the ``TkWKBIntegration`` of each of ``q`` and ``r``. Then it assembles the
``GkSource`` and computes the ``GkSourcePolicyData`` from them. Nothing is written to the store.

**Each producer is re-run from what A3 v2 stores** (prompt 06 section 6: "from its stored
``z_init``, initial data, ``z_sample`` and model, through the producer's own code"):
- the integration row is read by production's own reader, the factory, with
  ``_do_not_populate=True``, which reads the integration table only (at ``s`` the factories'
  value-table reads name ``theta_residual``, a column A3 v2 does not have). Its parent
  (``numeric_serial``, or ``PURE_WKB``) and its ``z_source`` are first read from the row itself
  with ``sqlite3`` ``mode=ro``, and the factory's object is checked to be that row (its serial);
- ``z_sample`` is the redshifts of the row's stored value rows (``sqlite3`` ``mode=ro``: the
  value table's ``serial`` and ``z_serial`` only), as the redshift objects the pool holds, in
  descending ``z``, the order the factory reads them in;
- the object is constructed as the factory's miss branch constructs it (``payload=None``), with
  the stored ``z_init``, ``G_init``/``Gprime_init`` (``T_init``/``Tprime_init``), parent, label,
  the stage's tags and the run's solver map; then ``compute()`` and ``store()``, with
  ``WKB_phase_function.remote`` stood in in-process by harness.py, as 05's harness stood in
  ``QuadSource``'s.

**What a later stage would read back** is formed as the factories form it from stored rows,
since nothing is stored:
- a ``GkWKBValue`` at ``z_response``, as ``GkWKBValue.read_batch`` returns it: the re-run value
  with ``_deserialized``, ``_k_exit``, ``_z_source`` and ``_parent_serial`` (the integration's
  serial) set, and the stored value row's serial as its ``store_id``;
- the ``GkSource``: ``GkSource.compute()`` with the numeric values production's reader returns
  (``GkNumericValue.read_batch``, unchanged by prompt 06) and those WKB values, the parent set the
  existence query was keyed on, and ``assemble_GkSource_values``' own body (stood in by
  harness.py); then the read-back the factory's ``build()`` forms: the stored row's serial and
  label, the values in descending ``z_source`` with the stored value rows' serials, ``z_sample``
  the values' redshifts, the metadata through JSON;
- the ``GkSourcePolicyData``: the factory's miss branch, ``compute()`` on a ``GkSourceProxy`` of
  that read-back with ``apply_GkSource_policy``'s own body (stood in by harness.py), then the
  read-back: the stored row's serial, ``crossover_z`` and ``Levin_z`` as the floats their columns
  hold, the metadata through JSON;
- each ``TkWKBIntegration``: the factory's ``_do_not_populate`` object (every integration-table
  field, as stored), copied, with the re-run's values, ``z_sample`` and coefficients in place,
  which is what the factory's populated ``build()`` forms from the same rows.

**Records.** Every re-run object's ``deltaTheta`` is captured, exactly, by wrapping ``atan2`` at
the producer module's attribute for the one ``store()`` call (it is called once there); the phase
probe needs it (prompt 06 section 6). The re-run values are compared with the stored ones
(descriptive under ``s``; under ``t5`` this is U0's producer-faithfulness clause): every column the
stored value table has, by bit pattern.

Read-only: every ``sqlite3`` connection is ``mode=ro``; the pool is read-only.
"""

import contextlib
import copy
import importlib
import json
import math
import sqlite3
import time
from math import ulp
from unittest import mock

import ray

GKW_MOD = importlib.import_module("ComputeTargets.GkWKBIntegration")
TKW_MOD = importlib.import_module("ComputeTargets.TkWKBIntegration")
GKS_MOD = importlib.import_module("ComputeTargets.GkSource")
GPD_MOD = importlib.import_module("ComputeTargets.GkSourcePolicyData")

from ComputeTargets.GkSource import GkSource, GkSourceProxy  # noqa: E402
from ComputeTargets.GkSourcePolicyData import GkSourcePolicyData  # noqa: E402
from ComputeTargets.GkWKBIntegration import GkWKBIntegration  # noqa: E402
from ComputeTargets.TkWKBIntegration import TkWKBIntegration  # noqa: E402
from CosmologyConcepts import redshift, redshift_array  # noqa: E402
from ObjectFactories.GkSourcePolicyData import (  # noqa: E402
    _quality_deserialize,
    _type_deserialize,
)

EPS = 2.0**-52

# the stored value tables' columns compared with the re-run values' attributes (every column but
# the keys); the stored GkSourceValue columns with the assembled values' attributes
GKW_COLS = (
    "H_ratio",
    "theta_mod_2pi",
    "theta_div_2pi",
    "omega_WKB_sq",
    "WKB_criterion",
    "G_WKB",
    "analytic_G_rad",
    "analytic_Gprime_rad",
    "analytic_G_w",
    "analytic_Gprime_w",
)
TKW_COLS = (
    "H_ratio",
    "theta_mod_2pi",
    "theta_div_2pi",
    "friction",
    "omega_WKB_sq",
    "WKB_criterion",
    "T_WKB",
    "analytic_T_rad",
    "analytic_Tprime_rad",
    "analytic_T_w",
    "analytic_Tprime_w",
)


def _bits(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, int):
        return ("i", x)
    return ("f", float(x).hex())


def same_bits(a, b):
    return _bits(a) == _bits(b)


@contextlib.contextmanager
def _capture_atan2(module):
    """Wrap ``atan2`` at ``module``'s attribute; yields the list of values it returned."""
    got = []
    orig = module.atan2

    def rec(y, x):
        v = orig(y, x)
        got.append(v)
        return v

    with mock.patch.object(module, "atan2", rec):
        yield got
    if module.atan2 is not orig:
        raise RuntimeError("rerun: atan2 was not restored")


def _ro(shard_file):
    return sqlite3.connect(f"{shard_file.as_uri()}?mode=ro", uri=True)


def _rows(conn, sql, args=()):
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


class _Stats:
    """Per-class comparison of re-run values with the stored ones."""

    def __init__(self, cols, theta_col="theta_mod_2pi", amp_col=None):
        self.cols = cols
        self.amp_col = amp_col
        self.n_values = 0
        self.n_identical_all = 0
        self.differ = {c: 0 for c in cols}
        self.first = []
        self.max_mod_over_eps_theta = 0.0
        self.div_changes = 0
        self.div_changes_not_near_boundary = []
        self.max_amp_rel = 0.0

    def add(self, stored_row, value, where):
        self.n_values += 1
        all_same = True
        for c in self.cols:
            s = stored_row[c]
            v = getattr(value, c)
            if not same_bits(s, v):
                all_same = False
                self.differ[c] += 1
                if len(self.first) < 8:
                    self.first.append((where, c, _bits(s), _bits(v)))
        if all_same:
            self.n_identical_all += 1
        # the phase: |theta| from the stored cycle count, the change of the remainder against
        # eps |theta|, and cycle changes (prompt 06 section 5.4's classes, descriptive here)
        theta_abs = abs(
            stored_row["theta_div_2pi"] * 2.0 * math.pi + stored_row["theta_mod_2pi"]
        )
        d_div = int(value.theta_div_2pi) - int(stored_row["theta_div_2pi"])
        if d_div == 0:
            dm = abs(float(value.theta_mod_2pi) - float(stored_row["theta_mod_2pi"]))
            if theta_abs > 0:
                self.max_mod_over_eps_theta = max(
                    self.max_mod_over_eps_theta, dm / (EPS * theta_abs)
                )
        else:
            self.div_changes += 1
            m = float(value.theta_mod_2pi)
            near = min(abs(m), abs(m + 2.0 * math.pi)) <= 4.0 * EPS * theta_abs
            if abs(d_div) != 1 or not near:
                if len(self.div_changes_not_near_boundary) < 8:
                    self.div_changes_not_near_boundary.append(
                        (where, d_div, m, theta_abs)
                    )
        if self.amp_col is not None:
            s, v = stored_row[self.amp_col], getattr(value, self.amp_col)
            if s is not None and v is not None and s != 0:
                self.max_amp_rel = max(self.max_amp_rel, abs((v - s) / s))

    def summary(self):
        return {
            "n_values": self.n_values,
            "n_bit_identical_in_every_column": self.n_identical_all,
            "n_differing_by_column": self.differ,
            "first_differences": self.first,
            "max_abs_dmod_over_eps_theta_same_div": self.max_mod_over_eps_theta,
            "theta_div_2pi_changes": self.div_changes,
            "theta_div_2pi_changes_not_one_cycle_near_boundary": self.div_changes_not_near_boundary,
            f"max_rel_change_{self.amp_col}": self.max_amp_rel,
        }


def rerun_row(
    ctx,
    pool,
    shard_file,
    k,
    q,
    r,
    z_response,
    GkSource_parents_set,
    GkSource_serial_obj,
    GkSourcePolicy_serial_obj,
    Gk_tags,
    Tk_tags,
    missing_Tk,
    Tk_numeric_results,
    log=print,
):
    """The re-run chain for one row (module docstring). Returns a dict with the read-back
    ``GkPolicy`` and ``Tk_WKB`` objects, the read-back ``GkSource``, every re-run object's
    ``deltaTheta``, timings and the comparisons with the stored values."""
    t_all = time.perf_counter()
    model_proxy = ctx["model_proxy"]
    solvers = ctx["solvers"]
    z_by_id = ctx["redshift_by_id"]
    out = {"timing": {}}

    # the shard of each wavenumber, from the primary's shard_keys table (sqlite3 mode=ro): G's
    # rows are on k's shard, which is the QuadSourceIntegral row's; each T on its own mode's
    primary_file = shard_file.parent / "handover-A3-v2-lambdacdm.sqlite"
    pconn = _ro(primary_file)
    try:
        shard_of = dict(pconn.execute("SELECT * FROM shard_keys").fetchall())
    finally:
        pconn.close()
    if shard_file.name != f"handover-A3-v2-lambdacdm-shard{shard_of[k.k.store_id]:04d}.sqlite":
        raise RuntimeError(f"rerun: k's shard is not the row's ({shard_file.name})")

    conn = _ro(shard_file)
    try:
        # ------------------------------------------------------------------ G's WKB objects
        t0 = time.perf_counter()
        members = [m for m in GkSource_parents_set if m[2] is not None]
        wkb_serials = [m[2] for m in members]
        if len(set(wkb_serials)) != len(wkb_serials):
            raise RuntimeError("rerun: a GkWKBIntegration serial appears twice in the parent set")
        integ = {}
        for chunk_start in range(0, len(wkb_serials), 500):
            chunk = wkb_serials[chunk_start : chunk_start + 500]
            qs = ",".join("?" * len(chunk))
            for row in _rows(
                conn,
                f'SELECT serial, z_source_serial, numeric_serial, z_init, G_init, Gprime_init, '
                f'sin_coeff, cos_coeff, z_samples, wavenumber_exit_serial, model_serial '
                f'FROM "GkWKBIntegration" WHERE serial IN ({qs})',
                chunk,
            ):
                integ[row["serial"]] = row
        values_by_wkb = {}
        for chunk_start in range(0, len(wkb_serials), 500):
            chunk = wkb_serials[chunk_start : chunk_start + 500]
            qs = ",".join("?" * len(chunk))
            for row in _rows(
                conn,
                f'SELECT serial, wkb_serial, z_serial, {", ".join(GKW_COLS)} '
                f'FROM "GkWKBValue" WHERE wkb_serial IN ({qs})',
                chunk,
            ):
                values_by_wkb.setdefault(row["wkb_serial"], []).append(row)
        out["timing"]["G_sql_read"] = time.perf_counter() - t0

        G_stats = _Stats(GKW_COLS, amp_col="G_WKB")
        coeff_mismatch = []
        WKB_data = {}
        deltaTheta_by_z_source = {}
        n_samples_rerun = 0
        t_lookup = 0.0
        t_compute = 0.0
        for z_source_serial, numeric_member, wkb_serial in members:
            row = integ.get(wkb_serial)
            if row is None:
                raise RuntimeError(f"rerun: no GkWKBIntegration row {wkb_serial}")
            if row["z_source_serial"] != z_source_serial or row["wavenumber_exit_serial"] != k.store_id or row["model_serial"] != model_proxy.store_id:
                raise RuntimeError(f"rerun: GkWKBIntegration {wkb_serial} is not this member's: {row}")
            z_source = z_by_id[z_source_serial]
            tl = time.perf_counter()
            stored = ray.get(
                pool.object_get(
                    "GkWKBIntegration",
                    solver_labels=solvers,
                    model=model_proxy,
                    k=k,
                    z_source=z_source,
                    numeric_serial=(
                        GkWKBIntegration.PURE_WKB
                        if row["numeric_serial"] is None
                        else row["numeric_serial"]
                    ),
                    z_sample=None,
                    tags=Gk_tags,
                    _do_not_populate=True,
                )
            )
            t_lookup += time.perf_counter() - tl
            if not stored.available or stored.store_id != wkb_serial:
                raise RuntimeError(
                    f"rerun: the factory's GkWKBIntegration for z_source serial {z_source_serial} "
                    f"is {stored.store_id if stored.available else None}, not {wkb_serial}"
                )
            stored_values = sorted(
                values_by_wkb[wkb_serial], key=lambda v: z_by_id[v["z_serial"]].z, reverse=True
            )
            if row["z_samples"] is not None and len(stored_values) != row["z_samples"]:
                raise RuntimeError(f"rerun: GkWKBIntegration {wkb_serial} value count")
            z_sample = redshift_array([z_by_id[v["z_serial"]] for v in stored_values])
            work = GkWKBIntegration(
                payload=None,
                solver_labels=solvers,
                model=model_proxy,
                k=k,
                z_source=z_source,
                numeric_serial=stored.numeric_serial,
                z_init=stored._z_init,
                G_init=stored.G_init,
                Gprime_init=stored.Gprime_init,
                z_sample=z_sample,
                label=stored.label,
                tags=Gk_tags,
            )
            tc = time.perf_counter()
            work.compute()
            with _capture_atan2(GKW_MOD) as got:
                if work.store() is not True:
                    raise RuntimeError("rerun: GkWKBIntegration.store() did not resolve")
            t_compute += time.perf_counter() - tc
            if len(got) != 1:
                raise RuntimeError(f"rerun: atan2 called {len(got)} times in one store()")
            n_samples_rerun += len(work.values)
            if not (same_bits(work.sin_coeff, row["sin_coeff"]) and same_bits(work.cos_coeff, row["cos_coeff"])):
                coeff_mismatch.append((wkb_serial, _bits(work.sin_coeff), _bits(row["sin_coeff"])))
            the_value = None
            for sv, v in zip(stored_values, work.values):
                if v.z.store_id != sv["z_serial"]:
                    raise RuntimeError("rerun: value order differs from z_sample")
                G_stats.add(sv, v, (wkb_serial, sv["z_serial"]))
                if sv["z_serial"] == z_response.store_id:
                    the_value = (sv, v)
            if the_value is None:
                raise RuntimeError(
                    f"rerun: GkWKBIntegration {wkb_serial} has no value at z_response serial "
                    f"{z_response.store_id}"
                )
            sv, v = the_value
            # as GkWKBValue.read_batch returns it (module docstring)
            v._my_id = sv["serial"]
            v._deserialized = True
            v._k_exit = k
            v._z_source = z_source
            v._parent_serial = wkb_serial
            WKB_data[z_source_serial] = v
            deltaTheta_by_z_source[z_source_serial] = {
                "deltaTheta": got[0],
                "z_init": stored._z_init,
                "pure_WKB": stored.numeric_serial is None,
                "wkb_serial": wkb_serial,
            }
        out["timing"]["G_factory_lookups"] = t_lookup
        out["timing"]["G_compute_store"] = t_compute
        out["G_objects_rerun"] = len(members)
        out["G_samples_rerun"] = n_samples_rerun
        out["G_coeff_mismatches"] = coeff_mismatch[:8]
        out["G_coeff_mismatch_count"] = len(coeff_mismatch)
        out["G_values_vs_stored"] = G_stats.summary()
        out["deltaTheta_by_z_source"] = deltaTheta_by_z_source

        # ------------------------------------------------------------------ G's numeric values
        t0 = time.perf_counter()
        GkNs = ray.get(
            pool.object_read_batch(
                "GkNumericValue",
                {"k": k},
                model=model_proxy,
                z=z_response,
                atol=ctx["Gk_numeric_atol"],
                rtol=ctx["Gk_numeric_rtol"],
                tags=Gk_tags,
            )
        )
        numeric_data = {}
        for GkN in GkNs:
            assert GkN._k_exit.store_id == k.store_id
            assert GkN.z.store_id == z_response.store_id
            numeric_data[GkN._z_source.store_id] = GkN
        out["timing"]["G_numeric_read"] = time.perf_counter() - t0

        # ------------------------------------------------------------------ the GkSource
        t0 = time.perf_counter()
        z_source_pool = ctx["z_source_sample"].truncate(z_response, keep="higher")
        work = GkSource(
            payload=None,
            model=model_proxy,
            k=k,
            z_response=z_response,
            z_sample=z_source_pool,
            tags=Gk_tags,
            parents=GkSource_parents_set,
        )
        work.compute(payload={"numeric": numeric_data, "WKB": WKB_data})
        work.store()
        if work.values is None:
            raise RuntimeError("rerun: GkSource.store() did not resolve")
        if work.assembled_parents != GkSource_parents_set:
            raise RuntimeError("rerun: the assembled parents are not the keyed set")
        stored_gsv = {
            row["z_source_serial"]: row
            for row in _rows(
                conn,
                'SELECT * FROM "GkSourceValue" WHERE parent_serial = ?',
                (GkSource_serial_obj.store_id,),
            )
        }
        assembled = sorted(work.values, key=lambda v: v.z_source.z, reverse=True)
        for v in assembled:
            v._my_id = stored_gsv[v.z_source.store_id]["serial"]
        gs_meta = json.loads(json.dumps(work.metadata)) if work.metadata is not None else None
        GkSource_rb = GkSource(
            payload={
                "store_id": GkSource_serial_obj.store_id,
                "values": assembled,
                "numeric_smallest_z": work.numeric_smallest_z,
                "primary_WKB_largest_z": work.primary_WKB_largest_z,
                "metadata": gs_meta,
            },
            k=k,
            model=model_proxy,
            label=GkSource_serial_obj.label,
            z_response=redshift(store_id=z_response.store_id, z=z_response.z),
            z_sample=redshift_array([v.z_source for v in assembled]),
            tags=Gk_tags,
            parents=GkSource_parents_set,
        )
        GkSource_rb._deserialized = True
        # the assembled values against the stored GkSourceValue rows, every stored column but the
        # keys (theta_residual is in no stored row)
        gs_cols = [
            c
            for c in next(iter(stored_gsv.values())).keys()
            if c not in ("serial", "parent_serial", "z_source_serial")
        ]
        gs_differ = {c: 0 for c in gs_cols}
        gs_first = []
        gs_identical = 0
        for v in assembled:
            sr = stored_gsv[v.z_source.store_id]
            ok = True
            for c in gs_cols:
                if c in ("G", "Gprime"):
                    a = v.numeric.G if c == "G" else v.numeric.Gprime
                elif c in ("omega_WKB_sq", "WKB_criterion", "analytic_G_rad", "analytic_Gprime_rad"):
                    a = getattr(v, c)
                else:
                    a = getattr(v.WKB, c)
                if not same_bits(a, sr[c]):
                    ok = False
                    gs_differ[c] += 1
                    if len(gs_first) < 8:
                        gs_first.append((v.z_source.store_id, c, _bits(sr[c]), _bits(a)))
            gs_identical += ok
        out["GkSource_vs_stored"] = {
            "n_values": len(assembled),
            "n_stored": len(stored_gsv),
            "n_bit_identical_in_every_column": gs_identical,
            "n_differing_by_column": gs_differ,
            "first_differences": gs_first,
            "numeric_smallest_z": None
            if work.numeric_smallest_z is None
            else work.numeric_smallest_z.store_id,
            "primary_WKB_largest_z": None
            if work.primary_WKB_largest_z is None
            else work.primary_WKB_largest_z.store_id,
        }
        out["timing"]["GkSource"] = time.perf_counter() - t0

        # ------------------------------------------------------------------ GkSourcePolicyData
        t0 = time.perf_counter()
        pol_work = GkSourcePolicyData(
            payload=None,
            source=GkSourceProxy(GkSource_rb),
            policy=ctx["GkSource_policy_1pt5"],
            k=k,
            label=None,
        )
        pol_work.compute()
        pol_work.store()
        stored_pol = _rows(
            conn,
            'SELECT serial, type, quality, crossover_z, Levin_z, metadata FROM "GkSourcePolicyData" WHERE serial = ?',
            (GkSourcePolicy_serial_obj.store_id,),
        )[0]
        GkPolicy_rb = GkSourcePolicyData(
            source=GkSourceProxy(GkSource_rb),
            policy=ctx["GkSource_policy_1pt5"],
            k=k,
            payload={
                "store_id": GkSourcePolicy_serial_obj.store_id,
                "type": pol_work.type,
                "quality": pol_work.quality,
                "crossover_z": None
                if pol_work.crossover_z is None
                else float(pol_work.crossover_z),
                "Levin_z": None if pol_work.Levin_z is None else float(pol_work.Levin_z),
                "metadata": json.loads(json.dumps(pol_work.metadata))
                if pol_work.metadata is not None
                else {},
            },
        )
        GkPolicy_rb._deserialized = True
        out["GkSourcePolicyData"] = {
            "rerun": {
                "type": pol_work.type,
                "quality": pol_work.quality,
                "crossover_z": None
                if pol_work.crossover_z is None
                else float(pol_work.crossover_z).hex(),
                "Levin_z": None if pol_work.Levin_z is None else float(pol_work.Levin_z).hex(),
                "Levin_z_store_id": getattr(pol_work.Levin_z, "store_id", None),
                "metadata": json.loads(json.dumps(pol_work.metadata)),
            },
            "stored": {
                "serial": stored_pol["serial"],
                # as the factory's build() deserialises them
                "type": None
                if stored_pol["type"] is None
                else _type_deserialize[stored_pol["type"]],
                "quality": None
                if stored_pol["quality"] is None
                else _quality_deserialize[stored_pol["quality"]],
                "crossover_z": None
                if stored_pol["crossover_z"] is None
                else float(stored_pol["crossover_z"]).hex(),
                "Levin_z": None
                if stored_pol["Levin_z"] is None
                else float(stored_pol["Levin_z"]).hex(),
                "metadata": json.loads(stored_pol["metadata"])
                if stored_pol["metadata"] is not None
                else None,
            },
        }
        out["timing"]["GkSourcePolicyData"] = time.perf_counter() - t0

        # ------------------------------------------------------------------ T's WKB objects
        # each mode's TkWKBIntegration is held on the shard of its own wavenumber, which the
        # primary's shard_keys table names (sqlite3 mode=ro)
        t0 = time.perf_counter()
        Tk_WKB_rb = {}
        T_stats = _Stats(TKW_COLS, amp_col="T_WKB")
        T_info = {}
        for k_exit, Tk_numeric in zip(missing_Tk, Tk_numeric_results):
            stored = ray.get(
                pool.object_get(
                    "TkWKBIntegration",
                    solver_labels=[],
                    model=model_proxy,
                    z_sample=None,
                    k=k_exit,
                    numeric_serial=Tk_numeric.store_id,
                    tags=Tk_tags,
                    _do_not_populate=True,
                )
            )
            if not stored.available:
                raise RuntimeError(f"rerun: no TkWKBIntegration for k serial {k_exit.store_id}")
            tconn = _ro(
                shard_file.parent
                / f"handover-A3-v2-lambdacdm-shard{shard_of[k_exit.k.store_id]:04d}.sqlite"
            )
            srow = _rows(
                tconn,
                'SELECT serial, numeric_serial, z_init, T_init, Tprime_init, sin_coeff, cos_coeff, z_samples FROM "TkWKBIntegration" WHERE serial = ?',
                (stored.store_id,),
            )[0]
            stored_values = sorted(
                _rows(
                    tconn,
                    f'SELECT serial, wkb_serial, z_serial, {", ".join(TKW_COLS)} FROM "TkWKBValue" WHERE wkb_serial = ?',
                    (stored.store_id,),
                ),
                key=lambda v: z_by_id[v["z_serial"]].z,
                reverse=True,
            )
            tconn.close()
            if len(stored_values) != srow["z_samples"]:
                raise RuntimeError("rerun: TkWKBIntegration value count")
            z_sample = redshift_array([z_by_id[v["z_serial"]] for v in stored_values])
            work = TkWKBIntegration(
                payload=None,
                solver_labels=solvers,
                model=model_proxy,
                k=k_exit,
                numeric_serial=stored.numeric_serial,
                z_init=stored.z_init,
                T_init=stored._T_init,
                Tprime_init=stored._Tprime_init,
                z_sample=z_sample,
                label=stored.label,
                tags=Tk_tags,
            )
            work.compute()
            with _capture_atan2(TKW_MOD) as got:
                if work.store() is not True:
                    raise RuntimeError("rerun: TkWKBIntegration.store() did not resolve")
            if len(got) != 1:
                raise RuntimeError(f"rerun: atan2 called {len(got)} times in one Tk store()")
            for sv, v in zip(stored_values, work.values):
                if v.z.store_id != sv["z_serial"]:
                    raise RuntimeError("rerun: Tk value order differs from z_sample")
                T_stats.add(sv, v, (stored.store_id, sv["z_serial"]))
                v._my_id = sv["serial"]
            rb = copy.copy(stored)
            del rb._do_not_populate
            rb._values = work.values
            rb._z_sample = z_sample
            rb._sin_coeff = work.sin_coeff
            rb._cos_coeff = work.cos_coeff
            Tk_WKB_rb[k_exit.store_id] = rb
            T_info[k_exit.store_id] = {
                "serial": stored.store_id,
                "deltaTheta": got[0],
                "z_init": stored.z_init,
                "n_values": len(work.values),
                "coeffs_bit_identical_to_stored": same_bits(work.sin_coeff, srow["sin_coeff"])
                and same_bits(work.cos_coeff, srow["cos_coeff"]),
            }
        out["T_objects"] = T_info
        out["T_values_vs_stored"] = T_stats.summary()
        out["timing"]["T"] = time.perf_counter() - t0
    finally:
        conn.close()

    out["timing"]["total"] = time.perf_counter() - t_all
    out["GkPolicy"] = GkPolicy_rb
    out["GkSource"] = GkSource_rb
    out["Tk_WKB"] = Tk_WKB_rb
    log(
        f"   rerun: {out['G_objects_rerun']} GkWKBIntegration ({out['G_samples_rerun']} samples), "
        f"{len(Tk_WKB_rb)} TkWKBIntegration, GkSource ({len(assembled)} values), "
        f"GkSourcePolicyData; {out['timing']['total']:.1f} s"
    )
    return out


# =============================================================================================
# prompt 06a section 0.3 change 1 and section 2 item 3: the producer cache. Each producer is
# re-run once, not once per row. The bodies below are rerun_row's, statement for statement, split
# at the points where rerun_row's work depends on less than the whole row:
# - rerun_k_G: rerun_row's per-member loop over G's WKB objects ("G's WKB objects"), run once on
#   every GkWKBIntegration of one k, keeping the values at the response redshifts;
# - assemble_Gk: rerun_row's "G's numeric values", "the GkSource" and "GkSourcePolicyData"
#   sections, once per (k, z_response), with the WKB values taken from rerun_k_G's cache;
# - rerun_T_mode: rerun_row's "T's WKB objects" loop body, once per mode.
# rerun_row itself is unchanged and unused by the census.
# =============================================================================================


def _shard_of(store_dir):
    pconn = _ro(store_dir / "handover-A3-v2-lambdacdm.sqlite")
    try:
        return dict(pconn.execute("SELECT * FROM shard_keys").fetchall())
    finally:
        pconn.close()


def _shard_file(store_dir, shard_of, wavenumber_serial):
    return store_dir / f"handover-A3-v2-lambdacdm-shard{shard_of[wavenumber_serial]:04d}.sqlite"


def rerun_k_G(ctx, pool, store_dir, k, Gk_tags, keep_z_serials, log=print, chunk=200):
    """prompt 06a: every GkWKBIntegration of k (the rows of k's shard with k's exit-time serial
    and the model's serial) re-run once, by rerun_row's per-member code. Returns
    ``(values, summary)``: ``values[(wkb_serial, z_serial)] = (stored value-row serial, re-run
    GkWKBValue)`` for every z_serial in ``keep_z_serials``, and the comparison of every re-run
    value with the stored one (06's W9, summarised for the k)."""
    t_all = time.perf_counter()
    model_proxy = ctx["model_proxy"]
    solvers = ctx["solvers"]
    z_by_id = ctx["redshift_by_id"]
    shard_of = _shard_of(store_dir)
    shard_file = _shard_file(store_dir, shard_of, k.k.store_id)
    keep_z_serials = set(keep_z_serials)
    summary = {"timing": {}, "shard_file": shard_file.name}
    conn = _ro(shard_file)
    try:
        wkb_serials = [
            s
            for (s,) in conn.execute(
                'SELECT serial FROM "GkWKBIntegration" WHERE wavenumber_exit_serial = ? '
                "AND model_serial = ? ORDER BY serial",
                (k.store_id, model_proxy.store_id),
            )
        ]
        G_stats = _Stats(GKW_COLS, amp_col="G_WKB")
        coeff_mismatch = []
        values = {}
        n_samples_rerun = 0
        t_sql = 0.0
        t_lookup = 0.0
        t_compute = 0.0
        n_pure = 0
        for chunk_start in range(0, len(wkb_serials), chunk):
            ch = wkb_serials[chunk_start : chunk_start + chunk]
            qs = ",".join("?" * len(ch))
            t0 = time.perf_counter()
            integ = {}
            for row in _rows(
                conn,
                f'SELECT serial, z_source_serial, numeric_serial, z_init, G_init, Gprime_init, '
                f'sin_coeff, cos_coeff, z_samples, wavenumber_exit_serial, model_serial '
                f'FROM "GkWKBIntegration" WHERE serial IN ({qs})',
                ch,
            ):
                integ[row["serial"]] = row
            values_by_wkb = {}
            for row in _rows(
                conn,
                f'SELECT serial, wkb_serial, z_serial, {", ".join(GKW_COLS)} '
                f'FROM "GkWKBValue" WHERE wkb_serial IN ({qs})',
                ch,
            ):
                values_by_wkb.setdefault(row["wkb_serial"], []).append(row)
            t_sql += time.perf_counter() - t0
            for wkb_serial in ch:
                # rerun_row's per-member body, with the member's z_source_serial read from the row
                row = integ.get(wkb_serial)
                if row is None:
                    raise RuntimeError(f"rerun: no GkWKBIntegration row {wkb_serial}")
                z_source_serial = row["z_source_serial"]
                if row["wavenumber_exit_serial"] != k.store_id or row["model_serial"] != model_proxy.store_id:
                    raise RuntimeError(f"rerun: GkWKBIntegration {wkb_serial} is not this k's: {row}")
                z_source = z_by_id[z_source_serial]
                tl = time.perf_counter()
                stored = ray.get(
                    pool.object_get(
                        "GkWKBIntegration",
                        solver_labels=solvers,
                        model=model_proxy,
                        k=k,
                        z_source=z_source,
                        numeric_serial=(
                            GkWKBIntegration.PURE_WKB
                            if row["numeric_serial"] is None
                            else row["numeric_serial"]
                        ),
                        z_sample=None,
                        tags=Gk_tags,
                        _do_not_populate=True,
                    )
                )
                t_lookup += time.perf_counter() - tl
                if not stored.available or stored.store_id != wkb_serial:
                    raise RuntimeError(
                        f"rerun: the factory's GkWKBIntegration for z_source serial {z_source_serial} "
                        f"is {stored.store_id if stored.available else None}, not {wkb_serial}"
                    )
                stored_values = sorted(
                    values_by_wkb[wkb_serial], key=lambda v: z_by_id[v["z_serial"]].z, reverse=True
                )
                if row["z_samples"] is not None and len(stored_values) != row["z_samples"]:
                    raise RuntimeError(f"rerun: GkWKBIntegration {wkb_serial} value count")
                z_sample = redshift_array([z_by_id[v["z_serial"]] for v in stored_values])
                work = GkWKBIntegration(
                    payload=None,
                    solver_labels=solvers,
                    model=model_proxy,
                    k=k,
                    z_source=z_source,
                    numeric_serial=stored.numeric_serial,
                    z_init=stored._z_init,
                    G_init=stored.G_init,
                    Gprime_init=stored.Gprime_init,
                    z_sample=z_sample,
                    label=stored.label,
                    tags=Gk_tags,
                )
                tc = time.perf_counter()
                work.compute()
                with _capture_atan2(GKW_MOD) as got:
                    if work.store() is not True:
                        raise RuntimeError("rerun: GkWKBIntegration.store() did not resolve")
                t_compute += time.perf_counter() - tc
                if len(got) != 1:
                    raise RuntimeError(f"rerun: atan2 called {len(got)} times in one store()")
                n_samples_rerun += len(work.values)
                n_pure += stored.numeric_serial is None
                if not (same_bits(work.sin_coeff, row["sin_coeff"]) and same_bits(work.cos_coeff, row["cos_coeff"])):
                    coeff_mismatch.append((wkb_serial, _bits(work.sin_coeff), _bits(row["sin_coeff"])))
                for sv, v in zip(stored_values, work.values):
                    if v.z.store_id != sv["z_serial"]:
                        raise RuntimeError("rerun: value order differs from z_sample")
                    G_stats.add(sv, v, (wkb_serial, sv["z_serial"]))
                    if sv["z_serial"] in keep_z_serials:
                        values[(wkb_serial, sv["z_serial"])] = (sv["serial"], z_source_serial, v)
                del work, stored_values
            del integ, values_by_wkb
    finally:
        conn.close()
    summary["timing"]["G_sql_read"] = t_sql
    summary["timing"]["G_factory_lookups"] = t_lookup
    summary["timing"]["G_compute_store"] = t_compute
    summary["G_objects_rerun"] = len(wkb_serials)
    summary["G_objects_pure_WKB"] = n_pure
    summary["G_samples_rerun"] = n_samples_rerun
    summary["G_values_kept"] = len(values)
    summary["G_coeff_mismatches"] = coeff_mismatch[:8]
    summary["G_coeff_mismatch_count"] = len(coeff_mismatch)
    summary["G_values_vs_stored"] = G_stats.summary()
    summary["timing"]["total"] = time.perf_counter() - t_all
    log(
        f"   rerun_k_G: k serial {k.store_id}: {len(wkb_serials)} GkWKBIntegration "
        f"({n_samples_rerun} samples, {len(values)} kept); {summary['timing']['total']:.1f} s"
    )
    return values, summary


def assemble_Gk(
    ctx,
    pool,
    store_dir,
    k,
    z_response,
    GkSource_parents_set,
    GkSource_serial_obj,
    GkSourcePolicy_serial_obj,
    Gk_tags,
    G_values,
):
    """prompt 06a: rerun_row's GkSource and GkSourcePolicyData sections for one (k, z_response),
    with the WKB values from rerun_k_G's cache (``G_values``). Returns a dict with the read-back
    ``GkPolicy`` and ``GkSource`` and the comparisons with the stored rows."""
    t_all = time.perf_counter()
    model_proxy = ctx["model_proxy"]
    z_by_id = ctx["redshift_by_id"]
    out = {"timing": {}}
    shard_of = _shard_of(store_dir)
    conn = _ro(_shard_file(store_dir, shard_of, k.k.store_id))
    try:
        # ------------------------------------------------------------------ G's WKB values
        members = [m for m in GkSource_parents_set if m[2] is not None]
        WKB_data = {}
        for z_source_serial, numeric_member, wkb_serial in members:
            got = G_values.get((wkb_serial, z_response.store_id))
            if got is None:
                raise RuntimeError(
                    f"rerun: GkWKBIntegration {wkb_serial} has no cached value at z_response "
                    f"serial {z_response.store_id}"
                )
            sv_serial, cached_z_source_serial, v = got
            if cached_z_source_serial != z_source_serial:
                raise RuntimeError(f"rerun: GkWKBIntegration {wkb_serial} is not this member's")
            z_source = z_by_id[z_source_serial]
            # as GkWKBValue.read_batch returns it (rerun_row; module docstring)
            v._my_id = sv_serial
            v._deserialized = True
            v._k_exit = k
            v._z_source = z_source
            v._parent_serial = wkb_serial
            WKB_data[z_source_serial] = v
        out["G_members_with_WKB"] = len(members)

        # ------------------------------------------------------------------ G's numeric values
        t0 = time.perf_counter()
        GkNs = ray.get(
            pool.object_read_batch(
                "GkNumericValue",
                {"k": k},
                model=model_proxy,
                z=z_response,
                atol=ctx["Gk_numeric_atol"],
                rtol=ctx["Gk_numeric_rtol"],
                tags=Gk_tags,
            )
        )
        numeric_data = {}
        for GkN in GkNs:
            assert GkN._k_exit.store_id == k.store_id
            assert GkN.z.store_id == z_response.store_id
            numeric_data[GkN._z_source.store_id] = GkN
        out["timing"]["G_numeric_read"] = time.perf_counter() - t0

        # ------------------------------------------------------------------ the GkSource
        t0 = time.perf_counter()
        z_source_pool = ctx["z_source_sample"].truncate(z_response, keep="higher")
        work = GkSource(
            payload=None,
            model=model_proxy,
            k=k,
            z_response=z_response,
            z_sample=z_source_pool,
            tags=Gk_tags,
            parents=GkSource_parents_set,
        )
        work.compute(payload={"numeric": numeric_data, "WKB": WKB_data})
        work.store()
        if work.values is None:
            raise RuntimeError("rerun: GkSource.store() did not resolve")
        if work.assembled_parents != GkSource_parents_set:
            raise RuntimeError("rerun: the assembled parents are not the keyed set")
        stored_gsv = {
            row["z_source_serial"]: row
            for row in _rows(
                conn,
                'SELECT * FROM "GkSourceValue" WHERE parent_serial = ?',
                (GkSource_serial_obj.store_id,),
            )
        }
        assembled = sorted(work.values, key=lambda v: v.z_source.z, reverse=True)
        for v in assembled:
            v._my_id = stored_gsv[v.z_source.store_id]["serial"]
        gs_meta = json.loads(json.dumps(work.metadata)) if work.metadata is not None else None
        GkSource_rb = GkSource(
            payload={
                "store_id": GkSource_serial_obj.store_id,
                "values": assembled,
                "numeric_smallest_z": work.numeric_smallest_z,
                "primary_WKB_largest_z": work.primary_WKB_largest_z,
                "metadata": gs_meta,
            },
            k=k,
            model=model_proxy,
            label=GkSource_serial_obj.label,
            z_response=redshift(store_id=z_response.store_id, z=z_response.z),
            z_sample=redshift_array([v.z_source for v in assembled]),
            tags=Gk_tags,
            parents=GkSource_parents_set,
        )
        GkSource_rb._deserialized = True
        gs_cols = [
            c
            for c in next(iter(stored_gsv.values())).keys()
            if c not in ("serial", "parent_serial", "z_source_serial")
        ]
        gs_differ = {c: 0 for c in gs_cols}
        gs_first = []
        gs_identical = 0
        for v in assembled:
            sr = stored_gsv[v.z_source.store_id]
            ok = True
            for c in gs_cols:
                if c in ("G", "Gprime"):
                    a = v.numeric.G if c == "G" else v.numeric.Gprime
                elif c in ("omega_WKB_sq", "WKB_criterion", "analytic_G_rad", "analytic_Gprime_rad"):
                    a = getattr(v, c)
                else:
                    a = getattr(v.WKB, c)
                if not same_bits(a, sr[c]):
                    ok = False
                    gs_differ[c] += 1
                    if len(gs_first) < 8:
                        gs_first.append((v.z_source.store_id, c, _bits(sr[c]), _bits(a)))
            gs_identical += ok
        out["GkSource_vs_stored"] = {
            "n_values": len(assembled),
            "n_stored": len(stored_gsv),
            "n_bit_identical_in_every_column": gs_identical,
            "n_differing_by_column": gs_differ,
            "first_differences": gs_first,
            "numeric_smallest_z": None
            if work.numeric_smallest_z is None
            else work.numeric_smallest_z.store_id,
            "primary_WKB_largest_z": None
            if work.primary_WKB_largest_z is None
            else work.primary_WKB_largest_z.store_id,
        }
        out["timing"]["GkSource"] = time.perf_counter() - t0

        # ------------------------------------------------------------------ GkSourcePolicyData
        t0 = time.perf_counter()
        pol_work = GkSourcePolicyData(
            payload=None,
            source=GkSourceProxy(GkSource_rb),
            policy=ctx["GkSource_policy_1pt5"],
            k=k,
            label=None,
        )
        pol_work.compute()
        pol_work.store()
        stored_pol = _rows(
            conn,
            'SELECT serial, type, quality, crossover_z, Levin_z, metadata FROM "GkSourcePolicyData" WHERE serial = ?',
            (GkSourcePolicy_serial_obj.store_id,),
        )[0]
        GkPolicy_rb = GkSourcePolicyData(
            source=GkSourceProxy(GkSource_rb),
            policy=ctx["GkSource_policy_1pt5"],
            k=k,
            payload={
                "store_id": GkSourcePolicy_serial_obj.store_id,
                "type": pol_work.type,
                "quality": pol_work.quality,
                "crossover_z": None
                if pol_work.crossover_z is None
                else float(pol_work.crossover_z),
                "Levin_z": None if pol_work.Levin_z is None else float(pol_work.Levin_z),
                "metadata": json.loads(json.dumps(pol_work.metadata))
                if pol_work.metadata is not None
                else {},
            },
        )
        GkPolicy_rb._deserialized = True
        out["GkSourcePolicyData"] = {
            "rerun": {
                "type": pol_work.type,
                "quality": pol_work.quality,
                "crossover_z": None
                if pol_work.crossover_z is None
                else float(pol_work.crossover_z).hex(),
                "Levin_z": None if pol_work.Levin_z is None else float(pol_work.Levin_z).hex(),
                "Levin_z_store_id": getattr(pol_work.Levin_z, "store_id", None),
            },
            "stored": {
                "serial": stored_pol["serial"],
                "type": None
                if stored_pol["type"] is None
                else _type_deserialize[stored_pol["type"]],
                "quality": None
                if stored_pol["quality"] is None
                else _quality_deserialize[stored_pol["quality"]],
                "crossover_z": None
                if stored_pol["crossover_z"] is None
                else float(stored_pol["crossover_z"]).hex(),
                "Levin_z": None
                if stored_pol["Levin_z"] is None
                else float(stored_pol["Levin_z"]).hex(),
            },
        }
        out["timing"]["GkSourcePolicyData"] = time.perf_counter() - t0
    finally:
        conn.close()
    out["timing"]["total"] = time.perf_counter() - t_all
    out["GkPolicy"] = GkPolicy_rb
    out["GkSource"] = GkSource_rb
    return out


def rerun_T_mode(ctx, pool, store_dir, k_exit, Tk_numeric, Tk_tags):
    """prompt 06a: rerun_row's "T's WKB objects" loop body for one mode, once. Returns
    ``(read_back, info, T_stats_summary)``."""
    t0 = time.perf_counter()
    model_proxy = ctx["model_proxy"]
    solvers = ctx["solvers"]
    z_by_id = ctx["redshift_by_id"]
    shard_of = _shard_of(store_dir)
    T_stats = _Stats(TKW_COLS, amp_col="T_WKB")
    stored = ray.get(
        pool.object_get(
            "TkWKBIntegration",
            solver_labels=[],
            model=model_proxy,
            z_sample=None,
            k=k_exit,
            numeric_serial=Tk_numeric.store_id,
            tags=Tk_tags,
            _do_not_populate=True,
        )
    )
    if not stored.available:
        raise RuntimeError(f"rerun: no TkWKBIntegration for k serial {k_exit.store_id}")
    tconn = _ro(_shard_file(store_dir, shard_of, k_exit.k.store_id))
    srow = _rows(
        tconn,
        'SELECT serial, numeric_serial, z_init, T_init, Tprime_init, sin_coeff, cos_coeff, z_samples FROM "TkWKBIntegration" WHERE serial = ?',
        (stored.store_id,),
    )[0]
    stored_values = sorted(
        _rows(
            tconn,
            f'SELECT serial, wkb_serial, z_serial, {", ".join(TKW_COLS)} FROM "TkWKBValue" WHERE wkb_serial = ?',
            (stored.store_id,),
        ),
        key=lambda v: z_by_id[v["z_serial"]].z,
        reverse=True,
    )
    tconn.close()
    if len(stored_values) != srow["z_samples"]:
        raise RuntimeError("rerun: TkWKBIntegration value count")
    z_sample = redshift_array([z_by_id[v["z_serial"]] for v in stored_values])
    work = TkWKBIntegration(
        payload=None,
        solver_labels=solvers,
        model=model_proxy,
        k=k_exit,
        numeric_serial=stored.numeric_serial,
        z_init=stored.z_init,
        T_init=stored._T_init,
        Tprime_init=stored._Tprime_init,
        z_sample=z_sample,
        label=stored.label,
        tags=Tk_tags,
    )
    work.compute()
    with _capture_atan2(TKW_MOD) as got:
        if work.store() is not True:
            raise RuntimeError("rerun: TkWKBIntegration.store() did not resolve")
    if len(got) != 1:
        raise RuntimeError(f"rerun: atan2 called {len(got)} times in one Tk store()")
    for sv, v in zip(stored_values, work.values):
        if v.z.store_id != sv["z_serial"]:
            raise RuntimeError("rerun: Tk value order differs from z_sample")
        T_stats.add(sv, v, (stored.store_id, sv["z_serial"]))
        v._my_id = sv["serial"]
    rb = copy.copy(stored)
    del rb._do_not_populate
    rb._values = work.values
    rb._z_sample = z_sample
    rb._sin_coeff = work.sin_coeff
    rb._cos_coeff = work.cos_coeff
    info = {
        "serial": stored.store_id,
        "z_init": stored.z_init,
        "n_values": len(work.values),
        "coeffs_bit_identical_to_stored": same_bits(work.sin_coeff, srow["sin_coeff"])
        and same_bits(work.cos_coeff, srow["cos_coeff"]),
        "seconds": time.perf_counter() - t0,
    }
    return rb, info, T_stats.summary()
