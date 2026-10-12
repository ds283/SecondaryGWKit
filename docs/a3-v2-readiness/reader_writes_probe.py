"""
R2 (a3-v2-readiness audit 00): every write one reader's lookup sequence makes, counted on a stand-in.

No Ray, nothing under ``var/``. A real ``ShardedPool``, the real ``Datastore`` actor code, every
factory and the real broker run on stand-in shards (``Datastore/tests/standin_pool.py``) in a
temporary directory, which is deleted at the end.

**The store.** ``build_full_store`` writes, through the pool, every row the reader
``extract_QuadSourceIntegral_data.py`` asks for in its setup and in one work item: the run's
``store_tag``, both cosmologies (through the real ``config.model_list.build_model_list``, as
``quadsource_atol_sweep.py --build``'s ``_filtered_build`` does before filtering), every tolerance
the six readers key on, three wavenumbers, their LambdaCDM exit times, a validated LambdaCDM
``BackgroundModel`` tagged with the run, the two ``GkSourcePolicy`` rows, the 1.5
``QuadSourcePolicy`` and one ``GkSourcePolicyData`` (sharded). It holds no ``QCD_Cosmology``
``BackgroundModel``, as A3 v2's store will not.

**The sequence.** ``qsi_sequence`` is that script's ``pool.`` calls, in its order, transcribed from
``extract_QuadSourceIntegral_data.py`` at ``HEAD`` (the script cannot be imported: it parses argv
and calls ``ray.init`` at module scope). ``resolve_run_selection`` and ``build_model_list`` are
imported and run as they are. Where the script draws a random subsample, the probe takes the first
element. ``ModelProxy`` and ``GkSourceProxy`` call ``ray.put``; the probe passes ``Proxy``, which
holds the object and answers ``store_id``/``get()`` the same way. The loop over models stops where
the script stops: at the ``RuntimeError`` it raises for a model with no ``BackgroundModel``.

**The instruments**, each shown counting a known write first (run 0):

- the SQLite *file change counter* (header bytes 24-27, big-endian) of every file: SQLite
  increments it once for every transaction that changes the file, in rollback-journal mode, which
  every store here uses. A transaction that only reads leaves it alone;
- the row count of every table in every file, read ``mode=ro``;
- every call of ``ShardedPool._replicated_write``, wrapped: its operation and class, and whether the
  controlling shard's answer was inserted or updated (``_new_insert`` / ``_updated``);
- ``standin_pool.in_flight_records(pool)`` after each run.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/reader_writes_probe.py \
        > docs/a3-v2-readiness/reader_writes_probe.txt 2>&1
"""

import contextlib
import hashlib
import io
import math
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import ray

from ObjectFactories.tests import standin_pool as sp
from config.defaults import (
    DEFAULT_GK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_GK_NUMERIC_REL_TOLERANCE,
    DEFAULT_HEXIT_ABS_TOLERANCE,
    DEFAULT_HEXIT_REL_TOLERANCE,
    DEFAULT_QUADRATURE_ATOL,
    DEFAULT_QUADRATURE_RTOL,
    DEFAULT_QUADSOURCE_POLICY_1PT5,
    DEFAULT_TK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_TK_NUMERIC_REL_TOLERANCE,
)
import config.model_list as model_list_module
from config.sharding import read_table_config
from extract_common import resolve_run_selection, run_label_tag

RUN_LABEL = "a3v2-probe"
K_VALUES = (1.0e5, 1.0e6, 1.0e7)
Z_VALUES = [10.0**e for e in (5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0.25, 0.0)]
RESPONSE_Z_MAX = 100.0

ALL_TOLERANCES = (
    DEFAULT_HEXIT_ABS_TOLERANCE,
    DEFAULT_HEXIT_REL_TOLERANCE,
    DEFAULT_GK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_GK_NUMERIC_REL_TOLERANCE,
    DEFAULT_TK_NUMERIC_ABS_TOLERANCE,
    DEFAULT_TK_NUMERIC_REL_TOLERANCE,
    DEFAULT_QUADRATURE_ATOL,
    DEFAULT_QUADRATURE_RTOL,
)

GKSOURCE_POLICY_1PT5 = dict(
    label='policy="maximize-WKB"-Levin-threshold="1.5"',
    Levin_threshold=1.5,
    numeric_policy="maximize-WKB",
)
GKSOURCE_POLICY_5PT0 = dict(
    label='policy="maximize-WKB"-Levin-threshold="5.0"',
    Levin_threshold=5.0,
    numeric_policy="maximize-WKB",
)


class Proxy:
    """``ModelProxy`` / ``GkSourceProxy`` without ``ray.put``: what the factories read of one."""

    def __init__(self, obj):
        self._obj = obj
        self._store_id = obj.store_id if getattr(obj, "available", True) else None
        self._units = getattr(getattr(obj, "cosmology", None), "units", None)
        self._cosmology = getattr(obj, "cosmology", None)

    @property
    def store_id(self):
        return self._store_id

    @property
    def available(self):
        return self._store_id is not None

    @property
    def units(self):
        return self._units

    @property
    def cosmology(self):
        return self._cosmology

    def get(self):
        return self._obj


# ------------------------------------------------------------------------------------------------
# the instruments
# ------------------------------------------------------------------------------------------------


def change_counter(path: Path) -> int:
    with open(path, "rb") as f:
        header = f.read(28)
    return int.from_bytes(header[24:28], "big")


def table_counts(path: Path) -> Dict[str, int]:
    names = [
        r[0]
        for r in sp._read(
            path, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    return {n: sp._read(path, f'SELECT count(*) FROM "{n}"')[0][0] for n in names}


def file_state(directory: Path) -> Dict[str, dict]:
    out = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        entry = {"sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12]}
        if p.suffix == ".sqlite":
            entry["counter"] = change_counter(p)
            entry["rows"] = table_counts(p)
        out[p.name] = entry
    return out


def state_delta(before: Dict[str, dict], after: Dict[str, dict]) -> Dict[str, dict]:
    """Per file: change-counter delta, rows inserted (+) or deleted (-) per table, files that
    appeared or went."""
    delta = {}
    for name in sorted(set(before) | set(after)):
        if name not in before:
            delta[name] = {"appeared": True}
            continue
        if name not in after:
            delta[name] = {"went": True}
            continue
        b, a = before[name], after[name]
        d = {}
        if "counter" in b:
            d["commits"] = a["counter"] - b["counter"]
            rows = {}
            for t in sorted(set(b["rows"]) | set(a["rows"])):
                n = a["rows"].get(t, 0) - b["rows"].get(t, 0)
                if n != 0:
                    rows[t] = n
            if rows:
                d["rows"] = rows
            new_tables = sorted(set(a["rows"]) - set(b["rows"]))
            if new_tables:
                d["new_tables"] = new_tables
        if a["sha"] != b["sha"]:
            d["changed"] = True
        if d.get("commits") or d.get("rows") or d.get("changed") or d.get("new_tables"):
            delta[name] = d
    return delta


def describe_delta(delta: Dict[str, dict]) -> str:
    if not delta:
        return "no file changed"
    parts = []
    for name, d in delta.items():
        short = name.replace("store", "").replace(".sqlite", "") or "primary"
        short = "primary" if short in ("", "-") else short.strip("-")
        bits = []
        if d.get("appeared"):
            bits.append("appeared")
        if d.get("went"):
            bits.append("went")
        if "commits" in d:
            bits.append(f"{d['commits']} commit(s)")
        if d.get("rows"):
            bits.append(
                "rows " + ", ".join(f"{t} {n:+d}" for t, n in d["rows"].items())
            )
        if d.get("new_tables"):
            bits.append("new tables " + ", ".join(d["new_tables"]))
        if d.get("changed"):
            bits.append("bytes changed")
        parts.append(f"{short}: " + "; ".join(bits))
    return " | ".join(parts)


class ReplicatedWriteLog:
    """Wraps ``ShardedPool._replicated_write`` for the duration of a ``with`` block."""

    def __init__(self):
        self.entries: List[dict] = []

    @contextlib.contextmanager
    def active(self):
        original = sp.sp_mod.ShardedPool._replicated_write
        log = self

        def wrapped(pool, operation, cls_name, submit_controller, replication, **kw):
            ref, answer = original(
                pool, operation, cls_name, submit_controller, replication, **kw
            )
            answers = answer if isinstance(answer, list) else [answer]
            log.entries.append(
                {
                    "operation": operation,
                    "class": cls_name,
                    "inserted": sum(
                        1
                        for a in answers
                        if hasattr(a, "_new_insert") or hasattr(a, "_updated")
                    ),
                    "serials": [
                        getattr(a, "store_id", None)
                        for a in answers
                        if hasattr(a, "_new_insert") or hasattr(a, "_updated")
                    ],
                }
            )
            return ref, answer

        sp.sp_mod.ShardedPool._replicated_write = wrapped
        try:
            yield self
        finally:
            sp.sp_mod.ShardedPool._replicated_write = original

    def since(self, n: int) -> List[dict]:
        return self.entries[n:]


def summarise_writes(entries: List[dict]) -> str:
    if not entries:
        return "none"
    by = {}
    for e in entries:
        key = (e["operation"], e["class"])
        calls, inserted, serials = by.get(key, (0, 0, []))
        by[key] = (calls + 1, inserted + e["inserted"], serials + e["serials"])
    return ", ".join(
        f"{op} {cls} x{calls}"
        + (f" ({ins} inserted or updated, serial(s) {serials})" if ins else "")
        for (op, cls), (calls, ins, serials) in by.items()
    )


class Step:
    """Records one step: file deltas, replicated writes, and any exception."""

    def __init__(self, recorder: "Recorder", name: str):
        self.recorder = recorder
        self.name = name

    def __enter__(self):
        self.detail = None
        self.before = file_state(self.recorder.directory)
        self.n = len(self.recorder.log.entries)
        return self

    def __exit__(self, exc_type, exc, tb):
        after = file_state(self.recorder.directory)
        delta = state_delta(self.before, after)
        writes = self.recorder.log.since(self.n)
        outcome = "ok" if exc is None else f"raised {exc_type.__name__}: {exc}"
        if self.detail is not None:
            outcome += f" [{self.detail}]"
        self.recorder.rows.append((self.name, delta, writes, outcome))
        return False


class Recorder:
    def __init__(self, directory: Path, log: ReplicatedWriteLog):
        self.directory = directory
        self.log = log
        self.rows = []

    def step(self, name: str) -> Step:
        return Step(self, name)

    def print(self, title: str):
        print(f"\n### {title}\n")
        total_commits = {}
        total_rows = {}
        total_writes = []
        for name, delta, writes, outcome in self.rows:
            print(f"- {name}: {outcome}")
            print(f"    files: {describe_delta(delta)}")
            print(f"    _replicated_write: {summarise_writes(writes)}")
            total_writes.extend(writes)
            for fname, d in delta.items():
                total_commits[fname] = total_commits.get(fname, 0) + d.get("commits", 0)
                for t, n in d.get("rows", {}).items():
                    total_rows[t] = total_rows.get(t, 0) + n
        print(f"  TOTAL commits per file: {dict(sorted(total_commits.items()))}")
        print(
            f"  TOTAL rows per table, summed over files: {dict(sorted(total_rows.items()))}"
        )
        print(
            f"  TOTAL _replicated_write: {len(total_writes)} call(s): "
            f"{summarise_writes(total_writes)}"
        )


# ------------------------------------------------------------------------------------------------
# the store
# ------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def build_full_store(primary: Path, cluster: sp.StandinCluster) -> dict:
    """Write every row the QuadSourceIntegral reader's sequence asks for, and close the store."""
    units = sp.make_units()
    pool = cluster.open_pool(primary, read_table_config=read_table_config)
    with quiet():
        run_tag = ray.get(pool.object_get("store_tag", label=run_label_tag(RUN_LABEL)))
        models = model_list_module.build_model_list(pool, units)
        lcdm = models[0]["cosmology"]
        tols = {t: sp.get_tolerance(pool, t) for t in ALL_TOLERANCES}
        ks = [
            ray.get(
                pool.object_get(
                    "wavenumber",
                    k_inv_Mpc=k,
                    units=units,
                    is_source=True,
                    is_response=True,
                )
            )
            for k in K_VALUES
        ]
        exits = [
            ray.get(
                pool.object_store(
                    sp.make_exit_time(
                        k,
                        lcdm,
                        tols[DEFAULT_HEXIT_ABS_TOLERANCE],
                        tols[DEFAULT_HEXIT_REL_TOLERANCE],
                        z_exit=1.0e6 * (i + 1),
                    )
                )
            )
            for i, k in enumerate(ks)
        ]
        model = ray.get(
            pool.object_store(
                sp.make_background_model(pool, lcdm, [run_tag], zs=tuple(Z_VALUES))
            )
        )
        validated = ray.get(pool.object_validate(model))
        # the lower redshifts are response redshifts too, as main.py's grid makes them
        ray.get(
            pool.object_get(
                "redshift",
                payload_data=[
                    {"z": z, "is_source": True, "is_response": True}
                    for z in Z_VALUES
                    if z <= RESPONSE_Z_MAX
                ],
            )
        )
        p15 = ray.get(pool.object_get("GkSourcePolicy", **GKSOURCE_POLICY_1PT5))
        p50 = ray.get(pool.object_get("GkSourcePolicy", **GKSOURCE_POLICY_5PT0))
        qp = ray.get(
            pool.object_get("QuadSourcePolicy", **DEFAULT_QUADSOURCE_POLICY_1PT5)
        )
        policy_data = ray.get(
            pool.object_store(
                sp.make_policy_data(
                    exits[0], source_serial=1, policy_serial=p15.store_id
                )
            )
        )
    facts = {
        "run_tag": run_tag.store_id,
        "LambdaCDM": lcdm.store_id,
        "QCD_Cosmology": models[1]["cosmology"].store_id,
        "BackgroundModel": model.store_id,
        "validated": validated,
        "exit_times": [e.store_id for e in exits],
        "GkSourcePolicy": (p15.store_id, p50.store_id),
        "QuadSourcePolicy": qp.store_id,
        "GkSourcePolicyData": policy_data.store_id,
        "tolerances": {t: o.store_id for t, o in tols.items()},
    }
    cluster.close_pool(pool)
    return facts


def copy_store(src_dir: Path, dst_dir: Path) -> Path:
    shutil.copytree(src_dir, dst_dir)
    return dst_dir / "store.sqlite"


def shard_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.glob("store-shard*.sqlite"))


def delete_rows(directory: Path, statements: List[str]):
    """Delete the same rows from every shard of a closed stand-in store (consistently, so that the
    check at open finds nothing to compare)."""
    for path in shard_files(directory):
        conn = sqlite3.connect(str(path))
        try:
            for sql in statements:
                conn.execute(sql)
            conn.commit()
        finally:
            conn.close()


# ------------------------------------------------------------------------------------------------
# the reader's sequence
# ------------------------------------------------------------------------------------------------


def qsi_sequence(pool, rec: Recorder, units):
    """extract_QuadSourceIntegral_data.py's pool calls, in order (line numbers at HEAD)."""
    from ComputeTargets import BackgroundModel
    from CosmologyConcepts import wavenumber_exit_time

    with rec.step("resolve_run_selection (L1308; extract_common)"):
        run_selection = resolve_run_selection(pool, None)
    with rec.step("build_model_list (L1313; config/model_list.py)"):
        model_list = model_list_module.build_model_list(pool, units)

    for model_data in model_list:  # L1315
        label = model_data["label"]
        cosmology = model_data["cosmology"]
        with rec.step(f"[{label}] tolerance x4 (L1051-1058)"):
            hexit_atol, hexit_rtol, quad_atol, quad_rtol = [
                ray.get(pool.object_get("tolerance", tol=t))
                for t in (
                    DEFAULT_HEXIT_ABS_TOLERANCE,
                    DEFAULT_HEXIT_REL_TOLERANCE,
                    DEFAULT_QUADRATURE_ATOL,
                    DEFAULT_QUADRATURE_RTOL,
                )
            ]
        with rec.step(f"[{label}] BackgroundModel (L1061-1076)") as s:
            model = ray.get(
                pool.object_get(
                    BackgroundModel,
                    solver_labels=[],
                    cosmology=cosmology,
                    z_sample=None,
                    tags=run_selection.tags,
                )
            )
            s.detail = f"available={model.available}, serial={model._my_id}"
            if not model.available:
                raise RuntimeError(
                    "Could not locate suitable background model instance in the datastore"
                )
        model_proxy = Proxy(model)
        with rec.step(f"[{label}] read_table wavenumber x2 (L1086-1089)") as s:
            source_k = ray.get(
                pool.read_table("wavenumber", units=units, is_source=True)
            )
            response_k = ray.get(
                pool.read_table("wavenumber", units=units, is_response=True)
            )
            s.detail = f"{len(source_k)} source, {len(response_k)} response"
        with rec.step(
            f"[{label}] wavenumber_exit_time x{len(source_k) + len(response_k)} (L1091-1128)"
        ) as s:
            source_exits = [
                ray.get(
                    pool.object_get(
                        wavenumber_exit_time,
                        k=k,
                        cosmology=cosmology,
                        atol=hexit_atol,
                        rtol=hexit_rtol,
                    )
                )
                for k in source_k
            ]
            response_exits = [
                ray.get(
                    pool.object_get(
                        wavenumber_exit_time,
                        k=k,
                        cosmology=cosmology,
                        atol=hexit_atol,
                        rtol=hexit_rtol,
                    )
                )
                for k in response_k
            ]
            s.detail = (
                f"{sum(e.available for e in source_exits + response_exits)} of "
                f"{len(source_exits + response_exits)} available"
            )
        with rec.step(f"[{label}] read_table redshift x2 (L1139-1146)") as s:
            z_source = ray.get(
                pool.read_table("redshift", is_source=True, model_proxy=model_proxy)
            )
            z_response = ray.get(
                pool.read_table("redshift", is_response=True, model_proxy=model_proxy)
            )
            s.detail = f"{len(z_source)} source, {len(z_response)} response"
        with rec.step(f"[{label}] GkSourcePolicy x2 (L1159-1174)") as s:
            policy_15 = ray.get(
                pool.object_get("GkSourcePolicy", **GKSOURCE_POLICY_1PT5)
            )
            policy_50 = ray.get(
                pool.object_get("GkSourcePolicy", **GKSOURCE_POLICY_5PT0)
            )
            s.detail = f"serials {policy_15.store_id}, {policy_50.store_id}"
        with rec.step(
            f"[{label}] read_table QuadSourcePolicy (L1179, find_QuadSource_policy)"
        ):
            params = DEFAULT_QUADSOURCE_POLICY_1PT5
            found = ray.get(
                pool.read_table(
                    "QuadSourcePolicy",
                    Levin_threshold=params["Levin_threshold"],
                    numeric_policy=params["numeric_policy"],
                )
            )
            if len(found) != 1:
                raise RuntimeError(f"find_QuadSource_policy: {len(found)} rows")
            quad_policy = found[0]

        # one work item (L1181-1255): the first response exit time and the first source pair
        k_exit = response_exits[0]
        q_exit = r_exit = source_exits[0]
        z_resp = min(z_response, key=lambda z: z.z)
        with rec.step(
            f"[{label}] work item: object_read_batch QuadSourceIntegral (L1193)"
        ) as s:
            batch = ray.get(
                pool.object_read_batch(
                    ObjectClass="QuadSourceIntegral",
                    shard_key={"k": k_exit},
                    model=model_proxy,
                    policy=policy_15,
                    quad_policy=quad_policy,
                    k=k_exit,
                    q=q_exit,
                    r=r_exit,
                    z_response=None,
                    z_source_max=None,
                    atol=quad_atol,
                    rtol=quad_rtol,
                    tags=run_selection.tags,
                )
            )
            s.detail = f"{len(batch)} row(s)"
        with rec.step(f"[{label}] work item: GkSource labels_only (L1212)") as s:
            gk_source = ray.get(
                pool.object_get(
                    "GkSource",
                    model=model_proxy,
                    k=k_exit,
                    z_response=z_resp,
                    z_sample=None,
                    labels_only=True,
                    tags=run_selection.tags,
                )
            )
            s.detail = f"available={gk_source.available}"
        with rec.step(
            f"[{label}] work item: GkSourcePolicyData (L1224), the stored row's source"
        ) as s:
            # the store's one GkSourcePolicyData names source serial 1; the probe asks for it, so
            # that this sharded lookup is a hit
            hit = ray.get(
                pool.object_get(
                    "GkSourcePolicyData",
                    source=sp.StandinSerial(1),
                    policy=policy_15,
                    k=source_exits[0],
                )
            )
            s.detail = f"available={hit.available}"
        with rec.step(f"[{label}] work item: GkSourcePolicyData, a miss") as s:
            miss = ray.get(
                pool.object_get(
                    "GkSourcePolicyData",
                    source=Proxy(gk_source),
                    policy=policy_15,
                    k=k_exit,
                )
            )
            s.detail = f"available={miss.available}"
        with rec.step(f"[{label}] work item: QuadSource labels_only (L1234)") as s:
            qs = ray.get(
                pool.object_get(
                    "QuadSource",
                    model=model_proxy,
                    z_sample=None,
                    q=q_exit,
                    r=r_exit,
                    labels_only=True,
                    tags=run_selection.tags,
                )
            )
            s.detail = f"available={qs.available}"
        CONTEXT.update(
            label=label,
            model_proxy=model_proxy,
            k_exit=k_exit,
            z_source=min(z_source, key=lambda z: z.z),
            tags=run_selection.tags,
        )


CONTEXT = {}
FACTS = {}


def other_sharded_lookups(pool, rec: Recorder):
    """The sharded lookups the other four readers make, each a miss on this store (it holds no
    such rows), in the form each script makes it (line numbers at HEAD)."""
    from config.defaults import (
        DEFAULT_GK_NUMERIC_ABS_TOLERANCE as GA,
        DEFAULT_GK_NUMERIC_REL_TOLERANCE as GR,
        DEFAULT_TK_NUMERIC_ABS_TOLERANCE as TA,
        DEFAULT_TK_NUMERIC_REL_TOLERANCE as TR,
    )

    from MetadataConcepts import tolerance

    c = CONTEXT
    label = c["label"]
    # the tolerance objects are made from the serials the store was built with, not looked up
    # through the pool, so that this step's counts are the sharded lookups' alone
    gk_atol, gk_rtol, tk_atol, tk_rtol = [
        tolerance(store_id=FACTS["tolerances"][t], log10_tol=math.log10(t))
        for t in (GA, GR, TA, TR)
    ]
    gk_payload = {
        "solver_labels": [],
        "model": c["model_proxy"],
        "k": c["k_exit"],
        "z_source": c["z_source"],
        "z_sample": None,
        "tags": c["tags"],
    }
    tk_payload = {
        "solver_labels": [],
        "model": c["model_proxy"],
        "k": c["k_exit"],
        "z_sample": None,
        "tags": c["tags"],
    }
    for name, cls, payload in (
        (
            "GkNumericIntegration (extract_Gk_data L422, extract_GkWKB_data L463)",
            "GkNumericIntegration",
            dict(gk_payload, atol=gk_atol, rtol=gk_rtol),
        ),
        (
            "GkWKBIntegration labels_only (extract_GkWKB_data L473)",
            "GkWKBIntegration",
            dict(gk_payload, labels_only=True),
        ),
        (
            "TkNumericIntegration (extract_TkWKB_data L453)",
            "TkNumericIntegration",
            dict(tk_payload, atol=tk_atol, rtol=tk_rtol),
        ),
        (
            "TkWKBIntegration labels_only (extract_TkWKB_data L462)",
            "TkWKBIntegration",
            dict(tk_payload, labels_only=True),
        ),
    ):
        with rec.step(f"[{label}] other readers: {name}") as s:
            obj = ray.get(pool.object_get(cls, **payload))
            s.detail = f"available={obj.available}"


def run_sequence(directory: Path, cluster: sp.StandinCluster, title: str, others=False):
    log = ReplicatedWriteLog()
    rec = Recorder(directory, log)
    units = sp.make_units()
    whole_before = file_state(directory)
    with log.active():
        with rec.step("ShardedPool open (constructor)"):
            pool = cluster.open_pool(
                directory / "store.sqlite", read_table_config=read_table_config
            )
        try:
            with quiet():
                qsi_sequence(pool, rec, units)
        except RuntimeError as e:
            print(f"  (the sequence stopped where the script would exit 1: {e})")
        if others:
            with quiet():
                other_sharded_lookups(pool, rec)
        records = sp.in_flight_records(pool)
        with rec.step("ShardedPool close (__exit__)"):
            cluster.close_pool(pool)
    rec.print(title)
    whole = state_delta(whole_before, file_state(directory))
    print(
        f"  WHOLE RUN, open to close, taken independently of the steps: {describe_delta(whole)}; "
        f"_replicated_write calls in all: {len(log.entries)}"
    )
    print(f"  in-flight records left on the primary: {records}")
    return rec


# ------------------------------------------------------------------------------------------------


def main():
    t0 = time.monotonic()
    print("# reader_writes_probe.py (a3-v2-readiness audit 00, R2)")
    print(f"python {sys.version.split()[0]}, sqlite {sqlite3.sqlite_version}")
    print(f"DEFAULT_QUADRATURE_RTOL = {DEFAULT_QUADRATURE_RTOL!r}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cluster = sp.StandinCluster()
        with cluster.active():
            full_dir = root / "full"
            facts = build_full_store(full_dir / "store.sqlite", cluster)
            FACTS.update(facts)
            print(f"\n## The full store (3 stand-in shards): {facts}")
            print(
                "files: "
                + ", ".join(
                    f"{n} counter={s['counter']}"
                    for n, s in file_state(full_dir).items()
                    if "counter" in s
                )
            )

            # ---- run 0: the instruments, on known writes
            print("\n## Run 0 - the instruments count known writes (A2)")
            d0 = root / "run0"
            copy_store(full_dir, d0)
            log = ReplicatedWriteLog()
            rec = Recorder(d0, log)
            with log.active():
                with rec.step("open"):
                    pool = cluster.open_pool(
                        d0 / "store.sqlite", read_table_config=read_table_config
                    )
                with rec.step(
                    "deliberate miss: object_get tolerance tol=1e-99 (inserts)"
                ):
                    ray.get(pool.object_get("tolerance", tol=1.0e-99))
                with rec.step("the same get again: a hit"):
                    ray.get(pool.object_get("tolerance", tol=1.0e-99))
                with rec.step(
                    "deliberate sharded object_store: one GkSourcePolicyData"
                ), quiet():
                    from CosmologyConcepts import wavenumber_exit_time

                    k0 = ray.get(
                        pool.read_table(
                            "wavenumber", units=sp.make_units(), is_source=True
                        )
                    )[0]
                    lcdm = ray.get(
                        pool.object_get(
                            "LambdaCDM",
                            params=__import__(
                                "CosmologyModels.LambdaCDM", fromlist=["Planck2018"]
                            ).Planck2018(),
                            units=sp.make_units(),
                        )
                    )
                    tA = ray.get(
                        pool.object_get("tolerance", tol=DEFAULT_HEXIT_ABS_TOLERANCE)
                    )
                    tR = ray.get(
                        pool.object_get("tolerance", tol=DEFAULT_HEXIT_REL_TOLERANCE)
                    )
                    e0 = ray.get(
                        pool.object_get(
                            wavenumber_exit_time, k=k0, cosmology=lcdm, atol=tA, rtol=tR
                        )
                    )
                    ray.get(
                        pool.object_store(
                            sp.make_policy_data(e0, source_serial=2, policy_serial=1)
                        )
                    )
                records = sp.in_flight_records(pool)
                with rec.step("close"):
                    cluster.close_pool(pool)
            rec.print("Run 0")
            print(f"  in-flight records left on the primary: {records}")

            # ---- run 1: the full store
            print("\n## Run 1 - the full store")
            d1 = root / "run1"
            copy_store(full_dir, d1)
            run_sequence(d1, cluster, "Run 1 - full store", others=True)

            # ---- runs 2-: one row missing
            l10 = lambda t: math.log10(t)
            missing = [
                (
                    "store_tag",
                    [
                        f"DELETE FROM store_tag WHERE label = '{run_label_tag(RUN_LABEL)}'"
                    ],
                ),
                ("LambdaCDM", ["DELETE FROM LambdaCDM"]),
                ("QCD_Cosmology", ["DELETE FROM QCD_Cosmology"]),
                (
                    "tolerance (DEFAULT_QUADRATURE_RTOL's row)",
                    [
                        f"DELETE FROM tolerance WHERE abs(log10_tol - ({l10(DEFAULT_QUADRATURE_RTOL)!r})) < 1e-9"
                    ],
                ),
                (
                    "BackgroundModel (LambdaCDM's, with its values and tags)",
                    [
                        f"DELETE FROM BackgroundModelValue WHERE model_serial = {facts['BackgroundModel']}",
                        f"DELETE FROM BackgroundModel_tags WHERE model_serial = {facts['BackgroundModel']}",
                        f"DELETE FROM BackgroundModel WHERE serial = {facts['BackgroundModel']}",
                    ],
                ),
                (
                    "wavenumber_exit_time (the first k's)",
                    [
                        f"DELETE FROM wavenumber_exit_time WHERE serial = {facts['exit_times'][0]}"
                    ],
                ),
                (
                    "GkSourcePolicy (the 5.0 policy)",
                    [
                        f"DELETE FROM GkSourcePolicy WHERE serial = {facts['GkSourcePolicy'][1]}"
                    ],
                ),
                (
                    "QuadSourcePolicy (the 1.5 policy, read with read_table)",
                    [
                        f"DELETE FROM QuadSourcePolicy WHERE serial = {facts['QuadSourcePolicy']}"
                    ],
                ),
            ]
            for i, (what, statements) in enumerate(missing, start=2):
                print(f"\n## Run {i} - the full store missing one row: {what}")
                print(f"  deleted on every shard, store closed: {statements}")
                d = root / f"run{i}"
                copy_store(full_dir, d)
                delete_rows(d, statements)
                run_sequence(d, cluster, f"Run {i} - missing {what}")

        print(f"\nray.is_initialized() = {ray.is_initialized()}")
    print(f"wall time {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
