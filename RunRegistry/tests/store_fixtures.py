"""Shared set-up for the store-sidecar tests (`prompts/datastore-portability` prompt 03).

Not a test module (no ``test_`` prefix); the test modules import it. Every store and every run is
built in one temporary directory: stores with `Datastore/tests/shard_store_fixtures.py` (a real
primary with the five tables, placeholder shard files), runs with the `RegistryTestCase` pattern in
a runs root inside the same directory, so that one `tree_state` covers both. No Ray is
initialised, the `ShardedPool` constructor is never called, and nothing under `var/` is opened.
"""

import json
import os
import tempfile
from pathlib import Path

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests.test_run_registry import RegistryTestCase

from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import read_pool, tree_state, write_new_store

N_SHARDS = 3

# the three layouts prompt §3 asks for: (destination directory, destination stem)
LAYOUTS = {
    "same directory, new stem": ("B", "renamed"),
    "new directory, same stem": ("C", "store"),
    "new directory, new stem": ("C", "renamed"),
}


def a3_shaped(directory: Path) -> dict:
    """A legacy sidecar with the shape of the hand-written
    `handover-A3-baseline-lambdacdm.manifest.json`: top-level provenance, a `status_files`
    object, nested `run_history` entries with lists inside them, a `restart` object and a
    `backup` object, with paths in several of them. Written inline; `var/` is never read.
    """
    return {
        "name": "store",
        "purpose": "Baseline datastore for a census. The before-picture. Keep.",
        "datastore": str(directory / "store.sqlite"),
        "created": "2026-09-20T21:02:30+01:00",
        "git_head": "0d7c05c0ddf5b6cd03763307c6a233d70c638fec",
        "git_dirty": True,
        "grid_criterion": "post prompt 15 (measured curvature criterion)",
        "driver": "docs/gktk-remedial/scoped_pipeline_run.py (NOT the other copy)",
        "scope": "8 log-spaced k over 1e5-3e8 /Mpc, --zend 0.1",
        "status_files": {
            "stdout": "var/runs/a3-pilot/run.out",
            "pid": "var/runs/a3-pilot/run.pid",
        },
        "note": "Pilot: measure cost first.",
        "run_history": [
            {
                "started": "2026-09-20T21:03:04+01:00",
                "elapsed": "10h 33m",
                "stages_complete": ["background", "APPLY GKSOURCE POLICIES"],
                "size_at_stop": "385 MB across 4 shards",
                "ratio": 0.9231,
                "count": 13,
                "nothing": None,
            },
            {
                "started": "2026-09-21T09:11",
                "outcome": "FAILED",
                "datastore_state": "UNDAMAGED. 290/290/290/290",
            },
        ],
        "restart": {
            "command": "./venv/bin/python -u driver.py --database var/datastores/store.sqlite",
            "note": "--allow-existing is REQUIRED",
        },
        "backup": {
            "path": "var/datastores/backup-pre-resume-20260921T091011",
            "retained": True,
            "reason": "kept until a resume completes",
        },
    }


def sweep_shaped(other: Path) -> dict:
    """A legacy sidecar with the shape `quadsource_atol_sweep.py` `prepare()` writes: its
    `datastore` and `copied_from` are paths into another directory."""
    return {
        "name": "store",
        "purpose": "Working copy for a sweep. Disposable.",
        "datastore": str(other / "store.sqlite"),
        "copied_from": str(other / "baseline.sqlite"),
        "created": "2026-09-23T12:41:04+0100",
    }


def write_sidecar_json(primary: Path, payload) -> Path:
    """Put ``payload`` at ``primary``'s sidecar name, by hand, as the legacy sidecars were."""
    path = stores.sidecar_path(primary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def unknown(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if k not in stores.KNOWN_FIELDS}


class StoreTestCase(RegistryTestCase):
    """One temporary directory holding the stores and, in ``runs/``, the runs root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # resolved: on macOS the temporary directory is reached through /var -> /private/var
        self.top = Path(self._tmp.name).resolve()
        self.root = str(self.top / "runs")  # RegistryTestCase.begin() puts runs here
        os.makedirs(self.root)

    # --- stores

    def a_store(self, directory: str = "B", stem: str = "store") -> Path:
        primary = self.top / directory / f"{stem}.sqlite"
        write_new_store(primary, shards=N_SHARDS)
        return primary

    def a_registry_store(self, directory: str = "B", stem: str = "store") -> Path:
        primary = self.a_store(directory, stem)
        stores.create_sidecar(primary, "a store in a temporary directory")
        return primary

    def destination(self, layout: str) -> Path:
        directory, stem = LAYOUTS[layout]
        return self.top / directory / f"{stem}.sqlite"

    # --- runs

    def a_running_run(self, slug: str, results) -> "RunRegistry.Run":
        return self.begin(slug, results=str(results))

    def a_stale_run(self, slug: str, results) -> "RunRegistry.Run":
        """`running`, with a fresh heartbeat and a pid that is gone, as `a_dead_run` builds it."""
        dead = self.a_dead_pid()
        run = self.begin(slug, results=str(results))
        run.heartbeat(pid=dead)
        return run

    # --- assertions

    def state(self):
        return tree_state(self.top)

    def assertOpens(self, primary: Path):
        """The constructor's read-and-check accepts ``primary`` and resolves its own shards."""
        pool = read_pool(primary)
        pool._check_shard_files()
        self.assertEqual(
            pool._shard_db_files,
            {i: primary.parent / shard_file_name(primary, i) for i in range(N_SHARDS)},
        )

    def assertRefusedStore(self, primary: Path):
        """``primary`` exists, and the constructor's read-and-check refuses it."""
        self.assertTrue(primary.is_file())
        with self.assertRaises(RuntimeError):
            read_pool(primary)._check_shard_files()

    def assertDescribes(self, primary: Path, store_id: str, history: int):
        """A problem-free registry sidecar beside ``primary``, with ``store_id`` and a complete
        history of ``history`` entries, the last of which put the store here."""
        reading = stores.read_sidecar(primary)
        self.assertTrue(reading.ok, (reading.kind, reading.problems))
        self.assertEqual(reading.fields["store_id"], store_id)
        self.assertEqual(len(reading.fields["history"]), history)
        self.assertEqual(
            reading.fields["history"][-1]["to"], RunRegistry._repo_path(primary)
        )
        return reading

    def assertProperty(self, src: Path, dst: Path, histories: dict):
        """Prompt §2 P11's property, at the two store names: each reachable sidecar is either
        problem-free, describing the store beside it with a complete history, or reported as
        having a problem, and no two problem-free sidecars share a store_id. ``histories`` maps
        each primary to the history length a problem-free sidecar there must have."""
        ids = []
        for primary in (src, dst):
            reading = stores.read_sidecar(primary)
            if reading.kind == "absent" or reading.problems:
                continue
            self.assertEqual(reading.kind, "registry", str(reading.path))
            self.assertIn(
                primary,
                histories,
                f"a problem-free sidecar at {reading.path}, where none may be",
            )
            self.assertDescribes(
                primary, reading.fields["store_id"], histories[primary]
            )
            ids.append(reading.fields["store_id"])
        self.assertEqual(len(ids), len(set(ids)), "two problem-free sidecars, one id")

    def assertNamesAndNothingChanged(self, before, error, *fragments):
        message = str(error)
        for fragment in fragments:
            self.assertIn(str(fragment), message)
        self.assertEqual(self.state(), before, "a refusal created, changed or removed")

    def leftovers(self):
        """Every sidecar, `.tmp` or `.incomplete-move` file outside the runs root."""
        found = []
        for path in sorted(self.top.rglob("*")):
            if Path(self.root) in path.parents:
                continue
            if path.name.endswith(
                (".manifest.json", ".tmp", stores.INCOMPLETE_MOVE_SUFFIX)
            ):
                found.append(path)
        return found
