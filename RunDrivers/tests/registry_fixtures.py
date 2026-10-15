"""This repository's own copy of the registry test helpers its driver tests use
(runregistry-extraction prompt 06; README §6.2, C14).

Not a test module (no ``test_`` prefix); the test modules import it. `runregistrykit` ships no test
support (RunRegistryKit's U19 and U21), so what the staying tests used of the registry's own test
helpers is copied here, and nothing else. The source is the frozen tree at `a45aefc`, the registry
package this repository held until runregistry-extraction prompt 06 deleted it:

- ``RegistryTestCase`` (its ``setUp`` and ``begin``), from its `tests/test_run_registry.py:113-126`:
  a run begun in ``self.root``, now through `runregistrykit.begin`. It names this repository's root
  as ``client_root`` (its `runregistrykit.toml`), so that it does not depend on the working
  directory; ``root=self.root`` keeps the run in the temporary runs root.
- ``StoreTestCase.setUp`` and ``state``, from its `tests/store_fixtures.py:114-124` and `:155-156`:
  one temporary directory, ``self.top`` (resolved), with the runs root ``self.root`` inside it, so
  that one `tree_state` covers the stores and the runs.

Every store and every run is built in that temporary directory. No Ray is initialised, and nothing
under `var/` is opened.
"""

import os
import tempfile
import unittest
from pathlib import Path

import runregistrykit

from StoreFixtures.shard_store_fixtures import tree_state

REPO_ROOT = Path(__file__).resolve().parents[2]


class RegistryTestCase(unittest.TestCase):
    """A temporary `var/runs/` for each test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def begin(self, slug="unit-test", **kwargs):
        kwargs.setdefault("purpose", "a test run, in a temporary directory")
        kwargs.setdefault("argv", ["unittest", slug])
        kwargs.setdefault("client_root", str(REPO_ROOT))
        return runregistrykit.begin(
            campaign="run-registry", prompt="01", slug=slug, root=self.root, **kwargs
        )


class StoreTestCase(RegistryTestCase):
    """One temporary directory holding the stores and, in ``runs/``, the runs root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # resolved: on macOS the temporary directory is reached through /var -> /private/var
        self.top = Path(self._tmp.name).resolve()
        self.root = str(self.top / "runs")  # RegistryTestCase.begin() puts runs here
        os.makedirs(self.root)

    def state(self):
        return tree_state(self.top)
