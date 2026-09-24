"""`python -m RunRegistry store …`, and what importing the registry loads.
`prompts/datastore-portability` prompt 03 §2 "The command line" and §3 test 10.

Every command is run in a child interpreter. The child runs the package exactly as `python -m
RunRegistry` does (`runpy.run_module("RunRegistry", run_name="__main__", alter_sys=True)`), and
then reports, from the same process, its exit code and whether `ray` and `sqlalchemy` were
imported and whether Ray was initialised. No Ray call is needed to ask. Everything is in a
temporary directory.
"""

import json
import os
import subprocess
import sys

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests.store_fixtures import (
    StoreTestCase,
    load,
    sweep_shaped,
    write_sidecar_json,
)

REPO_ROOT = RunRegistry.REPO_ROOT

CHILD = """\
import json, runpy, sys

sys.argv = ["RunRegistry"] + sys.argv[1:]
code = 0
try:
    runpy.run_module("RunRegistry", run_name="__main__", alter_sys=True)
except SystemExit as exit:
    code = exit.code
ray = sys.modules.get("ray")
print("@@" + json.dumps({
    "code": code,
    "ray": ray is not None,
    "ray_initialised": bool(ray is not None and ray.is_initialized()),
    "sqlalchemy": "sqlalchemy" in sys.modules,
}))
"""

IMPORTS = """\
import json, sys
import RunRegistry
import RunRegistry.stores
print(json.dumps({"ray": "ray" in sys.modules, "sqlalchemy": "sqlalchemy" in sys.modules}))
"""


class CommandLineTestCase(StoreTestCase):
    def command(self, *argv):
        """Run ``python -m RunRegistry *argv`` in a child; return (report, stdout, stderr)."""
        done = subprocess.run(
            [sys.executable, "-c", CHILD, *[str(a) for a in argv]],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            timeout=300,
        )
        lines = done.stdout.splitlines()
        self.assertTrue(lines and lines[-1].startswith("@@"), done.stdout + done.stderr)
        report = json.loads(lines[-1][2:])
        self.assertFalse(report["ray_initialised"], "a store command initialised Ray")
        return report, "\n".join(lines[:-1]), done.stderr

    def refused(self, *argv):
        before = self.state()
        report, out, err = self.command(*argv)
        self.assertNotEqual(report["code"], 0)
        self.assertIn("!!", err)
        self.assertEqual(self.state(), before, "a refused command wrote something")
        return err


class TestStoreCommands(CommandLineTestCase):
    def test_show_create_adopt_copy_and_move(self):
        runs = ["--runs-root", self.root]
        legacy = self.a_store("L")
        write_sidecar_json(legacy, sweep_shaped(self.top / "X"))

        # show: read-only, and it loads neither ray nor sqlalchemy
        before = self.state()
        report, out, _ = self.command("store", "show", legacy, *runs)
        self.assertEqual(report["code"], 0)
        self.assertEqual(self.state(), before)
        self.assertIn("kind:     legacy", out)
        self.assertIn("legacy path, read by its name 'store.sqlite'", out)
        self.assertIn("problems: none", out)
        self.assertFalse(report["ray"] or report["sqlalchemy"])

        # copy before adoption is refused
        err = self.refused(
            "store",
            "copy",
            legacy,
            self.top / "C" / "c.sqlite",
            "--purpose",
            "p",
            *runs,
        )
        self.assertIn("legacy", err)

        # adopt, and a second adopt is refused
        report, out, _ = self.command("store", "adopt", legacy)
        self.assertEqual(report["code"], 0)
        self.assertFalse(report["ray"])
        self.assertTrue(stores.read_sidecar(legacy).ok)
        self.assertIn(
            "already a registry sidecar", self.refused("store", "adopt", legacy)
        )

        # create, and a second create is refused
        bare = self.a_store("N")
        report, _, _ = self.command("store", "create", bare, "--purpose", "a new note")
        self.assertEqual(report["code"], 0)
        self.assertFalse(report["ray"])
        self.assertEqual(load(stores.sidecar_path(bare))["purpose"], "a new note")
        self.refused("store", "create", bare, "--purpose", "again")

        # a running run names the store: copy is refused, and show lists the run
        run = self.a_running_run("in-use", legacy)
        err = self.refused(
            "store",
            "copy",
            legacy,
            self.top / "C" / "c.sqlite",
            "--purpose",
            "p",
            *runs,
        )
        self.assertIn(run.id, err)
        _, out, _ = self.command("store", "show", legacy, *runs)
        self.assertRegex(
            out, rf"{run.id}\s+running\s+alive\s+by results and results_store_id"
        )
        run.finish("killed")
        _, out, _ = self.command("store", "show", legacy, *runs)
        self.assertRegex(out, rf"{run.id}\s+killed\s+finished")

        # copy, then move; ray is imported for ShardedPool and never initialised
        copied = self.top / "C" / "pcopy.sqlite"
        report, out, _ = self.command(
            "store", "copy", legacy, copied, "--purpose", "a copy", *runs
        )
        self.assertEqual(report["code"], 0)
        self.assertTrue(report["ray"])
        self.assertTrue(stores.read_sidecar(copied).ok)
        moved = self.top / "D" / "pmoved.sqlite"
        report, _, _ = self.command("store", "move", copied, moved, *runs)
        self.assertEqual(report["code"], 0)
        self.assertEqual(
            stores.read_sidecar(moved).store_id,
            load(stores.sidecar_path(moved))["store_id"],
        )
        self.assertEqual(
            [e["operation"] for e in load(stores.sidecar_path(moved))["history"]],
            ["adopt", "copy", "move"],
        )
        self.refused("store", "move", copied, self.top / "E" / "e.sqlite", *runs)


class TestImports(CommandLineTestCase):
    def test_import_and_list_load_neither_ray_nor_sqlalchemy(self):
        done = subprocess.run(
            [sys.executable, "-c", IMPORTS],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"ray": False, "sqlalchemy": False})

        self.begin("listed")
        report, out, _ = self.command("list", "--root", self.root)
        self.assertEqual(report["code"], 0)
        self.assertIn("listed", out)
        self.assertEqual((report["ray"], report["sqlalchemy"]), (False, False))
