"""The store sidecar's name, reader and writer, create and adopt, and the run manifest's
`results_store_id`. `prompts/datastore-portability` prompt 03 §2 P9, P10, P12 and §3 tests 1-4
and 9.

A `<stem>.manifest.json` beside a store used to be written by hand, or by a measurement script,
and one of them names a different store from the one it sits beside. `RunRegistry.stores` now
owns it. These tests pin what that means: one name, a reader that never writes and reads a legacy
`datastore` path by its name, writers that never overwrite, and unknown fields that survive.
Everything is in a temporary directory; nothing under `var/` is read.
"""

import ast
import json
import os
from pathlib import Path

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests.store_fixtures import (
    StoreTestCase,
    a3_shaped,
    load,
    sweep_shaped,
    unknown,
    write_sidecar_json,
)

from Datastore.tests.shard_store_fixtures import tree_state, write_placeholder


class TestName(StoreTestCase):
    def test_the_sidecar_is_stem_dot_manifest_dot_json_beside_the_primary(self):
        for primary, expected in (
            ("/a/b/store.sqlite", "/a/b/store.manifest.json"),
            ("pcopy.sqlite", "pcopy.manifest.json"),
            ("/x/handover-atol-sweep.sqlite", "/x/handover-atol-sweep.manifest.json"),
            ("/x/a.b.sqlite", "/x/a.b.manifest.json"),
            ("/x/noext", "/x/noext.manifest.json"),
        ):
            with self.subTest(primary=primary):
                self.assertEqual(stores.sidecar_path(primary), Path(expected))
        # what both existing sidecars and `prepare()` (`with_suffix(".manifest.json")`) use
        for stem in ("handover-A3-baseline-lambdacdm", "handover-atol-sweep"):
            primary = Path("/v") / f"{stem}.sqlite"
            self.assertEqual(
                stores.sidecar_path(primary), primary.with_suffix(".manifest.json")
            )

    def test_the_pattern_is_spelled_once(self):
        """Outside the tests and docstrings, `.manifest.json` appears in `sidecar_path` only."""
        package = Path(RunRegistry.__file__).parent
        found = []
        for source in sorted(package.glob("*.py")):
            tree = ast.parse(source.read_text())
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                ):
                    body = node.body
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                    ):
                        docstrings.add(id(body[0].value))
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and ".manifest.json" in node.value
                    and id(node) not in docstrings
                ):
                    owner = [
                        f.name
                        for f in functions
                        if f.lineno <= node.lineno <= f.end_lineno
                    ]
                    found.append((source.name, owner))
        self.assertEqual(found, [("stores.py", ["sidecar_path"])])


class TestReader(StoreTestCase):
    def test_it_classifies_absent_unreadable_legacy_and_registry(self):
        primary = self.a_store()
        self.assertEqual(stores.read_sidecar(primary).kind, "absent")
        self.assertEqual(stores.read_sidecar(primary).problems, [])

        path = stores.sidecar_path(primary)
        for text in ("{not json", "[1, 2]", '"a string"', ""):
            with self.subTest(text=text):
                path.write_text(text)
                reading = stores.read_sidecar(primary)
                self.assertEqual(reading.kind, "unreadable")
                self.assertTrue(reading.problems)
                self.assertIsNone(reading.store_id)

        path.write_text(json.dumps({"name": "store", "purpose": "a note"}))
        reading = stores.read_sidecar(primary)
        self.assertEqual((reading.kind, reading.problems), ("legacy", []))
        self.assertIsNone(reading.store_id)

        path.unlink()
        fields = stores.create_sidecar(primary, "a store")
        reading = stores.read_sidecar(primary)
        self.assertEqual((reading.kind, reading.problems), ("registry", []))
        self.assertEqual(reading.store_id, fields["store_id"])

    def test_a_legacy_datastore_path_is_read_by_its_name(self):
        """The backup's defect: its sidecar names the live store by path. Read by name, it names
        the sibling it sits beside, and the populated directory it names is not touched.
        """
        other = self.top / "X"
        self.a_store(
            "X"
        )  # another store of the same stem, which exists and is populated
        write_sidecar_json(other / "store.sqlite", {"name": "store", "purpose": "X"})
        primary = self.a_store("B")
        write_sidecar_json(primary, sweep_shaped(other))
        x_before = tree_state(other)

        reading = stores.read_sidecar(primary)
        self.assertEqual(reading.kind, "legacy")
        self.assertTrue(reading.legacy_path)
        self.assertEqual(reading.problems, [])
        self.assertEqual(reading.fields["datastore"], str(other / "store.sqlite"))

        # a relative path, and a bare name, read the same way; a bare name is not a legacy path
        for value, is_path in (
            ("var/datastores/store.sqlite", True),
            ("store.sqlite", False),
        ):
            with self.subTest(value=value):
                write_sidecar_json(primary, dict(sweep_shaped(other), datastore=value))
                before = self.state()
                reading = stores.read_sidecar(primary)
                self.assertEqual((reading.problems, reading.legacy_path), ([], is_path))
                self.assertEqual(self.state(), before)
        self.assertEqual(tree_state(other), x_before)

    def test_it_reports_each_problem(self):
        primary = self.a_registry_store()
        good = load(stores.sidecar_path(primary))

        def registry(**changes):
            fields = dict(good)
            for key, value in changes.items():
                if value is KeyError:
                    fields.pop(key)
                else:
                    fields[key] = value
            return fields

        cases = {
            "datastore names another file": (
                registry(datastore="other.sqlite"),
                "does not name",
            ),
            "datastore is a path": (
                registry(datastore=str(primary)),
                "is a path",
            ),
            "name is not the stem": (registry(name="other"), "stem"),
            "a malformed store_id": (registry(store_id="ABC"), "store_id"),
            "an empty purpose": (registry(purpose=""), "purpose"),
            "an unknown sidecar_format": (registry(sidecar_format=2), "sidecar_format"),
            "a malformed copied_from": (
                registry(copied_from={"store_id": "x"}),
                "copied_from",
            ),
            "an empty history": (registry(history=[]), "history"),
            "a history that does not begin with an identity": (
                registry(history=[dict(good["history"][0], operation="copy")]),
                "history entry #0",
            ),
            "a history entry missing keys": (
                registry(history=[{"operation": "create"}]),
                "lacks",
            ),
            "legacy datastore names another file": (
                {"name": "store", "datastore": "/else/where/other.sqlite"},
                '"other.sqlite"',
            ),
            "legacy name is not the stem": ({"name": "other"}, "stem"),
        }
        for name in stores.REQUIRED_FIELDS:
            if (
                name != "sidecar_format"
            ):  # without it, the sidecar is legacy by definition
                cases[f"no {name}"] = (registry(**{name: KeyError}), name)

        path = stores.sidecar_path(primary)
        for label, (fields, fragment) in cases.items():
            with self.subTest(label):
                path.write_text(json.dumps(fields))
                reading = stores.read_sidecar(primary)
                self.assertTrue(reading.problems, label)
                self.assertIn(fragment, "; ".join(reading.problems))
                self.assertIsNone(reading.store_id)

        # the primary: missing (an orphaned sidecar), a directory, a symbolic link
        path.write_text(json.dumps(good))
        moved = primary.with_name("elsewhere.sqlite")
        os.rename(primary, moved)
        self.assertIn(
            "does not exist", "; ".join(stores.read_sidecar(primary).problems)
        )
        os.mkdir(primary)
        self.assertIn(
            "not a regular file", "; ".join(stores.read_sidecar(primary).problems)
        )
        os.rmdir(primary)
        os.symlink(moved, primary)
        self.assertIn("symbolic link", "; ".join(stores.read_sidecar(primary).problems))

        # the sidecar itself: a symbolic link to a good one, or a directory
        os.unlink(primary)
        os.rename(moved, primary)
        self.assertTrue(stores.read_sidecar(primary).ok)
        elsewhere = self.top / "elsewhere.json"
        os.rename(path, elsewhere)
        os.symlink(elsewhere, path)
        self.assertIn(
            "sidecar itself is not a regular file",
            "; ".join(stores.read_sidecar(primary).problems),
        )
        os.unlink(path)
        os.mkdir(path)
        self.assertEqual(stores.read_sidecar(primary).kind, "unreadable")

    def test_it_never_writes(self):
        primary = self.a_store()
        other = self.top / "X"
        path = stores.sidecar_path(primary)
        payloads = [
            None,
            "{torn",
            sweep_shaped(other),
            a3_shaped(other),
            {"sidecar_format": 1, "datastore": "wrong.sqlite"},
        ]
        for payload in payloads:
            with self.subTest(payload=str(payload)[:40]):
                if payload is None:
                    if path.exists():
                        path.unlink()
                elif isinstance(payload, str):
                    path.write_text(payload)
                else:
                    write_sidecar_json(primary, payload)
                before = self.state()
                stores.read_sidecar(primary)
                stores.read_sidecar(self.top / "B" / "nothing-here.sqlite")
                self.assertEqual(self.state(), before)


class TestCreate(StoreTestCase):
    def test_the_fields_and_one_create_entry(self):
        primary = self.a_store()
        fields = stores.create_sidecar(primary, "a store in a temporary directory")
        on_disk = load(stores.sidecar_path(primary))
        self.assertEqual(on_disk, fields)
        self.assertEqual(
            sorted(fields),
            sorted(
                [
                    "sidecar_format",
                    "store_id",
                    "datastore",
                    "name",
                    "purpose",
                    "created",
                    "history",
                ]
            ),
        )
        self.assertEqual(fields["sidecar_format"], 1)
        self.assertRegex(fields["store_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(fields["datastore"], "store.sqlite")
        self.assertEqual(fields["name"], "store")
        self.assertEqual(fields["purpose"], "a store in a temporary directory")
        (entry,) = fields["history"]
        self.assertEqual(entry["operation"], "create")
        self.assertIsNone(entry["from"])
        self.assertEqual(entry["to"], RunRegistry._repo_path(primary))
        self.assertEqual(len(entry["git_head"]), 40)
        self.assertIn(entry["git_dirty"], (True, False))
        self.assertTrue(stores.read_sidecar(primary).ok)
        self.assertNotEqual(
            stores.create_sidecar(self.a_store("C"), "another")["store_id"],
            fields["store_id"],
        )

    def test_it_does_not_open_the_primary(self):
        primary = self.top / "B" / "junk.sqlite"
        write_placeholder(primary)  # not a database at all
        before = primary.read_bytes(), primary.stat().st_mtime_ns
        stores.create_sidecar(primary, "a note beside a file that is not a store")
        self.assertEqual((primary.read_bytes(), primary.stat().st_mtime_ns), before)

    def test_each_refusal(self):
        primary = self.a_store()
        path = stores.sidecar_path(primary)
        directory = self.top / "B" / "adir.sqlite"
        directory.mkdir()
        link = self.top / "B" / "link.sqlite"
        link.symlink_to(primary)

        def with_file(name, text="{}"):
            def arrange():
                (self.top / "B" / name).write_text(text)

            return arrange

        cases = [
            ("missing", self.top / "B" / "absent.sqlite", "p", None, "does not exist"),
            ("a directory", directory, "p", None, "not a regular file"),
            ("a symlink", link, "p", None, "symbolic link"),
            ("a sidecar exists", primary, "p", with_file(path.name), path.name),
            ("its .tmp exists", primary, "p", with_file(path.name + ".tmp"), ".tmp"),
            (
                "an interrupted move's sidecar exists",
                primary,
                "p",
                with_file(path.name + stores.INCOMPLETE_MOVE_SUFFIX),
                stores.INCOMPLETE_MOVE_SUFFIX,
            ),
            ("an empty purpose", primary, "", None, "purpose"),
            ("a blank purpose", primary, "   ", None, "purpose"),
        ]
        for label, target, purpose, arrange, fragment in cases:
            with self.subTest(label):
                for name in os.listdir(self.top / "B"):
                    if name.startswith("store.manifest"):
                        os.unlink(self.top / "B" / name)
                if arrange is not None:
                    arrange()
                before = self.state()
                with self.assertRaises(RuntimeError) as refused:
                    stores.create_sidecar(target, purpose)
                self.assertNamesAndNothingChanged(
                    before, refused.exception, target.name, fragment
                )


class TestAdopt(StoreTestCase):
    def test_a_sweep_shaped_legacy_sidecar(self):
        """`datastore` and `copied_from` are paths into another directory: `datastore` becomes
        bare, `copied_from` is kept verbatim, and the id and the one `adopt` entry are new.
        """
        other = self.top / "X"
        primary = self.a_store()
        legacy = sweep_shaped(other)
        write_sidecar_json(primary, legacy)

        fields = stores.adopt_sidecar(primary)
        self.assertEqual(load(stores.sidecar_path(primary)), fields)
        self.assertEqual(fields["sidecar_format"], 1)
        self.assertRegex(fields["store_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(fields["datastore"], "store.sqlite")
        self.assertEqual(fields["name"], "store")
        self.assertEqual(fields["copied_from"], legacy["copied_from"])
        self.assertEqual(fields["created"], legacy["created"])
        self.assertEqual(fields["purpose"], legacy["purpose"])
        (entry,) = fields["history"]
        self.assertEqual((entry["operation"], entry["from"]), ("adopt", None))
        self.assertEqual(entry["to"], RunRegistry._repo_path(primary))
        self.assertTrue(stores.read_sidecar(primary).ok)

    def test_an_a3_shaped_sidecar_keeps_every_unknown_field(self):
        primary = self.a_store()
        legacy = a3_shaped(self.top / "elsewhere")
        write_sidecar_json(primary, legacy)
        fields = stores.adopt_sidecar(primary)
        self.assertEqual(unknown(fields), unknown(legacy))
        self.assertEqual(unknown(load(stores.sidecar_path(primary))), unknown(legacy))
        self.assertEqual(len(unknown(legacy)), 10)
        self.assertNotIn("copied_from", fields)

    def test_a_purpose_is_taken_only_when_the_sidecar_has_none(self):
        primary = self.a_store()
        write_sidecar_json(primary, {"name": "store"})
        fields = stores.adopt_sidecar(primary, purpose="given now")
        self.assertEqual(fields["purpose"], "given now")
        self.assertTrue(fields["created"])

        other = self.a_store("C")
        write_sidecar_json(other, {"purpose": "the same"})
        self.assertEqual(
            stores.adopt_sidecar(other, purpose="the same")["purpose"], "the same"
        )

    def test_each_refusal(self):
        primary = self.a_store()
        path = stores.sidecar_path(primary)
        registry = self.a_registry_store("R")

        def legacy(payload):
            def arrange():
                write_sidecar_json(primary, payload)

            return arrange

        def text(value):
            def arrange():
                path.write_text(value)

            return arrange

        cases = [
            ("absent", primary, None, None, "use create"),
            ("already a registry sidecar", registry, None, None, "already a registry"),
            ("unreadable", primary, None, text("{torn"), "JSON"),
            (
                "the legacy datastore names a different file",
                primary,
                None,
                legacy({"datastore": "/else/where/other.sqlite", "purpose": "p"}),
                "other.sqlite",
            ),
            (
                "the legacy name is not the stem",
                primary,
                None,
                legacy({"name": "other", "purpose": "p"}),
                "stem",
            ),
            (
                "it already has a store_id",
                primary,
                None,
                legacy({"purpose": "p", "store_id": "0" * 32}),
                "store_id",
            ),
            (
                "it already has a history",
                primary,
                None,
                legacy({"purpose": "p", "history": []}),
                "history",
            ),
            (
                "no purpose and none given",
                primary,
                None,
                legacy({"name": "store"}),
                "purpose",
            ),
            (
                "a different purpose given",
                primary,
                "another",
                legacy({"purpose": "p"}),
                "already has a purpose",
            ),
            (
                "its .tmp exists",
                primary,
                None,
                lambda: (
                    write_sidecar_json(primary, {"purpose": "p"}),
                    Path(str(path) + ".tmp").write_text("{}"),
                ),
                ".tmp",
            ),
        ]
        for label, target, purpose, arrange, fragment in cases:
            with self.subTest(label):
                for name in os.listdir(self.top / "B"):
                    if name.startswith("store.manifest"):
                        os.unlink(self.top / "B" / name)
                if arrange is not None:
                    arrange()
                before = self.state()
                with self.assertRaises(RuntimeError) as refused:
                    stores.adopt_sidecar(target, purpose)
                self.assertNamesAndNothingChanged(
                    before,
                    refused.exception,
                    stores.sidecar_path(target).name,
                    fragment,
                )

        # the primary missing: the reader's orphan problem refuses the adopt
        write_sidecar_json(self.top / "O" / "gone.sqlite", {"purpose": "p"})
        before = self.state()
        with self.assertRaises(RuntimeError) as refused:
            stores.adopt_sidecar(self.top / "O" / "gone.sqlite")
        self.assertNamesAndNothingChanged(
            before, refused.exception, "gone.manifest.json", "does not exist"
        )


class TestUnknownFieldsSurvive(StoreTestCase):
    """Prompt §3 test 2. A writer that rebuilt the object from the fields it knows would pass
    every test that looks only at known fields."""

    def test_an_a3_shaped_sidecar_through_adopt_copy_and_move(self):
        primary = self.a_store()
        legacy = a3_shaped(self.top / "elsewhere")
        write_sidecar_json(primary, legacy)
        original = unknown(json.loads(json.dumps(legacy)))

        def check(label, returned, primary, created):
            with self.subTest(label):
                path = stores.sidecar_path(primary)
                self.assertEqual(unknown(returned), original)
                self.assertEqual(unknown(load(path)), original)
                self.assertEqual(load(path), returned)
                self.assertEqual(returned["created"], created)

        adopted = stores.adopt_sidecar(primary)
        check("adopt", adopted, primary, legacy["created"])  # adopt keeps it
        copied_to = self.top / "C" / "copied.sqlite"
        copied = stores.copy_store(primary, copied_to, "a copy", runs_root=self.root)
        self.assertNotEqual(copied["created"], legacy["created"])  # a new identity
        check("copy", copied, copied_to, copied["created"])
        moved_to = self.top / "D" / "moved.sqlite"
        moved = stores.move_store(copied_to, moved_to, runs_root=self.root)
        check("move", moved, moved_to, copied["created"])  # a move keeps it
        self.assertEqual(
            [e["operation"] for e in load(stores.sidecar_path(moved_to))["history"]],
            ["adopt", "copy", "move"],
        )
        # the source's sidecar still holds what adopt wrote
        self.assertEqual(load(stores.sidecar_path(primary)), adopted)


class TestRunManifests(StoreTestCase):
    """P12: `begin()` records the store_id of a problem-free registry sidecar beside `results`,
    reads the sidecar, and never writes one."""

    def test_the_store_id_of_a_registry_sidecar(self):
        primary = self.a_registry_store()
        run = self.begin("registry", results=str(primary))
        self.assertEqual(
            RunRegistry.read_json(run.manifest_path)["results_store_id"],
            stores.read_sidecar(primary).store_id,
        )

    def test_null_for_no_store_no_sidecar_legacy_and_a_problem(self):
        bare = self.a_store("N")
        legacy = self.a_store("L")
        write_sidecar_json(legacy, sweep_shaped(self.top / "X"))
        broken = self.a_registry_store("P")
        fields = load(stores.sidecar_path(broken))
        stores.sidecar_path(broken).write_text(
            json.dumps(dict(fields, datastore="other.sqlite"))
        )
        cases = {
            "no results": None,
            "results that do not exist": self.top / "nothing" / "here.sqlite",
            "no sidecar": bare,
            "a legacy sidecar": legacy,
            "a sidecar with a problem": broken,
        }
        for label, results in cases.items():
            with self.subTest(label):
                before = self.state()
                run = self.begin(
                    label.replace(" ", "-"),
                    results=str(results) if results is not None else None,
                )
                manifest = RunRegistry.read_json(run.manifest_path)
                self.assertIn("results_store_id", manifest)
                self.assertIsNone(manifest["results_store_id"])
                # the only things created or changed are the runs root and the run's own files
                after = self.state()
                changed = {
                    k for k in set(before) | set(after) if before.get(k) != after.get(k)
                }
                self.assertTrue(
                    all(k.startswith("runs") for k in changed), sorted(changed)
                )
        self.assertFalse(stores.sidecar_path(bare).exists())

    def test_a_manifest_without_the_field_lists_and_is_checked_by_path(self):
        """A manifest written before the field existed: listed as before, and still matched by
        its `results` path. Built by hand, as the older manifests under `var/runs/` were.
        """
        primary = self.a_registry_store()
        old = os.path.join(self.root, "an-older-run-20260101T000000")
        os.makedirs(old)
        manifest = {
            "run_id": os.path.basename(old),
            "created": RunRegistry.now_iso(),
            "purpose": "a run begun before results_store_id existed",
            "results": str(primary),
        }
        RunRegistry.write_json_atomic(os.path.join(old, "manifest.json"), manifest)
        RunRegistry.write_json_atomic(
            os.path.join(old, "status.json"),
            {
                "state": "running",
                "pid": os.getpid(),
                "heartbeat": RunRegistry.now_iso(),
            },
        )
        (entry,) = RunRegistry.list_runs(root=self.root)
        self.assertEqual(entry["purpose"], manifest["purpose"])
        self.assertEqual(entry["liveness"], "alive")
        self.assertEqual(RunRegistry.Run(old).results_path, str(primary))

        (named,) = stores.runs_naming(
            [primary], stores.read_sidecar(primary).store_id, runs_root=self.root
        )
        self.assertEqual(named["matched_by"], ["results"])
        before = self.state()
        with self.assertRaises(RuntimeError) as refused:
            stores.copy_store(
                primary, self.top / "C" / "c.sqlite", "p", runs_root=self.root
            )
        self.assertNamesAndNothingChanged(
            before, refused.exception, os.path.basename(old), "alive"
        )
