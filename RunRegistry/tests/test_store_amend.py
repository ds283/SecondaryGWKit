"""Amending one unknown field of a sidecar (store-retirement prompt 04):
`RunRegistry.stores.amend_sidecar`, the `amend` history entry and its markers, the D5 guards, and
`python -m RunRegistry store amend`.

1. replacing an unknown field;
2. removing one;
3. the live A3 shape: amending `backup` as prompt 05 will, then `store show`;
4. every refusal in the prompt's §2.1, each leaving the whole temporary tree unchanged;
5. the history rule: an `amend` entry's required keys, where it may stand, and that only it has
   them;
6. copy and move carry an amended field and its entry, unchanged;
7. two amendments of one field, read back as a sequence;
8. the command line, `--json`/`--remove`, bad JSON, and the exit codes.

Stores are the lightweight placeholder fixture (`Datastore.tests.shard_store_fixtures`), because
`amend_sidecar` never opens a store's files — it reads and writes only the sidecar. Every call
passes `runs_root`. No Ray is initialised and nothing under `var/` is opened.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import RunRegistry
from RunRegistry import stores
from RunRegistry.stores import KNOWN_FIELDS, amend_sidecar
from RunRegistry.tests.store_fixtures import (
    a3_shaped,
    load,
    sweep_shaped,
    write_sidecar_json,
)
from RunRegistry.tests.test_store_command_line import (
    IMPORTS,
    REPO_ROOT,
    CommandLineTestCase,
)

REASON = "disposable: this is a test reason a stranger can read"
_repo_path = RunRegistry._repo_path


class AmendTestCase(CommandLineTestCase):
    """A temporary directory holding the stores; every ``amend_sidecar`` call passes
    ``runs_root``, and every ``store amend`` command passes ``--runs-root``."""

    # --- the operation, always with runs_root

    def amend(self, primary, field, reason=REASON, **kwargs):
        return amend_sidecar(primary, field, reason, runs_root=self.root, **kwargs)

    def refused_amend(
        self, primary, *fragments, field="x", reason=REASON, **kwargs
    ) -> str:
        """``amend_sidecar`` refuses, naming ``fragments``, ending "Nothing was written", and
        nothing under the temporary directory is created, changed or removed."""
        before = self.state()
        with self.assertRaises(RuntimeError) as caught:
            self.amend(primary, field, reason, **kwargs)
        message = str(caught.exception)
        self.assertTrue(message.endswith("Nothing was written"), message[-160:])
        self.assertNamesAndNothingChanged(before, caught.exception, *fragments)
        return message

    # --- sidecars

    def sidecar(self, primary) -> dict:
        return load(stores.sidecar_path(primary))

    def a_tombstone(self, directory="T", *, complete=True) -> Path:
        """A sidecar hand-shaped as a tombstone, complete or not, of a store that
        ``a_registry_store`` built. Amend never opens a store's files, so nothing here calls
        `retire_store` or touches `ShardedPool`; the tombstone shape is built by hand, as
        `TestHistoryRule` builds one in `test_store_retire.py`. For a completed tombstone, the
        primary's own file is removed by hand, so the reading is a clean completed tombstone, not
        also "exists again"."""
        primary = self.a_registry_store(directory)
        fields = self.sidecar(primary)
        create = fields["history"][0]
        retire_entry = dict(
            create, operation="retire", **{"from": _repo_path(primary), "to": None}
        )
        tombstone = {
            "state": "retired" if complete else "retiring",
            "when": create["when"],
            "git_head": create["git_head"],
            "git_dirty": False,
            "reason": "disposable: superseded by a later store",
            "fingerprint": {"condition": "matched", "digest": "a" * 64},
            "files": [_repo_path(primary)],
            "files_present_only": False,
            "references": {
                "runs_root": "runs",
                "stores_root": "datastores",
                "runs": [],
                "sidecars": [],
            },
            "completed": create["when"] if complete else None,
        }
        new_fields = dict(
            fields, history=fields["history"] + [retire_entry], retired=tombstone
        )
        write_sidecar_json(primary, new_fields)
        if complete:
            primary.unlink()
        return primary

    # --- what an amendment leaves

    def assertOnlySidecarChanged(self, primary, before: dict) -> None:
        """Every file entry of `state()` is unchanged except the sidecar's, which is. A
        directory's mtime may bump when a file inside it is rewritten in place
        (`write_json_atomic`'s `os.replace`); its own kind and its set of entries must not.
        """
        after = self.state()
        rel = str(stores.sidecar_path(primary).relative_to(self.top))
        self.assertEqual(set(after), set(before), "something was created or removed")
        for key, value in before.items():
            if key == rel:
                continue
            if value[0] == "dir":
                self.assertEqual(after[key][0], "dir", key)
                continue
            self.assertEqual(after[key], value, key)
        self.assertNotEqual(after[rel], before[rel], "the sidecar was not written")


# =================================================================================================
# 1. replacing an unknown field


class TestAmend(AmendTestCase):
    def test_replace_an_unknown_field(self):
        primary = self.a_registry_store()
        fields = self.sidecar(primary)
        fields["note"] = {"nested": ["a", 1, None]}
        write_sidecar_json(primary, fields)

        before = self.state()
        result = self.amend(
            primary, "note", "correcting the note", value={"nested": ["b", 2]}
        )
        self.assertEqual(result["before"], {"nested": ["a", 1, None]})
        self.assertEqual(result["after"], {"nested": ["b", 2]})

        now = self.sidecar(primary)
        self.assertEqual(now["note"], {"nested": ["b", 2]})
        entry = now["history"][-1]
        self.assertEqual(entry["operation"], "amend")
        self.assertEqual(entry["field"], "note")
        self.assertEqual(entry["reason"], "correcting the note")
        self.assertIsNone(entry["from"])
        self.assertIsNone(entry["to"])
        self.assertEqual(
            entry["before"], {"present": True, "value": {"nested": ["a", 1, None]}}
        )
        self.assertEqual(
            entry["after"], {"present": True, "value": {"nested": ["b", 2]}}
        )

        # every other field, known or unknown, is value-identical
        for key in fields:
            if key in ("note", "history"):
                continue
            self.assertEqual(now[key], fields[key], key)
        self.assertEqual(now["history"][:-1], fields["history"])

        # the store's own files are unaffected: only the sidecar changed
        self.assertOnlySidecarChanged(primary, before)


# =================================================================================================
# 2. removing an unknown field


class TestRemove(AmendTestCase):
    def test_remove_an_unknown_field(self):
        primary = self.a_registry_store()
        fields = self.sidecar(primary)
        fields["scratch"] = "leftover"
        write_sidecar_json(primary, fields)

        before = self.state()
        result = self.amend(primary, "scratch", "no longer needed", remove=True)
        self.assertEqual(result["before"], "leftover")
        self.assertIsNone(result["after"])

        now = self.sidecar(primary)
        self.assertNotIn("scratch", now)
        entry = now["history"][-1]
        self.assertEqual(entry["field"], "scratch")
        self.assertEqual(entry["before"], {"present": True, "value": "leftover"})
        self.assertEqual(entry["after"], {"present": False})
        self.assertOnlySidecarChanged(primary, before)


# =================================================================================================
# 3. the live A3 shape


class TestLiveA3Shape(AmendTestCase):
    def test_amend_backup_as_prompt_05_will(self):
        primary = self.a_store()
        write_sidecar_json(primary, a3_shaped(primary.parent))
        stores.adopt_sidecar(primary)
        old_backup = self.sidecar(primary)["backup"]
        self.assertTrue(old_backup["retained"])

        new_backup = dict(
            old_backup,
            retained=False,
            reason="the backup was retired 2026-09-25 (store retire)",
        )
        before = self.state()
        result = self.amend(
            primary,
            "backup",
            "the backup this field names was retired by `store retire`",
            value=new_backup,
        )
        self.assertEqual(result["before"], old_backup)
        self.assertEqual(result["after"], new_backup)
        self.assertOnlySidecarChanged(primary, before)

        report, out, _ = self.command(
            "store", "show", primary, "--runs-root", self.root
        )
        self.assertEqual(report["code"], 0)
        # the new value is visible as the current field, the old one inside history: both the
        # literal texts are printed, and the reading confirms where each sits
        self.assertIn('"retained": false', out)
        self.assertIn('"retained": true', out)
        reading = stores.read_sidecar(primary)
        self.assertFalse(reading.fields["backup"]["retained"])
        self.assertTrue(reading.fields["history"][-1]["before"]["value"]["retained"])


# =================================================================================================
# 4. every refusal, each leaving the whole temporary tree unchanged


class TestRefusals(AmendTestCase):
    def test_no_reason_and_a_blank_reason(self):
        primary = self.a_registry_store()
        for reason in (None, "", "   "):
            with self.subTest(reason=reason):
                self.refused_amend(
                    primary, "a reason is required", reason=reason, value=1
                )

    def test_both_value_and_remove_or_neither(self):
        primary = self.a_registry_store()
        self.refused_amend(primary, "both a value and --remove", value=1, remove=True)
        self.refused_amend(primary, "neither a value nor --remove")

    def test_every_known_field_is_refused_naming_its_owner(self):
        primary = self.a_registry_store()
        for field in KNOWN_FIELDS:
            with self.subTest(field):
                message = self.refused_amend(
                    primary,
                    f"{field!r} is a known field",
                    "amend replaces or removes " "only an unknown field",
                    field=field,
                    value="x",
                )
                self.assertIn(stores._FIELD_OWNERS[field], message)
        # remove of a known field is refused the same way, retired included
        self.refused_amend(
            primary, "'retired' is a known field", field="retired", remove=True
        )

    def test_remove_of_an_absent_field(self):
        primary = self.a_registry_store()
        self.refused_amend(primary, "'ghost' is absent", field="ghost", remove=True)

    def test_a_value_identical_after_a_json_round_trip(self):
        primary = self.a_registry_store()
        self.amend(primary, "note", "add", value=[1, 2, {"a": None}])
        self.refused_amend(
            primary,
            "identical, after a JSON round trip",
            field="note",
            value=(
                1,
                2,
                {"a": None},
            ),  # a tuple, which json round-trips to the same list
        )

    def test_a_value_that_is_not_json_serialisable(self):
        primary = self.a_registry_store()
        self.refused_amend(primary, "not JSON-serialisable", field="x", value={1, 2, 3})

    def test_an_absent_an_unreadable_a_legacy_and_a_problem_sidecar(self):
        absent = self.a_store("absent")
        unreadable = self.a_store("unreadable")
        stores.sidecar_path(unreadable).write_text("{torn")
        legacy = self.a_store("legacy")
        write_sidecar_json(legacy, sweep_shaped(self.top / "X"))
        problem = self.a_registry_store("problem")
        fields = self.sidecar(problem)
        fields["purpose"] = ""
        write_sidecar_json(problem, fields)
        for primary, fragments in (
            (absent, ("there is none", "store create", "store adopt")),
            (unreadable, ("unreadable", "store create")),
            (legacy, ("legacy sidecar", "store adopt", "never upgrades")),
            (problem, ("is not a problem-free registry sidecar", "purpose")),
        ):
            with self.subTest(primary.parent.name):
                self.refused_amend(
                    primary, stores.sidecar_path(primary), *fragments, value=1
                )

    def test_a_tombstone_complete_or_not(self):
        for label, complete in (("complete", True), ("incomplete", False)):
            with self.subTest(label):
                primary = self.a_tombstone(label, complete=complete)
                self.refused_amend(primary, "is a tombstone", "never reused", value=1)

    def test_an_alive_and_a_stale_running_run(self):
        primary = self.a_registry_store("alive")
        run = self.a_running_run("in-use", primary)
        self.refused_amend(primary, run.id, "(alive,", "Run.finish", value=1)
        run.finish("killed")

        stale = self.a_registry_store("stale")
        run2 = self.a_stale_run("in-use-stale", stale)
        self.refused_amend(stale, run2.id, "(stale,", "Run.finish", value=1)

    def test_a_running_run_by_store_id_alone(self):
        primary = self.a_registry_store()
        run = self.a_running_run("by-id", primary)
        manifest = load(Path(run.manifest_path))
        manifest["results"] = str(self.top / "elsewhere" / "other.sqlite")
        Path(run.manifest_path).write_text(json.dumps(manifest))
        self.refused_amend(primary, run.id, "by its results_store_id", value=1)

    def test_a_runs_root_that_is_not_a_directory(self):
        primary = self.a_registry_store()
        before = self.state()
        with self.assertRaises(RuntimeError) as caught:
            amend_sidecar(
                primary,
                "x",
                REASON,
                value=1,
                runs_root=str(self.top / "no-such-root"),
            )
        self.assertNamesAndNothingChanged(
            before, caught.exception, "the runs root", "not a directory"
        )


# =================================================================================================
# 5. the history rule


class TestHistoryRule(AmendTestCase):
    def entries(self):
        primary = self.a_registry_store()
        create = self.sidecar(primary)["history"][0]

        def base(operation, **extra):
            return dict(create, operation=operation, **extra)

        copy = base("copy", **{"from": "B/store.sqlite", "to": "C/copy.sqlite"})
        retire = base("retire", **{"from": "B/store.sqlite", "to": None})
        amend = dict(
            base("amend", **{"from": None, "to": None}),
            field="note",
            reason="r",
            before={"present": False},
            after={"present": True, "value": 1},
        )
        return primary, create, copy, retire, amend

    def test_amend_stands_after_index_0_and_before_any_retire(self):
        _, create, copy, retire, amend = self.entries()
        cases = {
            "amend at index 0": ([amend], "operation 'amend'"),
            "amend after retire": ([create, retire, amend], "only as the last entry"),
            "amend has a from": (
                [create, dict(amend, **{"from": "B/store.sqlite"})],
                "where an amend has none",
            ),
            "amend has a to": (
                [create, dict(amend, to="C/copy.sqlite")],
                "where an amend has none",
            ),
            "copy carries amend's extra keys": (
                [
                    create,
                    dict(
                        copy,
                        field="note",
                        reason="r",
                        before={"present": False},
                        after={"present": True, "value": 1},
                    ),
                ],
                "which only an amend entry has",
            ),
        }
        for key in stores.AMEND_KEYS:
            cases[f"amend missing {key}"] = (
                [create, {k: v for k, v in amend.items() if k != key}],
                "lacks",
            )
        # a blank reason or field: present as a key (so not caught by the "lacks" check above),
        # but not a non-empty string
        cases["amend with a blank reason"] = (
            [create, dict(amend, reason="  ")],
            "malformed reason",
        )
        cases["amend with a blank field"] = (
            [create, dict(amend, field="")],
            "malformed field",
        )
        for label, (history, fragment) in cases.items():
            with self.subTest(label):
                found = stores._history_problems(history)
                self.assertTrue(found, label)
                self.assertIn(fragment, "; ".join(found))

        # well-formed sequences: two amendments in a row, and one before a retirement
        for label, history in {
            "create, amend": [create, amend],
            "create, amend, amend": [create, amend, amend],
            "create, amend, retire": [create, amend, retire],
        }.items():
            with self.subTest(label):
                self.assertEqual(stores._history_problems(history), [])

    def test_the_amend_markers_shape(self):
        _, create, _, _, amend = self.entries()
        cases = {
            "before not an object": (
                dict(amend, before="absent"),
                "not a well-formed amend marker",
            ),
            "before present with no value": (
                dict(amend, before={"present": True}),
                "before",
            ),
            "before absent but carries a value": (
                dict(amend, before={"present": False, "value": 1}),
                "before",
            ),
            "after present not a bool": (
                dict(amend, after={"present": "yes", "value": 1}),
                "not a well-formed amend marker",
            ),
        }
        for label, (entry, fragment) in cases.items():
            with self.subTest(label):
                found = stores._history_problems([create, entry])
                self.assertTrue(found, label)
                self.assertIn(fragment, "; ".join(found))


# =================================================================================================
# 6. copy and move carry an amended field and its entry


class TestCopyAndMove(AmendTestCase):
    def test_copy_carries_an_amended_field_and_its_entry(self):
        primary = self.a_registry_store()
        self.amend(primary, "note", "add a note", value="kept")
        dst = self.destination("same directory, new stem")
        fields = stores.copy_store(primary, dst, "a copy", runs_root=self.root)
        self.assertEqual(fields["note"], "kept")
        amends = [e for e in fields["history"] if e["operation"] == "amend"]
        self.assertEqual(len(amends), 1)
        self.assertEqual(amends[0]["field"], "note")
        self.assertEqual(amends[0]["after"], {"present": True, "value": "kept"})
        # the source is untouched
        self.assertEqual(self.sidecar(primary)["note"], "kept")

    def test_move_carries_an_amended_field_and_its_entry(self):
        primary = self.a_registry_store()
        self.amend(primary, "note", "add a note", value="kept")
        dst = self.destination("new directory, same stem")
        fields = stores.move_store(primary, dst, runs_root=self.root)
        self.assertEqual(fields["note"], "kept")
        amends = [e for e in fields["history"] if e["operation"] == "amend"]
        self.assertEqual(len(amends), 1)
        self.assertEqual(amends[0]["field"], "note")


# =================================================================================================
# 7. two amendments of one field


class TestTwoAmendments(AmendTestCase):
    def test_two_amendments_of_one_field_read_back_as_a_sequence(self):
        primary = self.a_registry_store()
        self.amend(primary, "note", "first", value="v1")
        self.amend(primary, "note", "second", value="v2")
        now = self.sidecar(primary)
        self.assertEqual(now["note"], "v2")
        amends = [e for e in now["history"] if e["operation"] == "amend"]
        self.assertEqual(len(amends), 2)
        self.assertEqual(amends[0]["reason"], "first")
        self.assertEqual(amends[0]["before"], {"present": False})
        self.assertEqual(amends[0]["after"], {"present": True, "value": "v1"})
        self.assertEqual(amends[1]["reason"], "second")
        self.assertEqual(amends[1]["before"], {"present": True, "value": "v1"})
        self.assertEqual(amends[1]["after"], {"present": True, "value": "v2"})


# =================================================================================================
# 8. the command line


class TestCommandLine(AmendTestCase):
    def test_json_remove_bad_json_and_exit_codes(self):
        primary = self.a_registry_store()
        runs = ["--runs-root", self.root]

        # a missing required flag is a usage error, which argparse reports with 2
        before = self.state()
        report, _, err = self.command(
            "store", "amend", primary, "--json", "1", "--reason", "r", *runs
        )
        self.assertEqual(report["code"], 2)
        self.assertIn("--field", err)
        self.assertEqual(self.state(), before)

        report, _, err = self.command(
            "store", "amend", primary, "--field", "note", "--json", "1", *runs
        )
        self.assertEqual(report["code"], 2)
        self.assertIn("--reason", err)
        self.assertEqual(self.state(), before)

        # bad JSON is a refusal, exit 1, and writes nothing
        err = self.refused(
            "store",
            "amend",
            primary,
            "--field",
            "note",
            "--json",
            "not json",
            "--reason",
            "r",
            *runs,
        )
        self.assertIn("does not parse", err)

        # neither --json nor --remove: a refusal
        err = self.refused(
            "store", "amend", primary, "--field", "note", "--reason", "r", *runs
        )
        self.assertIn("neither a value nor --remove", err)

        # both --json and --remove: a refusal
        err = self.refused(
            "store",
            "amend",
            primary,
            "--field",
            "note",
            "--json",
            "1",
            "--remove",
            "--reason",
            "r",
            *runs,
        )
        self.assertIn("both a value and --remove", err)

        # a successful --json amendment: exit 0, and prints the field, before, after and entry
        report, out, _ = self.command(
            "store",
            "amend",
            primary,
            "--field",
            "note",
            "--json",
            '{"a": 1}',
            "--reason",
            "add a note",
            *runs,
        )
        self.assertEqual(report["code"], 0)
        self.assertFalse(report["ray"] or report["sqlalchemy"])
        self.assertIn("field:    note", out)
        self.assertIn("before:   null", out)
        self.assertIn('after:    {"a": 1}', out)
        self.assertIn('"operation": "amend"', out)
        self.assertIn(">> amended:", out)
        self.assertEqual(self.sidecar(primary)["note"], {"a": 1})

        # --remove: exit 0
        report, out, _ = self.command(
            "store",
            "amend",
            primary,
            "--field",
            "note",
            "--remove",
            "--reason",
            "cleanup",
            *runs,
        )
        self.assertEqual(report["code"], 0)
        self.assertIn('before:   {"a": 1}', out)
        self.assertIn("after:    null", out)
        self.assertNotIn("note", self.sidecar(primary))

        # a known field is refused, exit 1
        err = self.refused(
            "store",
            "amend",
            primary,
            "--field",
            "purpose",
            "--json",
            '"x"',
            "--reason",
            "r",
            *runs,
        )
        self.assertIn("is a known field", err)

    def test_import_loads_neither_ray_nor_sqlalchemy(self):
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
