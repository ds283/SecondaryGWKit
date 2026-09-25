"""Retiring a store (store-retirement prompt 03): `RunRegistry.stores.retire_store`, the sidecar's
`retired` field and `retire` history entry, the reader's tombstones, the D6 guards, and
`python -m RunRegistry store retire`.

1. a retirement: the files go, the sidecar stays as a completed tombstone, the references are
   reported, and nothing else under the temporary directory changes;
2. the legacy shape: a primary naming another populated store's shards by absolute path;
3. every refusal, each leaving the whole temporary tree unchanged;
4. interruption: every point, the reading there, and its completion by a second `store retire`
   (the log's interruption table, one test per row);
5. D4, `--without-fingerprint`;
6. the dry run;
7. D6: a retired name is never reused;
8. the history rule;
9. the command line, and what importing the registry loads.

Stores are real multi-shard fixture stores (`Datastore.tests.real_store_fixtures`), which can be
fingerprinted, in a temporary directory with a runs root and a stores root of its own. Every call
passes both roots. No Ray is initialised and nothing under `var/` is opened.
"""

import contextlib
import errno
import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest import mock

import RunRegistry
from RunRegistry import stores
from RunRegistry.__main__ import main
from RunRegistry.stores import compare_fingerprints, fingerprint_store, retire_store
from RunRegistry.tests.store_fixtures import load, sweep_shaped, write_sidecar_json
from RunRegistry.tests.test_store_command_line import (
    IMPORTS,
    REPO_ROOT,
    CommandLineTestCase,
)

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.store_inventory import read_inventory
from Datastore.tests.real_store_fixtures import build_full_store
from Datastore.tests.shard_store_fixtures import (
    stored_records,
    tree_state,
    write_new_store,
)

N_SHARDS = 4
REASON = "disposable: every number it held is in a committed document"
QSI = "QuadSourceIntegral"
_repo_path = RunRegistry._repo_path


def _fail_on_call(real, n, *, after=False):
    """``real``, except that its ``n``th call raises; with ``after``, once it has taken effect."""
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            if after:
                real(*args, **kwargs)
            raise OSError(errno.EIO, "injected failure", str(args[0]))
        return real(*args, **kwargs)

    return fake


def _without(tombstone, *keys) -> dict:
    return {k: v for k, v in tombstone.items() if k not in keys}


class RetireTestCase(CommandLineTestCase):
    """A temporary directory with ``runs/`` (the runs root) and ``datastores/`` (the stores root)
    in it, and full stores built under either."""

    def setUp(self):
        super().setUp()
        self.stores_root = str(self.top / "datastores")
        os.makedirs(self.stores_root)

    # --- stores

    def a_full_store(
        self,
        directory="S",
        stem="store",
        *,
        sidecar=True,
        fingerprint=True,
        inside=True,
        **kwargs,
    ) -> Path:
        """A full four-shard store, with a registry sidecar and a recorded fingerprint unless
        told otherwise, under the stores root unless ``inside`` is false."""
        parent = (Path(self.stores_root) if inside else self.top) / directory
        parent.mkdir(parents=True, exist_ok=True)
        primary = build_full_store(parent, stem=stem, shards=N_SHARDS, **kwargs).primary
        if sidecar:
            stores.create_sidecar(primary, "a full store in a temporary directory")
            if fingerprint:
                fingerprint_store(primary, write=True, runs_root=self.root)
        return primary

    def store_files(self, primary) -> list:
        """The store's files in the order they are deleted: the shards, then the primary."""
        return [
            primary.parent / shard_file_name(primary, i) for i in range(N_SHARDS)
        ] + [primary]

    def sidecar(self, primary) -> dict:
        return load(stores.sidecar_path(primary))

    # --- retirement, always with both roots

    def retire(self, primary, reason=REASON, **kwargs):
        return retire_store(
            primary,
            reason,
            runs_root=self.root,
            stores_root=self.stores_root,
            **kwargs,
        )

    def refused_retire(self, primary, *fragments, reason=REASON, **kwargs) -> str:
        """``retire_store`` refuses, naming ``fragments``, and nothing under the temporary
        directory is created, changed or removed."""
        before = self.state()
        with self.assertRaises(RuntimeError) as caught:
            self.retire(primary, reason, **kwargs)
        message = str(caught.exception)
        self.assertTrue(
            message.endswith("Nothing was written or deleted"), message[-200:]
        )
        self.assertNamesAndNothingChanged(before, caught.exception, *fragments)
        return message

    # --- what a retirement leaves

    def assertCompletedTombstone(self, primary, before: dict) -> dict:
        """The sidecar beside ``primary`` is a completed tombstone of the store whose sidecar was
        ``before``, and every file it lists is gone. Returns its `retired`."""
        reading = stores.read_sidecar(primary)
        self.assertEqual(reading.kind, "registry")
        self.assertTrue(reading.retired)
        self.assertEqual(reading.problems, [])
        self.assertFalse(reading.ok)
        self.assertIsNone(reading.store_id)
        self.assertIsNone(reading.incomplete_retirement)

        fields = reading.fields
        history = fields["history"]
        self.assertEqual(history[:-1], before["history"])
        self.assertEqual([e["operation"] for e in history].count("retire"), 1)
        entry = history[-1]
        self.assertEqual(
            (entry["operation"], entry["from"], entry["to"]),
            ("retire", _repo_path(primary), None),
        )
        # every other field, the recorded fingerprint among them, value-identical
        self.assertEqual(fields.get("fingerprint"), before.get("fingerprint"))
        self.assertEqual(
            {k: v for k, v in fields.items() if k not in ("retired", "history")},
            {k: v for k, v in before.items() if k != "history"},
        )

        retired = fields["retired"]
        self.assertEqual(retired["state"], "retired")
        self.assertEqual(retired["when"], entry["when"])
        self.assertEqual(retired["git_head"], entry["git_head"])
        self.assertEqual(retired["reason"], REASON)
        self.assertTrue(retired["completed"])
        for path in retired["files"]:
            self.assertFalse(os.path.lexists(path), path)
        self.assertFalse(os.path.lexists(primary))
        self.assertFalse(os.path.lexists(str(stores.sidecar_path(primary)) + ".tmp"))
        return retired

    def assertIncomplete(self, primary, remaining) -> dict:
        """The reading is an incomplete retirement naming exactly ``remaining``."""
        reading = stores.read_sidecar(primary)
        self.assertTrue(reading.retired)
        self.assertFalse(reading.ok)
        self.assertIsNone(reading.store_id)
        self.assertIsNotNone(reading.incomplete_retirement)
        self.assertEqual(reading.problems, [reading.incomplete_retirement])
        text = reading.incomplete_retirement
        self.assertIn("incomplete", text)
        # what remains, and then the remedy, which names the primary
        listed, _, remedy = text.partition(". Run `python -m RunRegistry store retire")
        self.assertIn(f'"{primary}"', remedy)
        for path in self.store_files(primary):
            if path in remaining:
                self.assertIn(f'"{path}"', listed)
            else:
                self.assertNotIn(f'"{path}"', listed)
        if not remaining:
            self.assertIn("no listed file remains", text)
        self.assertEqual(reading.fields["retired"]["state"], "retiring")
        self.assertIsNone(reading.fields["retired"]["completed"])
        self.assertEqual(
            [e["operation"] for e in reading.fields["history"]].count("retire"), 1
        )
        return reading.fields["retired"]


# =================================================================================================
# 1. a retirement


class TestRetirement(RetireTestCase):
    def test_a_retirement(self):
        primary = self.a_full_store("S")
        store_id = self.sidecar(primary)["store_id"]
        files = self.store_files(primary)

        # a second store in the same directory, and one elsewhere, outside the stores root
        other = self.a_full_store("S", stem="other")
        elsewhere = self.a_full_store("E", inside=False)
        fields = self.sidecar(elsewhere)
        fields["notes"] = {"about": str(primary.parent)}
        write_sidecar_json(elsewhere, fields)

        # the three references the report must find
        finished = self.begin("finished", results=str(primary))
        finished.finish("done", exit_code=0)
        copied = Path(self.stores_root) / "C" / "copy.sqlite"
        stores.copy_store(primary, copied, "a copy of S", runs_root=self.root)
        backup = self.a_registry_store("datastores/D")
        fields = self.sidecar(backup)
        fields["backup"] = {
            "path": str(primary.parent),
            "relative": os.path.relpath(primary.parent, REPO_ROOT),
            "retained": True,
            "reason": "kept until a resume completes",
        }
        write_sidecar_json(backup, fields)
        # and a sidecar that is not JSON, which is listed and does not stop the search
        junk = Path(self.stores_root) / "J" / "junk.manifest.json"
        junk.parent.mkdir()
        junk.write_text("{torn")

        before_sidecar = self.sidecar(primary)
        before = self.state()
        result = self.retire(primary)
        after = self.state()

        # the five files are gone, and nothing else under the temporary directory changed but
        # the sidecar and the store's own directory entry
        self.assertEqual(len(files), 5)
        for path in files:
            self.assertFalse(os.path.lexists(path), path)
        rel = lambda path: str(Path(path).relative_to(self.top))
        sidecar = rel(stores.sidecar_path(primary))
        changed = {sidecar, rel(primary.parent)}
        self.assertEqual(
            {k: v for k, v in after.items() if k not in changed},
            {
                k: v
                for k, v in before.items()
                if k not in changed and k not in {rel(p) for p in files}
            },
        )
        self.assertEqual(
            sorted(os.listdir(primary.parent)),
            sorted(
                ["store.manifest.json", "other.manifest.json", "other.sqlite"]
                + [shard_file_name(other, i) for i in range(N_SHARDS)]
            ),
        )

        retired = self.assertCompletedTombstone(primary, before_sidecar)
        self.assertEqual(retired["files"], [_repo_path(p) for p in files])
        self.assertFalse(retired["files_present_only"])
        self.assertEqual(
            retired["fingerprint"],
            {"condition": "matched", "digest": before_sidecar["fingerprint"]["digest"]},
        )
        self.assertEqual(result["deleted"], retired["files"])
        self.assertEqual(result["tombstone"], retired)
        self.assertEqual(result["fields"], self.sidecar(primary))

        references = retired["references"]
        self.assertEqual(result["references"], references)
        self.assertEqual(references["runs_root"], _repo_path(self.root))
        self.assertEqual(references["stores_root"], _repo_path(self.stores_root))
        self.assertIn(_repo_path(self.stores_root), references["not_searched"])
        self.assertIn("docs, boards, logs", references["not_searched"])
        (run,) = references["runs"]
        self.assertEqual((run["id"], run["state"]), (finished.id, "done"))
        self.assertIn("results", run["matched_by"])

        by_sidecar = {entry["sidecar"]: entry for entry in references["sidecars"]}
        self.assertEqual(
            set(by_sidecar),
            {
                _repo_path(stores.sidecar_path(copied)),
                _repo_path(stores.sidecar_path(backup)),
                _repo_path(junk),
            },
        )
        self.assertIn(
            "$.copied_from.store_id",
            by_sidecar[_repo_path(stores.sidecar_path(copied))]["fields"],
        )
        self.assertEqual(
            self.sidecar(copied)["copied_from"]["store_id"], store_id
        )  # a precondition
        self.assertEqual(
            by_sidecar[_repo_path(stores.sidecar_path(backup))]["fields"],
            ["$.backup.path", "$.backup.relative"],
        )
        self.assertIn("unreadable", by_sidecar[_repo_path(junk)])
        # the other store in the same directory, and the one elsewhere, are still stores
        self.assertTrue(stores.read_sidecar(other).ok)
        self.assertTrue(stores.read_sidecar(elsewhere).ok)


# =================================================================================================
# 2. the legacy shape


class TestLegacyShape(RetireTestCase):
    def test_a_primary_naming_another_populated_stores_shards(self):
        """The backup's shape: B's `shards` rows are absolute paths to A's shards, which exist.
        Retiring B deletes B's files, and A's files and sidecar are byte-identical."""
        a = self.a_full_store("A")
        b = self.a_full_store("B", sidecar=False)
        conn = sqlite3.connect(b)
        with conn:
            for serial in range(N_SHARDS):
                conn.execute(
                    "UPDATE shards SET filename = ? WHERE serial = ?",
                    (str(a.parent / shard_file_name(a, serial)), serial),
                )
        conn.close()
        records = stored_records(b)
        self.assertEqual(
            sorted(records.values()),
            sorted(str(p) for p in self.store_files(a)[:-1]),
        )
        self.assertTrue(all(Path(p).is_file() for p in records.values()))
        stores.create_sidecar(b, "the backup's shape")
        fingerprint_store(b, write=True, runs_root=self.root)
        before_b = self.sidecar(b)

        a_before = tree_state(a.parent)
        with contextlib.redirect_stdout(io.StringIO()):  # the resolver's `!!` notice
            result = self.retire(b)

        self.assertEqual(tree_state(a.parent), a_before)
        self.assertTrue(stores.read_sidecar(a).ok)
        retired = self.assertCompletedTombstone(b, before_b)
        self.assertEqual(retired["files"], [_repo_path(p) for p in self.store_files(b)])
        for path in result["deleted"]:
            self.assertEqual(Path(path).parent, b.parent)


# =================================================================================================
# 3. refusals


class TestRefusals(RetireTestCase):
    """Each leaves the whole temporary tree unchanged, and names its reason."""

    def test_no_reason_and_a_blank_reason(self):
        primary = self.a_full_store()
        for reason in (None, "", "   "):
            with self.subTest(reason=reason):
                self.refused_retire(primary, "a reason is required", reason=reason)

    def test_an_absent_an_unreadable_a_legacy_and_a_problem_sidecar(self):
        absent = self.a_full_store("absent", sidecar=False)
        unreadable = self.a_full_store("unreadable", sidecar=False)
        stores.sidecar_path(unreadable).write_text("{torn")
        legacy = self.a_full_store("legacy", sidecar=False)
        write_sidecar_json(legacy, sweep_shaped(self.top / "X"))
        problem = self.a_full_store("problem")
        fields = self.sidecar(problem)
        fields["purpose"] = ""
        write_sidecar_json(problem, fields)
        for primary, fragments in (
            (absent, ("there is none", "store create", "store adopt")),
            (unreadable, ("unreadable", "store create")),
            (legacy, ("legacy sidecar", "store adopt", "never upgrades")),
            (problem, ("has problems", "purpose")),
        ):
            with self.subTest(primary.parent.name):
                self.refused_retire(primary, stores.sidecar_path(primary), *fragments)

    def test_a_completed_tombstone(self):
        primary = self.a_full_store()
        self.retire(primary)
        retired = self.sidecar(primary)["retired"]
        self.refused_retire(
            primary, "already retired", retired["when"], retired["completed"], REASON
        )

    def test_an_alive_running_run(self):
        primary = self.a_full_store()
        run = self.a_running_run("alive", primary)
        self.refused_retire(primary, run.id, "(alive,", "Run.finish")

    def test_a_stale_running_run_and_its_remedy(self):
        primary = self.a_full_store()
        run = self.a_stale_run("stale", primary)
        self.assertEqual(RunRegistry.list_runs(root=self.root)[0]["liveness"], "stale")
        self.refused_retire(
            primary,
            run.id,
            "(stale,",
            "A stale run is ended by a person, with Run.finish",
            "store fingerprint",
            "--write",
        )

    def test_a_running_run_by_store_id_alone(self):
        primary = self.a_full_store()
        run = self.a_running_run("by-id", primary)
        manifest = load(Path(run.manifest_path))
        manifest["results"] = str(self.top / "elsewhere" / "other.sqlite")
        Path(run.manifest_path).write_text(json.dumps(manifest))
        self.refused_retire(primary, run.id, "by its results_store_id")

    def test_no_recorded_fingerprint_names_its_remedy(self):
        primary = self.a_full_store(fingerprint=False)
        self.refused_retire(
            primary, "records no fingerprint", "store fingerprint", "--write"
        )

    def test_a_row_changed_after_fingerprinting(self):
        primary = self.a_full_store()
        conn = sqlite3.connect(primary.parent / shard_file_name(primary, 1))
        with conn:
            conn.execute(f"DELETE FROM {QSI} WHERE serial = 3")
            conn.execute(f"DELETE FROM {QSI}_tags WHERE parent_serial = 3")
        conn.close()
        self.refused_retire(
            primary,
            "does not match the fingerprint its sidecar records",
            "1 difference(s)",
            f"{QSI}: tag set [Run_fixture] differs: 2 recorded, 1 now",
        )

    def test_a_mismatch_in_problems_alone(self):
        primary = self.a_full_store()
        conn = sqlite3.connect(primary.parent / shard_file_name(primary, 1))
        with conn:
            conn.execute(
                f"INSERT INTO {QSI}_tags (timestamp, parent_serial, tag_serial) "
                f"VALUES ('2026-09-24 12:00:00.000000', 999, 3)"
            )
        conn.close()
        # a precondition: the digests agree, and only the problems differ
        now = fingerprint_store(primary, runs_root=self.root)["fingerprint"]
        comparison = compare_fingerprints(self.sidecar(primary)["fingerprint"], now)
        self.assertEqual([d["kind"] for d in comparison], ["problems"])
        self.assertEqual(now["digest"], self.sidecar(primary)["fingerprint"]["digest"])
        self.refused_retire(
            primary,
            "does not match",
            f"{QSI}: orphan-tag problems: 0 recorded, 1 now",
        )

    def test_a_journal_beside_a_shard(self):
        primary = self.a_full_store()
        journal = Path(str(primary.parent / shard_file_name(primary, 2)) + "-journal")
        journal.write_bytes(b"")
        self.refused_retire(primary, journal, "not closed cleanly")

    def test_a_tmp_beside_the_sidecar(self):
        primary = self.a_full_store()
        tmp = Path(str(stores.sidecar_path(primary)) + ".tmp")
        tmp.write_text("{}")
        self.refused_retire(primary, tmp, "interrupted write")

    def test_roots_that_are_not_directories(self):
        primary = self.a_full_store()
        for keyword, fragment in (
            ("runs_root", "the runs root"),
            ("stores_root", "the stores root"),
        ):
            with self.subTest(keyword):
                before = self.state()
                roots = {"runs_root": self.root, "stores_root": self.stores_root}
                roots[keyword] = str(self.top / "no-such-root")
                with self.assertRaises(RuntimeError) as caught:
                    retire_store(primary, REASON, **roots)
                self.assertNamesAndNothingChanged(
                    before,
                    caught.exception,
                    fragment,
                    "no-such-root",
                    "not a directory",
                )


# =================================================================================================
# 4. interruption: one test per row of the log's table


class TestInterruption(RetireTestCase):
    def interrupted_then_completed(self, primary, patcher, remaining, step):
        """Retire under ``patcher``, which makes it fail at ``step``. The reading must then be an
        incomplete retirement naming ``remaining``; a different reason is refused; and a second
        retirement completes it, leaving what an uninterrupted one leaves."""
        before = self.sidecar(primary)
        with patcher:
            with self.assertRaises(RuntimeError) as failed:
                self.retire(primary)
        message = str(failed.exception)
        self.assertIn(f'failed at step "{step}"', message)
        self.assertIn("injected failure", message)
        self.assertIn("store retire", message)
        for path in remaining:
            self.assertIn(f'"{path}"', message)

        first = self.assertIncomplete(primary, remaining)
        self.assertFalse(os.path.lexists(str(stores.sidecar_path(primary)) + ".tmp"))
        self.refused_retire(
            primary, "a different reason given is refused", reason="another reason"
        )

        result = self.retire(primary)
        self.assertTrue(result["completing"])
        self.assertEqual(result["deleted"], [_repo_path(p) for p in remaining])
        retired = self.assertCompletedTombstone(primary, before)
        # the tombstone the first write recorded, completed: nothing re-planned or re-searched
        self.assertEqual(
            _without(retired, "state", "completed"),
            _without(first, "state", "completed"),
        )
        self.assertEqual(os.listdir(primary.parent), ["store.manifest.json"])

    def test_I1_to_I5_os_unlink_fails_before_the_nth_deletion(self):
        for n in range(1, 6):
            with self.subTest(n=n):
                primary = self.a_full_store(f"U{n}")
                files = self.store_files(primary)
                self.interrupted_then_completed(
                    primary,
                    mock.patch("os.unlink", _fail_on_call(os.unlink, n)),
                    files[n - 1 :],
                    "delete the store's files",
                )

    def test_I6_os_unlink_fails_after_deleting_the_primary(self):
        primary = self.a_full_store()
        self.interrupted_then_completed(
            primary,
            mock.patch("os.unlink", _fail_on_call(os.unlink, 5, after=True)),
            [],
            "delete the store's files",
        )

    def test_I6_the_completion_write_fails(self):
        primary = self.a_full_store()
        self.interrupted_then_completed(
            primary,
            mock.patch.object(
                stores, "_update_sidecar", _fail_on_call(stores._update_sidecar, 2)
            ),
            [],
            "mark the tombstone retired",
        )

    def test_I5_the_deletion_reports_success_and_leaves_a_listed_file(self):
        """The check before completion: a deletion that says it succeeded, and left the
        primary, is not marked retired."""

        def partial(primary, *, resume=False):
            files = ShardedPool.closed_store_files(primary, resume=resume)
            for path in files[:-1]:
                os.unlink(path)
            return files[:-1]

        primary = self.a_full_store()
        before = self.sidecar(primary)
        with mock.patch.object(ShardedPool, "delete_store", partial):
            with self.assertRaises(RuntimeError) as failed:
                self.retire(primary)
        self.assertIn(
            'failed at step "check that no listed file remains"', str(failed.exception)
        )
        self.assertIncomplete(primary, [primary])
        self.refused_retire(
            primary, "a different reason given is refused", reason="another reason"
        )
        self.retire(primary)
        self.assertCompletedTombstone(primary, before)

    def test_I0_the_tombstone_write_fails(self):
        primary = self.a_full_store()
        before = self.state()
        with mock.patch.object(
            stores, "_update_sidecar", _fail_on_call(stores._update_sidecar, 1)
        ):
            with self.assertRaises(RuntimeError) as failed:
                self.retire(primary)
        message = str(failed.exception)
        self.assertIn('failed at step "write the tombstone"', message)
        self.assertIn("Nothing was deleted", message)
        self.assertIn("the live store", message)
        # a live store, untouched, and no .tmp: nothing was written at all
        self.assertEqual(self.state(), before)
        self.assertTrue(stores.read_sidecar(primary).ok)
        sidecar = self.sidecar(primary)
        self.retire(primary)
        self.assertCompletedTombstone(primary, sidecar)

    def killed_write(self, primary, nth):
        """The ``nth`` write of the sidecar is killed between its `.tmp` and its `os.replace`,
        which leaves the `.tmp` file, as a process that died there would."""
        real = os.replace
        target = str(stores.sidecar_path(primary))
        calls = {"n": 0}

        def fake(src, dst, *args, **kwargs):
            if str(dst) == target:
                calls["n"] += 1
                if calls["n"] == nth:
                    raise OSError(errno.EIO, "injected failure", str(dst))
            return real(src, dst, *args, **kwargs)

        return mock.patch("os.replace", fake)

    def test_I0t_the_tombstone_write_is_killed_and_leaves_its_tmp(self):
        primary = self.a_full_store()
        sidecar = self.sidecar(primary)
        tmp = Path(str(stores.sidecar_path(primary)) + ".tmp")
        with self.killed_write(primary, 1):
            with self.assertRaises(RuntimeError) as failed:
                self.retire(primary)
        self.assertIn(str(tmp), str(failed.exception))
        # the live store, untouched; the .tmp is not at a sidecar name, and nothing reads it
        self.assertTrue(stores.read_sidecar(primary).ok)
        self.assertEqual(self.sidecar(primary), sidecar)
        self.assertEqual(load(tmp)["retired"]["state"], "retiring")
        for path in self.store_files(primary):
            self.assertTrue(path.exists())
        # a second retirement refuses until a person removes it, and then retires
        self.refused_retire(primary, tmp, "interrupted write")
        tmp.unlink()
        self.retire(primary)
        self.assertCompletedTombstone(primary, sidecar)

    def test_I6t_the_completion_write_is_killed_and_leaves_its_tmp(self):
        primary = self.a_full_store()
        sidecar = self.sidecar(primary)
        tmp = Path(str(stores.sidecar_path(primary)) + ".tmp")
        with self.killed_write(primary, 2):
            with self.assertRaises(RuntimeError) as failed:
                self.retire(primary)
        self.assertIn(
            'failed at step "mark the tombstone retired"', str(failed.exception)
        )
        self.assertIn(str(tmp), str(failed.exception))
        self.assertIncomplete(primary, [])
        self.assertEqual(load(tmp)["retired"]["state"], "retired")
        self.refused_retire(
            primary, "a different reason given is refused", reason="another reason"
        )
        self.refused_retire(primary, tmp, "interrupted write")
        tmp.unlink()
        self.retire(primary)
        self.assertCompletedTombstone(primary, sidecar)

    def test_a_listed_file_that_remains_with_no_primary(self):
        primary = self.a_full_store()
        with mock.patch.object(
            stores, "_update_sidecar", _fail_on_call(stores._update_sidecar, 2)
        ):
            with self.assertRaises(RuntimeError):
                self.retire(primary)
        shard = self.store_files(primary)[1]
        shard.write_bytes(b"somebody put a file back\n")
        self.assertIncomplete(primary, [shard])
        self.refused_retire(primary, shard, "only by a person, outside the registry")


# =================================================================================================
# 5. D4: --without-fingerprint


class TestWithoutFingerprint(RetireTestCase):
    def test_a_corrupt_shard(self):
        primary = self.a_full_store()
        before = self.sidecar(primary)
        self.store_files(primary)[1].write_bytes(b"this is not a database\n" * 100)
        with self.assertRaises(Exception) as unreadable:
            read_inventory(primary)
        self.refused_retire(
            primary, "a fresh fingerprint could not be taken", "--without-fingerprint"
        )

        result = self.retire(primary, without_fingerprint=True)
        retired = self.assertCompletedTombstone(primary, before)
        self.assertEqual(
            retired["fingerprint"],
            {
                "condition": "without",
                "error": str(unreadable.exception),
                "error_type": type(unreadable.exception).__name__,
            },
        )
        self.assertFalse(retired["files_present_only"])
        self.assertEqual(
            result["deleted"], [_repo_path(p) for p in self.store_files(primary)]
        )

    def test_a_store_that_can_be_fingerprinted_is_refused(self):
        for recorded in (True, False):
            with self.subTest(recorded=recorded):
                primary = self.a_full_store(f"F{recorded}", fingerprint=recorded)
                self.refused_retire(
                    primary,
                    "can be fingerprinted, so --without-fingerprint does not apply",
                    "store fingerprint",
                    "--write",
                    without_fingerprint=True,
                )

    def test_a_running_run_is_still_refused(self):
        """D4: the flag still refuses a running run, alive or stale. The store has a corrupt
        shard, so the flag would apply, and `fingerprint_store`'s own refusal of the run would
        otherwise be taken as the error that prevents a fingerprint."""
        for liveness, make in (
            ("alive", self.a_running_run),
            ("stale", self.a_stale_run),
        ):
            with self.subTest(liveness):
                primary = self.a_full_store(f"R-{liveness}")
                self.store_files(primary)[1].write_bytes(b"this is not a database\n")
                run = make(f"flag-{liveness}", primary)
                self.refused_retire(
                    primary,
                    run.id,
                    f"({liveness},",
                    "A stale run is ended by a person, with Run.finish",
                    without_fingerprint=True,
                )

    def test_a_journal_is_refused(self):
        primary = self.a_full_store()
        journal = Path(str(primary) + "-wal")
        journal.write_bytes(b"")
        self.refused_retire(primary, journal, without_fingerprint=True)

    def test_a_missing_shard(self):
        primary = self.a_full_store()
        before = self.sidecar(primary)
        files = self.store_files(primary)
        files[2].unlink()
        self.refused_retire(primary, files[2], "does not exist")

        result = self.retire(primary, without_fingerprint=True)
        retired = self.assertCompletedTombstone(primary, before)
        present = [_repo_path(p) for p in files if p != files[2]]
        self.assertEqual(retired["files"], present)
        self.assertEqual(result["deleted"], present)
        self.assertTrue(retired["files_present_only"])
        self.assertEqual(retired["fingerprint"]["condition"], "without")
        self.assertIn(str(files[2]), retired["fingerprint"]["error"])

    def test_an_unreadable_shards_table(self):
        primary = self.a_full_store()
        conn = sqlite3.connect(primary)
        with conn:
            conn.execute("DROP TABLE shards")
        conn.close()
        for flag in (True, False):
            with self.subTest(without_fingerprint=flag):
                self.refused_retire(
                    primary,
                    "its shards table could not be read",
                    "refused even under --without-fingerprint",
                    "only by a person, outside the registry",
                    without_fingerprint=flag,
                )


# =================================================================================================
# 6. the dry run


class TestDryRun(RetireTestCase):
    def test_a_store_that_would_retire(self):
        primary = self.a_full_store()
        before = self.state()
        dry = self.retire(primary, dry_run=True)
        self.assertEqual(self.state(), before)
        self.assertTrue(dry["dry_run"])
        self.assertEqual(dry["deleted"], [])
        self.assertEqual(
            dry["to_delete"], [_repo_path(p) for p in self.store_files(primary)]
        )
        self.assertEqual(dry["tombstone"]["state"], "retiring")

        real = self.retire(primary)
        self.assertEqual(real["deleted"], dry["to_delete"])
        self.assertEqual(real["references"], dry["references"])
        self.assertEqual(real["fingerprint"], dry["fingerprint"])
        self.assertEqual(
            _without(real["tombstone"], "when", "state", "completed"),
            _without(dry["tombstone"], "when", "state", "completed"),
        )

    def test_refusals_are_the_real_runs(self):
        def no_fingerprint():
            return self.a_full_store("N", fingerprint=False)

        def a_stale_run():
            primary = self.a_full_store("R")
            self.a_stale_run("dry-stale", primary)
            return primary

        def a_journal():
            primary = self.a_full_store("J")
            Path(str(primary) + "-journal").write_bytes(b"")
            return primary

        def a_completed_tombstone():
            primary = self.a_full_store("T")
            self.retire(primary)
            return primary

        for make in (no_fingerprint, a_stale_run, a_journal, a_completed_tombstone):
            with self.subTest(make.__name__):
                primary = make()
                dry = self.refused_retire(primary, dry_run=True)
                real = self.refused_retire(primary)
                self.assertEqual(dry, real)

    def test_an_incomplete_retirement(self):
        primary = self.a_full_store()
        files = self.store_files(primary)
        with mock.patch("os.unlink", _fail_on_call(os.unlink, 3)):
            with self.assertRaises(RuntimeError):
                self.retire(primary)
        before = self.state()
        dry = self.retire(primary, dry_run=True)
        self.assertEqual(self.state(), before)
        self.assertTrue(dry["completing"])
        self.assertEqual(dry["to_delete"], [_repo_path(p) for p in files[2:]])


# =================================================================================================
# 7. D6: a retired name is never reused


class TestNeverReused(RetireTestCase):
    def tombstones(self):
        """A completed tombstone, and an incomplete one."""
        done = self.a_full_store("done")
        self.retire(done)
        partial = self.a_full_store("partial")
        with mock.patch("os.unlink", _fail_on_call(os.unlink, 2)):
            with self.assertRaises(RuntimeError):
                self.retire(partial)
        return {"completed": done, "incomplete": partial}

    def test_begin_refuses_a_retired_store_and_creates_no_run_directory(self):
        for label, primary in self.tombstones().items():
            with self.subTest(label):
                retired = self.sidecar(primary)["retired"]
                runs_before = tree_state(Path(self.root))
                with self.assertRaises(RuntimeError) as refused:
                    self.begin(f"reuse-{label}", results=str(primary))
                message = str(refused.exception)
                for fragment in (
                    "tombstone",
                    retired["when"],
                    REASON,
                    "never reused",
                    "No run directory was created",
                ):
                    self.assertIn(fragment, message)
                self.assertEqual(tree_state(Path(self.root)), runs_before)

    def test_every_other_operation_refuses_a_tombstone(self):
        live = self.a_registry_store("L")
        for label, primary in self.tombstones().items():
            operations = {
                "copy from": lambda: stores.copy_store(
                    primary, self.top / "C" / "c.sqlite", "p", runs_root=self.root
                ),
                "copy to": lambda: stores.copy_store(
                    live, primary, "p", runs_root=self.root
                ),
                "move from": lambda: stores.move_store(
                    primary, self.top / "C" / "c.sqlite", runs_root=self.root
                ),
                "move to": lambda: stores.move_store(
                    live, primary, runs_root=self.root
                ),
                "fingerprint": lambda: fingerprint_store(primary, runs_root=self.root),
                "fingerprint --write": lambda: fingerprint_store(
                    primary, write=True, runs_root=self.root
                ),
                "adopt": lambda: stores.adopt_sidecar(primary),
                "create": lambda: stores.create_sidecar(primary, "a new store"),
            }
            for operation, call in operations.items():
                with self.subTest(label, operation=operation):
                    before = self.state()
                    with self.assertRaises(RuntimeError) as refused:
                        call()
                    self.assertNamesAndNothingChanged(
                        before,
                        refused.exception,
                        "is a tombstone",
                        stores.sidecar_path(primary),
                        REASON,
                        "never reused",
                    )
        self.assertTrue(stores.read_sidecar(live).ok)

    def test_a_store_written_at_the_retired_name(self):
        primary = self.a_full_store()
        self.retire(primary)
        write_new_store(primary, shards=3)  # what an unregistered opener would leave
        reading = stores.read_sidecar(primary)
        self.assertTrue(reading.retired)
        self.assertFalse(reading.ok)
        self.assertIsNone(reading.incomplete_retirement)
        (problem,) = reading.problems
        self.assertIn("exists again", problem)
        self.assertIn(f'"{primary}"', problem)
        self.assertIn("describes the store that was retired", problem)

        report, out, _ = self.command(
            "store", "show", primary, "--runs-root", self.root
        )
        self.assertEqual(report["code"], 1)
        self.assertIn("exists again", out)
        self.assertLess(out.index("retirement:"), out.index("sidecar:"))
        self.refused_retire(primary, "tombstone with problems", "exists again")
        with self.assertRaises(RuntimeError):
            self.begin("reappeared", results=str(primary))


# =================================================================================================
# 8. the history rule, and the tombstone's shape


class TestHistoryRule(RetireTestCase):
    def entries(self):
        primary = self.a_registry_store()
        create = self.sidecar(primary)["history"][0]

        def entry(operation, source, destination):
            return dict(
                create, operation=operation, **{"from": source, "to": destination}
            )

        return (
            primary,
            create,
            entry("copy", "B/store.sqlite", "C/copy.sqlite"),
            entry("move", "C/copy.sqlite", "D/moved.sqlite"),
            entry("retire", "D/moved.sqlite", None),
        )

    def test_retire_stands_only_last_once_and_with_no_to(self):
        _, create, copy, move, retire = self.entries()
        problems = {
            "an entry after retire": ([create, retire, copy], "only as the last entry"),
            "two retire entries": ([create, retire, retire], "only as the last entry"),
            "retire at index 0": (
                [retire],
                "history entry #0 records operation 'retire'",
            ),
            "retire with a to": (
                [create, dict(retire, to="E/elsewhere.sqlite")],
                "where a retirement has none",
            ),
            "retire with no from": (
                [create, dict(retire, **{"from": None})],
                "no from",
            ),
            "copy with retire's to": ([create, dict(copy, to=None)], "malformed to"),
            "move with retire's to": ([create, dict(move, to=None)], "malformed to"),
        }
        for label, (history, fragment) in problems.items():
            with self.subTest(label):
                found = stores._history_problems(history)
                self.assertTrue(found, label)
                self.assertIn(fragment, "; ".join(found))
        for label, history in {
            "create": [create],
            "create, copy, move": [create, copy, move],
            "create, move, copy": [create, move, copy],
            "create, copy, move, retire": [create, copy, move, retire],
        }.items():
            with self.subTest(label):
                self.assertEqual(stores._history_problems(history), [])

    def test_the_tombstone_and_its_history_agree(self):
        primary, create, _, _, retire = self.entries()
        good = self.sidecar(primary)
        path = stores.sidecar_path(primary)
        tombstone = {
            "state": "retired",
            "when": create["when"],
            "git_head": create["git_head"],
            "git_dirty": False,
            "reason": REASON,
            "fingerprint": {"condition": "matched", "digest": "a" * 64},
            "files": ["B/store.sqlite"],
            "files_present_only": False,
            "references": {
                "runs_root": "runs",
                "stores_root": "datastores",
                "runs": [],
                "sidecars": [],
            },
            "completed": create["when"],
        }
        cases = {
            "a retire entry and no retired": (
                dict(good, history=[create, retire]),
                "carries no retired field",
            ),
            "retired and no retire entry": (
                dict(good, retired=tombstone),
                "does not end in a retire entry",
            ),
            "retired not an object": (
                dict(good, history=[create, retire], retired="yes"),
                "not an object",
            ),
            "an unknown state": (
                dict(
                    good,
                    history=[create, retire],
                    retired=dict(tombstone, state="gone"),
                ),
                "retired's state",
            ),
            "retiring and completed": (
                dict(
                    good,
                    history=[create, retire],
                    retired=dict(tombstone, state="retiring"),
                ),
                "while its state is retiring",
            ),
            "no files": (
                dict(good, history=[create, retire], retired=dict(tombstone, files=[])),
                "retired's files",
            ),
            "a condition of neither kind": (
                dict(
                    good,
                    history=[create, retire],
                    retired=dict(tombstone, fingerprint={"condition": "trust me"}),
                ),
                "neither a matched digest",
            ),
        }
        for label, (fields, fragment) in cases.items():
            with self.subTest(label):
                path.write_text(json.dumps(fields))
                reading = stores.read_sidecar(primary)
                self.assertEqual(reading.kind, "registry")
                self.assertFalse(reading.ok)
                self.assertIn(fragment, "; ".join(reading.problems))
        # well-formed, with the primary still there: a store at a retired name
        path.write_text(
            json.dumps(dict(good, history=[create, retire], retired=tombstone))
        )
        self.assertIn("exists again", "; ".join(stores.read_sidecar(primary).problems))

    def test_a_legacy_sidecar_with_a_retired_key_reads_as_before(self):
        primary = self.a_store("L")
        payload = {"name": "store", "purpose": "p", "retired": {"state": "retired"}}
        write_sidecar_json(primary, payload)
        reading = stores.read_sidecar(primary)
        self.assertEqual((reading.kind, reading.problems), ("legacy", []))
        self.assertFalse(reading.retired)
        self.assertEqual(reading.unknown_fields, {"retired": {"state": "retired"}})


# =================================================================================================
# 9. the command line


class TestCommandLine(RetireTestCase):
    def roots(self):
        return ["--runs-root", self.root, "--stores-root", self.stores_root]

    def test_retire_its_exit_codes_and_what_it_prints(self):
        primary = self.a_full_store()
        files = [_repo_path(p) for p in self.store_files(primary)]

        # a refusal exits 1
        err = self.refused("store", "retire", primary, "--reason", "  ", *self.roots())
        self.assertIn("a reason is required", err)
        # no --reason at all is a usage error, which argparse reports with 2
        before = self.state()
        report, _, err = self.command("store", "retire", primary, *self.roots())
        self.assertEqual(report["code"], 2)
        self.assertIn("--reason", err)
        self.assertEqual(self.state(), before)

        # a dry run exits 0 and changes nothing
        report, out, _ = self.command(
            "store", "retire", primary, "--reason", REASON, "--dry-run", *self.roots()
        )
        self.assertEqual(report["code"], 0)
        self.assertEqual(self.state(), before)
        self.assertIn("dry run:  nothing was written or deleted", out)
        self.assertIn("files to be deleted:", out)
        self.assertNotIn(">> retired", out)

        # a retirement exits 0 and prints each section
        report, out, _ = self.command(
            "store", "retire", primary, "--reason", REASON, *self.roots()
        )
        self.assertEqual(report["code"], 0)
        sections = ["references:", "fingerprint check:", "files deleted:", "tombstone:"]
        positions = [out.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        for path in files:
            self.assertIn(f"  {path}", out)
        self.assertIn("matched: a fresh read-only fingerprint matched", out)
        self.assertIn('"state": "retired"', out)
        self.assertIn(">> retired:", out)
        self.assertEqual(stores.read_sidecar(primary).problems, [])

        # store show on the tombstone: the retirement first, then the sidecar, then the runs;
        # 0, and neither ray nor sqlalchemy
        report, out, _ = self.command(
            "store", "show", primary, "--runs-root", self.root
        )
        self.assertEqual(report["code"], 0)
        self.assertFalse(report["ray"] or report["sqlalchemy"])
        self.assertTrue(out.startswith("retirement:"), out[:200])
        order = [
            out.index(text)
            for text in (
                "retirement:",
                "  state:     retired",
                f"  reason:    {REASON}",
                "  fingerprint:",
                "  files (every file of the store):",
                "  references, when it was retired:",
                "sidecar:",
                "runs naming this store",
            )
        ]
        self.assertEqual(order, sorted(order))

        # and a second retirement is refused
        self.assertIn(
            "already retired",
            self.refused("store", "retire", primary, "--reason", REASON, *self.roots()),
        )

    def test_a_completion_and_show_on_an_incomplete_retirement(self):
        primary = self.a_full_store()
        with mock.patch("os.unlink", _fail_on_call(os.unlink, 2)):
            with self.assertRaises(RuntimeError):
                self.retire(primary)
        report, out, _ = self.command(
            "store", "show", primary, "--runs-root", self.root
        )
        self.assertEqual(report["code"], 1)
        self.assertIn("  state:     retiring", out)
        self.assertIn("is incomplete", out)
        report, out, _ = self.command(
            "store", "retire", primary, "--reason", REASON, *self.roots()
        )
        self.assertEqual(report["code"], 0)
        self.assertIn("completing a retirement that was interrupted", out)
        self.assertEqual(stores.read_sidecar(primary).problems, [])

    def test_a_failure_exits_one(self):
        primary = self.a_full_store()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(
            ShardedPool,
            "delete_store",
            side_effect=OSError(errno.EIO, "injected failure"),
        ):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(
                    ["store", "retire", str(primary), "--reason", REASON, *self.roots()]
                )
        self.assertEqual(code, 1)
        self.assertIn("!! retire of store", err.getvalue())
        self.assertIn('failed at step "delete the store\'s files"', err.getvalue())
        self.assertIncomplete(primary, self.store_files(primary))

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
