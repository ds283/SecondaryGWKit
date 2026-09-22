"""The pipeline driver's side of the registry: what it reports, and what it refuses to claim.

`prompts/run-registry` prompt 02. The subject is `docs/gktk-remedial/scoped_pipeline_run.py`
with `--register`, and the two things that can go wrong with it (prompt §3):

  * a heartbeat from a Ray-parallel job that means less than it looks like it means, and
  * a progress denominator that does not exist and gets invented anyway.

Nothing here starts Ray or opens a datastore; every run directory is in a temporary one. Two
cases run the driver as a subprocess, and both exit before it reaches `import ray`. Two read
files under `var/`, read-only, and **skip** when they are absent, because `var/` is gitignored
and is not there on a fresh clone: the hand-written A3 manifest, and the default run root, which
is only listed to assert that nothing was added to it.
"""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import RunRegistry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRIVER = os.path.join(REPO_ROOT, "docs", "gktk-remedial", "scoped_pipeline_run.py")

# Six lines in the vocabulary `RayWorkPool` prints, in the order it prints them: the banner from
# its constructor (:166), two progress notifications (:548) and the completion pair (:566). They
# are transcribed from `var/runs/a3-pilot/run.out`, the stdout of the 10 h 33 m A3 baseline run,
# rather than read from it: that file is gitignored evidence, and a test that needs it would not
# run on a fresh clone. The whole of that file, fed through `StageTracker`, ends at the last
# stage asserted below.
A3_OUTPUT = """
** CALCULATE QUADRATIC SOURCE INTEGRALS
   -- 2026-09-21 02:14:03+0100 (5h 11m running): 4/13 work items remaining = 69.23% complete
      inflight: 0 lookup, 4 compute, 0 store | completed: 9 lookup, 9 compute, 9 store
   -- 2026-09-21 07:36:58+0100 (10h 33m running): 1/13 work items remaining = 92.31% complete
      inflight: 0 lookup, 1 compute, 0 store | completed: 12 lookup, 12 compute, 12 store
"""

HAND_WRITTEN = (
    "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)"
)


def load_driver():
    """Import the driver by path. It imports numpy and argparse at module scope and nothing
    else; `ray`, `config.model_list` and `main.py` are all reached from inside `main()`.
    """
    spec = importlib.util.spec_from_file_location("scoped_pipeline_run", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineAdoptionTestCase(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(DRIVER):
            self.skipTest(f"{DRIVER} is not in this tree")
        self.driver = load_driver()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def a_run(self, **kwargs):
        kwargs.setdefault("purpose", "a test run, in a temporary directory")
        kwargs.setdefault("argv", ["unittest"])
        return RunRegistry.begin(
            campaign="run-registry",
            prompt="02",
            slug="pipeline",
            root=self.root,
            **kwargs,
        )


class TestStage(PipelineAdoptionTestCase):
    """The progress a pipeline run can honestly report is a sentence, not a fraction."""

    def test_the_stage_is_the_line_that_had_to_be_reconstructed_by_hand(self):
        tracker = self.driver.StageTracker()
        for line in A3_OUTPUT.splitlines():
            tracker.feed(line)
        self.assertEqual(tracker.stage, HAND_WRITTEN)

    def test_the_stage_is_the_hand_written_manifests_own_field(self):
        """`var/datastores/handover-A3-baseline-lambdacdm.manifest.json` carried this string as
        `stage_reached`, typed in by a human reading a stdout file after the run was killed.
        That is the field, and this is the evidence that it is the right one."""
        manifest = os.path.join(
            REPO_ROOT,
            "var",
            "datastores",
            "handover-A3-baseline-lambdacdm.manifest.json",
        )
        if not os.path.exists(manifest):  # gitignored; absent on a fresh clone
            self.skipTest("the hand-written A3 manifest is not in this tree")
        with open(manifest, "r") as handle:
            recorded = json.load(handle)["run_history"][0]["stage_reached"]
        tracker = self.driver.StageTracker()
        for line in A3_OUTPUT.splitlines():
            tracker.feed(line)
        for fragment in (
            "CALCULATE QUADRATIC SOURCE INTEGRALS",
            "92.31%",
            "1 of 13 work items remaining",
        ):
            self.assertIn(fragment, recorded, "the hand-written manifest has moved")
            self.assertIn(fragment, tracker.stage)

    def test_a_banner_alone_is_a_stage_and_a_completion_closes_it(self):
        tracker = self.driver.StageTracker()
        self.assertTrue(
            tracker.feed("\n** BUILDING BESSEL FUNCTION SPLINES FOR LambdaCDM")
        )
        self.assertEqual(
            tracker.stage, "BUILDING BESSEL FUNCTION SPLINES FOR LambdaCDM"
        )
        self.assertFalse(tracker.feed("      inflight: 0 lookup | completed: 0 lookup"))
        self.assertTrue(tracker.feed("   -- ALL WORK ITEMS COMPLETE in time 4.3s"))
        self.assertEqual(
            tracker.stage,
            "BUILDING BESSEL FUNCTION SPLINES FOR LambdaCDM, all work items complete",
        )

    def test_no_denominator_is_invented(self):
        """Prompt §3 item 3. The pipeline has no total, so `units_done`/`units_total` stay empty
        and the lister prints `-` rather than a number that looks measured."""
        run = self.a_run()
        run.heartbeat(stage="CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31%")
        status = run.status()
        self.assertIsNone(status["units_total"], "a denominator was invented")
        self.assertEqual(
            status["units_done"], 0, "the pipeline records no units at all"
        )
        entry = RunRegistry.list_runs(root=self.root)[0]
        self.assertIsNone(entry["units_total"])
        self.assertEqual(entry["stage"], "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31%")
        done = subprocess.run(
            [sys.executable, "-m", "RunRegistry", "list", "--root", self.root],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300,
        )
        self.assertRegex(done.stdout, r"running\s+-\s")

    def test_the_lister_prints_the_stage(self):
        run = self.a_run()
        run.heartbeat(stage=HAND_WRITTEN)
        done = subprocess.run(
            [sys.executable, "-m", "RunRegistry", "list", "--root", self.root],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("92.31%", done.stdout)
        self.assertIn("CALCULATE QUADRATIC SOURCE INTEGRALS", done.stdout)


class TestHeartbeatHonesty(PipelineAdoptionTestCase):
    def test_the_manifest_says_what_the_heartbeat_does_not_assert(self):
        """Prompt §3 item 2: if the heartbeat cannot mean "an actor is progressing", the manifest
        must say so rather than implying a guarantee it does not give."""
        run = self.a_run(
            heartbeat_means=self.driver.HEARTBEAT_MEANS.format(
                interval=self.driver.BEAT_MIN_INTERVAL
            ),
            scope="2 log-spaced k over 1e5-1e6 /Mpc, LambdaCDM, 2 cpus",
        )
        said = RunRegistry.read_json(run.manifest_path)["heartbeat_means"]
        self.assertIn("does NOT mean any Ray actor is progressing", said)
        self.assertIn("stale", said)
        self.assertEqual(
            RunRegistry.read_json(run.manifest_path)["scope"],
            "2 log-spaced k over 1e5-1e6 /Mpc, LambdaCDM, 2 cpus",
        )

    def test_the_stream_passes_everything_through_and_beats_on_a_new_stage(self):
        """Every byte reaches the real stream unaltered — a registry that eats a line of a ten
        hour run's output has cost more than it saved — and the run directory gets a copy.
        """
        run = self.a_run()
        real = StringWriter()
        stream = self.driver.RegisteredStream(
            real,
            run,
            self.driver.StageTracker(),
            copy_to=run.stdout_path,
            interval=10_000.0,  # only a *change of stage* may beat within this test
        )
        self.addCleanup(stream.close_copy)
        for line in A3_OUTPUT.splitlines():
            print(line, file=stream)
        stream.flush()

        self.assertEqual(real.text(), A3_OUTPUT)
        with open(run.stdout_path, "r") as handle:
            self.assertEqual(handle.read(), A3_OUTPUT)
        self.assertEqual(run.status()["stage"], HAND_WRITTEN)

    def test_a_failing_tee_costs_neither_output_nor_the_heartbeat(self):
        """The copy is a convenience; the output and the status are not. A tee that has died
        takes neither with it."""
        run = self.a_run()
        real = StringWriter()
        stream = self.driver.RegisteredStream(
            real, run, self.driver.StageTracker(), copy_to=run.stdout_path
        )
        stream.close_copy()
        stream._copy = open(os.devnull, "r")  # a handle that cannot be written to
        self.addCleanup(stream._copy.close)

        print("** CALCULATE QUADRATIC SOURCE INTEGRALS", file=stream)
        self.assertEqual(real.text(), "** CALCULATE QUADRATIC SOURCE INTEGRALS\n")
        self.assertEqual(run.status()["stage"], "CALCULATE QUADRATIC SOURCE INTEGRALS")

    def test_attributes_are_the_real_streams(self):
        run = self.a_run()
        stream = self.driver.RegisteredStream(
            sys.stdout, run, self.driver.StageTracker()
        )
        self.assertEqual(stream.fileno(), sys.stdout.fileno())
        self.assertEqual(stream.encoding, sys.stdout.encoding)


class TestTerminalState(PipelineAdoptionTestCase):
    """A deliberate interruption is not a failure, and the registry must not call it one."""

    def test_a_sigterm_that_arrives_as_rays_systemexit_is_killed_not_failed(self):
        """`ray.init` replaces our SIGTERM handler with `sys.exit(signum)`
        (`ray/_private/worker.py:1498`), so after Ray is up a SIGTERM reaches the driver as
        `SystemExit(15)` — observed in this prompt's demonstration."""
        self.assertEqual(self.driver.terminal_state(int(signal.SIGTERM)), "killed")

    def test_the_ordinary_exits_are_unchanged(self):
        self.assertEqual(self.driver.terminal_state(None), "done")
        self.assertEqual(self.driver.terminal_state(0), "done")
        self.assertEqual(self.driver.terminal_state(1), "failed")
        self.assertEqual(self.driver.terminal_state("a message"), "failed")

    def test_every_state_it_can_return_is_one_the_registry_accepts(self):
        for code in (None, 0, 1, int(signal.SIGTERM), "a message"):
            self.assertIn(self.driver.terminal_state(code), RunRegistry.TERMINAL_STATES)


class TestOptIn(PipelineAdoptionTestCase):
    """`--register` is opt-in, and it never runs before the checks that protect a datastore."""

    def driver_run(self, *argv):
        return subprocess.run(
            [sys.executable, DRIVER, *argv],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            timeout=300,
        )

    def test_register_without_a_purpose_is_refused(self):
        done = self.driver_run(
            "--register", "smoke", "--k-min", "1e5", "--k-max", "1e6", "--k-count", "2"
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--purpose", done.stderr)

    def test_a_refused_datastore_leaves_no_run_behind(self):
        """The existing-datastore guard runs *before* `begin()`, so a run that is never going to
        start does not leave a manifest claiming it did."""
        root = RunRegistry.DEFAULT_ROOT
        before = sorted(os.listdir(root)) if os.path.isdir(root) else None
        done = self.driver_run(
            "--register",
            "smoke",
            "--purpose",
            "a run that must not happen",
            "--k-min",
            "1e5",
            "--k-max",
            "1e6",
            "--k-count",
            "2",
            "--",
            "--database",
            os.path.join(self.root, "no", "such", "directory", "store.sqlite"),
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("is not a directory", done.stderr)
        after = sorted(os.listdir(root)) if os.path.isdir(root) else None
        self.assertEqual(
            after, before, "a run that never started left a directory behind"
        )


class TestTheDatastoreIsNotALedger(PipelineAdoptionTestCase):
    """Prompt 03. This driver's results live in a SQLite datastore, which the registry names and
    never opens; it has no unit ledger and records no units. The registration is exercised for
    real — the manifest on disk is the assertion — but in a temporary run root, with no Ray, no
    datastore and nothing under `var/` touched.
    """

    def register(self, database):
        """Run the driver's own registration block against a temporary run root, and put back
        everything it takes over: both output streams and the two signal handlers."""
        args = argparse.Namespace(
            campaign="run-registry",
            prompt="03",
            register="pipeline",
            purpose="a registration, in a temporary directory, that runs no pipeline",
            models=None,
            k_min=1e5,
            k_max=1e6,
            cpus=2,
        )
        handlers = {
            number: signal.getsignal(number)
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = StringWriter(), StringWriter()
        try:
            with mock.patch.object(RunRegistry, "DEFAULT_ROOT", self.root):
                return self.driver.register(
                    args, ["--database", database], [1e5, 1e6], database
                )
        finally:
            for stream in (sys.stdout, sys.stderr):
                close_copy = getattr(stream, "close_copy", None)
                if close_copy is not None:
                    close_copy()
            sys.stdout, sys.stderr = streams
            for number, handler in handlers.items():
                signal.signal(number, handler)

    def test_the_datastore_is_registered_as_results_and_not_as_a_ledger(self):
        database = os.path.join(self.root, "store.sqlite")
        run = self.register(database)

        manifest = RunRegistry.read_json(run.manifest_path)
        self.assertEqual(manifest["results"], database)
        self.assertIsNone(
            manifest["checkpoint"], "the datastore is declared as a unit ledger"
        )
        for field in ("purpose", "scope", "heartbeat_means", "script", "git_head"):
            self.assertTrue(
                manifest[field],
                f"a stranger reading this manifest learns nothing from {field}",
            )

        # The script that copies this pattern and then adds units is told what to do instead,
        # and the datastore is not created, opened or written to on the way.
        with self.assertRaises(ValueError) as refused:
            run.record("a unit this pipeline does not have")
        self.assertIn("checkpoint=True", str(refused.exception))
        self.assertIn(database, str(refused.exception))
        self.assertFalse(os.path.exists(database))
        self.assertEqual(run.known(), {})


class StringWriter:
    """A file-like stand-in for `sys.stdout`."""

    def __init__(self):
        self._parts = []

    def write(self, text):
        self._parts.append(text)
        return len(text)

    def flush(self):
        pass

    def text(self):
        return "".join(self._parts)


if __name__ == "__main__":
    unittest.main()
