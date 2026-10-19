"""A build killed outright is resumed by `end`, then the resume (rrk-v020-adoption prompt 02).

Under `runregistrykit` v0.2.0 a run killed by `SIGKILL` never records its end, so its status says
`running` for ever, and `begin` on its store is refused until a person has ended it with
`python -m runregistrykit end`. This driver's `register` (`RunDrivers/scoped_pipeline_run.py`)
neither catches that refusal nor works around it (campaign README §6.2, C3). These tests pin that,
so that a later change to `register` cannot lose the refusal unnoticed:

  1. the refusal reaches the caller, names the dead run, and leaves the root and the process as
     they were -- `register` takes over nothing before `begin` raises;
  2. `end` records the dead run `lost`, keeps every other field, and the second registration on
     the same store is then accepted;
  3. `end` cannot end a live run, which is what `CLAUDE.md`'s rule 5 leans on.

The dead run is made by a real child process, built as RunRegistryKit's own `test_end` builds one
(this repository's installed `runregistrykit`, not the package's test helpers), and sent `SIGKILL`.
Every child is reaped by a cleanup registered before its first read, so none outlives its test.
No Ray, no datastore: the database is a path that is never created. Every run is in a temporary
runs root, and nothing under `var/` is read or written.
"""

import argparse
import os
import signal
import subprocess
import sys
import unittest

import runregistrykit

from RunDrivers.tests.registry_fixtures import REPO_ROOT, StoreTestCase
from RunDrivers.tests.test_pipeline_adoption import DRIVER, StringWriter, load_driver

# a child that begins a run, heartbeats once, says where its run is, and waits to be killed
CHILD = """\
import sys, time

import runregistrykit

run = runregistrykit.begin(
    campaign="example",
    prompt="01",
    slug="hard-killed",
    purpose="a run that is killed outright",
    root=sys.argv[1],
    client_root=sys.argv[2],
    argv=["a-child"],
    results=sys.argv[3],
)
run.heartbeat(units_done=1, stage="working")
print(run.path, flush=True)
time.sleep(600)
"""

# what `end` (and any status write) changes: the rest of the status is kept as it was
WRITTEN = ("state", "exit_code", "ended", "heartbeat")


def _reap(child) -> None:
    """Kill ``child`` if it is still running, and wait for it, whatever has happened."""
    if child.poll() is None:
        child.kill()
    child.wait(timeout=60)
    for pipe in (child.stdout, child.stderr):
        if pipe is not None:
            pipe.close()


def read_bytes(path) -> bytes:
    with open(os.path.join(path, "status.json"), "rb") as handle:
        return handle.read()


class HardKilledPredecessorTestCase(StoreTestCase):
    def setUp(self):
        if not os.path.exists(DRIVER):
            self.skipTest(f"{DRIVER} is not in this tree")
        super().setUp()
        self.driver = load_driver()
        self.database = os.path.join(self.root, "store.sqlite")  # never created

    def a_child_run(self):
        """Begin a run in a child process, which heartbeats and waits. Returns (the child, the
        run's path, its status's bytes)."""
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                CHILD,
                self.root,
                str(REPO_ROOT),
                self.database,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.addCleanup(_reap, child)
        path = child.stdout.readline().strip()
        if not path:
            child.kill()
            child.wait(timeout=60)
            self.fail(f"the child began no run: {child.stderr.read()}")
        return child, path, read_bytes(path)

    def a_hard_killed_run(self):
        """A run begun by a child, which is then sent `SIGKILL` and reaped. Returns (the run's
        path, its status's bytes before the kill)."""
        child, path, before = self.a_child_run()
        child.kill()
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)
        self.assertEqual(read_bytes(path), before, "the kill changed the status")
        status = runregistrykit.read_json(os.path.join(path, "status.json"))
        self.assertEqual(status["state"], "running")
        self.assertIsNotNone(status.get("host"))
        return path, before

    def register(self):
        """Call the driver's `register` against the temporary runs root, and put back everything
        it takes over: both output streams and the two signal handlers.

        Returns (the `Run` or the exception it raised, what it had taken over at that moment).
        The observation is made *before* the restoration: after it, every assertion about what
        `register` took over would pass whatever `register` did.
        """
        args = argparse.Namespace(
            campaign="c",
            prompt="02",
            register="resume",
            purpose="a resume, registered in a temporary directory",
            models=None,
            k_min=1e5,
            k_max=1e6,
            cpus=2,
        )
        numbers = (signal.SIGTERM, signal.SIGINT)
        handlers = {number: signal.getsignal(number) for number in numbers}
        streams = (sys.stdout, sys.stderr)
        installed = (StringWriter(), StringWriter())
        sys.stdout, sys.stderr = installed
        try:
            try:
                outcome = self.driver.register(
                    args,
                    ["--database", self.database],
                    [1e5, 1e6],
                    self.database,
                    runs_root=self.root,
                )
            except RuntimeError as exc:
                outcome = exc
            taken = {
                "stdout": sys.stdout is not installed[0],
                "stderr": sys.stderr is not installed[1],
                "handlers": any(
                    signal.getsignal(number) is not handlers[number]
                    for number in numbers
                ),
            }
            return outcome, taken
        finally:
            for stream in (sys.stdout, sys.stderr):
                close_copy = getattr(stream, "close_copy", None)
                if close_copy is not None:
                    close_copy()
            sys.stdout, sys.stderr = streams
            for number, handler in handlers.items():
                signal.signal(number, handler)

    def end(self, run_id, reason="the test ended it"):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "runregistrykit",
                "end",
                run_id,
                "--reason",
                reason,
                "--root",
                self.root,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )


class TestTheRefusal(HardKilledPredecessorTestCase):
    def test_register_meets_the_refusal_and_takes_nothing_over(self):
        path, _ = self.a_hard_killed_run()
        run_id = os.path.basename(path)
        listing = sorted(os.listdir(self.root))
        self.assertEqual(listing, [run_id])

        outcome, taken = self.register()

        self.assertIsInstance(outcome, RuntimeError, "begin was not refused")
        message = str(outcome)
        self.assertIn(run_id, message)
        self.assertIn("stale", message)
        self.assertIn("python -m runregistrykit end", message)
        self.assertTrue(message.endswith("No run directory was created"), message)
        self.assertEqual(sorted(os.listdir(self.root)), listing)
        self.assertEqual(
            taken,
            {"stdout": False, "stderr": False, "handlers": False},
            "register took something over before begin refused",
        )


class TestEndThenTheResume(HardKilledPredecessorTestCase):
    def test_end_records_the_run_lost_and_the_second_registration_is_accepted(self):
        path, before = self.a_hard_killed_run()
        run_id = os.path.basename(path)
        status_before = runregistrykit.read_json(os.path.join(path, "status.json"))
        outcome, _ = self.register()
        self.assertIsInstance(outcome, RuntimeError)

        result = self.end(run_id)
        self.assertEqual(result.returncode, 0, result.stderr)

        status = runregistrykit.read_json(os.path.join(path, "status.json"))
        self.assertEqual(status["state"], "lost")
        self.assertIsNone(status["exit_code"])
        ended = status["ended"]
        self.assertIs(ended["verified"], True)
        self.assertIsNone(ended["not_verified_because"])
        self.assertEqual(ended["evidence"]["host"], "this")
        self.assertIs(ended["evidence"]["pid_answers"], False)
        for key, value in status_before.items():
            if key not in WRITTEN:
                self.assertEqual(status[key], value, f"end changed {key!r}")

        before_second = set(os.listdir(self.root))
        run, taken = self.register()
        self.assertNotIsInstance(run, Exception, str(run))
        made = set(os.listdir(self.root)) - before_second
        self.assertEqual(len(made), 1, "the second registration made no run directory")
        self.assertEqual(os.path.join(self.root, made.pop()), run.path)

    def test_a_second_end_on_the_lost_run_is_refused(self):
        path, _ = self.a_hard_killed_run()
        run_id = os.path.basename(path)
        self.assertEqual(self.end(run_id).returncode, 0)
        ended = read_bytes(path)
        again = self.end(run_id)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(read_bytes(path), ended)


class TestEndCannotEndALiveRun(HardKilledPredecessorTestCase):
    def test_end_is_refused_on_a_run_whose_pid_answers(self):
        child, path, before = self.a_child_run()
        result = self.end(os.path.basename(path))
        self.assertNotEqual(result.returncode, 0, "end ended a live run")
        self.assertIn("kill -0", result.stderr)
        self.assertEqual(read_bytes(path), before, "a refused end wrote the status")
        # --unverified does not override it either
        forced = subprocess.run(
            [
                sys.executable,
                "-m",
                "runregistrykit",
                "end",
                os.path.basename(path),
                "--reason",
                "forced",
                "--unverified",
                "--root",
                self.root,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertNotEqual(forced.returncode, 0)
        self.assertEqual(read_bytes(path), before)
        child.kill()
        self.assertEqual(child.wait(timeout=60), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
