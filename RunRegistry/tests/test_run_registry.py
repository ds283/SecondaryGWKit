"""Tests for `RunRegistry`. No Ray, no datastore, no network; every run directory these build is
in a temporary directory, and nothing under `var/` is read or written.

The first test is the campaign's whole justification in miniature and is the reason the others are
here at all — see `TestSelfMatch`.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import RunRegistry

REPO_ROOT = RunRegistry.REPO_ROOT

# A liveness check performed from a process whose own command line contains the target's name.
# `--waiting-for <token>` is argv[1:3] so that it survives any command-line truncation `ps` may
# apply, and the script's own file name carries the token too — exactly the shape of the poll loop
# that could never exit (README §0 item 3).
CHECKER = '''\
import json, os, subprocess, sys

sys.path.insert(0, {repo!r})
import RunRegistry

token, run_dir = sys.argv[2], sys.argv[3]
status = RunRegistry.read_json(os.path.join(run_dir, "status.json"))


def _ps(args):
    for extra in (["-ww"], []):
        try:
            done = subprocess.run(
                ["ps"] + extra + args, capture_output=True, text=True, timeout=60
            )
        except OSError:
            return ""
        if done.returncode == 0:
            return done.stdout
    return ""


def own_command_line():
    return _ps(["-p", str(os.getpid()), "-o", "command="]).strip()


def pattern_match_pids(pattern):
    """What `pgrep -f <name>` answers: the naive implementation, kept here as the oracle that
    proves this test can bite rather than as anything the registry may use."""
    try:
        done = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=60
        )
        if done.returncode in (0, 1):
            return [int(p) for p in done.stdout.split()]
    except (OSError, ValueError):
        pass
    pids = []
    for line in _ps(["-axo", "pid=,command="]).splitlines():
        head, _, rest = line.strip().partition(" ")
        if head.isdigit() and pattern in rest:
            pids.append(int(head))
    return pids


print(
    json.dumps(
        {{
            "self_pid": os.getpid(),
            "self_command_line": own_command_line(),
            "pattern_match_pids": pattern_match_pids(token),
            "recorded_pid": status.get("pid"),
            "verdict": RunRegistry.liveness(status, stale_after=3600.0),
        }}
    )
)
'''

# A second poll loop waiting for the same job, doing nothing but existing with the target's name
# in its command line. There were nineteen of these.
SIBLING = """\
import sys, time

time.sleep(float(sys.argv[3]))
"""

# Begin a run and die in the first instant, without unwinding: the manifest must survive it.
CRASHER = """\
import os, sys

sys.path.insert(0, {repo!r})
import RunRegistry

RunRegistry.begin(
    campaign="run-registry",
    prompt="01",
    slug="crash-in-the-first-second",
    purpose="a job that dies before it does anything",
    root={root!r},
    argv=["crasher"],
)
os._exit(1)
"""


class RegistryTestCase(unittest.TestCase):
    """A temporary `var/runs/` for each test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def begin(self, slug="unit-test", **kwargs):
        kwargs.setdefault("purpose", "a test run, in a temporary directory")
        kwargs.setdefault("argv", ["unittest", slug])
        return RunRegistry.begin(
            campaign="run-registry", prompt="01", slug=slug, root=self.root, **kwargs
        )

    def a_dead_pid(self):
        """The pid of a process that has exited and been reaped."""
        corpse = subprocess.Popen([sys.executable, "-c", "pass"])
        corpse.wait()
        self.assertFalse(
            RunRegistry.pid_alive(corpse.pid),
            "the corpse's pid still answers kill -0; the test cannot mean anything",
        )
        return corpse.pid

    def a_dead_run(self, slug="unit-test"):
        """A run that says `running`, with a **fresh** heartbeat and a pid that is gone — so that
        only `kill -0` can tell it is over."""
        dead = self.a_dead_pid()
        run = self.begin(slug)
        run.heartbeat(pid=dead)
        return run, dead


class TestSelfMatch(RegistryTestCase):
    """The regression that this whole campaign is for.

    `pgrep -f <script-name>`, run from a poll loop whose own command line contains that name,
    matches a poll loop rather than the job. Nineteen such loops accumulated in one session and
    none of them could ever exit; fifteen were waiting on a job that had already been killed
    (`prompts/run-registry/README.md` §0 item 3).

    The cohort, not the single process, is what makes the condition unsatisfiable, and this test
    reproduces it as such: one process performs the check while a second waits for the same job,
    both carrying the target's name. On macOS `pgrep` excludes itself *and its ancestors* unless
    `-a` is given, so a lone poller cannot match itself there — with nineteen of them each matched
    the other eighteen, and the more there are the more certainly none can exit. The checking
    process is named after the target as well, so the property the prompt states is the one under
    test whichever platform's exclusion rule applies.
    """

    def test_a_poller_named_after_the_target_does_not_report_it_alive(self):
        run, dead_pid = self.a_dead_run(slug="selfmatch")
        token = run.id

        sibling_script = os.path.join(self.root, f"poll-{token}.py")
        with open(sibling_script, "w") as handle:
            handle.write(SIBLING)
        sibling = subprocess.Popen(
            [sys.executable, sibling_script, "--waiting-for", token, "120"]
        )
        self.addCleanup(sibling.wait)
        self.addCleanup(sibling.kill)

        checker = os.path.join(self.root, f"watch-{token}.py")
        with open(checker, "w") as handle:
            handle.write(CHECKER.format(repo=REPO_ROOT))
        done = subprocess.run(
            [sys.executable, checker, "--waiting-for", token, run.path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(done.stdout)

        # The trap is armed. If any of these fails the test is vacuous, and says so rather than
        # passing: the checking process's own command line must really carry the target's name,
        # a command-line pattern match must really find a live process, and that process must
        # really be another poller rather than the job.
        self.assertIn(
            token,
            result["self_command_line"],
            "the checking process's command line does not contain the target's name, so "
            "nothing here exercises the self-match failure",
        )
        self.assertIn(
            sibling.pid,
            result["pattern_match_pids"],
            "pgrep -f <target> did not match the other poller, so the naive implementation "
            "would not have been fooled and this test proves nothing",
        )
        self.assertNotIn(
            dead_pid,
            result["pattern_match_pids"],
            "the target's own pid is still running; it was supposed to be dead",
        )
        self.assertNotEqual(result["self_pid"], dead_pid)

        # And the registry is not fooled: `kill -0` on the recorded pid, never a pattern.
        self.assertEqual(result["recorded_pid"], dead_pid)
        self.assertEqual(
            result["verdict"],
            "stale",
            "the registry reported a dead run as alive from a process named after it, with "
            "another poller for the same job alive beside it",
        )


class TestManifest(RegistryTestCase):
    def test_manifest_is_on_disk_before_the_work_starts(self):
        script = os.path.join(self.root, "crasher.py")
        with open(script, "w") as handle:
            handle.write(CRASHER.format(repo=REPO_ROOT, root=self.root))
        done = subprocess.run(
            [sys.executable, script], capture_output=True, text=True, timeout=300
        )
        self.assertEqual(done.returncode, 1, done.stderr)

        directories = [
            name
            for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]
        self.assertEqual(len(directories), 1, directories)
        path = os.path.join(self.root, directories[0])
        manifest = RunRegistry.read_json(os.path.join(path, "manifest.json"))
        self.assertIsNotNone(manifest, "the crash left no manifest behind")
        self.assertEqual(manifest["purpose"], "a job that dies before it does anything")
        self.assertEqual(manifest["argv"], ["crasher"])
        status = RunRegistry.read_json(os.path.join(path, "status.json"))
        self.assertEqual(status["state"], "running")

    def test_manifest_is_not_mutated_afterwards(self):
        run = self.begin(script=__file__, checkpoint=True, expected_units=2)
        with open(run.manifest_path, "rb") as handle:
            written = handle.read()

        run.record("one", {"value": 1})
        run.heartbeat(units_done=1, units_total=7, pid=os.getpid())
        run.record("two", {"value": 2})
        run.finish("done", exit_code=0)

        with open(run.manifest_path, "rb") as handle:
            self.assertEqual(handle.read(), written, "the manifest moved under us")
        self.assertEqual(run.status()["state"], "done")
        self.assertEqual(run.status()["exit_code"], 0)
        self.assertEqual(run.status()["units_total"], 7)

    def test_a_second_begin_will_not_overwrite_a_manifest(self):
        when = time.time()
        self.begin(when=when)
        with self.assertRaises(FileExistsError):
            self.begin(when=when)

    def test_manifest_carries_the_provenance_triple(self):
        run = self.begin(script=__file__)
        self.assertEqual(
            run.manifest["script_sha256"], RunRegistry.script_sha256(__file__)
        )
        self.assertEqual(len(run.manifest["git_head"]), 40)
        self.assertIn(run.manifest["git_dirty"], (True, False))


class TestCheckpoint(RegistryTestCase):
    def test_round_trip_and_resume_skips_what_is_known(self):
        run = self.begin(
            script=__file__, checkpoint=True, unit="cell", expected_units=3
        )
        for unit in ("a", "b"):
            run.record(unit, {"answer": unit.upper()})

        known = run.known()
        self.assertEqual(sorted(known), ["a", "b"])
        self.assertEqual(known["a"]["data"], {"answer": "A"})
        self.assertEqual(known["a"]["git_head"], run.manifest["git_head"])
        self.assertEqual(run.status()["units_done"], 2)

        computed = [unit for unit in ("a", "b", "c") if unit not in known]
        self.assertEqual(computed, ["c"], "a resume recomputed a unit it already held")

    def test_a_record_from_another_script_is_discarded_and_never_re_stamped(self):
        run = self.begin(script=__file__, checkpoint=True)
        run.record("mine", {"kept": True})
        foreign = {
            "unit": "theirs",
            "recorded": RunRegistry.now_iso(),
            "script_sha256": "0" * 64,
            "git_head": "deadbeef",
            "git_dirty": False,
            "data": {"kept": False},
        }
        with open(run.checkpoint_path, "a") as handle:
            handle.write(json.dumps(foreign) + "\n")
        with open(run.checkpoint_path, "rb") as handle:
            before = handle.read()

        notice = StringWriter()
        known = run.known(notice=notice)

        # Discarded, which is the documented policy, and announced.
        self.assertEqual(sorted(known), ["mine"])
        self.assertIn("discarding 1 record", notice.text())
        # Never re-stamped: the file on disk is untouched, foreign stamp and all.
        with open(run.checkpoint_path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertIn("0" * 64, before.decode())

    def test_a_record_shadowed_by_a_foreign_one_is_dropped_not_blended(self):
        run = self.begin(script=__file__, checkpoint=True)
        run.record("cell", {"generation": 1})
        with open(run.checkpoint_path, "a") as handle:
            handle.write(
                json.dumps({"unit": "cell", "script_sha256": "f" * 64, "data": None})
                + "\n"
            )
        self.assertEqual(run.known(notice=None), {})

    def test_a_torn_final_line_is_tolerated(self):
        run = self.begin(script=__file__, checkpoint=True)
        run.record("a", {"value": 1})
        run.record("b", {"value": 2})
        with open(run.checkpoint_path, "r") as handle:
            text = handle.read()
        with open(run.checkpoint_path, "w") as handle:  # a kill mid-append
            handle.write(text + '{"unit": "c", "data": {"val')

        known = run.known(notice=None)
        self.assertEqual(sorted(known), ["a", "b"])


class TestStatus(RegistryTestCase):
    def test_status_is_replaced_atomically_from_the_same_directory(self):
        run = self.begin()
        with mock.patch("os.replace", wraps=os.replace) as replace:
            run.heartbeat(units_done=1)
        replace.assert_called_once()
        source, destination = replace.call_args[0]
        self.assertEqual(destination, run.status_path)
        self.assertTrue(source.endswith(".tmp"))
        self.assertEqual(
            os.path.dirname(source),
            os.path.dirname(destination),
            "the temporary file is not in the destination's directory, so the replace is not "
            "guaranteed to be a rename",
        )
        self.assertFalse([n for n in os.listdir(run.path) if n.endswith(".tmp")])

    def test_a_reader_never_sees_a_half_written_status(self):
        run = self.begin()
        stop = threading.Event()
        torn, reads = [], []

        def reader():
            while not stop.is_set():
                try:
                    with open(run.status_path, "rb") as handle:
                        payload = handle.read()
                except OSError as exc:
                    torn.append(repr(exc))
                    continue
                try:
                    reads.append(json.loads(payload)["units_done"])
                except (ValueError, KeyError, TypeError):
                    torn.append(payload[:80])

        watcher = threading.Thread(target=reader, daemon=True)
        watcher.start()
        try:
            for index in range(400):
                run.heartbeat(units_done=index)
        finally:
            stop.set()
            watcher.join(timeout=30)

        self.assertEqual(torn, [], "a reader saw a status file mid-write")
        self.assertGreater(
            len(reads), 50, "the reader hardly ran; the test proves little"
        )
        self.assertTrue(set(reads) <= set(range(400)))


class TestLiveness(RegistryTestCase):
    def test_a_run_whose_process_is_gone_is_stale_not_alive(self):
        run, dead = self.a_dead_run()
        status = run.status()
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["pid"], dead)
        # The heartbeat is fresh, so only kill -0 can tell, which is the point.
        self.assertLess(RunRegistry.heartbeat_age(status), 60.0)
        self.assertEqual(RunRegistry.liveness(status, stale_after=3600.0), "stale")

    def test_a_live_process_with_an_old_heartbeat_is_stale(self):
        run = self.begin()  # pid is this process, which is certainly alive
        status = run.status()
        self.assertTrue(RunRegistry.pid_alive(status["pid"]))
        self.assertEqual(RunRegistry.liveness(status, stale_after=1e9), "alive")
        self.assertEqual(RunRegistry.liveness(status, stale_after=-1.0), "stale")

    def test_a_finished_run_is_finished_whatever_its_pid_says(self):
        run = self.begin()
        run.finish("failed", exit_code=3)
        self.assertEqual(
            RunRegistry.liveness(run.status(), stale_after=1e9), "finished"
        )
        with self.assertRaises(ValueError):
            run.finish("nearly")


class TestLister(RegistryTestCase):
    def test_a_directory_with_no_manifest_is_reported_not_skipped(self):
        run = self.begin(slug="registered")
        run.finish("done", exit_code=0)
        os.makedirs(os.path.join(self.root, "a3-pilot"))
        with open(os.path.join(self.root, "a3-pilot", "run.out"), "w") as handle:
            handle.write("a run that predates the registry\n")
        with open(os.path.join(self.root, "loose-file.jsonl"), "w") as handle:
            handle.write("{}\n")

        entries = RunRegistry.list_runs(root=self.root)
        by_id = {entry["id"]: entry for entry in entries}
        self.assertEqual(sorted(by_id), sorted(["a3-pilot", run.id]))
        self.assertFalse(by_id["a3-pilot"]["has_manifest"])
        self.assertIsNone(by_id["a3-pilot"]["state"])
        self.assertEqual(by_id["a3-pilot"]["liveness"], "unknown")
        self.assertTrue(by_id[run.id]["has_manifest"])
        self.assertEqual(by_id[run.id]["liveness"], "finished")

    def test_the_command_line_prints_every_run_and_shouts_about_stale_ones(self):
        self.a_dead_run(slug="crashed")
        os.makedirs(os.path.join(self.root, "a3-pilot"))
        done = subprocess.run(
            [sys.executable, "-m", "RunRegistry", "list", "--root", self.root],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("a3-pilot", done.stdout)
        self.assertIn("no manifest", done.stdout)
        self.assertIn("!!", done.stdout)
        self.assertIn("say they are running and are not", done.stderr)


class StringWriter:
    """A file-like stand-in for `sys.stderr`."""

    def __init__(self):
        self._parts = []

    def write(self, text):
        self._parts.append(text)

    def flush(self):
        pass

    def text(self):
        return "".join(self._parts)


if __name__ == "__main__":
    unittest.main()
