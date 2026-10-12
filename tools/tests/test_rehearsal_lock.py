"""
Tests for ``tools/rehearsal_lock.py``, SGK's side of the shared rehearsal lock
(``runregistry-extraction`` prompt 02a; the contract is StochasticInstantons' brief at ``d6d550b``).

Every test runs with ``HOME`` patched to a temporary directory, as well as passing an explicit lock
directory, so that a forgotten argument cannot reach the real ``~/ScienceData``. ``ps`` is replaced
by fixed text through ``rehearsal_lock._ps_output`` (empty unless a test says otherwise). No Ray;
nothing under ``var/``.

Each refusal test leaves exactly one reason standing, so that a mutation of one check cannot hide
behind another: another project's owner is tested with a dead or absent pid and no processes; a live
pid with SGK's owner and no processes; processes with SGK's owner and ``pid: -``.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from tools import rehearsal_lock as rl

PAST = "2026-01-01T00:00:00+00:00"

# The owner files SI's and ChamPBH's ``write_owner`` produce (their ``ray_cluster.sh`` at
# ``8e04a7c`` and ``288b385``), as the scripts wrote them under a temporary HOME.
SI_OWNER = """project: StochasticInstantons
purpose: orch 02a dry run
acquired: 2026-10-12T14:19:31+01:00
host: mps026088
ray_temp_dir: /tmp/ray-si
ray_address: 127.0.0.1:6390
pid: -
"""
CHAMPBH_OWNER = """project: ChamPBH
purpose: orch 02a dry run
acquired: 2026-10-12T14:19:40+01:00
host: mps026088
ray_temp_dir: /tmp/ray-cpbh
ray_address: 127.0.0.1:6380
pid: -
"""
# A project that is not SGK but, hypothetically, runs in-process: only the project differs.
OTHER_IN_PROCESS_OWNER = SI_OWNER.replace("/tmp/ray-si", "in-process").replace(
    "127.0.0.1:6390", "in-process"
)
# SGK's project name with a cluster's temp dir: only ray_temp_dir differs.
SGK_WITH_TEMP_DIR_OWNER = rl.owner_text("x", "-", acquired=PAST, host="h").replace(
    "ray_temp_dir: in-process", "ray_temp_dir: /tmp/ray-si"
)


def sgk_owner(pid="-", purpose="test"):
    return rl.owner_text(purpose, pid, acquired=PAST, host="testhost")


def dead_pid():
    """A pid that has just exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


VENV = rl.venv_roots()[0]
RAY = f"{VENV}/lib/python3.12/site-packages/ray"


def ps_line(pid, ppid, command):
    return f"{pid:>6} {ppid:>6} {command}\n"


SGK_PS = "".join(
    [
        ps_line(100, 1, f"{RAY}/core/src/ray/gcs/gcs_server --gcs_server_port=1"),
        ps_line(101, 100, f"{RAY}/core/src/ray/raylet/raylet --raylet_socket_name=x"),
        ps_line(102, 100, f"python -u {RAY}/dashboard/agent.py --node-ip-address=1"),
        ps_line(103, 101, "ray::IDLE"),
    ]
)


class LockTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        patch_env = mock.patch.dict(os.environ, {"HOME": str(self.home)})
        patch_env.start()
        self.addCleanup(patch_env.stop)
        self.ps = ""
        patch_ps = mock.patch.object(rl, "_ps_output", lambda: self.ps)
        patch_ps.start()
        self.addCleanup(patch_ps.stop)
        self.lock = self.tmp / "ScienceData" / ".rehearsal-lock"

    def run_cmd(self, func, *args, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = func(*args, **kwargs)
        return rc, out.getvalue(), err.getvalue()

    def make_lock(self, text):
        """Create the lock directory by hand; ``text=None`` leaves it without an owner file."""
        self.lock.mkdir(parents=True)
        if text is not None:
            (self.lock / "owner").write_bytes(text.encode("utf-8"))

    def snapshot(self):
        if not self.lock.exists():
            return None
        return {p.name: p.read_bytes() for p in sorted(self.lock.iterdir())}

    def assertUntouched(self, before):
        self.assertEqual(self.snapshot(), before)


class TestAcquire(LockTestCase):
    def test_free_lock_is_created_with_the_seven_keys(self):
        rc, _, _ = self.run_cmd(rl.acquire, "a purpose", self.lock)
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(p.name for p in self.lock.iterdir()), ["owner"])
        text = (self.lock / "owner").read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertEqual([l.split(": ")[0] for l in lines], list(rl.OWNER_KEYS))
        owner = rl.parse_owner(text)
        self.assertEqual(owner["project"], "SecondaryGWKit")
        self.assertEqual(owner["purpose"], "a purpose")
        self.assertEqual(owner["ray_temp_dir"], "in-process")
        self.assertEqual(owner["ray_address"], "in-process")
        self.assertEqual(owner["pid"], "-")
        stamp = datetime.fromisoformat(owner["acquired"])
        self.assertIsNotNone(stamp.utcoffset())
        self.assertTrue(owner["host"] and "." not in owner["host"])
        self.assertTrue(text.endswith("\n"))

    def test_parent_is_created_but_never_the_lock_with_exist_ok(self):
        self.assertFalse(self.lock.parent.exists())
        rc, _, _ = self.run_cmd(rl.acquire, "p", self.lock)
        self.assertEqual(rc, 0)
        self.assertTrue(self.lock.parent.is_dir())

    def held_case(self, text):
        self.make_lock(text)
        before = self.snapshot()
        rc, out, err = self.run_cmd(rl.acquire, "new", self.lock)
        self.assertEqual(rc, 3)
        self.assertUntouched(before)
        return out + err

    def test_held_by_sgk_returns_3_and_prints_the_owner(self):
        shown = self.held_case(sgk_owner(purpose="earlier"))
        self.assertIn("purpose: earlier", shown)

    def test_held_by_si_returns_3_and_prints_the_owner(self):
        shown = self.held_case(SI_OWNER)
        self.assertIn("project: StochasticInstantons", shown)
        self.assertIn("ray_address: 127.0.0.1:6390", shown)

    def test_held_with_no_owner_file_returns_3_and_says_so(self):
        shown = self.held_case(None)
        self.assertIn("no owner file", shown)
        self.assertEqual(list(self.lock.iterdir()), [])

    def test_ray_processes_return_4_and_leave_no_lock(self):
        self.ps = SGK_PS
        rc, out, err = self.run_cmd(rl.acquire, "p", self.lock)
        self.assertEqual(rc, 4)
        self.assertFalse(self.lock.exists())
        self.assertIn("gcs_server", err)
        self.assertTrue(self.lock.parent.is_dir())


class TestSetPid(LockTestCase):
    def test_live_pid_changes_pid_and_only_pid(self):
        before_text = sgk_owner()
        self.make_lock(before_text)
        rc, _, _ = self.run_cmd(rl.set_pid, os.getpid(), self.lock)
        self.assertEqual(rc, 0)
        after = (self.lock / "owner").read_text(encoding="utf-8")
        self.assertEqual(after, before_text.replace("pid: -", f"pid: {os.getpid()}"))
        self.assertIn(f"acquired: {PAST}", after)
        self.assertEqual(sorted(p.name for p in self.lock.iterdir()), ["owner"])

    def refused(self, text, pid):
        self.make_lock(text)
        before = self.snapshot()
        rc, _, err = self.run_cmd(rl.set_pid, pid, self.lock)
        self.assertEqual(rc, 1, err)
        self.assertUntouched(before)

    def test_dead_pid_is_refused(self):
        self.refused(sgk_owner(), dead_pid())

    def test_pid_zero_is_refused(self):
        self.refused(sgk_owner(), 0)

    def test_si_lock_is_refused(self):
        self.refused(SI_OWNER, os.getpid())

    def test_champbh_lock_is_refused(self):
        self.refused(CHAMPBH_OWNER, os.getpid())

    def test_other_project_with_in_process_temp_dir_is_refused(self):
        self.refused(OTHER_IN_PROCESS_OWNER, os.getpid())

    def test_sgk_project_with_another_temp_dir_is_refused(self):
        self.refused(SGK_WITH_TEMP_DIR_OWNER, os.getpid())

    def test_missing_owner_is_refused(self):
        self.refused(None, os.getpid())

    def test_missing_lock_is_refused(self):
        rc, _, _ = self.run_cmd(rl.set_pid, os.getpid(), self.lock)
        self.assertEqual(rc, 1)
        self.assertFalse(self.lock.exists())


class TestRelease(LockTestCase):
    def test_pid_dash_with_no_processes_releases(self):
        self.make_lock(sgk_owner("-"))
        (self.lock / "owner.tmp").write_text("leftover")
        rc, _, _ = self.run_cmd(rl.release, self.lock)
        self.assertEqual(rc, 0)
        self.assertFalse(self.lock.exists())
        self.assertTrue(self.lock.parent.is_dir())

    def test_dead_pid_with_no_processes_releases(self):
        self.make_lock(sgk_owner(dead_pid()))
        rc, _, _ = self.run_cmd(rl.release, self.lock)
        self.assertEqual(rc, 0)
        self.assertFalse(self.lock.exists())

    def test_no_lock_returns_0(self):
        rc, out, _ = self.run_cmd(rl.release, self.lock)
        self.assertEqual(rc, 0)
        self.assertIn("no lock to release", out)

    def refused(self, text):
        self.make_lock(text)
        before = self.snapshot()
        rc, out, err = self.run_cmd(rl.release, self.lock)
        self.assertEqual(rc, 1, out + err)
        self.assertUntouched(before)
        return err

    def test_si_owner_is_refused_and_both_sides_printed(self):
        err = self.refused(SI_OWNER)
        self.assertIn("StochasticInstantons", err)
        self.assertIn("SecondaryGWKit", err)

    def test_champbh_owner_is_refused(self):
        self.refused(CHAMPBH_OWNER)

    def test_other_project_with_in_process_temp_dir_is_refused(self):
        self.refused(OTHER_IN_PROCESS_OWNER)

    def test_sgk_project_with_another_temp_dir_is_refused(self):
        self.refused(SGK_WITH_TEMP_DIR_OWNER)

    def test_missing_owner_is_refused(self):
        self.refused(None)

    def test_live_pid_is_refused(self):
        err = self.refused(sgk_owner(os.getpid()))
        self.assertIn(str(os.getpid()), err)

    def test_ray_processes_are_refused(self):
        self.ps = SGK_PS
        err = self.refused(sgk_owner("-"))
        self.assertIn("gcs_server", err)

    def test_ray_processes_are_refused_with_a_dead_pid(self):
        self.ps = SGK_PS
        self.refused(sgk_owner(dead_pid()))


class TestRayProcesses(unittest.TestCase):
    def pids(self, text, **kwargs):
        return [p for p, _, _ in rl.ray_processes(text, **kwargs)]

    def test_finds_gcs_raylet_agent_and_the_workers_below_a_raylet(self):
        text = SGK_PS + ps_line(104, 103, "ray::Actor.method")
        self.assertEqual(self.pids(text, self_pid=9999), [100, 101, 102, 103, 104])

    def test_a_worker_child_of_a_raylet_is_listed_though_it_names_no_path(self):
        found = rl.ray_processes(SGK_PS, self_pid=9999)
        self.assertIn((103, 101, "ray::IDLE"), found)

    def test_skips_other_projects_and_search_commands(self):
        text = "".join(
            [
                ps_line(
                    200,
                    1,
                    "/Users/x/StochasticInstantons/venv/lib/python3.12/site-packages/ray/"
                    "core/src/ray/raylet/raylet --temp-dir=/tmp/ray-si",
                ),
                ps_line(201, 200, "ray::IDLE"),
                ps_line(
                    210,
                    1,
                    "/Users/x/ChamPBH/venv/lib/python3.12/site-packages/ray/core/src/ray/"
                    "gcs/gcs_server --temp-dir=/tmp/ray-cpbh",
                ),
                ps_line(211, 210, "ray::IDLE"),
                ps_line(300, 50, "pgrep -lf gcs_server|raylet|ray::"),
                ps_line(301, 50, "grep ray"),
                ps_line(302, 50, f"{VENV}/bin/python -m tools.rehearsal_lock status"),
                ps_line(
                    303,
                    1,
                    f"{VENV}2/lib/python3.12/site-packages/ray/dashboard/agent.py",
                ),
            ]
        )
        self.assertEqual(self.pids(text, self_pid=9999), [])

    def test_skips_its_own_pid_and_its_ancestors(self):
        text = "".join(
            [
                ps_line(399, 1, f"{RAY}/core/src/ray/raylet/raylet"),
                ps_line(400, 399, f"{RAY}/dashboard/agent.py"),
                ps_line(401, 400, "ray::IDLE"),
                ps_line(402, 399, "ray::IDLE"),
            ]
        )
        self.assertEqual(self.pids(text, self_pid=400), [402])

    def test_matches_the_path_as_written_and_its_realpath(self):
        roots = ["/written/venv", "/real/venv"]
        text = "".join(
            [
                ps_line(
                    500, 1, "/written/venv/lib/python3.12/site-packages/ray/x/raylet"
                ),
                ps_line(
                    501, 1, "/real/venv/lib/python3.12/site-packages/ray/x/gcs_server"
                ),
                ps_line(
                    502, 1, "/other/venv/lib/python3.12/site-packages/ray/x/raylet"
                ),
            ]
        )
        self.assertEqual(self.pids(text, roots=roots, self_pid=9999), [500, 501])

    def test_venv_roots_includes_the_realpath(self):
        roots = rl.venv_roots()
        self.assertIn(os.path.realpath(roots[0]), roots)

    def test_reads_the_process_table_through_one_function(self):
        with mock.patch.object(rl, "_ps_output", lambda: SGK_PS):
            self.assertEqual(
                [p for p, _, _ in rl.ray_processes(self_pid=9999)], [100, 101, 102, 103]
            )


class TestDefaultLockDirectory(LockTestCase):
    def test_default_is_computed_from_home_at_call_time(self):
        self.assertEqual(
            rl.default_lock_dir(), self.home / "ScienceData" / ".rehearsal-lock"
        )
        self.assertEqual(list(self.home.iterdir()), [])
        other = self.tmp / "other"
        other.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(other)}):
            self.assertEqual(
                rl.default_lock_dir(), other / "ScienceData" / ".rehearsal-lock"
            )

    def test_commands_without_an_argument_use_the_patched_home(self):
        rc, _, _ = self.run_cmd(rl.acquire, "p")
        self.assertEqual(rc, 0)
        self.assertTrue(
            (self.home / "ScienceData" / ".rehearsal-lock" / "owner").is_file()
        )
        rc, _, _ = self.run_cmd(rl.release)
        self.assertEqual(rc, 0)
        self.assertFalse((self.home / "ScienceData" / ".rehearsal-lock").exists())


class TestOwnerParser(unittest.TestCase):
    def test_reads_si_and_champbh_owner_files(self):
        si = rl.parse_owner(SI_OWNER)
        self.assertEqual(list(si), list(rl.OWNER_KEYS))
        self.assertEqual(si["project"], "StochasticInstantons")
        self.assertEqual(si["ray_temp_dir"], "/tmp/ray-si")
        self.assertEqual(si["ray_address"], "127.0.0.1:6390")
        self.assertEqual(si["acquired"], "2026-10-12T14:19:31+01:00")
        cp = rl.parse_owner(CHAMPBH_OWNER)
        self.assertEqual(list(cp), list(rl.OWNER_KEYS))
        self.assertEqual(cp["project"], "ChamPBH")
        self.assertEqual(cp["ray_temp_dir"], "/tmp/ray-cpbh")
        self.assertEqual(cp["ray_address"], "127.0.0.1:6380")

    def test_value_is_everything_after_the_first_colon_space(self):
        owner = rl.parse_owner("purpose: a: b: c\npid: -\n")
        self.assertEqual(owner["purpose"], "a: b: c")

    def test_round_trips_its_own_file(self):
        text = rl.owner_text("p: q", "123", acquired=PAST, host="h")
        owner = rl.parse_owner(text)
        self.assertEqual(
            owner,
            {
                "project": "SecondaryGWKit",
                "purpose": "p: q",
                "acquired": PAST,
                "host": "h",
                "ray_temp_dir": "in-process",
                "ray_address": "in-process",
                "pid": "123",
            },
        )


class TestCommandLine(LockTestCase):
    def test_commands_and_exit_statuses(self):
        self.assertEqual(self.run_cmd(rl.main, ["status"])[0], 0)
        self.assertEqual(self.run_cmd(rl.main, ["acquire", "--purpose", "p"])[0], 0)
        self.assertEqual(self.run_cmd(rl.main, ["acquire", "--purpose", "p"])[0], 3)
        self.assertEqual(self.run_cmd(rl.main, ["set-pid", str(os.getpid())])[0], 0)
        self.assertEqual(self.run_cmd(rl.main, ["release"])[0], 1)
        self.assertEqual(self.run_cmd(rl.main, ["status"])[0], 0)
        self.assertEqual(self.run_cmd(rl.main, ["ray-processes"])[0], 0)
        self.ps = SGK_PS
        rc, out, _ = self.run_cmd(rl.main, ["ray-processes"])
        self.assertEqual(rc, 1)
        self.assertEqual(len(out.splitlines()), 4)
        self.assertTrue(all(len(l) <= 150 for l in out.splitlines()))

    def test_status_shows_owner_pid_and_processes(self):
        self.ps = SGK_PS
        # the lock lives under the patched HOME
        home_lock = self.home / "ScienceData" / ".rehearsal-lock"
        home_lock.mkdir(parents=True)
        (home_lock / "owner").write_text(sgk_owner(os.getpid()))
        rc, out, _ = self.run_cmd(rl.status)
        self.assertEqual(rc, 0)
        self.assertIn("HELD", out)
        self.assertIn(f"owner pid {os.getpid()}: alive", out)
        self.assertIn("gcs_server", out)

    def test_usage_errors_exit_2(self):
        for argv in ([], ["acquire"], ["set-pid", "x"], ["nonsense"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        rl.main(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_there_is_no_option_to_break_wait_or_retry(self):
        for argv in (
            ["acquire", "--purpose", "p", "--force"],
            ["acquire", "--purpose", "p", "--wait"],
            ["release", "--force"],
            ["release", "--break"],
            ["set-pid", "1", "--retry"],
        ):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        rl.main(argv)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
