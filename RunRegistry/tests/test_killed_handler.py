"""A signal to a pipeline driver is recorded `killed`, and nothing it then does can say otherwise.

`prompts/a3-v2-readiness` prompt 04. The subject is three functions of
`docs/gktk-remedial/scoped_pipeline_run.py`, which both pipeline drivers share:

  * `make_killed_handler` / `install_killed_handler`: the handler each driver installs straight
    after `ray.init`, which records `killed`, `128 + signum` at once (prompt §2, K1);
  * `finish_after_exec`: what each driver does once `main.py`'s `exec` is over (K2);

and the two drivers' wiring of them, read with `ast` (K1, K2, K4).

Nothing here starts Ray or opens a datastore (README §5 rule 9). The handler is called directly,
with a stand-in run or a real `Run` in a temporary run root; `ray.shutdown` and `signal.alarm` are
stand-ins. What these show on real Ray is `docs/a3-v2-readiness/sigterm_handler_probe.py`'s.
"""

import ast
import io
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import RunRegistry
from RunRegistry.tests.test_pipeline_adoption import DRIVER, REPO_ROOT, load_driver

SWEEP = os.path.join(REPO_ROOT, "docs", "handover", "quadsource_atol_sweep.py")

SIGTERM = int(signal.SIGTERM)
SIGINT = int(signal.SIGINT)


class StandInRun:
    """Records every `finish` call as `(state, exit_code, fingerprint)`; `during_fingerprint`,
    if given, is called inside a `finish(..., fingerprint=True)`, as a signal arriving while the
    fingerprint is taken would run."""

    def __init__(self, during_fingerprint=None, fail=None):
        self.calls = []
        self.during_fingerprint = during_fingerprint
        self.fail = fail

    def finish(self, state, exit_code=None, *, fingerprint=False):
        if self.fail is not None:
            raise self.fail
        if fingerprint and self.during_fingerprint is not None:
            hook, self.during_fingerprint = self.during_fingerprint, None
            hook()
        self.calls.append((state, exit_code, fingerprint))
        return {"state": state, "exit_code": exit_code}


class Recorder:
    """A stand-in for `ray.shutdown` or `signal.alarm`: counts its calls, returns `value`."""

    def __init__(self, value=None):
        self.calls = []
        self.value = value

    def __call__(self, *args):
        self.calls.append(args)
        return self.value


class FailingStream:
    """A stream whose write raises what CPython raises for a reentrant write to a buffered
    stream."""

    def write(self, _text):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stderr>'>")

    def flush(self):
        pass


class DriverTestCase(unittest.TestCase):
    def setUp(self):
        self.driver = load_driver()
        self.latch = self.driver.SignalLatch()
        self.out = io.StringIO()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def handler(self, run):
        return self.driver.make_killed_handler(run, self.latch, stream=self.out)

    def a_real_run(self):
        return RunRegistry.begin(
            campaign="a3-v2-readiness",
            prompt="04",
            slug="killed-handler",
            purpose="a unit test of the killed handler, in a temporary directory",
            argv=["unittest"],
            root=self._tmp.name,
        )

    def after_exec(self, run, exc, shutdown=None, alarm=None):
        """`finish_after_exec` with stand-ins; returns `(SystemExit raised or None, shutdown,
        alarm)`."""
        shutdown = Recorder() if shutdown is None else shutdown
        alarm = Recorder(0) if alarm is None else alarm
        try:
            self.driver.finish_after_exec(
                run, exc, self.latch, shutdown, alarm=alarm, stream=self.out
            )
        except SystemExit as e:
            return e, shutdown, alarm
        return None, shutdown, alarm


# =================================================================================================
# 1. the handler, called directly


class TestTheHandler(DriverTestCase):
    def test_its_first_call_writes_killed_143_remembers_prints_and_raises(self):
        run = StandInRun()
        with self.assertRaises(SystemExit) as raised:
            self.handler(run)(SIGTERM, None)
        self.assertEqual(raised.exception.code, 143)
        self.assertEqual(run.calls, [("killed", 143, False)])
        self.assertEqual(self.latch.signum, SIGTERM)
        lines = self.out.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("SIGTERM", lines[0])
        self.assertIn("killed", lines[0])
        # the handler's own line is not Ray's banner, which the probe looks for
        self.assertNotIn("*** SIGTERM received", lines[0])

    def test_its_second_call_writes_nothing_and_raises_nothing(self):
        run = StandInRun()
        handler = self.handler(run)
        with self.assertRaises(SystemExit):
            handler(SIGTERM, None)
        self.assertIsNone(handler(SIGTERM, None))
        self.assertIsNone(handler(SIGINT, None))
        self.assertEqual(run.calls, [("killed", 143, False)])
        self.assertEqual(
            self.latch.signum, SIGTERM, "a later signal replaced the first"
        )
        lines = self.out.getvalue().splitlines()
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("ignored", lines[1])
        self.assertIn("SIGINT", lines[2])

    def test_sigint_gives_130(self):
        run = StandInRun()
        with self.assertRaises(SystemExit) as raised:
            self.handler(run)(SIGINT, None)
        self.assertEqual(raised.exception.code, 130)
        self.assertEqual(run.calls, [("killed", 130, False)])
        self.assertEqual(self.latch.signum, SIGINT)

    def test_unregistered_it_writes_nothing_and_still_raises(self):
        with self.assertRaises(SystemExit) as raised:
            self.handler(None)(SIGTERM, None)
        self.assertEqual(raised.exception.code, 143)
        self.assertEqual(self.latch.signum, SIGTERM)

    def test_a_print_that_fails_does_not_stop_the_raise(self):
        run = StandInRun()
        handler = self.driver.make_killed_handler(
            run, self.latch, stream=FailingStream()
        )
        with self.assertRaises(SystemExit) as raised:
            handler(SIGTERM, None)
        self.assertEqual(raised.exception.code, 143)
        self.assertEqual(run.calls, [("killed", 143, False)])
        self.assertEqual(self.latch.signum, SIGTERM)
        self.assertIsNone(handler(SIGTERM, None), "a later call raised")

    def test_a_write_that_fails_does_not_stop_the_raise(self):
        run = StandInRun(fail=OSError("no space left on device"))
        with self.assertRaises(SystemExit) as raised:
            self.handler(run)(SIGTERM, None)
        self.assertEqual(raised.exception.code, 143)
        self.assertEqual(self.latch.signum, SIGTERM)
        self.assertIn("could not record killed", self.out.getvalue())

    def test_on_a_real_run_status_json_says_killed_143_and_holds_no_fingerprint(self):
        run = self.a_real_run()
        with self.assertRaises(SystemExit):
            self.handler(run)(SIGTERM, None)
        status = RunRegistry.read_json(run.status_path)
        self.assertEqual((status["state"], status["exit_code"]), ("killed", 143))
        for key in ("fingerprint", "fingerprint_error", "fingerprint_sidecar"):
            self.assertNotIn(key, status)

    def test_once_the_outcome_is_decided_a_signal_is_only_noted(self):
        run = StandInRun()
        self.assertIsNone(self.latch.decide())
        self.assertIsNone(self.handler(run)(SIGTERM, None))
        self.assertEqual(run.calls, [])
        self.assertIsNone(self.latch.signum)
        self.assertIn("ignored", self.out.getvalue())


class TestInstall(DriverTestCase):
    def setUp(self):
        super().setUp()
        saved = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT)}

        def restore():
            for number, handler in saved.items():
                signal.signal(number, handler)

        self.addCleanup(restore)

    def test_it_takes_sigterm_and_sigint(self):
        handler = self.driver.install_killed_handler(None, self.latch, stream=self.out)
        self.assertIs(signal.getsignal(signal.SIGTERM), handler)
        self.assertIs(signal.getsignal(signal.SIGINT), handler)

    def test_installing_again_keeps_the_remembered_signal(self):
        """`ignore_reinit_error=True` lets a second `ray.init` reach the installer again."""
        run = StandInRun()
        first = self.driver.install_killed_handler(run, self.latch, stream=self.out)
        with self.assertRaises(SystemExit):
            first(SIGTERM, None)
        second = self.driver.install_killed_handler(run, self.latch, stream=self.out)
        self.assertEqual(self.latch.signum, SIGTERM)
        self.assertIsNone(second(SIGTERM, None))
        self.assertEqual(run.calls, [("killed", 143, False)])


# =================================================================================================
# 2. the post-exec function


def fired(test, run, signum=SIGTERM):
    """Fire the handler as a signal during the pipeline would, and swallow its SystemExit."""
    with test.assertRaises(SystemExit):
        test.handler(run)(signum, None)


class TestSignalled(DriverTestCase):
    """The handler fired during the pipeline: `killed`, whatever reached the driver."""

    WHAT_REACHES_THE_DRIVER = (
        ("SystemExit(1), out of ray.wait", SystemExit(1)),
        ("SystemExit(143), raised anywhere else", SystemExit(143)),
        ("SystemExit(None)", SystemExit(None)),
        ("KeyboardInterrupt", KeyboardInterrupt()),
        ("RuntimeError from a guarded step", RuntimeError("interrupted")),
        ("no exception: the exec returned", None),
    )

    def test_it_is_killed_143_with_the_fingerprint_whatever_reaches_the_driver(self):
        for label, exc in self.WHAT_REACHES_THE_DRIVER:
            with self.subTest(label):
                self.latch = self.driver.SignalLatch()
                run = StandInRun()
                fired(self, run)
                raised, shutdown, alarm = self.after_exec(run, exc)
                # the record first, so that a breakage shows what it wrote
                self.assertEqual(
                    run.calls, [("killed", 143, False), ("killed", 143, True)]
                )
                self.assertNotIn("failed", [state for state, _, _ in run.calls])
                self.assertIsNotNone(raised, "it did not exit")
                self.assertEqual(raised.code, 143)
                self.assertEqual(alarm.calls, [(0,)])
                self.assertEqual(len(shutdown.calls), 1)

    def test_sigint_is_killed_130(self):
        run = StandInRun()
        fired(self, run, SIGINT)
        raised, shutdown, _ = self.after_exec(run, SystemExit(1))
        self.assertEqual(raised.code, 130)
        self.assertEqual(run.calls, [("killed", 130, False), ("killed", 130, True)])
        self.assertEqual(len(shutdown.calls), 1)

    def test_unregistered_it_shuts_down_and_exits_143(self):
        fired(self, None)
        raised, shutdown, alarm = self.after_exec(None, SystemExit(1))
        self.assertEqual(raised.code, 143)
        self.assertEqual(len(shutdown.calls), 1)
        self.assertEqual(alarm.calls, [(0,)])

    def test_the_alarm_is_cancelled_before_the_fingerprint(self):
        order = []
        run = StandInRun(during_fingerprint=lambda: order.append("fingerprint"))
        fired(self, run)

        def alarm(seconds):
            order.append(("alarm", seconds))
            return 3  # what absl's armed alarm leaves (audit S2)

        raised, _, _ = self.after_exec(
            run,
            SystemExit(1),
            shutdown=lambda: order.append("shutdown"),
            alarm=alarm,
        )
        self.assertEqual(raised.code, 143)
        self.assertEqual(order, [("alarm", 0), "fingerprint", "shutdown"])
        self.assertIn("cancelled an alarm with 3 s left", self.out.getvalue())

    def test_a_write_that_fails_still_shuts_down_and_exits(self):
        run = StandInRun()
        fired(self, run)
        run.fail = OSError("no space left on device")
        raised, shutdown, _ = self.after_exec(run, SystemExit(1))
        self.assertEqual(raised.code, 143)
        self.assertEqual(len(shutdown.calls), 1)

    def test_a_shutdown_that_fails_still_exits_143(self):
        run = StandInRun()
        fired(self, run)

        def failing():
            raise RuntimeError("the raylet is gone")

        raised, _, _ = self.after_exec(run, SystemExit(1), shutdown=failing)
        self.assertEqual(raised.code, 143)
        self.assertIn("ray.shutdown() raised", self.out.getvalue())

    def test_a_second_signal_during_the_fingerprint_is_ignored(self):
        """Prompt §3, the probe's third variant, without Ray: on a real `Run`, the second signal
        arrives inside `finish(..., fingerprint=True)`."""
        run = self.a_real_run()
        handler = self.handler(run)
        with self.assertRaises(SystemExit):
            handler(SIGTERM, None)
        real = run._fingerprint_results

        def interrupted():
            self.assertIsNone(handler(SIGTERM, None))
            return real()

        with mock.patch.object(run, "_fingerprint_results", interrupted):
            raised, shutdown, _ = self.after_exec(run, SystemExit(1))
        self.assertEqual(raised.code, 143)
        status = RunRegistry.read_json(run.status_path)
        self.assertEqual((status["state"], status["exit_code"]), ("killed", 143))
        self.assertIn("names no results store", status["fingerprint_error"])
        self.assertEqual(len(shutdown.calls), 1)

    def test_a_signal_inside_decide_is_a_signal_that_came_first(self):
        """The handler runs between `decide()`'s test and its assignment, and raises."""
        run = StandInRun()
        handler = self.handler(run)
        latch = self.latch

        def decide():
            if latch.signum is None:
                handler(SIGTERM, None)  # raises SystemExit(143)
                latch.decided = True
            return latch.signum

        with mock.patch.object(latch, "decide", decide):
            raised, shutdown, _ = self.after_exec(run, None)
        self.assertEqual(raised.code, 143)
        self.assertEqual(run.calls, [("killed", 143, False), ("killed", 143, True)])
        self.assertEqual(len(shutdown.calls), 1)


def today(driver, exc):
    """What the drivers recorded at `74270dd` with no signal: `(state, exit_code)`."""
    if exc is None:
        return ("done", 0)
    if isinstance(exc, SystemExit):
        return (driver.terminal_state(exc.code), exc.code)
    return ("failed", 1)


class TestNotSignalled(DriverTestCase):
    """No signal: every existing outcome, exactly as before prompt 04, then Ray shut down."""

    OUTCOMES = (
        ("the exec returned", None, ("done", 0)),
        ("SystemExit(None)", SystemExit(None), ("done", None)),
        ("SystemExit(0)", SystemExit(0), ("done", 0)),
        ("SystemExit(1)", SystemExit(1), ("failed", 1)),
        ("SystemExit(15), Ray's own handler", SystemExit(15), ("killed", 15)),
        ("SystemExit('a message')", SystemExit("a message"), ("failed", "a message")),
        ("RuntimeError", RuntimeError("a real failure"), ("failed", 1)),
        ("KeyboardInterrupt", KeyboardInterrupt(), ("failed", 1)),
    )

    def test_every_outcome_is_recorded_as_it_was(self):
        for label, exc, expected in self.OUTCOMES:
            with self.subTest(label):
                self.latch = self.driver.SignalLatch()
                self.assertEqual(today(self.driver, exc), expected)
                run = StandInRun()
                raised, shutdown, alarm = self.after_exec(run, exc)
                self.assertIsNone(
                    raised, "it exited instead of returning for the re-raise"
                )
                self.assertEqual(run.calls, [expected + (True,)])
                self.assertEqual(len(shutdown.calls), 1)
                self.assertEqual(alarm.calls, [(0,)])
                self.assertTrue(self.latch.decided)

    def test_unregistered_it_records_nothing_and_returns(self):
        for label, exc, _ in self.OUTCOMES:
            with self.subTest(label):
                self.latch = self.driver.SignalLatch()
                raised, shutdown, _ = self.after_exec(None, exc)
                self.assertIsNone(raised)
                self.assertEqual(len(shutdown.calls), 1)

    def test_a_signal_during_its_own_fingerprint_changes_nothing(self):
        """Prompt 04's choice (§2 K2, amended): once the pipeline's outcome has reached the
        driver, a signal is noted and does not change the record or the exit."""
        for label, exc, expected in self.OUTCOMES:
            with self.subTest(label):
                self.latch = self.driver.SignalLatch()
                run = StandInRun()
                handler = self.handler(run)
                run.during_fingerprint = lambda: self.assertIsNone(
                    handler(SIGTERM, None)
                )
                raised, shutdown, _ = self.after_exec(run, exc)
                self.assertIsNone(raised)
                self.assertEqual(run.calls, [expected + (True,)])
                self.assertIsNone(self.latch.signum)
                self.assertEqual(len(shutdown.calls), 1)

    def test_on_a_real_run_the_outcome_keeps_its_fingerprint(self):
        run = self.a_real_run()
        handler = self.handler(run)
        real = run._fingerprint_results

        def interrupted():
            self.assertIsNone(handler(SIGTERM, None))
            return real()

        with mock.patch.object(run, "_fingerprint_results", interrupted):
            raised, _, _ = self.after_exec(run, RuntimeError("a real failure"))
        self.assertIsNone(raised)
        status = RunRegistry.read_json(run.status_path)
        self.assertEqual((status["state"], status["exit_code"]), ("failed", 1))
        self.assertIn("names no results store", status["fingerprint_error"])
        self.assertNotIn("interrupted", status["fingerprint_error"])
        self.assertIn("ignored", self.out.getvalue())

    def test_a_shutdown_that_fails_does_not_replace_the_exception(self):
        def failing():
            raise RuntimeError("the raylet is gone")

        raised, _, _ = self.after_exec(StandInRun(), SystemExit(1), shutdown=failing)
        self.assertIsNone(raised)
        self.assertIn("ray.shutdown() raised", self.out.getvalue())


# =================================================================================================
# 3. both drivers install it after ray.init, and both finish through it


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(node, name):
    return [
        n for n in ast.walk(node) if isinstance(n, ast.Call) and _called_name(n) == name
    ]


class TestTheDrivers(unittest.TestCase):
    """Read with `ast`: `main.py` cannot be run here, and neither can the drivers."""

    DRIVERS = ((DRIVER, "main"), (SWEEP, "run_build"))

    def outer(self, path, name):
        return _function(ast.parse(Path(path).read_text()), name)

    def test_each_ray_init_wrapper_starts_a_local_instance_then_installs_the_handler(
        self,
    ):
        for path, name in self.DRIVERS:
            with self.subTest(driver=os.path.basename(path)):
                wrapper = _function(self.outer(path, name), "_local_ray_init")
                real = _calls(wrapper, "_real_ray_init")
                self.assertEqual(len(real), 1)
                address = [k.value for k in real[0].keywords if k.arg == "address"]
                self.assertEqual(len(address), 1, "the real ray.init passes no address")
                self.assertIsInstance(address[0], ast.Constant)
                self.assertEqual(address[0].value, "local")
                # the installer is a statement of the wrapper's own body, after the real
                # ray.init's: reached on every call, and only once Ray is up
                body = wrapper.body
                at = [
                    i
                    for i, stmt in enumerate(body)
                    if any(
                        _called_name(c) == "install_killed_handler"
                        for c in _calls(stmt, "install_killed_handler")
                    )
                ]
                ray_init = [
                    i for i, stmt in enumerate(body) if _calls(stmt, "_real_ray_init")
                ]
                self.assertEqual(len(at), 1, "the wrapper does not install the handler")
                self.assertEqual(len(ray_init), 1)
                self.assertGreater(at[0], ray_init[0], "installed before ray.init")
                self.assertIsInstance(body[at[0]], ast.Expr)
                args = [
                    a.id for a in _calls(body[at[0]], "install_killed_handler")[0].args
                ]
                self.assertEqual(args, ["run", "latch"])

    def test_the_one_exec_ends_through_finish_after_exec_registered_or_not(self):
        for path, name in self.DRIVERS:
            with self.subTest(driver=os.path.basename(path)):
                outer = self.outer(path, name)
                execs = _calls(outer, "exec")
                self.assertEqual(len(execs), 1, "an exec outside the try")
                tries = [
                    t
                    for t in ast.walk(outer)
                    if isinstance(t, ast.Try) and _calls(ast.Module(t.body, []), "exec")
                ]
                self.assertEqual(len(tries), 1)
                node = tries[0]
                self.assertEqual(len(node.handlers), 1)
                handler = node.handlers[0]
                self.assertIsInstance(handler.type, ast.Name)
                self.assertEqual(handler.type.id, "BaseException")
                self.assertEqual(len(handler.body), 2)
                call = handler.body[0]
                self.assertIsInstance(call, ast.Expr)
                self.assertEqual(_called_name(call.value), "finish_after_exec")
                self.assertEqual(
                    [ast.unparse(a) for a in call.value.args],
                    ["run", handler.name, "latch", "ray.shutdown"],
                )
                self.assertIsInstance(handler.body[1], ast.Raise)
                self.assertIsNone(handler.body[1].exc, "not a bare re-raise")
                # and the clean path, after the try
                after = [
                    c
                    for c in _calls(outer, "finish_after_exec")
                    if c.lineno > node.end_lineno
                ]
                self.assertEqual(len(after), 1)
                self.assertEqual(
                    [ast.unparse(a) for a in after[0].args],
                    ["run", "None", "latch", "ray.shutdown"],
                )

    def test_the_shared_functions_import_nothing_from_ray_at_module_scope(self):
        tree = ast.parse(Path(DRIVER).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("ray", [a.name.split(".")[0] for a in node.names])
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or "").split(".")[0], "ray")
        names = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for name in (
            "SignalLatch",
            "make_killed_handler",
            "install_killed_handler",
            "finish_after_exec",
        ):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
