"""M2: what every pool constructor raised, printed and recorded, test by test.

usage (from the tree, PYTHONPATH=. PYTHONHASHSEED=0): python m2_open_behaviour.py <out.json>

Runs the five check-at-open test modules with ShardedPool.__init__ wrapped. For every
constructor call it records: the exception type and message if it raised; every line it
printed (stdout teed, so the tests' own redirections are untouched); and pool.reconciliation
and pool.pruned, made JSON-safe. Temporary paths, datetimes and object addresses are
normalised so that two runs of one tree can be compared byte for byte.
"""
import datetime
import io
import json
import random
import re
import sys
import unittest

MODULES = [
    "Datastore.tests.test_reconcile_at_open",
    "Datastore.tests.test_prune_at_open",
    "Datastore.tests.test_version_row_at_open",
    "Datastore.tests.test_replicated_write",
    "Datastore.tests.test_read_only_pool",
]

from Datastore.SQL import ShardedPool as _unused  # noqa: F401  (the package rebinds names)
import importlib

sp_mod = importlib.import_module("Datastore.SQL.ShardedPool")
ShardedPool = sp_mod.ShardedPool

events = {}
current = {"test": None}

_TMP = re.compile(r"(/private)?/(var/folders|tmp)/[^\s\"',)\]]*")
_DT = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?")
_ADDR = re.compile(r" at 0x[0-9a-f]+")
_DTREPR = re.compile(r"datetime\.datetime\([0-9, ]+\)")


def norm(text: str) -> str:
    text = _TMP.sub("<tmp>", text)
    text = _DTREPR.sub("<datetime>", text)
    text = _DT.sub("<datetime>", text)
    text = _ADDR.sub(" at <addr>", text)
    return text


def safe(x):
    if isinstance(x, dict):
        return {norm(str(k)): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = [safe(v) for v in x]
        if isinstance(x, (set, frozenset)):
            items = sorted(items, key=repr)
        return items
    if isinstance(x, (datetime.datetime, datetime.date)):
        return "<datetime>"
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    return norm(str(x))


class Tee(io.TextIOBase):
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def write(self, s):
        self.sink.write(s)
        return self.inner.write(s)

    def flush(self):
        self.inner.flush()


original_init = ShardedPool.__init__


def wrapped_init(self, *args, **kwargs):
    buf = io.StringIO()
    previous = sys.stdout
    sys.stdout = Tee(previous, buf)
    entry = {"read_only": bool(kwargs.get("read_only", False))}
    try:
        original_init(self, *args, **kwargs)
    except BaseException as e:
        entry["raised"] = type(e).__name__
        entry["message"] = norm(str(e))
        raise
    else:
        entry["reconciliation"] = safe(getattr(self, "reconciliation", None))
        entry["pruned"] = safe(getattr(self, "pruned", None))
        version = getattr(self, "_version", None)
        entry["version"] = (
            None
            if version is None
            else {
                "type": type(version).__name__,
                "store_id": version.store_id,
                "label": getattr(version, "label", None),
            }
        )
    finally:
        sys.stdout = previous
        entry["printed"] = [norm(l) for l in buf.getvalue().splitlines()]
        events.setdefault(current["test"], []).append(entry)


ShardedPool.__init__ = wrapped_init


class Result(unittest.TextTestResult):
    def startTest(self, test):
        current["test"] = test.id()
        # the stand-in cluster draws an unpinned controlling shard with the random module: seed
        # it per test, so that two runs of one tree draw the same shards (run under
        # PYTHONHASHSEED=0 for the same reason)
        random.seed(20261006)
        super().startTest(test)


suite = unittest.TestSuite()
loader = unittest.TestLoader()
for m in MODULES:
    suite.addTests(loader.loadTestsFromName(m))
runner = unittest.TextTestRunner(stream=sys.stderr, resultclass=Result, verbosity=0)
result = runner.run(suite)
outcomes = {
    "testsRun": result.testsRun,
    "failures": [t.id() for t, _ in result.failures],
    "errors": [t.id() for t, _ in result.errors],
}
with open(sys.argv[1], "w") as f:
    json.dump({"outcomes": outcomes, "events": events}, f, indent=1, sort_keys=True)
print(json.dumps(outcomes), file=sys.stderr)
print("constructor calls:", sum(len(v) for v in events.values()), file=sys.stderr)
