"""levin-phase-floor prompt 06k: copied from 06j's scripts/refs_common.py (a record, not edited; its
copy is in state/scripts_06j_reference/), changed only where marked "prompt 06k": ROOT is
var/levin-phase-floor-06k/ (the same depth as 06j's), and G_TREE/G_SHA are 06k's tree (the worktree
at 06k's dispatch SHA 89370609, production code 1c5336e's). 06j's text follows.

levin-phase-floor prompt 06j: copied from prompt 06g's references scripts/refs_common.py (a record,
not edited; its copy is in state/scripts_06g_reference/), changed only where marked "prompt 06j": ROOT
is var/levin-phase-floor-06j/ (one level shallower, so the main-checkout assertion counts one level
fewer), and G_TREE/G_SHA are 06j's tree (the worktree at 06j's dispatch SHA dfa19d98, production
code 1c5336e's, tool b5626ecb's). Every input it names (06g's rows.json, 06h's census, 06a's census,
the scratch copy) is read only, as in 06g. 06g's text follows.

levin-phase-floor prompt 06g section 5, the references: paths and helpers shared by the references
agent's own scripts (draw_refs.py, select_refs.py, check_captures.py, slice_refs.py, compare13.py,
one_reading.py, compare06i.py, run_refs.py, launch_refs.py, score_refs.py). New in 06g.

Everything here is read only, except ROOT (var/levin-phase-floor-06g/references/), where the
references agent writes. The census harness (prov.py, worker.py, harness.py, ...) is the census
agent's, copied unchanged into this directory; this module does not change it.

Trees (Part B's dispatch brief):
- G_TREE, the code worktree at Part B's dispatch SHA 1c5336e (code 06g's 3ae6f3a): the captures,
  the Filon tool and every reading run here, with this directory second on PYTHONPATH (the captured
  calls' pickles name the census harness's functions, e.g. ``harness._rebuild_lru_cache``);
- nothing runs at the parent tree 4f4f9ed here.
"""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # var/levin-phase-floor-06g/references; prompt 06j: var/levin-phase-floor-06j; prompt 06k: var/levin-phase-floor-06k
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[2] == MAIN, HERE  # prompt 06j: one level shallower than 06g's
PYTHON = MAIN / "venv" / "bin" / "python"

G_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060")  # prompt 06k (06j: -lpf06j-dfa19d9; 06g: -lpf06g-1c5336e)
G_SHA = "89370609a64d58993289dcfe991350a420ea698d"  # prompt 06k (06j: dfa19d98...; 06g: 1c5336e7e0ab8ed4495601a4033841f2df17ebdd)

# 06h's census, the "before" (read only)
RUN06H = MAIN / "var" / "runs" / "levin-phase-floor-06h-census-20261101T005037"
CHECKPOINT06H_SHA256 = (
    "3104c18ed8b119f1798aec75be98d5b39bce641f3b5623b4c1cdc5b18b5ef372"
)
# 06a's census: its 1e-7 scan defines the 720 stationary-point rows (spscan.py), read only
RUN06A = MAIN / "var" / "runs" / "levin-phase-floor-06a-census-20261027T023143"
# the g census: the census agent's run, read only, found when it is complete
RUNS = MAIN / "var" / "runs"
CENSUS06G_GLOB = "levin-phase-floor-06g-census-*"
# the census agent's rows.json, read only (equal to 06e's and 06h's)
ROWS = MAIN / "var" / "levin-phase-floor-06g" / "census" / "state" / "rows.json"
ROWS_SHA256 = "aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417"

# Z3's 30 calls (06c's depth20-tiling), the committed copy in the code worktree, read only
Z3_RESULTS = (
    G_TREE / "docs" / "levin-phase-floor" / "06c" / "depth20-tiling" / "results.json"
)
# 06i's 13 samples, its readings and scores (committed), and its inputs on this host (06e's copies
# of depth20-tiling's captures), read only
I06 = G_TREE / "docs" / "levin-phase-floor" / "06i"
I06_READINGS = I06 / "out" / "readings"
I06_SCORES = I06 / "out" / "scores.json"
I06_SUMS = I06 / "inputs_SHA256SUMS.txt"
I06_INPUTS = MAIN / "var" / "levin-phase-floor-06e" / "census" / "references" / "inputs"
I06_SAMPLES = (1, 2, 3, 8, 9, 10, 13, 14, 15, 22, 23, 25, 27)
V4_SAMPLES = (2, 8, 9, 14)

# addendum 2 (the user's ruling, 2026-11-01): V8's 12 lost calls, selected by the orchestrator, read only
V8_SAMPLE = MAIN / "var" / "levin-phase-floor-06g" / "review" / "v8_sample.json"
V8_SAMPLE_SHA256 = "874e17db6e60afd37aba815043fb5783ffebab52c44273c7b0ba55a3c0dc9605"

# 06's certified scratch copy of A3 v2, read only (hashed before and after)
COPY_DIR = MAIN / "var" / "levin-phase-floor-06" / "wider" / "state" / "a3v2_copy"
COPY_CERT = (
    MAIN
    / "var"
    / "levin-phase-floor-06"
    / "wider"
    / "state"
    / "copy_files_certified.txt"
)

RTOLS = ("1e-7", "1e-8")
KINDS = (
    "analytic",
    "group",
)  # census "other_calls" (the analytic blocks) and "calls" (groups)
IMPLS = ("qawo", "fcc")  # tools.filon_reference.IMPLEMENTATIONS, in its order
RESS = (1, 2)  # tools.filon_reference.RESOLUTIONS
READING_LIMIT_S = 1800.0  # a reading over 30 minutes is stopped (the brief)
TOOL_WORKERS = 6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def H(x):
    """A census float ({"repr", "hex"} or a hex string) as a float, exactly."""
    if x is None:
        return None
    return float.fromhex(x["hex"] if isinstance(x, dict) else x)


def hexof(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def rest_of(unit):
    """'g-1e-7-2-1627' -> '1e-7-2-1627' (each census names its units by its tree)."""
    return unit.split("-", 1)[1]


def parse_rest(rest):
    """'1e-7-2-1627' -> ('1e-7', '2', '1627'); the rtol holds a '-'."""
    parts = rest.split("-")
    return "-".join(parts[:-2]), parts[-2], parts[-1]


def rowkey_of(unit):
    return "-".join(unit.rsplit("-", 2)[-2:])


def census_run_06g():
    """The g census's run directory: exactly one, and finished ``done``."""
    runs = sorted(RUNS.glob(CENSUS06G_GLOB))
    if len(runs) != 1:
        raise SystemExit(
            f"expected exactly one {CENSUS06G_GLOB}, found {[r.name for r in runs]}"
        )
    status = json.loads((runs[0] / "status.json").read_text())
    state = status.get("state") or status.get("status")
    if state != "done":
        raise SystemExit(f"{runs[0].name}: state {state!r}, not done")
    return runs[0]


def scan06a_sp():
    """06a's 720 stationary-point rows, as 06e's and 06h's analyse scripts read them: the rows whose
    1e-7 unit's scan shows a sign change of theta' in any call."""
    sp = set()
    with open(RUN06A / "checkpoint.jsonl") as f:
        for line in f:
            if '"rtol": "1e-7"' not in line:
                continue
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            if any(s and s["n_sign_changes"] > 0 for s in (d.get("scan") or [])):
                sp.add(f"{d['shard']}-{d['serial']}")
    return sp


def call_key(c):
    """What a capture must reproduce of a census call record (06c's x1_capture.py's tuple)."""
    return (
        c["value"]["hex"],
        c["abserr"]["hex"],
        c["num_regions"],
        c["max_depth"],
        bool(c["converged"]),
        c["notify_label"],
    )


def scan_checkpoint(checkpoint, wanted):
    """{(rest, index): [(census unit, unit status, kind, call record, unit data), ...]} for
    ``wanted``, a set of (rest, index); every record of a unit is read."""
    want = {}
    for rest, index in wanted:
        want.setdefault(rest, set()).add(index)
    found = {}
    with open(checkpoint) as f:
        for line in f:
            head = line[:300]
            if not any(f'-{r}"' in head for r in want):
                continue
            rec = json.loads(line)
            unit = rec.get("unit", "")
            rest = unit.split("-", 1)[-1]
            if rest not in want:
                continue
            data = rec.get("data") or {}
            for kind, key in (("group", "calls"), ("analytic", "other_calls")):
                for c in data.get(key) or []:
                    if c["index"] in want[rest]:
                        found.setdefault((rest, c["index"]), []).append(
                            (unit, data.get("status"), kind, c, data)
                        )
    return found


def finite(x):
    return x is not None and math.isfinite(x)
