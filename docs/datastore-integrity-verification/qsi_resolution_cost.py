"""The `QuadSourceIntegral` stage's time from its banner to its first compute, per launch
(prompt 10 §2.1, the 09c log's estimate measured).

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/qsi_resolution_cost.py \\
        <primary> <run dir> [<run dir> ...]

- **The banner's time** is `RayWorkPool`'s start, which is when it prints the banner: a progress
  line's wall-clock stamp less the `(… running)` time it prints beside it, or, for a stage that
  printed no progress line, its `ALL WORK ITEMS COMPLETE in time …` less nothing (the stage's whole
  duration is then reported instead).
- **The first compute's start** is the earliest `timestamp - compute_time` over the
  `QuadSourceIntegral` rows the launch stored, read from every shard `mode=ro`. A row's
  `timestamp` is when it was stored; `compute_time` is the integration's own duration, so this is
  the earliest start of a computation whose result reached the store.
- A launch's rows are those whose `timestamp` falls between its manifest's `created` and its
  `status.json` heartbeat.
"""

import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROGRESS = re.compile(
    r"-- (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \((?:(\d+)h )?(?:(\d+)m )?([\d.]+)s running\)"
)
COMPLETE = re.compile(r"ALL WORK ITEMS COMPLETE in time (?:(\d+)m )?([\d.]+)s")
BANNER = "** CALCULATE QUADRATIC SOURCE INTEGRALS"


def stage_lines(run_dir: Path):
    lines = (run_dir / "stdout.log").read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == BANNER:
            return lines[i + 1 :]
    return None


def banner_time(lines):
    for line in lines:
        m = PROGRESS.search(line)
        if m:
            stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            h, mi, s = (float(x) if x else 0.0 for x in m.groups()[1:])
            return stamp - timedelta(hours=h, minutes=mi, seconds=s), None
        m = COMPLETE.search(line)
        if m:
            mi, s = (float(x) if x else 0.0 for x in m.groups())
            return None, mi * 60 + s
    return None, None


def shards(primary: Path):
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    names = [
        row[0] for row in con.execute("select filename from shards order by serial")
    ]
    con.close()
    return [primary.parent / name for name in names]


def first_compute(primary: Path, start, end):
    earliest, rows = None, 0
    for shard in shards(primary):
        con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        for stamp, compute_time in con.execute(
            "select timestamp, compute_time from QuadSourceIntegral"
        ):
            when = datetime.fromisoformat(stamp)
            if not (start <= when <= end):
                continue
            rows += 1
            began = when - timedelta(seconds=compute_time or 0.0)
            earliest = began if earliest is None or began < earliest else earliest
        con.close()
    return earliest, rows


def main(primary, run_dirs) -> int:
    primary = Path(primary)
    print(f"store {primary}")
    for run_dir in map(Path, run_dirs):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        status = json.loads((run_dir / "status.json").read_text())
        start = datetime.fromisoformat(manifest["created"]).replace(tzinfo=None)
        end = datetime.fromisoformat(status["heartbeat"]).replace(tzinfo=None)
        lines = stage_lines(run_dir)
        if lines is None:
            print(f"  {run_dir.name}: the stage never started")
            continue
        banner, whole = banner_time(lines)
        earliest, rows = first_compute(primary, start, end + timedelta(seconds=5))
        if banner is None:
            print(
                f"  {run_dir.name}: no progress line; whole stage {whole:.2f} s; "
                f"{rows} row(s) stored by this launch"
            )
            continue
        gap = (earliest - banner).total_seconds() if earliest is not None else None
        print(
            f"  {run_dir.name}: banner {banner:%H:%M:%S.%f}; first compute began "
            f"{earliest:%H:%M:%S.%f}; banner to first compute {gap:.1f} s; "
            f"{rows} row(s) stored by this launch"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
