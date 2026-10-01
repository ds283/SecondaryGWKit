"""Tabulate each stage's final lookup / compute / store counts from registered runs' `stdout.log`
(prompt 10 §2.1).

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/stage_counts.py \\
        var/runs/<run id> [var/runs/<run id> ...]

For a stage that completed, the counts are its `Queue summary` line. For the stage a launch was
interrupted in, they are its last `completed:` progress line, marked `(last progress)`, or
`(no progress line)` if it printed none. `-` means the stage never started in that launch. It
reads only the log files.
"""

import re
import sys
from pathlib import Path

BANNER = re.compile(r"^\*\* (CALCULATE .*|APPLY .*)$")
SUMMARY = re.compile(r"Queue summary: (.*)$")
PROGRESS = re.compile(r"\| completed: (.*)$")
COUNT = re.compile(r"(\d+) (lookup|compute|store|validation)")


def counts(text):
    found = {kind: int(n) for n, kind in COUNT.findall(text)}
    parts = [f"{found[k]}" for k in ("lookup", "compute", "store") if k in found]
    return "/".join(parts)


def read(run_dir: Path):
    stages, current = {}, None
    for raw in (run_dir / "stdout.log").read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        banner = BANNER.match(line)
        if banner:
            current = banner.group(1)
            stages[current] = "(no progress line)"
            continue
        if current is None:
            continue
        summary = SUMMARY.search(line)
        if summary:
            stages[current] = counts(summary.group(1))
            current = None
            continue
        progress = PROGRESS.search(line)
        if progress:
            stages[current] = counts(progress.group(1)) + " (last progress)"
    return stages


def main(paths) -> int:
    runs = [Path(p) for p in paths]
    tables = [read(run) for run in runs]
    names = []
    for table in tables:
        names += [n for n in table if n not in names]
    slugs = [run.name.split("-10-", 1)[1].rsplit("-", 1)[0] for run in runs]
    width = max(len(n) for n in names)
    cols = [
        max(len(s), max(len(t.get(n, "-")) for n in names))
        for s, t in zip(slugs, tables)
    ]
    print("lookup/compute/store per stage, from each run's stdout.log")
    print(
        f"  {'stage':<{width}}  " + "  ".join(f"{s:<{c}}" for s, c in zip(slugs, cols))
    )
    for name in names:
        print(
            f"  {name:<{width}}  "
            + "  ".join(f"{t.get(name, '-'):<{c}}" for t, c in zip(tables, cols))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
