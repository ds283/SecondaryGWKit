"""Prompt 07's stage-count table (§2.2), through `stage_counts.read` as it stands.

`docs/datastore-integrity-verification/stage_counts.py`'s `main` takes each column's label from
the run id by splitting on `-10-`, prompt 10's number, so it raises `IndexError` on a run of any
other prompt. This prints the same table from the same `read`, with the label split on `-07-`.

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/stage_counts_07.py \\
        var/runs/<run id> [var/runs/<run id> ...]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "docs" / "datastore-integrity-verification"))

from stage_counts import read  # noqa: E402


def main(paths) -> int:
    runs = [Path(p) for p in paths]
    tables = [read(run) for run in runs]
    names = []
    for table in tables:
        names += [n for n in table if n not in names]
    slugs = [run.name.split("-07-", 1)[1].rsplit("-", 1)[0] for run in runs]
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
