"""Each stage's final lookup / compute / store counts, one column per registered run.

    PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py \\
        var/runs/<run id> [var/runs/<run id> ...]

The table is read through `docs/datastore-integrity-verification/stage_counts.read`, unchanged:
for a stage that completed, its `Queue summary` line; for the stage a launch was interrupted in,
its last `completed:` progress line, marked `(last progress)`, or `(no progress line)` if it
printed none; `-` where the stage never started in that launch. Only the log files are read.

Each column is labelled by the run's slug, taken from the run's own `manifest.json`: its `run_id`
less the `<campaign>-<prompt>-` prefix its `campaign` and `prompt` give, and less the trailing
timestamp. Nothing here splits on a prompt number, so the script labels a run of any campaign or
prompt (dk-and-rpk-migration README §4; `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]`).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "docs" / "datastore-integrity-verification"))

import runregistrykit  # noqa: E402
from stage_counts import read  # noqa: E402


def slug_of(run_dir: Path) -> str:
    """The slug `runregistrykit.run_id` was given, recovered from the manifest's own fields."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = manifest["run_id"]
    prefix = (
        "-".join(
            (
                runregistrykit.slugify(manifest["campaign"]),
                runregistrykit.slugify(manifest["prompt"]),
            )
        )
        + "-"
    )
    if not run_id.startswith(prefix):
        raise ValueError(
            f"{run_dir}: run_id {run_id!r} does not begin with its manifest's "
            f"campaign and prompt ({prefix!r})"
        )
    return run_id[len(prefix) :].rsplit("-", 1)[0]


def main(paths) -> int:
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    runs = [Path(p) for p in paths]
    tables = [read(run) for run in runs]
    names = []
    for table in tables:
        names += [n for n in table if n not in names]
    slugs = [slug_of(run) for run in runs]
    width = max(len(n) for n in names)
    cols = [
        max(len(s), max(len(t.get(n, "-")) for n in names))
        for s, t in zip(slugs, tables)
    ]
    print("lookup/compute/store per stage, from each run's stdout.log")
    for slug, run in zip(slugs, runs):
        print(f"  {slug}: {run.name}")
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
