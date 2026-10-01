"""Prompt 10 §2.4: `main.py --inventory` and `--inventory-verbose` on both stores, and what their
reports say.

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/inventory_reports.py \\
        <reference primary> <resumed primary> <output dir>

For each of the four runs it records the exit code, the wall time, the report's size and its
longest record line by class, and writes the report to `<output dir>/<store>[-verbose].txt`.

- **Every label is defined once.** Every `<Class>#<n>` and `S<n>` the report names
  (`references`) has exactly one defining line (`definitions`), and every defined label is
  defined once. Both helpers are imported from prompt 09d's test module, not copied, as is
  `HEX64`; no 64-hex digest may appear.
- **The two stores' reports are identical** once each report's first line (which names the
  store) and every class header's timestamp range (`| <from> – <to>`) are masked.

It runs `main.py --inventory`, which is read-only and needs no Ray, and writes nothing else.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datastore.tests.inventory_report_parsing import HEX64, sections  # noqa: E402
from Datastore.tests.test_inventory_report_labels import (  # noqa: E402
    definitions,
    references,
)

PYTHON = REPO_ROOT / "venv" / "bin" / "python"
RANGE = re.compile(r" \| \d{4}-\d\d-\d\d \d\d:\d\d – \d{4}-\d\d-\d\d \d\d:\d\d$")


def run(primary: Path, verbose: bool, out: Path):
    argv = [str(PYTHON), "main.py", "--database", str(primary), "--inventory"]
    if verbose:
        argv.append("--inventory-verbose")
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)
    wall = time.monotonic() - start
    out.write_text(proc.stdout)
    return proc.returncode, wall, proc.stdout, proc.stderr


def masked(text: str) -> str:
    lines = text.split("\n")
    lines[0] = "<first line: names the store>"
    return "\n".join(RANGE.sub(" | <timestamp range>", line) for line in lines)


def check_labels(text: str):
    rows, sets = definitions(text)
    names, numbers = references(text)
    twice = {k: len(v) for k, v in rows.items() if len(v) != 1}
    twice.update({f"S{k}": len(v) for k, v in sets.items() if len(v) != 1})
    undefined = sorted(names - set(rows)) + [
        f"S{n}" for n in sorted(numbers - set(sets))
    ]
    return len(rows), len(sets), len(names), len(numbers), twice, undefined


def main(reference, resumed, out_dir) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    reports = {}
    for name, primary in (("reference", Path(reference)), ("resumed", Path(resumed))):
        for verbose in (False, True):
            tag = f"{name}{'-verbose' if verbose else ''}"
            code, wall, text, err = run(primary, verbose, out_dir / f"{tag}.txt")
            reports[tag] = text
            print(
                f"== main.py --database {primary} --inventory{' --inventory-verbose' if verbose else ''}"
            )
            print(
                f"   exit code {code}; wall time {wall:.2f} s; {len(text.encode()):,} bytes, {text.count(chr(10)):,} lines"
            )
            if err.strip():
                print(f"   stderr: {err.strip()[:500]}")
            ok = ok and code == 0
            longest = {
                cls: max((len(line) for _, line in sec.records), default=0)
                for cls, sec in sections(text).items()
            }
            print("   longest record line by class (characters, without the bullet):")
            for cls, n in longest.items():
                print(f"     {cls:22s} {n:6d}")
            rows, sets, names, numbers, twice, undefined = check_labels(text)
            print(
                f"   labels: {rows} row label(s) and {sets} set label(s) defined; "
                f"{names} row label(s) and {numbers} set label(s) named"
            )
            print(f"   defined more than once: {twice or 'none'}")
            print(f"   named and not defined: {undefined or 'none'}")
            digest = HEX64.search(text)
            print(f"   64-hex digest: {digest.group(0) if digest else 'none'}")
            ok = ok and not twice and not undefined and digest is None
    for suffix in ("", "-verbose"):
        a, b = masked(reports[f"reference{suffix}"]), masked(
            reports[f"resumed{suffix}"]
        )
        same = a == b
        print(
            f"== reference{suffix} vs resumed{suffix}, masked: "
            + ("identical" if same else "DIFFER")
        )
        if not same:
            la, lb = a.split("\n"), b.split("\n")
            diffs = [
                i
                for i in range(max(len(la), len(lb)))
                if la[i : i + 1] != lb[i : i + 1]
            ]
            print(f"   {len(diffs)} line(s) differ; first at line {diffs[0] + 1}")
        ok = ok and same
    print("verdict:", "clean" if ok else "NOT clean")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
