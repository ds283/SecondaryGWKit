"""Print a run's `** GkSourcePolicyData SUMMARY STATISTICS` block from its `stdout.log`, verbatim.

    ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py var/runs/<run id>

The block begins at the line `** GkSourcePolicyData SUMMARY STATISTICS` and ends before the first
line after it that begins a new section of the pipeline's output: a line that, after optional
leading whitespace, starts with `** ` or `@@ ` (for a rehearsal build, the `@@ QuadSourceIntegral
triangle filter` line, then the `** CALCULATE QUADRATIC SOURCE INTEGRALS` banner), or the end of
the file. Every line of the block is printed exactly as the log holds it, blank lines and trailing
whitespace included, and nothing else is printed.

Exit status: 0 when the block appears exactly once; 1 when it is absent or appears more than once
(a message then goes to stderr, and nothing to stdout); 2 on a usage error.
"""

import re
import sys
from pathlib import Path

HEADER = "** GkSourcePolicyData SUMMARY STATISTICS"
NEXT_SECTION = re.compile(r"^\s*(\*\*|@@) ")


def main(argv) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    log = Path(argv[0]) / "stdout.log"
    lines = log.read_text(errors="replace").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == HEADER]
    if len(starts) != 1:
        print(
            f"{log}: {HEADER!r} appears {len(starts)} time(s); exactly one is required",
            file=sys.stderr,
        )
        return 1
    start = starts[0]
    end = start + 1
    while end < len(lines) and not NEXT_SECTION.match(lines[end]):
        end += 1
    sys.stdout.write("".join(lines[start:end]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
