"""
Reading the text of ``tools/inventory_report.format_inventory_report`` back into its parts, for
the tests of the display (store-fingerprint prompt 03). Not a test module; the tests import it.

A report is a title and summary lines, then categories (``   -- <title>``), each holding class
sections. A section starts with its header (``      @@ <class> (<where>): ...``) and holds, in
order: an optional ``common to every record:`` line, a ``tag set`` line per tag set where the class
is tagged, its record lines (``            - ...``), an ``... and N more`` line where records are
not shown, and its problems (``            !! ...``).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HEADER = re.compile(r"^      @@ (\w+) \((replicated|sharded)\): ([\d,]+) records?(.*)$")
TAG_SET = re.compile(r"^         tag set (\d+) of (\d+) \(([\d,]+) records?\): (.*)$")
COMMON = "         common to every record: "
RECORD = "            - "
MORE = re.compile(r"^            \.\.\. and ([\d,]+) more$")
PROBLEM = "            !! "

HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass
class Section:
    name: str
    where: str
    count: int
    header: str
    common: Optional[str] = None
    # (tag-set number, or 0 for an untagged class; the tag-set line's labels; record count)
    tag_sets: List[Tuple[int, str, int]] = field(default_factory=list)
    # (tag-set number, or 0; the record line without its bullet)
    records: List[Tuple[int, str]] = field(default_factory=list)
    more: int = 0
    problems: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def sections(report: str) -> Dict[str, Section]:
    """class name -> its section."""
    out: Dict[str, Section] = {}
    current: Optional[Section] = None
    group = 0
    for line in report.split("\n"):
        header = HEADER.match(line)
        if header is not None:
            current = Section(
                name=header.group(1),
                where=header.group(2),
                count=int(header.group(3).replace(",", "")),
                header=line,
            )
            if current.name in out:
                raise AssertionError(f"class {current.name} is reported twice")
            out[current.name] = current
            group = 0
            current.lines.append(line)
            continue
        if line.startswith("   -- ") or current is None:
            current = None if line.startswith("   -- ") else current
            continue
        current.lines.append(line)
        tag_set = TAG_SET.match(line)
        more = MORE.match(line)
        if line.startswith(COMMON):
            current.common = line[len(COMMON) :]
        elif tag_set is not None:
            group = int(tag_set.group(1))
            current.tag_sets.append(
                (group, tag_set.group(4), int(tag_set.group(3).replace(",", "")))
            )
        elif line.startswith(RECORD):
            current.records.append((group, line[len(RECORD) :]))
        elif more is not None:
            current.more += int(more.group(1).replace(",", ""))
        elif line.startswith(PROBLEM):
            current.problems.append(line[len(PROBLEM) :])
    return out


def split_top_level(text: str) -> List[str]:
    """``text`` split at every ", " that is outside braces, brackets and quotes."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        c = text[i]
        if quote is not None:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        elif depth == 0 and text.startswith(", ", i):
            parts.append(text[start:i])
            start = i + 2
            i += 1
        i += 1
    parts.append(text[start:])
    return parts


def fields_of(line: str) -> List[Tuple[str, str]]:
    """The top-level ``field=value`` pairs of a record line or a common line, in order, without the
    ``[unvalidated]`` / ``| values:`` markers that follow them."""
    body = line.split("  ")[0]
    out = []
    for part in split_top_level(body):
        name, _, value = part.partition("=")
        out.append((name, value))
    return out
