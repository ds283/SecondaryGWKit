"""levin-phase-floor prompt 06b section 5 (06a's census re-run at the code commit, on the Mac Studio):
provenance shared by every script. Copied from prompt 06a's ``scripts/prov.py``
(docs/levin-phase-floor/06a/scripts/prov.py, a record, not edited; its copy is kept unchanged in
state/scripts_06a_reference/); changed here only in its trees and its paths (prompt 06b section 5:
"Change only: the tree and its paths; ..."), each change marked "prompt 06b".

Two trees take part, both detached git worktrees (the dispatch note, B.5 addendum):
- ``p``, the code commit 4ddc3c2's parent ``da5c24d``
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-da5c24d``): Gate W0's reproduction of 06a;
- ``b``, Part B's dispatch SHA ``92bc62f``, whose code is the code commit 4ddc3c2's
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f``): the census.
06a's own tree was ``e05c34f`` (06a's h); it is read only through git history.
The main checkout (``MAIN``, at ``28ebaf0``) holds the venv, the stores, ``var/`` and the registry,
and no tree code runs from it. Nothing is written in either worktree. Both trees read, read-only,
06's certified scratch copy of A3 v2 (route (B), as in 06 and 06a).

(06a's text follows, unchanged.)

levin-phase-floor prompt 06a (the depth census of every A3 v2 QuadSourceIntegral row at the
current code, on the Mac Studio): provenance shared by every script. Copied from prompt 06's
``scripts/prov.py`` (docs/levin-phase-floor/06/wider/scripts/prov.py, a record, not edited; its copy
is kept unchanged in state/scripts_06_reference/); changed here only in its tree and its paths
(prompt 06a section 2 item 1: "One tree, h, in place of t5 and s"), each change marked "prompt 06a".

One tree takes part, ``h``: the detached worktree at the dispatch SHA ``e05c34f``
(``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06a-e05c34f``), whose code outside docs/ and
prompts/ is 06's s tree's (f5cee4c) byte for byte (the Gate's ``git diff --stat`` shows CLAUDE.md
alone). The main checkout (``MAIN``, at ``28ebaf0``) holds the venv, the stores, ``var/`` and the
registry, and no tree code runs from it. Nothing is written in the worktree.

Route (B), as in 06: h reads, read-only, 06's certified scratch copy of A3 v2
(var/levin-phase-floor-06/wider/state/a3v2_copy/), since no post-06 tree can open A3 v2 itself.

(06's text follows, unchanged; it describes 06's two trees, which are not used here.)

levin-phase-floor prompt 06 section 6 (the re-measurement of 05's 39 rows on A3 v2, with the
producers re-run in-process): provenance shared by every script. Copied from prompt 05a's
``scripts/prov.py`` (docs/levin-phase-floor/05a/wider/scripts/prov.py, a record, not edited);
changed here only in its trees and its paths (prompt 06 section 6: "Change only: the tree names and
paths; ..."), each change marked "prompt 06".

Two trees take part (prompt 06 section 6), both git worktrees, detached (orchestrator note B.5):
- T5_TREE, the code commit's parent ``9f66622``, whose code is 05a's t: the worktree
  ``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-9f66622``. Its consumers, and the stored
  producer samples. U0's producer-faithfulness clause runs here too.
- S_TREE, Part B's dispatch SHA ``f5cee4c``, whose code is the code commit ``dc9d28c``'s (the
  dispatch commit adds only records under ``prompts/``): the worktree
  ``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-f5cee4c``. Its producers re-run on each row's
  objects, and its consumers.
The main checkout (``MAIN``, at ``28ebaf0``) holds the venv, the store, ``var/`` and the registry,
and no tree runs from it. Nothing is written in either worktree.

(05a's text follows, unchanged; read its c as t5 and its t as s.)

levin-phase-floor prompt 05a section 6 (the wider re-measurement, with section 9's T1): provenance
shared by every script. Copied from prompt 05's ``scripts/prov.py``
(docs/levin-phase-floor/05/wider/scripts/prov.py, sha256 e8579509..., a record, not edited); changed
here only in its trees and its paths (prompt 05a section 6: "the scripts name 05's paths, its trees
p and c ... Change only those"), each change marked "prompt 05a".

Every output header records the SHA-256 of the script that wrote it and both trees' HEADs and
``git status --short``. Read-only git only (rev-parse, status).
"""

import datetime
import hashlib
import os
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDER = HERE.parent  # prompt 06b: var/levin-phase-floor-06b/census (the name WIDER kept)
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE  # prompt 06b: one directory deeper than 06a's

# prompt 06a: one tree, h, the worktree at the dispatch SHA (code = 06's s tree's, f5cee4c)
H_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06a-e05c34f")
H_SHA = "e05c34f2fcc6ef20fc3323cb3a6f81d130c5a01c"
S06_SHA = "f5cee4c7a76e755119d923341f204f139b29be7e"  # 06's s tree, whose code h's equals
# prompt 06b: the two trees of 06b section 5 (06a's h is kept above, by name, for the record)
P_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-da5c24d")
P_SHA = "da5c24deb430172f68d983f56577fa48def19903"  # the code commit 4ddc3c2's parent
B_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f")
B_SHA = "92bc62f09711d10ef0035663d4bbe5629e145e17"  # Part B's dispatch SHA; code = 4ddc3c2's
CODE_SHA = "4ddc3c2"  # the code commit (short), for the record
TREES = {"p": (P_TREE, P_SHA), "b": (B_TREE, B_SHA)}

# prompt 06b: 06a's run (the "before") and records, read only
RUN06A = MAIN / "var" / "runs" / "levin-phase-floor-06a-census-20261027T023143"
REC06A = MAIN / "var" / "levin-phase-floor-06a"
ROWS06A_SHA256 = "aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417"

# prompt 06a: 06's run (C0 (c) and C4 compare against its s units), 06's records (read only)
RUN06 = MAIN / "var" / "runs" / "levin-phase-floor-06-wider-20261026T223634"
REC06 = MAIN / "var" / "levin-phase-floor-06" / "wider"
SAMPLE05 = REC06 / "state" / "sample.json"  # 05's sample, as 06 kept it
SAMPLE_SHA256 = "73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7"

PYTHON = MAIN / "venv" / "bin" / "python"
STORE_DIR = MAIN / "var" / "datastores"
PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"
CENSUS = MAIN / "var" / "levin-phase-floor-01" / "out" / "groups_01.csv"
A3V2_DIGEST = "0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56"

# route (B), as in 06 (the user's decision, 2026-10-26): h opens, read-only, 06's certified scratch
# copy of A3 v2. prompt 06a: the copy is 06's, under var/levin-phase-floor-06/wider/state/.
COPY_DIR = REC06 / "state" / "a3v2_copy"
STORE_DIR_BY_TREE = {"p": COPY_DIR, "b": COPY_DIR}  # prompt 06b: both trees read the copy


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_state(tree):
    """(HEAD, git status --short) of a tree, read-only."""
    # prompt 06: --no-optional-locks, so that ``status`` does not refresh (write) the index
    head = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(tree), "status", "--short"],
        capture_output=True,
        text=True,
    ).stdout
    return head, status


def header(script_path, title, inputs=()):
    """Provenance header lines (each starting with '# ')."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# {title}",
        f"# written {now} on {socket.gethostname()}",
        f"# script {os.path.basename(str(script_path))} sha256 {sha256(script_path)}",
    ]
    for name, (tree, _sha) in TREES.items():
        head, status = tree_state(tree)
        lines.append(
            f"# tree {name}: {tree} HEAD {head} (status --short: "
            f"{'empty' if status.strip() == '' else repr(status)})"
        )
    # prompt 06a: the main checkout too (it supplies the venv, the stores and var/)
    mhead, mstatus = tree_state(MAIN)
    lines.append(
        f"# main checkout: {MAIN} HEAD {mhead} (status --short: "
        f"{'empty' if mstatus.strip() == '' else repr(mstatus)})"
    )
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
