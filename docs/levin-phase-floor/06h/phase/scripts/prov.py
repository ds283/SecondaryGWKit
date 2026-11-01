"""levin-phase-floor prompt 06h section 5, the phase re-measurement (06f's M2 items 1 and 4 repeated at
06h's code, on the Mac Studio): provenance shared by every script. Copied from prompt 06f's
``scripts/prov.py`` (docs/levin-phase-floor/06f/scripts/prov.py, a record, not edited; its copy is
kept unchanged in state/scripts_06f_reference/scripts/); changed here only in its trees and its
paths, each change marked "prompt 06h":
- two trees, both detached worktrees (Part B's dispatch brief): ``h``, Part B's dispatch SHA
  ``c1ec3b2`` (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2``), whose code is 06h's
  code commit 7172a8f's plus its tests-only follow-up ac1c6f4; and ``p``, the code commit's parent
  ``2c325f5`` (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-2c325f5``), whose code is 06e's
  (H1's parent column). 06f's ``g`` is named below for the record and not used;
- the scripts sit one directory deeper than 06f's (``var/levin-phase-floor-06h/phase/scripts``), so
  ``WIDER`` is ``var/levin-phase-floor-06h/phase`` and the depth assertion is one level more;
- the census record each tree's captures are checked against: at ``p`` 06e's census run (read only,
  as in 06f), at ``h`` 06h's census run (``levin-phase-floor-06h-census-*``, the census agent's,
  read only), located by ``census_run_06h()`` once it exists.

(06f's text follows, unchanged.)

levin-phase-floor prompt 06f section 1 (a measurement of two Clenshaw-Curtis floor terms on calls
captured from 06e's census, on the Mac Studio): provenance shared by every script. Copied from
prompt 06e's census ``scripts/prov.py`` (docs/levin-phase-floor/06e/census/scripts/prov.py, a
record, not edited; its copy is kept unchanged in state/scripts_06e_reference/census_scripts/);
changed here only in its tree and its paths, each change marked "prompt 06f":
- one tree, ``g``, the detached worktree at 06f's dispatch SHA ``f4770d13``
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1``), whose code is 06e's code commit
  f4d8e17's (``git diff --stat f4d8e17 f4770d13`` outside docs/ and prompts/ shows CLAUDE.md
  only); 06e's ``p`` and ``g`` worktrees are named below for the record and not used;
- the scripts sit one directory shallower than 06e's (``var/levin-phase-floor-06f/scripts``), so
  ``WIDER`` is ``var/levin-phase-floor-06f`` and the depth assertion is one level less;
- 06e's census run, its records and its ``rows.json``, read only.

(06e's text follows, unchanged.)

levin-phase-floor prompt 06e section 5 (06c's census re-run at 06e's code, exact end nodes and
the measured node term, on the Mac Studio): provenance shared by every script. Copied from prompt
06c's census ``scripts/prov.py`` (docs/levin-phase-floor/06c/census/scripts/prov.py, a record, not
edited; its copy is kept unchanged in state/scripts_06c_reference/); changed here only in its trees
and its paths (prompt 06e section 5: "Change only: the tree and its paths; ..."), each change marked
"prompt 06e".

Two trees take part, both detached git worktrees (Part B's dispatch brief):
- ``p``, 06e's code commit f4d8e17's parent ``50277f3``
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-50277f3``), whose code is 06c's census's
  (e5764a6's; ``git diff --stat e5764a6 50277f3`` outside docs/ and prompts/ shows CLAUDE.md only):
  Gate Z0's reproduction of 06c;
- ``g``, Part B's dispatch SHA ``6d81fad``, whose code is 06e's code commit f4d8e17's
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad``): the census and Z3's references.
06c's census tree was ``e5764a6``; it is read only through git history. 06c's worktrees
(``-lpf06c-e5764a6``, ``-lpf06c-fef401c``) are not used. The main checkout (``MAIN``, at
``28ebaf0``) holds the venv, the stores, ``var/`` and the registry, and no tree code runs from it.
Nothing is written in either worktree. Both trees read, read-only, 06's certified scratch copy of
A3 v2.

(06c's text follows, unchanged.)

levin-phase-floor prompt 06c section 5 (06b's census re-run under the global error budget, on the
Mac Studio): provenance shared by every script. Copied from prompt 06b's census ``scripts/prov.py``
(docs/levin-phase-floor/06b/census/scripts/prov.py, a record, not edited; its copy is kept
unchanged in state/scripts_06b_reference/); changed here only in its trees and its paths (prompt
06c section 5: "Change only: the tree and its paths; ..."), each change marked "prompt 06c".

Two trees take part, both detached git worktrees (Part B's dispatch brief):
- ``p``, 06c's code commit ad9a5f4's parent ``fef401c``
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c``), whose code is 06b's census's
  (92bc62f's): Gate X0's reproduction of 06b;
- ``g``, Part B's dispatch SHA ``e5764a6``, whose code is 06c's ad9a5f4 plus 06d's 3d70db9
  (``/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6``): the census.
06b's census tree was ``92bc62f``; it is read only through git history. The main checkout
(``MAIN``, at ``28ebaf0``) holds the venv, the stores, ``var/`` and the registry, and no tree code
runs from it. Nothing is written in either worktree. Both trees read, read-only, 06's certified
scratch copy of A3 v2.

(06b's text follows, unchanged.)

levin-phase-floor prompt 06b section 5 (06a's census re-run at the code commit, on the Mac Studio):
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
WIDER = HERE.parent  # prompt 06h: var/levin-phase-floor-06h/phase; prompt 06c: var/levin-phase-floor-06c/census (the name WIDER kept); prompt 06e: var/levin-phase-floor-06e/census
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE  # prompt 06h: var/levin-phase-floor-06h/phase/scripts (06f: parents[2]; 06b-06e: parents[3])

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
# prompt 06c: 06b's trees are kept above, by name, for the record; the two trees of 06c section 5
B06B_TREE, B06B_SHA = B_TREE, B_SHA  # 06b's census tree (b), read only through git history
P_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c")
P_SHA = "fef401c2614da3dbbf1b4261edd31f5f7f80cdbf"  # 06c's code commit ad9a5f4's parent
G_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6")
G_SHA = "e5764a6708757cb11704b95bcd66825baab4ba29"  # Part B's dispatch SHA; code = ad9a5f4 + 3d70db9
CODE_SHA_06C = "ad9a5f4"  # 06c's code commit (short), for the record
CODE_SHA_06D = "3d70db9"  # 06d's code commit (short), for the record
TREES = {"p": (P_TREE, P_SHA), "g": (G_TREE, G_SHA)}  # prompt 06c
# prompt 06e: 06c's trees are kept above, by name, for the record; the two trees of 06e section 5
P06C_TREE, P06C_SHA = P_TREE, P_SHA  # 06c's parent tree (fef401c), not used
G06C_TREE, G06C_SHA = G_TREE, G_SHA  # 06c's census tree (e5764a6), read only through git history
P_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-50277f3")
P_SHA = "50277f3e019bb37df6d91a5c2340877755498508"  # 06e's code commit f4d8e17's parent (06d's code)
G_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad")
G_SHA = "6d81fad43cbf1f17cff8d0f48fc7b65966e343ab"  # Part B's dispatch SHA; code = f4d8e17's
CODE_SHA_06E = "f4d8e17"  # 06e's code commit (short), for the record
TREES = {"p": (P_TREE, P_SHA), "g": (G_TREE, G_SHA)}  # prompt 06e
# prompt 06f: 06e's trees are kept above, by name, for the record; the one tree of 06f section 1
P06E_TREE, P06E_SHA = P_TREE, P_SHA  # 06e's parent tree (50277f3), not used
G06E_TREE, G06E_SHA = G_TREE, G_SHA  # 06e's census tree (6d81fad), not used
G_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1")
G_SHA = "f4770d13f86674381b3a0ef48b7ba05f8dda4934"  # 06f's dispatch SHA; code = f4d8e17's
TREES = {"g": (G_TREE, G_SHA)}  # prompt 06f
# prompt 06f: 06e's census run (the population) and its records, read only
RUN06E = MAIN / "var" / "runs" / "levin-phase-floor-06e-census-20261030T081318"
CHECKPOINT06E_SHA256 = "a0dc4d7f5c47eb28d8691b450559d005a76e1d9f3bd41442e249a419fb15904c"
REC06E = MAIN / "var" / "levin-phase-floor-06e" / "census"
ROWS06E = REC06E / "state" / "rows.json"
ROWS06E_SHA256 = "aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417"

# prompt 06h: 06f's tree is kept above, by name, for the record; the two trees of 06h section 5's
# phase re-measurement. Each capture names the tree it runs in, and is checked against that tree's
# census: p against 06e's census (RUN06E above), h against 06h's.
G06F_TREE, G06F_SHA = G_TREE, G_SHA  # 06f's tree (f4770d13), not used
H_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2")
H_SHA = "c1ec3b2c819cc5df22ddf4b303f69c7129a54d62"  # Part B's dispatch SHA; code = 7172a8f + ac1c6f4
P_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-2c325f5")
P_SHA = "2c325f56641666b45df483adb8c1202575721c42"  # 06h's code commit's parent; code = 06e's
CODE_SHA_06H = "7172a8f"  # 06h's code commit (short), for the record
TREES = {"h": (H_TREE, H_SHA), "p": (P_TREE, P_SHA)}  # prompt 06h
# prompt 06h: 06f's records (the sample, the 06f nodes 06f measured), read only
REC06F = MAIN / "var" / "levin-phase-floor-06f"
RUNS = MAIN / "var" / "runs"


def census_run_06h():
    """prompt 06h: 06h's census run directory, the census agent's (read only). Exactly one
    ``levin-phase-floor-06h-census-*`` directory must exist, and it must hold a checkpoint."""
    found = sorted(RUNS.glob("levin-phase-floor-06h-census-*"))
    if len(found) != 1 or not (found[0] / "checkpoint.jsonl").exists():
        raise SystemExit(f"06h's census run: expected exactly one with a checkpoint, found {found}")
    return found[0]


def census_run(tree_name):
    """prompt 06h: the census whose record a capture at ``tree_name`` must reproduce."""
    return {"p": RUN06E, "h": None}[tree_name] or census_run_06h()


# prompt 06e: 06c's census run (the "before") and records, read only
RUN06C = MAIN / "var" / "runs" / "levin-phase-floor-06c-census-20261029T130214"
REC06C = MAIN / "var" / "levin-phase-floor-06c" / "census"
CHECKPOINT06C_SHA256 = "b61cb034c1ee95217fae6edb3d14ace0193d2ff88ce66ee0bd642099873012f7"
# prompt 06e: depth20-tiling's records (Z3's sample, captured inputs and references), read only
D20T = MAIN / "var" / "levin-phase-floor-06c" / "depth20-tiling"

# prompt 06c: 06b's census run (the "before") and records, read only
RUN06B = MAIN / "var" / "runs" / "levin-phase-floor-06b-census-20261028T161212"
REC06B = MAIN / "var" / "levin-phase-floor-06b" / "census"
CHECKPOINT06B_SHA256 = "f934ea245c0086529fb9372a9ad457d4d8668114ce346dca5440404c766646ad"

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
STORE_DIR_BY_TREE = {"p": COPY_DIR, "g": COPY_DIR, "h": COPY_DIR}  # prompt 06c: both trees read the copy; prompt 06h: h too


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
