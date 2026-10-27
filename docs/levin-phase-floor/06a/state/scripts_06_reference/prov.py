"""levin-phase-floor prompt 06 section 6 (the re-measurement of 05's 39 rows on A3 v2, with the
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
WIDER = HERE.parent  # var/levin-phase-floor-06/wider  (prompt 06)
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
assert HERE.parents[3] == MAIN, HERE

# prompt 06: the trees t5 (the code commit's parent, 05a's t) and s (the code commit), both worktrees
T5_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-9f66622")
T5_SHA = "9f66622d5ced82de9a6ec0c20baa9a2e715c6d27"
S_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-f5cee4c")
S_SHA = "f5cee4c7a76e755119d923341f204f139b29be7e"  # Part B's dispatch SHA; its code is dc9d28c's
CODE_COMMIT = "dc9d28c"
TREES = {"t5": (T5_TREE, T5_SHA), "s": (S_TREE, S_SHA)}

# prompt 06: 05a's run, U0's reference for t5; 05's run, convergence_kinds.py's --capped-run
RUN05A = MAIN / "var" / "runs" / "levin-phase-floor-05a-wider-20261026T123047"
RUN05 = MAIN / "var" / "runs" / "levin-phase-floor-05-wider-20261026T020005"
SAMPLE_SHA256 = "73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7"

PYTHON = MAIN / "venv" / "bin" / "python"
STORE_DIR = MAIN / "var" / "datastores"
PRIMARY = STORE_DIR / "handover-A3-v2-lambdacdm.sqlite"
CENSUS = MAIN / "var" / "levin-phase-floor-01" / "out" / "groups_01.csv"
A3V2_DIGEST = "0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56"

# prompt 06, route (B) (the user's decision, 2026-10-26): s cannot open A3 v2 (its factories
# declare theta_residual, which A3 v2's value tables lack), so it opens, read-only, the unregistered
# scratch copy make_copy.py built and certify_copy.py / fingerprint_copy.py certified: A3 v2 with
# the three nullable, all-NULL columns added. t5 opens A3 v2 itself.
COPY_DIR = WIDER / "state" / "a3v2_copy"
STORE_DIR_BY_TREE = {"t5": STORE_DIR, "s": COPY_DIR}


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
    for p in inputs:
        lines.append(f"# input {p} sha256 {sha256(p)}")
    return lines
