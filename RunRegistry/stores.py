"""The store sidecar, and the registry's operations on the stores it manages.

A `ShardedPool` datastore is a primary SQLite file and its shards. Beside the primary there may be
a **sidecar**, `<stem>.manifest.json`, which is where a person learns what the store is for and
where it came from. `ShardedPool` and `tools/sharded_store.py` know nothing about it
(`prompts/datastore-portability` README §6.3). This module is its one owner (README §6.5): the
name (`sidecar_path`), the format, the reader (`read_sidecar`) and the writer are defined here and
nowhere else, and only the operations below write a sidecar.

**The format.** A *registry sidecar* is a JSON object with `"sidecar_format": 1`. Its fields:

    sidecar_format  1; its presence is what tells a registry sidecar from a legacy one
    store_id        32 lowercase hex characters. New on create, adopt and copy; kept on a move
    datastore       the primary's bare file name. Never a path: a path goes stale when the store
                    moves, and the backup's legacy sidecar names the live store for that reason
    name            the primary's stem
    purpose         one line a stranger can read
    created         when this store identity began (`now_iso()`); adopt keeps a legacy value
    copied_from     optional: the immediate parent, {"store_id", "datastore"} for a registry
                    copy; a legacy string is kept verbatim
    history         append-only, one entry per operation, the first being the create or adopt
                    that assigned the store_id: {"operation", "from", "to", "when", "git_head",
                    "git_dirty"}. After it, `copy`, `move` and `amend`; and `retire`, which may
                    stand only as the last entry, and only once. A `retire` entry's `from` is the
                    primary's path and its `to` is null, because a retirement has no destination.
                    An `amend` entry's `from` and `to` are both null, because it has neither a
                    source nor a destination, and it carries four more keys: `field`, `reason`,
                    and `before` and `after`, each {"present": true, "value": <the field's value>}
                    or {"present": false} — a marker one level above the value, never the value
                    itself, so that no JSON value a field could hold is ever mistaken for it.
                    Every other entry's `from` and `to` are both non-empty paths
    fingerprint     optional: the store's content fingerprint (`fingerprint_of`), as
                    {"fingerprint_format", "classes", "digest", "problems", "taken"}. Written only
                    by `fingerprint_store` when asked, which a registered run's
                    `Run.finish(..., fingerprint=True)` does. Copy and move carry it verbatim,
                    `taken` included, because neither changes the content it describes. A
                    retirement keeps it unchanged: it is what says what the retired store held
    retired         present only on a **tombstone**, the sidecar `retire_store` leaves behind
                    when it deletes the store's files, and then always with a `retire` entry
                    last in `history`:
                      state        "retiring" from the first write, while the deletion is under
                                   way; "retired" once no listed file exists
                      when         `now_iso()` at the first write, the `retire` entry's `when`
                      git_head     `git_provenance()` at the first write
                      git_dirty
                      reason       the required text saying why (README D3)
                      fingerprint  how the fingerprint condition was met: {"condition":
                                   "matched", "digest": <the recorded overall digest, which a
                                   fresh read-only fingerprint matched>}, or under
                                   `--without-fingerprint` {"condition": "without", "error": <the
                                   text of what prevented a fingerprint, verbatim>,
                                   "error_type": <its exception's name>}
                      files        the files to be deleted, written before any is: the shards in
                                   ascending serial, then the primary, as `_repo_path` writes
                                   paths (`ShardedPool.closed_store_files`)
                      files_present_only  true when a shard was already missing, so that
                                   `files` lists the files present (`--without-fingerprint` only)
                      references   what referenced the store when it was retired: {"runs_root",
                                   "stores_root", "runs": [{"id", "state", "matched_by"}],
                                   "sidecars": [{"sidecar", "fields": [JSON paths]} or
                                   {"sidecar", "unreadable"}], "not_searched"}
                      completed    `now_iso()` when the state became "retired", null until then

Every other field is **unknown**, and every write preserves it verbatim, meaning value-identical
after a JSON round trip (`write_json_atomic` re-indents and sorts keys). Unknown fields are carried
and never interpreted, even ones holding paths, which may be stale after a move.

**A tombstone** reads as a registry sidecar whose `retired` property is true and whose `ok` is
**false**: `ok` means a problem-free sidecar describing a store that is there, and every operation
that tests it refuses a tombstone. A completed tombstone, whose listed files and primary are all
gone, has no problems. An incomplete retirement is a problem that names what remains, and so is a
completed one whose primary or listed files exist again, since a process that opened the retired
path without the registry has made a new store at a name that is never reused.

The paths in `copied_from` and `history` say where something was **at the time**, written as
`_repo_path` writes paths. Nothing opens them or finds a store from them. Only `datastore` says
where the store is, and it says it by name, relative to the sidecar it sits in.

A *legacy* sidecar is any other JSON object, such as the two written by hand or by a script
before this module existed. It is read as it is and never rewritten except by an explicit
`adopt_sidecar`. A legacy `datastore` that is a path is read by its final component, never as a
path: that is prompt 01's rule for legacy shard records, applied to JSON.

**The operations** are `create_sidecar`, `adopt_sidecar`, `copy_store`, `move_store`,
`fingerprint_store`, `retire_store` and `amend_sidecar`. Copy and move call
`ShardedPool.copy_store` / `move_store` for the store's files, then carry the sidecar. Copy, move,
fingerprint, retire and amend refuse a store that any `running` run names, alive or stale; that
check is what the bare script cannot make. It is a check and not a lock: a run begun after it is
not seen. Every operation but a retirement's completion refuses a tombstone.

`retire_store` alone deletes, and only a store's own files: the shards its primary names, read
through the one resolver, and then the primary, through `ShardedPool.delete_store`. It keeps the
sidecar, as the store's tombstone, and it deletes no sidecar, run directory or other record.
Nothing else here deletes a file. Nothing overwrites one except the in-place updates
`_update_sidecar` names (adopt, the move's temporary sidecar, recording a fingerprint, a
retirement's two writes, and an amendment), and nothing cleans up after a failure or decides that a
run is over.

`amend_sidecar` (store-retirement prompt 04) replaces or removes exactly one **unknown** field of
a registry sidecar, with a required reason, and records the field's old value in an `amend` history
entry, so that a present-tense claim an unknown field makes (such as the live A3 sidecar's
`backup.retained`) can be corrected by a person without a hand edit of a sidecar, and nothing is
ever lost. It never touches a known field, `retired` included: those belong to the operation that
owns them. It never opens the store's files; it reads and writes only the sidecar.

**The fingerprint** (store-fingerprint prompt 04) says what a closed store holds, as digests of its
structured inventory (`Datastore.store_inventory.read_inventory`): per class and per tag set a
count and a SHA-256 digest, and one overall digest. It never holds a listing; `listing_lines`
generates one on demand, whose lines hash to the digests it lists. `fingerprint_of` and
`compare_fingerprints` are pure. `fingerprint_store` reads the store read-only, with no Ray, and
writes only the sidecar's `fingerprint` field, and only when asked.

`ShardedPool` is imported inside `copy_store`, `move_store` and `retire_store` only, and
`Datastore.store_inventory` inside the fingerprint functions only, so that importing this module,
like `import RunRegistry`, loads neither `ray` nor `sqlalchemy`.
"""

import copy as _copy
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import (
    DEFAULT_ROOT,
    DEFAULT_STALE_AFTER,
    REPO_ROOT,
    _repo_path,
    _resolve,
    git_provenance,
    list_runs,
    now_iso,
    read_json,
    write_json_atomic,
)

SIDECAR_FORMAT = 1

# the fields this module defines; every other field is unknown and is carried verbatim
REQUIRED_FIELDS = (
    "sidecar_format",
    "store_id",
    "datastore",
    "name",
    "purpose",
    "created",
    "history",
)
KNOWN_FIELDS = REQUIRED_FIELDS + ("copied_from", "fingerprint", "retired")

# the operations a history entry may record. The first entry assigned the store_id; a retirement
# may stand only as the last
IDENTITY_OPERATIONS = ("create", "adopt")
RELOCATION_OPERATIONS = ("copy", "move")
RETIRE_OPERATION = "retire"
AMEND_OPERATION = "amend"
HISTORY_KEYS = ("operation", "from", "to", "when", "git_head", "git_dirty")
# an `amend` entry carries these four keys beside `HISTORY_KEYS`, and no other operation may
AMEND_KEYS = ("field", "reason", "before", "after")

# which operation owns each known field, named in `amend_sidecar`'s refusal of one (store-retirement
# prompt 04 §2.1)
_FIELD_OWNERS = {
    "sidecar_format": "create or adopt",
    "store_id": "create, adopt or copy",
    "datastore": "create, adopt, copy or move",
    "name": "create, adopt, copy or move",
    "purpose": "create, adopt or copy",
    "created": "create, adopt or copy",
    "history": "create, adopt, copy, move, retire or amend",
    "copied_from": "copy",
    "fingerprint": "fingerprint",
    "retired": "retire",
}

# a tombstone's `retired.state`: under way, then complete
RETIREMENT_STATES = ("retiring", "retired")

# where `retire_store` looks for sidecars that reference the store it retires, by default
DEFAULT_STORES_ROOT = os.path.join(REPO_ROOT, "var", "datastores")

# a move updates the sidecar under this name, in the destination directory, and renames it onto
# the destination's sidecar name last
INCOMPLETE_MOVE_SUFFIX = ".incomplete-move"

_STORE_ID = re.compile(r"[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")

# the fingerprint's format. A fingerprint is compared only with one of the same format; a change to
# what the inventory records, or to how a digest is formed, bumps this and regenerates the golden
# fingerprint (`RunRegistry/tests/data/full_store_fingerprint.json`) in the same commit
FINGERPRINT_FORMAT = 1
TAKEN_KEYS = ("when", "git_head", "git_dirty", "run_id")

# both separators, on every platform, as `Datastore/shard_paths.py` treats a shard record
_SEPARATORS = re.compile(r"[\\/]")


# =================================================================================================
# the name


def sidecar_path(primary) -> Path:
    """The sidecar of the store whose primary is ``primary``: `<stem>.manifest.json` beside it.
    This is the one place the pattern is spelled."""
    primary = Path(primary)
    return primary.parent / f"{primary.stem}.manifest.json"


def _tmp_path(path) -> Path:
    """The temporary name `write_json_atomic` writes through before its `os.replace`."""
    return Path(str(path) + ".tmp")


def _incomplete_move_path(sidecar) -> Path:
    sidecar = Path(sidecar)
    return sidecar.with_name(sidecar.name + INCOMPLETE_MOVE_SUFFIX)


def new_store_id() -> str:
    return uuid.uuid4().hex


# =================================================================================================
# the reader


@dataclass
class SidecarReading:
    """What `read_sidecar` found. ``kind`` is `absent`, `unreadable`, `legacy` or `registry`.
    ``problems`` is empty when the sidecar describes this primary. ``legacy_path`` is true when a
    legacy `datastore` is a path, which is read by its final component and is not a problem.
    ``incomplete_retirement`` is the text of the one problem an interrupted retirement is, when
    the sidecar is a tombstone in state `retiring`, and `None` otherwise.
    """

    primary: Path
    path: Path
    kind: str
    fields: Optional[dict] = None
    problems: List[str] = field(default_factory=list)
    legacy_path: bool = False
    incomplete_retirement: Optional[str] = None

    @property
    def retired(self) -> bool:
        """A registry sidecar that carries `retired`: a tombstone, complete or not."""
        return self.kind == "registry" and "retired" in (self.fields or {})

    @property
    def ok(self) -> bool:
        """A problem-free registry sidecar describing a store that is there. False for a
        tombstone, even a problem-free one, so that every operation testing it refuses one.
        """
        return self.kind == "registry" and not self.problems and not self.retired

    @property
    def store_id(self) -> Optional[str]:
        """The store_id of a problem-free registry sidecar, and `None` otherwise, a tombstone
        included. A caller that needs a tombstone's recorded store_id reads the field.
        """
        return self.fields["store_id"] if self.ok else None

    @property
    def unknown_fields(self) -> dict:
        # a legacy sidecar cannot be retired, so a `retired` key in one is unknown, as before
        known = (
            KNOWN_FIELDS
            if self.kind == "registry"
            else tuple(name for name in KNOWN_FIELDS if name != "retired")
        )
        return {k: v for k, v in (self.fields or {}).items() if k not in known}


def _final_component(value: str) -> str:
    return _SEPARATORS.split(value)[-1]


def _primary_problems(primary: Path) -> List[str]:
    if os.path.islink(primary):
        return [f'the primary "{primary}" is a symbolic link, not a regular file']
    if not os.path.lexists(primary):
        return [
            f'the primary "{primary}" does not exist, so this sidecar describes no store beside it'
        ]
    if not os.path.isfile(primary):
        return [f'the primary "{primary}" is not a regular file']
    return []


def _nonempty_string(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _history_problems(history) -> List[str]:
    if not isinstance(history, list) or len(history) == 0:
        return [f"history is {history!r}, not a non-empty list"]
    problems = []
    for index, entry in enumerate(history):
        where = f"history entry #{index}"
        if not isinstance(entry, dict):
            problems.append(f"{where} is {entry!r}, not an object")
            continue
        missing = [key for key in HISTORY_KEYS if key not in entry]
        if missing:
            problems.append(f"{where} lacks {missing}")
            continue
        operation = entry["operation"]
        allowed = (
            IDENTITY_OPERATIONS
            if index == 0
            else RELOCATION_OPERATIONS + (RETIRE_OPERATION, AMEND_OPERATION)
        )
        if operation not in allowed:
            problems.append(
                f"{where} records operation {operation!r}, where only {allowed} can stand"
            )
        if operation == RETIRE_OPERATION and index != len(history) - 1:
            problems.append(
                f"{where} records a retirement, which may stand only as the last entry, and "
                f"{len(history) - 1 - index} entr(y/ies) follow it"
            )
        if index == 0 and entry["from"] is not None:
            problems.append(f"{where} ({operation}) has a from, {entry['from']!r}")
        # an amend has neither a source nor a destination: both `from` and `to` are null. A
        # retirement has no destination: its `to` is null, and every other entry's `from` and
        # `to` are both non-empty paths
        if operation == AMEND_OPERATION:
            if entry["from"] is not None:
                problems.append(
                    f"{where} (amend) has a from, {entry['from']!r}, where an amend has none"
                )
        elif index > 0 and not _nonempty_string(entry["from"]):
            problems.append(f"{where} ({operation}) has no from")
        if operation == RETIRE_OPERATION:
            if entry["to"] is not None:
                problems.append(
                    f"{where} (retire) has a to, {entry['to']!r}, where a retirement has none"
                )
        elif operation == AMEND_OPERATION:
            if entry["to"] is not None:
                problems.append(
                    f"{where} (amend) has a to, {entry['to']!r}, where an amend has none"
                )
        elif not _nonempty_string(entry["to"]):
            problems.append(f"{where} has a malformed to, {entry['to']!r}")
        for key in ("when", "git_head"):
            if not _nonempty_string(entry[key]):
                problems.append(f"{where} has a malformed {key}, {entry[key]!r}")
        if not isinstance(entry["git_dirty"], bool):
            problems.append(
                f"{where} has a malformed git_dirty, {entry['git_dirty']!r}"
            )
        # AMEND_KEYS are required on an amend entry, and a problem on every other kind
        if operation == AMEND_OPERATION:
            missing_amend = [key for key in AMEND_KEYS if key not in entry]
            if missing_amend:
                problems.append(f"{where} (amend) lacks {missing_amend}")
            else:
                if not _nonempty_string(entry.get("field")):
                    problems.append(
                        f"{where} (amend) has a malformed field, {entry.get('field')!r}"
                    )
                if not _nonempty_string(entry.get("reason")):
                    problems.append(
                        f"{where} (amend) has a malformed reason, {entry.get('reason')!r}"
                    )
                for key in ("before", "after"):
                    problems.extend(
                        _amend_slot_problems(entry.get(key), f"{where} (amend)'s {key}")
                    )
        else:
            carried = [key for key in AMEND_KEYS if key in entry]
            if carried:
                problems.append(
                    f"{where} ({operation}) carries {carried}, which only an amend entry has"
                )
    return problems


def _amend_slot_problems(value, where: str) -> List[str]:
    """Every way ``value`` fails to be an amend history entry's `before` or `after`: an object
    ``{"present": true, "value": <the field's value>}`` or ``{"present": false}``. This tagged
    shape, not a sentinel string, is what an `amend` entry uses to say a field was absent or
    removed, so that no JSON value a field could hold is ever mistaken for the marker (README D5,
    prompt 04 §2.2): the marker is a wrapper one level above the value, never the value itself.
    """
    if not isinstance(value, dict) or not isinstance(value.get("present"), bool):
        return [f"{where} is {value!r}, not a well-formed amend marker"]
    keys = set(value)
    if value["present"]:
        if keys != {"present", "value"}:
            return [
                f"{where} is {value!r}: a present marker holds exactly present and value"
            ]
    else:
        if keys != {"present"}:
            return [f"{where} is {value!r}: an absent marker holds nothing but present"]
    return []


def _fingerprint_problems(value) -> List[str]:
    """Every way ``value`` fails to have a fingerprint's shape: a JSON object with an integer
    `fingerprint_format`, a `classes` object, a 64-hex `digest` and a `taken` object."""
    if not isinstance(value, dict):
        return [f"fingerprint is {value!r}, not an object"]
    problems = []
    fmt = value.get("fingerprint_format")
    if isinstance(fmt, bool) or not isinstance(fmt, int):
        problems.append(f"fingerprint's fingerprint_format {fmt!r} is not an integer")
    if not isinstance(value.get("classes"), dict):
        problems.append(
            f"fingerprint's classes {value.get('classes')!r} is not an object"
        )
    digest = value.get("digest")
    if not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
        problems.append(
            f"fingerprint's digest {digest!r} is not 64 lowercase hexadecimal characters"
        )
    if not isinstance(value.get("taken"), dict):
        problems.append(f"fingerprint's taken {value.get('taken')!r} is not an object")
    return problems


def _retired_problems(value) -> List[str]:
    """Every way ``value`` fails to have the shape of a tombstone's `retired` field (the module
    docstring's format table)."""
    if not isinstance(value, dict):
        return [f"retired is {value!r}, not an object"]
    problems = []
    state = value.get("state")
    if state not in RETIREMENT_STATES:
        problems.append(f"retired's state {state!r} is not one of {RETIREMENT_STATES}")
    for key in ("when", "git_head", "reason"):
        if not _nonempty_string(value.get(key)):
            problems.append(
                f"retired's {key} {value.get(key)!r} is not a non-empty string"
            )
    if not isinstance(value.get("git_dirty"), bool):
        problems.append(f"retired's git_dirty {value.get('git_dirty')!r} is malformed")

    condition = value.get("fingerprint")
    kind = condition.get("condition") if isinstance(condition, dict) else None
    if kind == "matched":
        digest = condition.get("digest")
        if not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
            problems.append(f"retired's fingerprint digest {digest!r} is malformed")
    elif kind == "without":
        for key in ("error", "error_type"):
            if not _nonempty_string(condition.get(key)):
                problems.append(
                    f"retired's fingerprint {key} {condition.get(key)!r} is malformed"
                )
    else:
        problems.append(
            f"retired's fingerprint {condition!r} is neither a matched digest nor a "
            f"retirement without one"
        )

    files = value.get("files")
    if not (
        isinstance(files, list)
        and len(files) > 0
        and all(_nonempty_string(f) for f in files)
    ):
        problems.append(f"retired's files {files!r} is not a non-empty list of paths")
    present_only = value.get("files_present_only")
    if not isinstance(present_only, bool):
        problems.append(
            f"retired's files_present_only {present_only!r} is not a boolean"
        )
    elif present_only and kind != "without":
        problems.append(
            "retired's files are of the files present, which only a retirement without a "
            "fingerprint records"
        )

    references = value.get("references")
    if not (
        isinstance(references, dict)
        and isinstance(references.get("runs"), list)
        and isinstance(references.get("sidecars"), list)
        and _nonempty_string(references.get("runs_root"))
        and _nonempty_string(references.get("stores_root"))
    ):
        problems.append(f"retired's references {references!r} are malformed")

    completed = value.get("completed")
    if state == "retiring" and completed is not None:
        problems.append(
            f"retired's completed is {completed!r} while its state is retiring"
        )
    if state == "retired" and not _nonempty_string(completed):
        problems.append(
            f"retired's completed is {completed!r} while its state is retired"
        )
    return problems


def _present(files, primary: Path) -> List[Path]:
    """Which of a tombstone's listed ``files`` exist, as entries of any kind, and the primary if
    it exists and is not among them."""
    present = [Path(_resolve(f)) for f in files if os.path.lexists(_resolve(f))]
    if os.path.lexists(primary) and os.path.realpath(primary) not in {
        os.path.realpath(p) for p in present
    }:
        present.append(primary)
    return present


def _tombstone_problems(
    retired: dict, primary: Path
) -> Tuple[List[str], Optional[str]]:
    """What a well-formed tombstone's files say: ``(problems, the incomplete retirement's text
    or None)``. A completed tombstone with nothing at its name has none."""
    present = _present(retired["files"], primary)
    names = ", ".join(f'"{p}"' for p in present)
    if retired["state"] == "retiring":
        if present:
            text = (
                f"the retirement begun {retired['when']} is incomplete: these files are still "
                f"present: [{names}]"
            )
        else:
            text = (
                f"the retirement begun {retired['when']} is incomplete: no listed file remains, "
                f"but its completion was not recorded"
            )
        text += (
            f'. Run `python -m RunRegistry store retire "{primary}" --reason <the recorded '
            f"reason>` again to complete it"
        )
        return [text], text
    if present:
        return [
            f"retired {retired['when']}, but {names} exists again. This sidecar describes the "
            f"store that was retired, not whatever now sits at its name; a process opened the "
            f"retired path without the registry, and a retired name is never reused"
        ], None
    return [], None


def _tombstone_text(reading) -> str:
    """What every operation but a retirement's completion says when it refuses a tombstone."""
    retired = (reading.fields or {}).get("retired")
    retired = retired if isinstance(retired, dict) else {}
    text = (
        f'its sidecar "{reading.path}" is a tombstone: the store was retired '
        f"{retired.get('when')}, because {retired.get('reason')!r}"
    )
    # an incomplete retirement, or a primary that exists again, is among the problems
    if reading.problems:
        text += "; " + "; ".join(reading.problems)
    return (
        text
        + ". A retired name is never reused, and a tombstone describes a store that is gone"
    )


def _registry_problems(fields: dict, primary: Path) -> List[str]:
    """Every way ``fields`` fails to be a registry sidecar describing ``primary``. Used by the
    reader, and by every writer on what it is about to write."""
    problems = []
    fmt = fields.get("sidecar_format")
    if isinstance(fmt, bool) or fmt != SIDECAR_FORMAT:
        problems.append(
            f"sidecar_format is {fmt!r}; this reader knows format {SIDECAR_FORMAT} only"
        )
    for name in REQUIRED_FIELDS:
        if name not in fields:
            problems.append(f"it lacks the required field {name!r}")

    if "store_id" in fields:
        value = fields["store_id"]
        if not (isinstance(value, str) and _STORE_ID.fullmatch(value)):
            problems.append(
                f"store_id {value!r} is not 32 lowercase hexadecimal characters"
            )
    if "datastore" in fields:
        value = fields["datastore"]
        if not _nonempty_string(value):
            problems.append(f"datastore {value!r} is not a file name")
        elif _SEPARATORS.search(value):
            problems.append(
                f"datastore {value!r} is a path; a registry sidecar names its store by bare "
                f"file name, and the registry never writes one"
            )
        elif value != primary.name:
            problems.append(
                f'datastore {value!r} does not name the primary beside it, "{primary.name}"'
            )
    if "name" in fields and fields["name"] != primary.stem:
        problems.append(
            f'name {fields["name"]!r} is not the primary\'s stem, "{primary.stem}"'
        )
    for name in ("purpose", "created"):
        if name in fields and not _nonempty_string(fields[name]):
            problems.append(f"{name} {fields[name]!r} is not a non-empty string")
    if "copied_from" in fields:
        value = fields["copied_from"]
        if isinstance(value, dict):
            parent = value.get("store_id")
            if not (isinstance(parent, str) and _STORE_ID.fullmatch(parent)):
                problems.append(f"copied_from's store_id {parent!r} is malformed")
            if not _nonempty_string(value.get("datastore")):
                problems.append(
                    f"copied_from's datastore {value.get('datastore')!r} is malformed"
                )
        elif value is not None and not isinstance(value, str):
            problems.append(
                f"copied_from {value!r} is neither a registry parent nor a legacy string"
            )
    if "history" in fields:
        problems.extend(_history_problems(fields["history"]))
    if "fingerprint" in fields:
        problems.extend(_fingerprint_problems(fields["fingerprint"]))

    # a tombstone and its history say the same thing: `retired` iff a `retire` entry, last
    history = fields.get("history")
    last = history[-1] if isinstance(history, list) and history else None
    ends_retired = isinstance(last, dict) and last.get("operation") == RETIRE_OPERATION
    if "retired" in fields:
        problems.extend(_retired_problems(fields["retired"]))
        if not ends_retired:
            problems.append(
                "it carries retired, but its history does not end in a retire entry"
            )
    elif ends_retired:
        problems.append(
            "its history ends in a retire entry, but it carries no retired field"
        )
    return problems


def read_sidecar(primary) -> SidecarReading:
    """What sits at `sidecar_path(primary)`, and whether it describes ``primary``.

    Never writes, and never raises on bad content. A legacy `datastore` that is a path is read by
    its final component, never as a path, so a legacy sidecar copied beside a copy of its store
    reads as naming the sibling it sits beside.
    """
    primary = Path(os.path.abspath(primary))
    path = sidecar_path(primary)
    if not os.path.lexists(path):
        return SidecarReading(primary, path, "absent")

    try:
        with open(path, "r") as handle:
            fields = json.load(handle)
    except (OSError, ValueError) as e:
        return SidecarReading(
            primary,
            path,
            "unreadable",
            problems=[f"it is not readable as JSON ({type(e).__name__}: {e})"],
        )
    if not isinstance(fields, dict):
        return SidecarReading(
            primary,
            path,
            "unreadable",
            problems=[f"it holds a JSON {type(fields).__name__}, not an object"],
        )

    problems = []
    if os.path.islink(path) or not os.path.isfile(path):
        problems.append("the sidecar itself is not a regular file")

    if "sidecar_format" in fields:
        # a well-formed tombstone describes a store that is gone, so the primary's absence is
        # not a problem; what is at its name is, instead. A malformed `retired` says nothing,
        # and the primary is checked as for any sidecar
        incomplete = None
        if "retired" in fields and not _retired_problems(fields["retired"]):
            tombstone, incomplete = _tombstone_problems(fields["retired"], primary)
            problems.extend(tombstone)
        else:
            problems.extend(_primary_problems(primary))
        problems.extend(_registry_problems(fields, primary))
        return SidecarReading(
            primary,
            path,
            "registry",
            fields,
            problems,
            incomplete_retirement=incomplete,
        )

    problems.extend(_primary_problems(primary))

    # legacy: read as it is. A path in `datastore` is read by its name; a missing `datastore` or
    # `name` makes no claim, and the sidecar's own file name is what ties it to this primary. A
    # `retired` key here is an unknown field, and uninterpreted: legacy sidecars cannot be retired
    legacy_path = False
    if "datastore" in fields:
        value = fields["datastore"]
        if not _nonempty_string(value):
            problems.append(f"datastore {value!r} is not a file name or path")
        else:
            named = _final_component(value)
            legacy_path = named != value
            if named != primary.name:
                problems.append(
                    f'datastore {value!r} names "{named}", not the primary beside it, "{primary.name}"'
                )
    if "name" in fields and fields["name"] != primary.stem:
        problems.append(
            f'name {fields["name"]!r} is not the primary\'s stem, "{primary.stem}"'
        )
    return SidecarReading(primary, path, "legacy", fields, problems, legacy_path)


# =================================================================================================
# the writer: two private functions over `write_json_atomic`, and nothing else writes a sidecar


def _check_before_writing(path: Path, fields: dict, primary: Path) -> None:
    problems = _registry_problems(fields, primary)
    if problems:
        raise RuntimeError(
            f'refusing to write the sidecar "{path}": what would be written has problems: '
            + "; ".join(problems)
        )


def _write_new_sidecar(path: Path, fields: dict, primary: Path) -> None:
    """Write a sidecar that does not exist yet. Refuses if ``path`` or its `.tmp` name exists:
    nothing is ever overwritten."""
    _check_before_writing(path, fields, primary)
    taken = [f'"{p}"' for p in (path, _tmp_path(path)) if os.path.lexists(p)]
    if taken:
        raise RuntimeError(
            f'refusing to write a new sidecar at "{path}": {", ".join(taken)} already exist, '
            f"and nothing is ever overwritten"
        )
    write_json_atomic(str(path), fields)


def _update_sidecar(path: Path, fields: dict, primary: Path) -> None:
    """Replace an existing sidecar's contents in place. Used in exactly five places: adopt, the
    move's temporary sidecar, `fingerprint_store` recording a fingerprint when asked (by a
    person, or by a registered run's finish), `retire_store`'s two writes of the tombstone,
    before and after the deletion, and `amend_sidecar`'s one write. Refuses if the `.tmp` name
    exists."""
    _check_before_writing(path, fields, primary)
    if os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError(
            f'refusing to update the sidecar "{path}": it is not an existing regular file'
        )
    if os.path.lexists(_tmp_path(path)):
        raise RuntimeError(
            f'refusing to update the sidecar "{path}": "{_tmp_path(path)}" already exists'
        )
    write_json_atomic(str(path), fields)


def _history_entry(operation, source, destination, provenance) -> dict:
    return {
        "operation": operation,
        "from": _repo_path(source) if source is not None else None,
        # null only for a retirement, which has no destination
        "to": _repo_path(destination) if destination is not None else None,
        "when": now_iso(),
        "git_head": provenance["git_head"],
        "git_dirty": provenance["git_dirty"],
    }


# =================================================================================================
# create and adopt


def _refuse_interrupted_move(primary: Path, operation: str) -> None:
    pending = _incomplete_move_path(sidecar_path(primary))
    if os.path.lexists(pending):
        raise RuntimeError(
            f'Cannot {operation} a sidecar for "{primary}": "{pending}" exists, which is the '
            f"sidecar of a move that did not finish, carrying this store's identity. Rename it "
            f'to "{sidecar_path(primary).name}" by hand if the move is to stand. Nothing was written'
        )


def create_sidecar(primary, purpose) -> dict:
    """Give the existing store ``primary``, which has no sidecar, a registry sidecar, and return
    it. It does not open the primary: whether the store is sound is the constructor's question,
    not the note's."""
    primary = Path(os.path.abspath(primary))
    path = sidecar_path(primary)

    def refuse(reason):
        return RuntimeError(
            f'Cannot create a sidecar for "{primary}": {reason}. Nothing was written'
        )

    # a retired name is never reused, and the tombstone says so before anything else does
    reading = read_sidecar(primary)
    if reading.retired:
        raise refuse(_tombstone_text(reading))
    problems = _primary_problems(primary)
    if problems:
        raise refuse(problems[0])
    taken = [f'"{p}"' for p in (path, _tmp_path(path)) if os.path.lexists(p)]
    if taken:
        raise refuse(
            f"{', '.join(taken)} already exist(s); adopt a legacy sidecar instead, and nothing "
            f"is ever overwritten"
        )
    _refuse_interrupted_move(primary, "create")
    if not _nonempty_string(purpose):
        raise refuse("a purpose is required, one line a stranger can read")

    fields = {
        "sidecar_format": SIDECAR_FORMAT,
        "store_id": new_store_id(),
        "datastore": primary.name,
        "name": primary.stem,
        "purpose": purpose,
        "created": now_iso(),
        "history": [_history_entry("create", None, primary, git_provenance())],
    }
    _write_new_sidecar(path, fields, primary)
    return fields


def adopt_sidecar(primary, purpose=None) -> dict:
    """Upgrade the legacy sidecar beside ``primary`` to a registry sidecar, in place, and return
    it. The only operation that rewrites a legacy file, and it runs only when a person asks.

    It adds `sidecar_format`, a new `store_id` and an `adopt` history entry, sets `datastore` to
    the bare name and `name` to the stem, keeps `created` and `purpose` if present, keeps a legacy
    `copied_from` verbatim, and keeps every unknown field verbatim.
    """
    primary = Path(os.path.abspath(primary))
    reading = read_sidecar(primary)
    path = reading.path

    def refuse(reason):
        return RuntimeError(
            f'Cannot adopt the sidecar "{path}": {reason}. Nothing was written'
        )

    if reading.kind == "absent":
        raise refuse("there is no sidecar; use create")
    if reading.retired:
        raise refuse(_tombstone_text(reading))
    if reading.kind == "registry":
        raise refuse("it is already a registry sidecar")
    if reading.kind != "legacy":
        raise refuse("; ".join(reading.problems))
    if reading.problems:
        raise refuse("; ".join(reading.problems))
    guessed = [name for name in ("store_id", "history") if name in reading.fields]
    if guessed:
        raise refuse(
            f"the legacy object already has {guessed}, and the registry would be guessing at "
            f"what they mean"
        )
    _refuse_interrupted_move(primary, "adopt")

    kept_purpose = reading.fields.get("purpose")
    if _nonempty_string(kept_purpose):
        if purpose is not None and purpose != kept_purpose:
            raise refuse(
                f"it already has a purpose, {kept_purpose!r}, which adopt keeps; a different "
                f"one given is refused rather than ignored"
            )
        purpose = kept_purpose
    elif not _nonempty_string(purpose):
        raise refuse("it has no purpose, and none was given")

    fields = _copy.deepcopy(reading.fields)
    fields["sidecar_format"] = SIDECAR_FORMAT
    fields["store_id"] = new_store_id()
    fields["datastore"] = primary.name
    fields["name"] = primary.stem
    fields["purpose"] = purpose
    if not _nonempty_string(fields.get("created")):
        fields["created"] = now_iso()
    fields["history"] = [_history_entry("adopt", None, primary, git_provenance())]
    _update_sidecar(path, fields, primary)
    return fields


# =================================================================================================
# which runs name a store


def runs_naming(primaries, store_id=None, runs_root=None, stale_after=None) -> list:
    """Every run under ``runs_root`` whose manifest names one of ``primaries`` by its `results`
    (both `os.path.realpath`'d, since the file may no longer exist) or names ``store_id`` by its
    `results_store_id`. A manifest without `results_store_id`, which every manifest written before
    the field existed is, is matched by path alone. Runs with no manifest name nothing.
    """
    targets = {os.path.realpath(p) for p in primaries}
    found = []
    for entry in list_runs(
        root=runs_root or DEFAULT_ROOT,
        stale_after=DEFAULT_STALE_AFTER if stale_after is None else stale_after,
    ):
        if not entry["has_manifest"]:
            continue
        manifest = read_json(os.path.join(entry["path"], "manifest.json")) or {}
        matched = []
        results = manifest.get("results")
        if _nonempty_string(results) and os.path.realpath(_resolve(results)) in targets:
            matched.append("results")
        recorded = manifest.get("results_store_id")
        if store_id is not None and recorded == store_id:
            matched.append("results_store_id")
        if matched:
            found.append(
                dict(
                    entry,
                    matched_by=matched,
                    results=results,
                    results_store_id=recorded,
                )
            )
    return found


def _running_runs_naming(operation, primaries, store_id, runs_root, exclude=None):
    """Every run whose state is `running`, alive **or stale**, that names one of ``primaries`` or
    ``store_id``, except the run ``exclude`` (a `Run`, its caller). Refuses if the runs root is not
    a directory, since then the check cannot be made."""
    root = runs_root or DEFAULT_ROOT
    if not os.path.isdir(root):
        raise RuntimeError(
            f'Cannot {operation} "{primaries[0]}": the runs root "{root}" is not a directory, so '
            f"whether a running run names this store cannot be checked. Nothing was written"
        )
    excluded = os.path.realpath(exclude.path) if exclude is not None else None
    return [
        entry
        for entry in runs_naming(primaries, store_id, runs_root=root)
        if entry["state"] == "running" and os.path.realpath(entry["path"]) != excluded
    ]


def _describe_running(running) -> str:
    return "; ".join(
        f"run {entry['id']} is running ({entry['liveness']}, pid {entry['pid']}) and names "
        f"this store by its {' and '.join(entry['matched_by'])}"
        for entry in running
    )


def _refuse_if_in_use(operation, src, dst, store_id, runs_root) -> None:
    """Refuse if any run whose state is `running`, alive **or stale**, names the source or the
    destination. Liveness is evidence, not proof, and the registry does not decide that a run is
    dead: a stale run is ended by a person, with `Run.finish`, before its store is moved.
    """
    running = _running_runs_naming(operation, [src, dst], store_id, runs_root)
    if running:
        described = _describe_running(running)
        raise RuntimeError(
            f'Cannot {operation} "{src}" to "{dst}": {described}. A stale run is ended by a '
            f"person, with Run.finish, before its store is moved; the registry does not decide "
            f"that a run is dead. Nothing was written"
        )


# =================================================================================================
# copy and move


def _files_beside(primary: Path) -> List[str]:
    """Every file in ``primary``'s directory whose name begins with its stem: the store, its
    sidecar, and any `.tmp`, `.incomplete-copy` or `.incomplete-move` file."""
    try:
        names = os.listdir(primary.parent)
    except OSError:
        return []
    return sorted(
        f'"{primary.parent / name}"'
        for name in names
        if name.startswith(primary.stem + ".") or name.startswith(primary.stem + "-")
    )


def _failure(operation, src, dst, step, error) -> RuntimeError:
    return RuntimeError(
        f'{operation} of store "{src}" to "{dst}" failed at step "{step}": '
        f"{type(error).__name__}: {error}. Nothing has been deleted or cleaned up; that is for a "
        f"person. Files now at the destination: [{', '.join(_files_beside(dst))}]; at the "
        f"source: [{', '.join(_files_beside(src))}]"
    )


def _prepare(operation, src, dst, runs_root):
    """The refusals copy and move share, all before anything is written."""
    src = Path(os.path.abspath(src))
    dst = Path(os.path.abspath(dst))
    reading = read_sidecar(src)
    if reading.retired:
        raise RuntimeError(
            f'Cannot {operation} "{src}": {_tombstone_text(reading)}. Nothing was written'
        )
    if not reading.ok:
        why = (
            "there is none"
            if reading.kind == "absent"
            else (
                "it is a legacy sidecar"
                if reading.kind == "legacy" and not reading.problems
                else f"it is {reading.kind}: " + "; ".join(reading.problems)
            )
        )
        raise RuntimeError(
            f'Cannot {operation} "{src}": its sidecar "{reading.path}" is not a problem-free '
            f"registry sidecar ({why}). Run `python -m RunRegistry store create` or `store "
            f"adopt` on it first; {operation} never upgrades a sidecar implicitly. Nothing was "
            f"written"
        )
    destination = read_sidecar(dst)
    if destination.retired:
        raise RuntimeError(
            f'Cannot {operation} "{src}" to "{dst}", a retired name: '
            f"{_tombstone_text(destination)}. Nothing was written"
        )
    _refuse_if_in_use(operation, src, dst, reading.store_id, runs_root)

    dst_sidecar = sidecar_path(dst)
    names = [dst_sidecar, _tmp_path(dst_sidecar)]
    if operation == "move":
        pending = _incomplete_move_path(dst_sidecar)
        names += [pending, _tmp_path(pending)]
    taken = [f'"{p}"' for p in names if os.path.lexists(p)]
    if taken:
        raise RuntimeError(
            f'Cannot {operation} "{src}" to "{dst}": the destination sidecar name(s) '
            f"{', '.join(taken)} already exist, and nothing is ever overwritten. Nothing was "
            f"written"
        )
    return src, dst, reading, dst_sidecar


def _store_files(operation, method, src, dst):
    """Prompt 02's interface, for the store's files. Its refusals come through unchanged."""
    try:
        method(src, dst)
    except Exception as e:
        raise RuntimeError(
            f"{e}. No sidecar was written or moved: sidecar files at the destination "
            f"[{', '.join(_files_beside(dst))}]; at the source [{', '.join(_files_beside(src))}]"
        ) from e


def _no_overwrite(path: Path) -> None:
    # checked once before anything is written, and again just before each rename, because
    # os.rename replaces an existing file silently
    if os.path.lexists(path):
        raise FileExistsError(f'refusing to overwrite "{path}"')


def copy_store(src, dst, purpose, runs_root=None) -> dict:
    """Copy the closed store ``src`` to ``dst`` with `ShardedPool.copy_store`, then write the
    destination's sidecar as a new file, and return it.

    The copy is a new store: a new `store_id`, the given `purpose`, `created` now, `copied_from`
    the source, and the source's history plus one `copy` entry. Every other field, unknown ones
    included, is the source's, verbatim. The source's sidecar is never opened for writing.
    """
    if not _nonempty_string(purpose):
        raise RuntimeError(
            f'Cannot copy "{src}": a copy is a new store and needs its own purpose. Nothing was '
            f"written"
        )
    src, dst, reading, dst_sidecar = _prepare("copy", src, dst, runs_root)
    from Datastore.SQL.ShardedPool import ShardedPool

    _store_files("copy", ShardedPool.copy_store, src, dst)

    step = "write the destination's sidecar"
    try:
        fields = _copy.deepcopy(reading.fields)
        fields["store_id"] = new_store_id()
        fields["datastore"] = dst.name
        fields["name"] = dst.stem
        fields["purpose"] = purpose
        fields["created"] = now_iso()
        fields["copied_from"] = {
            "store_id": reading.fields["store_id"],
            "datastore": _repo_path(src),
        }
        fields["history"] = list(fields["history"]) + [
            _history_entry("copy", src, dst, git_provenance())
        ]
        _write_new_sidecar(dst_sidecar, fields, dst)
    except Exception as e:
        raise _failure("copy", src, dst, step, e) from e
    return fields


def move_store(src, dst, runs_root=None) -> dict:
    """Move the closed store ``src`` to ``dst`` with `ShardedPool.move_store`, then carry its
    sidecar, and return it.

    In this order: the store; `os.rename` of the source's sidecar to
    `<destination sidecar>.incomplete-move`; that file updated in place (`datastore`, `name`, and
    one `move` entry appended to `history`; `store_id` and everything else kept); `os.rename` of
    it onto the destination's sidecar name. The sidecar that validly describes the store appears
    only once the store is in place, and a half-updated sidecar never sits under a sidecar name.
    Nothing is written anew and nothing is deleted: the one sidecar is renamed.
    """
    src, dst, reading, dst_sidecar = _prepare("move", src, dst, runs_root)
    from Datastore.SQL.ShardedPool import ShardedPool

    _store_files("move", ShardedPool.move_store, src, dst)

    pending = _incomplete_move_path(dst_sidecar)
    step = "rename the source's sidecar to its temporary name"
    try:
        _no_overwrite(pending)
        os.rename(reading.path, pending)

        step = "update the temporary sidecar"
        if read_json(pending) != reading.fields:
            raise RuntimeError(
                f'"{pending}" no longer holds what was read from "{reading.path}" before the move'
            )
        fields = _copy.deepcopy(reading.fields)
        fields["datastore"] = dst.name
        fields["name"] = dst.stem
        fields["history"] = list(fields["history"]) + [
            _history_entry("move", src, dst, git_provenance())
        ]
        _update_sidecar(pending, fields, dst)

        step = "rename the temporary sidecar to the destination's sidecar name"
        _no_overwrite(dst_sidecar)
        os.rename(pending, dst_sidecar)
    except Exception as e:
        raise _failure("move", src, dst, step, e) from e
    return fields


# =================================================================================================
# the fingerprint (store-fingerprint prompt 04)
#
# A fingerprint digests the structured inventory, `Datastore.store_inventory.read_inventory`. A
# record's canonical form is `record.canonical_json()`, which is `canonical_json(record.as_json())`:
# its key, its tags, `validated` and `value_count`, and no timestamp. A digest is the SHA-256 of
# records' canonical JSON, one per line, each followed by "\n", in the order the inventory gives,
# which is sorted by canonical JSON. So a listing written as those lines hashes to the digest it
# lists, and filtering a class's lines to one tag set keeps them sorted. Problems are counted beside
# the digests, never in them: their text names shards and serials, which are store-local.


def _sha256_lines(lines) -> str:
    """SHA-256 (hex) of ``lines``, each followed by a newline."""
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _tag_sets(records) -> list:
    """``[(tags, lines)]``: each distinct tag set, sorted, with the canonical lines of the records
    that carry exactly that set, in the records' own order."""
    groups = {}
    for record in records:
        groups.setdefault(tuple(record.tags), []).append(record.canonical_json())
    return sorted(groups.items())


def fingerprint_of(inventory, taken=None) -> dict:
    """The fingerprint of ``inventory``, what `read_inventory` returns. Pure: it reads nothing.

    - `fingerprint_format`: `FINGERPRINT_FORMAT`;
    - `classes`: for every class, in the inventory's order, its `count`, its `digest` over every
      record, and `tag_sets`, one `{"tags", "count", "digest"}` per distinct tag set, sorted by
      tags. A class with no association table has one, with `tags` `[]`; an empty class has none;
    - `digest`: SHA-256 of `canonical_json({"fingerprint_format", "classes": {name: {"count",
      "digest"}}})`;
    - `problems`: per class that has any, the number of its named problems of each kind (the word
      before the first colon). Outside every digest;
    - `taken`: ``taken`` as given (`{"when", "git_head", "git_dirty", "run_id"}`), or `None`.
      Outside every digest.
    """
    # here, not at module scope: that module imports sqlalchemy, and `import RunRegistry` must not
    from Datastore.store_inventory import canonical_json

    classes = {}
    problems = {}
    for name, cls in inventory.classes.items():
        lines = [record.canonical_json() for record in cls.records]
        classes[name] = {
            "count": len(lines),
            "digest": _sha256_lines(lines),
            "tag_sets": [
                {
                    "tags": list(tags),
                    "count": len(group),
                    "digest": _sha256_lines(group),
                }
                for tags, group in _tag_sets(cls.records)
            ],
        }
        kinds = Counter(problem.split(":", 1)[0] for problem in cls.problems)
        if kinds:
            problems[name] = dict(sorted(kinds.items()))

    overall = {
        "fingerprint_format": FINGERPRINT_FORMAT,
        "classes": {
            name: {"count": entry["count"], "digest": entry["digest"]}
            for name, entry in classes.items()
        },
    }
    return {
        "fingerprint_format": FINGERPRINT_FORMAT,
        "classes": classes,
        "digest": hashlib.sha256(canonical_json(overall).encode("utf-8")).hexdigest(),
        "problems": problems,
        "taken": dict(taken) if taken is not None else None,
    }


def _tags_text(tags) -> str:
    return "[" + ", ".join(tags) + "]"


def _difference(kind, name, tags, recorded, current, text) -> dict:
    return {
        "kind": kind,
        "class": name,
        "tags": None if tags is None else list(tags),
        "recorded": recorded,
        "current": current,
        "text": text,
    }


def _presence(recorded, current) -> str:
    if recorded is None:
        return "is only in the current fingerprint"
    return "is only in the recorded fingerprint"


def compare_fingerprints(recorded, current) -> list:
    """The differences between two fingerprints, one entry per difference. Pure.

    Two fingerprints of different formats are not compared: the answer is one entry saying so,
    because the remedy is to recompute both from the stores. Otherwise each entry names the class,
    the tag set (or that a tag set, or a class, is present in only one) and the two counts, `None`
    where absent. Equal overall digests give no content entry. Differences in `problems` are
    entries of their own, compared whatever the digests say, because problems are outside every
    digest. `taken` is never compared. Each entry is `{"kind", "class", "tags", "recorded",
    "current", "text"}`, `kind` being `format`, `class`, `tag_set`, `digest` or `problems`.
    """
    recorded_format = recorded.get("fingerprint_format")
    current_format = current.get("fingerprint_format")
    if recorded_format != current_format:
        return [
            _difference(
                "format",
                None,
                None,
                recorded_format,
                current_format,
                f"the recorded fingerprint is format {recorded_format!r} and this one is format "
                f"{current_format!r}; fingerprints of different formats are not compared. "
                f"Recompute both from the stores",
            )
        ]

    out = []
    if recorded.get("digest") != current.get("digest"):
        before = recorded.get("classes") or {}
        after = current.get("classes") or {}
        names = list(after) + [name for name in before if name not in after]
        for name in names:
            old, new = before.get(name), after.get(name)
            if old is None or new is None:
                out.append(
                    _difference(
                        "class",
                        name,
                        None,
                        None if old is None else old.get("count"),
                        None if new is None else new.get("count"),
                        f"{name}: the class {_presence(old, new)}",
                    )
                )
                continue
            if old.get("digest") == new.get("digest"):
                continue
            old_sets = {tuple(t["tags"]): t for t in old.get("tag_sets") or []}
            new_sets = {tuple(t["tags"]): t for t in new.get("tag_sets") or []}
            found = False
            for tags in sorted(set(old_sets) | set(new_sets)):
                a, b = old_sets.get(tags), new_sets.get(tags)
                if (
                    a is not None
                    and b is not None
                    and a.get("digest") == b.get("digest")
                ):
                    continue
                found = True
                counts = (
                    None if a is None else a.get("count"),
                    None if b is None else b.get("count"),
                )
                if a is None or b is None:
                    text = (
                        f"{name}: tag set {_tags_text(tags)} {_presence(a, b)} "
                        f"({counts[0]} recorded, {counts[1]} now)"
                    )
                else:
                    text = (
                        f"{name}: tag set {_tags_text(tags)} differs: {counts[0]} recorded, "
                        f"{counts[1]} now"
                    )
                out.append(_difference("tag_set", name, tags, *counts, text))
            if not found:
                out.append(
                    _difference(
                        "class",
                        name,
                        None,
                        old.get("count"),
                        new.get("count"),
                        f"{name}: the class digest differs, though no tag set's does "
                        f"({old.get('count')} recorded, {new.get('count')} now)",
                    )
                )
        if not out:
            out.append(
                _difference(
                    "digest",
                    None,
                    None,
                    recorded.get("digest"),
                    current.get("digest"),
                    "the overall digests differ, though no class's does",
                )
            )

    before = recorded.get("problems") or {}
    after = current.get("problems") or {}
    names = list(after) + [name for name in before if name not in after]
    for name in names:
        old, new = before.get(name) or {}, after.get(name) or {}
        for kind in sorted(set(old) | set(new)):
            if old.get(kind, 0) != new.get(kind, 0):
                out.append(
                    _difference(
                        "problems",
                        name,
                        None,
                        old.get(kind, 0),
                        new.get(kind, 0),
                        f"{name}: {kind} problems: {old.get(kind, 0)} recorded, "
                        f"{new.get(kind, 0)} now",
                    )
                )
    return out


def listing_lines(inventory, fingerprint):
    """The full listing of ``inventory``, generated on demand and never stored: a header line for
    the fingerprint, then per class a header and, per tag set, a header and the canonical record
    lines. Header lines begin `#`; record lines begin `{`. The lines under a tag-set header hash
    (SHA-256, each line followed by a newline) to that tag set's digest, and a class's record lines,
    sorted bytewise, hash to the class's digest. ``fingerprint`` is `fingerprint_of(inventory)`.
    """
    from Datastore.store_inventory import canonical_json

    yield (
        f"# fingerprint_format {fingerprint['fingerprint_format']} "
        f"digest {fingerprint['digest']}"
    )
    for name, cls in inventory.classes.items():
        entry = fingerprint["classes"][name]
        yield f"# class {name} count {entry['count']} digest {entry['digest']}"
        by_tags = {tuple(t["tags"]): t for t in entry["tag_sets"]}
        for tags, lines in _tag_sets(cls.records):
            recorded = by_tags[tags]
            yield (
                f"# tags {canonical_json(list(tags))} count {recorded['count']} "
                f"digest {recorded['digest']}"
            )
            yield from lines


def check_listing_path(path, primary) -> Path:
    """Refuse a listing path that exists, whose directory does not, or that is inside the store's
    own directory; return it, absolute. A listing is never written beside the store."""
    path = Path(os.path.abspath(path))
    primary = Path(os.path.abspath(primary))
    if os.path.lexists(path):
        raise RuntimeError(
            f'Cannot write a listing to "{path}": it already exists, and nothing is ever '
            f"overwritten. Nothing was written"
        )
    if not os.path.isdir(path.parent):
        raise RuntimeError(
            f'Cannot write a listing to "{path}": "{path.parent}" is not a directory. Nothing was '
            f"written"
        )
    store_dir = os.path.realpath(primary.parent)
    where = os.path.realpath(path.parent)
    if where == store_dir or where.startswith(store_dir + os.sep):
        raise RuntimeError(
            f'Cannot write a listing to "{path}": it is inside the store\'s own directory '
            f'"{primary.parent}", and a listing is never written beside the store. Nothing was '
            f"written"
        )
    return path


def write_listing(inventory, fingerprint, path, primary) -> Path:
    """Write `listing_lines` to the **new** file ``path`` (`check_listing_path`), and return it."""
    path = check_listing_path(path, primary)
    with open(path, "x") as handle:
        for line in listing_lines(inventory, fingerprint):
            handle.write(line + "\n")
    return path


def _sidecar_refusal(reading) -> str:
    if reading.kind == "absent":
        return "there is none"
    if reading.kind == "legacy" and not reading.problems:
        return "it is a legacy sidecar"
    return f"it is {reading.kind}: " + "; ".join(reading.problems)


def fingerprint_store(primary, *, write=False, runs_root=None, taken_by=None) -> dict:
    """Fingerprint the closed store ``primary``, compare it with the fingerprint its sidecar
    records, and, only if ``write``, record it there. In this order:

    1. refuse a tombstone (`SidecarReading.retired`), whose store is gone; then refuse if any
       `running` run under ``runs_root``, alive or stale, names the store by path or
       by `store_id` (read from the sidecar for this check). The one exception is ``taken_by``,
       the `Run` calling this from its finish, which is still `running` because it has not
       finished. A fingerprint of a store being written describes no instant;
    2. import `read_inventory`, here, so that `import RunRegistry` loads neither ray nor
       sqlalchemy;
    3. `read_inventory(primary)`, read-only and with no Ray. A reader refusal comes through;
    4. `fingerprint_of`, with `taken` from `now_iso()`, `git_provenance()` and ``taken_by``'s id;
    5. `read_sidecar`, and `compare_fingerprints` against a registry sidecar's `fingerprint`;
    6. only if ``write``: refuse unless the sidecar is a problem-free registry sidecar, then
       replace **only** its `fingerprint` field, in place, through `_update_sidecar`.

    Returns `{"primary", "fingerprint", "recorded", "comparison", "wrote", "sidecar",
    "inventory"}`: `comparison` is `None` when nothing is recorded, `sidecar` is the sidecar's
    kind, and `inventory` is what was fingerprinted, for a listing. It never writes the store,
    never initialises Ray, and never deletes anything.
    """
    primary = Path(os.path.abspath(primary))

    # 1. the running-run refusal, after the refusal of a tombstone: there is no store to read
    first = read_sidecar(primary)
    if first.retired:
        raise RuntimeError(
            f'Cannot fingerprint "{primary}": {_tombstone_text(first)}. Nothing was read or '
            f"written"
        )
    store_id = (
        (first.fields or {}).get("store_id") if first.kind == "registry" else None
    )
    if not (isinstance(store_id, str) and _STORE_ID.fullmatch(store_id)):
        store_id = None
    running = _running_runs_naming(
        "fingerprint", [primary], store_id, runs_root, exclude=taken_by
    )
    if running:
        raise RuntimeError(
            f'Cannot fingerprint "{primary}": {_describe_running(running)}. A fingerprint of a '
            f"store that is being written describes no instant; a stale run is ended by a "
            f"person, with Run.finish, first. Nothing was read or written"
        )

    # 2, 3. the inventory, read-only
    from Datastore.store_inventory import read_inventory

    inventory = read_inventory(primary)

    # 4. the fingerprint
    provenance = git_provenance()
    fingerprint = fingerprint_of(
        inventory,
        {
            "when": now_iso(),
            "git_head": provenance["git_head"],
            "git_dirty": provenance["git_dirty"],
            "run_id": taken_by.id if taken_by is not None else None,
        },
    )

    # 5. the comparison with what the sidecar records
    reading = read_sidecar(primary)
    recorded = None
    comparison = None
    if reading.kind == "registry" and "fingerprint" in reading.fields:
        recorded = reading.fields["fingerprint"]
        malformed = _fingerprint_problems(recorded)
        if malformed:
            comparison = [
                _difference(
                    "malformed",
                    None,
                    None,
                    None,
                    None,
                    "the recorded fingerprint is malformed: " + "; ".join(malformed),
                )
            ]
        else:
            comparison = compare_fingerprints(recorded, fingerprint)

    # 6. the write, only when asked
    wrote = False
    if write:
        if not reading.ok:
            raise RuntimeError(
                f'Cannot record a fingerprint for "{primary}": its sidecar "{reading.path}" is not '
                f"a problem-free registry sidecar ({_sidecar_refusal(reading)}). Run `python -m "
                f"RunRegistry store create` or `store adopt` on it first; a fingerprint never "
                f"upgrades a sidecar implicitly. Nothing was written"
            )
        fields = _copy.deepcopy(reading.fields)
        fields["fingerprint"] = fingerprint
        _update_sidecar(reading.path, fields, primary)
        wrote = True

    return {
        "primary": primary,
        "fingerprint": fingerprint,
        "recorded": recorded,
        "comparison": comparison,
        "wrote": wrote,
        "sidecar": reading.kind,
        "inventory": inventory,
    }


# =================================================================================================
# retirement (store-retirement prompt 03)
#
# `retire_store` deletes a closed store's own files through `ShardedPool.delete_store` and keeps
# its sidecar as a tombstone. The order of the writes is the point: the tombstone, with the list of
# files to be deleted, is on disk in state `retiring` before anything is deleted, and it is marked
# `retired` only once none of those files exists. So every state an interruption can leave reads,
# through `read_sidecar`, as the live store untouched, an incomplete retirement naming what
# remains, or a completed tombstone; and `store retire` again completes an incomplete one.

# the text `ShardedPool._read_closed_store` gives a `shards` table it cannot read (prompt 01's log)
_SHARDS_TABLE_UNREADABLE = "its shards table could not be read"


def _json_path(where: str, key) -> str:
    if isinstance(key, int):
        return f"{where}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{where}.{key}"
    return f"{where}[{json.dumps(key)}]"


def _strings(value, where="$"):
    """``(JSON path, string)`` for every string value anywhere in ``value``, recursively."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _strings(value[key], _json_path(where, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, _json_path(where, index))


def _names_one_of(text: str, targets) -> bool:
    """Whether ``text``, read as a path by `_resolve`, is one of ``targets`` (realpaths)."""
    if not text.strip():
        return False
    try:
        return os.path.realpath(_resolve(text)) in targets
    except (OSError, ValueError):
        return False


def _find_references(primary, planned, store_id, runs_root, stores_root, own) -> dict:
    """D5's report: the runs naming the store, and every sidecar under ``stores_root`` that names
    it. Reads, and never writes anything."""
    runs = [
        {"id": entry["id"], "state": entry["state"], "matched_by": entry["matched_by"]}
        for entry in runs_naming([primary], store_id, runs_root=runs_root)
    ]

    targets = {os.path.realpath(p) for p in [primary, *planned, primary.parent]}
    own = os.path.realpath(own)
    # the sidecar suffix, taken from `sidecar_path`, where alone the pattern is spelled
    suffix = sidecar_path(Path("x.sqlite")).name[len("x") :]
    sidecars = []
    for directory, subdirectories, names in os.walk(stores_root):
        subdirectories.sort()
        for name in sorted(names):
            if not name.endswith(suffix):
                continue
            path = os.path.join(directory, name)
            if os.path.realpath(path) == own:
                continue
            try:
                with open(path, "r") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as e:
                sidecars.append(
                    {
                        "sidecar": _repo_path(path),
                        "unreadable": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            matched = []
            parent = payload.get("copied_from") if isinstance(payload, dict) else None
            if isinstance(parent, dict) and parent.get("store_id") == store_id:
                matched.append("$.copied_from.store_id")
            matched += [
                where
                for where, text in _strings(payload)
                if _names_one_of(text, targets)
            ]
            if matched:
                sidecars.append({"sidecar": _repo_path(path), "fields": matched})

    return {
        "runs_root": _repo_path(runs_root),
        "stores_root": _repo_path(stores_root),
        "runs": runs,
        "sidecars": sidecars,
        "not_searched": (
            f"sidecars outside {_repo_path(stores_root)}, and the `.tmp` and "
            f"`{INCOMPLETE_MOVE_SUFFIX}` files beside sidecars; run manifests outside "
            f"{_repo_path(runs_root)}; and every reference that is not in a run manifest or a "
            f"sidecar, such as docs, boards, logs and other committed files"
        ),
    }


def _retire_failure(primary, step, error, files, sidecar) -> RuntimeError:
    """A failure after the tombstone was written: it stays `retiring`, and nothing is cleaned up
    or retried (README §5 rule 11)."""
    present = ", ".join(f'"{p}"' for p in _present(files, primary))
    tmp = _tmp_path(sidecar)
    left = (
        f' "{tmp}" exists, left by the interrupted write; it is not the sidecar, and a person '
        f"removes it before running `store retire` again."
        if os.path.lexists(tmp)
        else ""
    )
    return RuntimeError(
        f'retire of store "{primary}" failed at step "{step}": {type(error).__name__}: '
        f"{error}. Nothing has been cleaned up, and nothing was retried. The sidecar "
        f'"{sidecar}" is a tombstone in state retiring. Store files still present: '
        f"[{present}].{left} Once the failure is understood, run `python -m RunRegistry store "
        f'retire "{primary}" --reason <the same reason>` again to complete it'
    )


def retire_store(
    primary,
    reason,
    *,
    runs_root=None,
    stores_root=None,
    without_fingerprint=False,
    dry_run=False,
) -> dict:
    """Retire the closed store ``primary``: delete its primary and shards, and keep its sidecar as
    a tombstone saying when, at what tree and why, what the store held, which files went, and
    what referenced it. Returns what it found and did (see the end of this function).

    Every refusal is a `RuntimeError` naming the store and the reason and ending "Nothing was
    written or deleted". In order:

    1. **The sidecar.** A reason is required (D3). The sidecar must be a problem-free registry
       sidecar, or a tombstone whose one problem is an incomplete retirement, which this call then
       completes. It never upgrades a sidecar, and it refuses a completed tombstone.
    2. **In use.** Any `running` run under ``runs_root``, alive **or stale**, naming the store by
       path or by its recorded `store_id`.
    3. **The files and the fingerprint.** The files are `ShardedPool.closed_store_files`, whose
       refusals come through, a hot journal among them. Without ``without_fingerprint``, the
       sidecar must record a fingerprint, and a fresh read-only one must compare empty with it,
       `problems` included. With it (D4), the fingerprint is attempted all the same: if it
       succeeds the call refuses, since the flag does not apply, and if it fails its error is kept
       verbatim. Under the flag only, a missing shard is left out of the files, planned and
       deleted with ``resume=True``. A `shards` table that cannot be read is refused even so.
    4. **The references** (D5): the runs naming the store, and the sidecars under
       ``stores_root`` (default `DEFAULT_STORES_ROOT`) that name it. None is ever rewritten.
    5. With ``dry_run``, it stops here, having written and deleted nothing.
    6. **The writes:** the tombstone in state `retiring`, with the file list and the references,
       and one `retire` history entry; `ShardedPool.delete_store`; the check that no listed file
       exists; the tombstone in state `retired`. A failure after the first write raises, naming
       the step and the files still present; the tombstone stays `retiring`.

    On an incomplete retirement it re-makes checks 1 and 2, refuses a reason different from the
    recorded one, deletes what remains of the listed files (``resume=True``) while the primary
    exists, or checks that none remains if it does not, and completes. It appends no history and
    rewrites no reference.
    """
    primary = Path(os.path.abspath(primary))
    runs_root = str(runs_root or DEFAULT_ROOT)
    stores_root = str(stores_root or DEFAULT_STORES_ROOT)

    def refuse(why):
        return RuntimeError(
            f'Cannot retire "{primary}": {why}. Nothing was written or deleted'
        )

    # 1. the reason and the sidecar
    if not _nonempty_string(reason):
        raise refuse(
            "a reason is required, saying why the store is retired, which only a person can say"
        )
    reading = read_sidecar(primary)
    if reading.kind != "registry":
        raise refuse(
            f'its sidecar "{reading.path}" is not a registry sidecar '
            f"({_sidecar_refusal(reading)}). Run `python -m RunRegistry store create` or `store "
            f"adopt` on it first; retirement never upgrades a sidecar"
        )
    completing = False
    if reading.retired:
        retired = reading.fields["retired"]
        if reading.incomplete_retirement is not None and reading.problems == [
            reading.incomplete_retirement
        ]:
            completing = True
        elif not reading.problems:
            raise refuse(
                f'it is already retired: its sidecar "{reading.path}" is a completed tombstone, '
                f"retired {retired['when']} at {retired['git_head']} and completed "
                f"{retired['completed']}, because {retired['reason']!r}"
            )
        else:
            raise refuse(
                f'its sidecar "{reading.path}" is a tombstone with problems: '
                + "; ".join(reading.problems)
            )
    elif reading.problems:
        raise refuse(
            f'its sidecar "{reading.path}" has problems: ' + "; ".join(reading.problems)
        )
    fields = reading.fields
    # the recorded field, read directly: SidecarReading.store_id is None for a tombstone
    store_id = fields["store_id"]
    if completing and reason != fields["retired"]["reason"]:
        raise refuse(
            f"its retirement is under way with the reason {fields['retired']['reason']!r}, which "
            f"its completion keeps; a different reason given is refused rather than ignored"
        )
    tmp = _tmp_path(reading.path)
    if os.path.lexists(tmp):
        raise refuse(
            f'"{tmp}" exists, left by an interrupted write of its sidecar. It is not the sidecar, '
            f"and nothing reads it; a person looks at it and removes it, and then runs `store "
            f"retire` again"
        )

    # 2. in use: any running run, alive or stale, by path or by the recorded store_id
    if not os.path.isdir(runs_root):
        raise refuse(
            f'the runs root "{runs_root}" is not a directory, so whether a running run names '
            f"this store cannot be checked"
        )
    running = _running_runs_naming("retire", [primary], store_id, runs_root)
    if running:
        raise refuse(
            f"{_describe_running(running)}. A stale run is ended by a person, with Run.finish, "
            f"before its store is retired; the registry does not decide that a run is dead. Once "
            f'it is ended, `python -m RunRegistry store fingerprint "{primary}" --write` records '
            f"the fingerprint that retirement needs"
        )

    from Datastore.SQL.ShardedPool import ShardedPool

    # 3. the files, and (on a first retirement) the fingerprint. The plan's refusals come through
    def plan_refusal(e):
        if _SHARDS_TABLE_UNREADABLE in str(e):
            return refuse(
                f"{e}. Its shards table is the only list of which files are this store's, so "
                f"nothing can say which files to delete, and deleting by the naming rule would "
                f"be a guess. This is refused even under --without-fingerprint: this store's "
                f"files can be removed only by a person, outside the registry"
            )
        return refuse(f"the plan of its files was refused: {e}")

    def plan(resume):
        try:
            return ShardedPool.closed_store_files(primary, resume=resume)
        except RuntimeError as e:
            raise plan_refusal(e) from e

    if completing:
        tombstone = _copy.deepcopy(fields["retired"])
        files = tombstone["files"]
        condition = tombstone["fingerprint"]
        references = tombstone["references"]
        resume = True
        if os.path.lexists(primary):
            planned = plan(True)
            unlisted = [str(p) for p in planned if _repo_path(p) not in files]
            if unlisted:
                raise refuse(
                    f"its files now include {unlisted}, which its tombstone does not list, so "
                    f"they are not the files whose deletion was begun"
                )
        else:
            planned = []
            remaining = _present(files, primary)
            if remaining:
                raise refuse(
                    f"its primary is gone, but these listed files remain: "
                    f"{[str(p) for p in remaining]}. With no primary nothing names them as the "
                    f"store's through the resolver, so they can be removed only by a person, "
                    f"outside the registry"
                )
        new_fields = _copy.deepcopy(fields)
    else:
        resume = False
        try:
            planned = ShardedPool.closed_store_files(primary)
        except RuntimeError as e:
            if not without_fingerprint or _SHARDS_TABLE_UNREADABLE in str(e):
                raise plan_refusal(e) from e
            # under the flag only, a missing shard is left out; resume relaxes that alone, so
            # every other refusal comes through again here
            planned = plan(True)
            resume = True
        files = [_repo_path(p) for p in planned]

        if not without_fingerprint:
            recorded = fields.get("fingerprint")
            if recorded is None:
                raise refuse(
                    f"its sidecar records no fingerprint, so nothing would say what the "
                    f'retirement deleted. Run `python -m RunRegistry store fingerprint "{primary}" '
                    f"--write` to record one, and then retire it"
                )
            try:
                taken = fingerprint_store(primary, write=False, runs_root=runs_root)
            except Exception as e:
                raise refuse(
                    f"a fresh fingerprint could not be taken to compare with the recorded one "
                    f"({type(e).__name__}: {e}). If the store cannot be fingerprinted at all, "
                    f"`--without-fingerprint` retires it and records this error"
                ) from e
            differences = list(taken["comparison"])
            if differences:
                raise refuse(
                    f"its content does not match the fingerprint its sidecar records, taken "
                    f"{(recorded.get('taken') or {}).get('when')}: {len(differences)} "
                    f"difference(s): " + "; ".join(d["text"] for d in differences)
                )
            condition = {"condition": "matched", "digest": recorded["digest"]}
        else:
            try:
                fingerprint_store(primary, write=False, runs_root=runs_root)
            except Exception as e:
                condition = {
                    "condition": "without",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            else:
                raise refuse(
                    f"it can be fingerprinted, so --without-fingerprint does not apply. Run "
                    f'`python -m RunRegistry store fingerprint "{primary}" --write`, then `store '
                    f"retire` without the flag"
                )

        # 4. the references
        if not os.path.isdir(stores_root):
            raise refuse(
                f'the stores root "{stores_root}" is not a directory, so the sidecars that '
                f"reference this store cannot be searched"
            )
        references = _find_references(
            primary, planned, store_id, runs_root, stores_root, reading.path
        )

        provenance = git_provenance()
        entry = _history_entry(RETIRE_OPERATION, primary, None, provenance)
        tombstone = {
            "state": "retiring",
            "when": entry["when"],
            "git_head": provenance["git_head"],
            "git_dirty": provenance["git_dirty"],
            "reason": reason,
            "fingerprint": condition,
            "files": files,
            "files_present_only": resume,
            "references": references,
            "completed": None,
        }
        new_fields = _copy.deepcopy(fields)
        new_fields["retired"] = tombstone
        new_fields["history"] = list(new_fields["history"]) + [entry]
        problems = _registry_problems(new_fields, primary)
        if problems:
            raise refuse("the tombstone would have problems: " + "; ".join(problems))

    result = {
        "primary": str(primary),
        "sidecar": str(reading.path),
        "store_id": store_id,
        "dry_run": bool(dry_run),
        "completing": completing,
        "fingerprint": condition,
        "files": list(files),
        "to_delete": [_repo_path(p) for p in planned],
        "deleted": [],
        "references": references,
        "tombstone": tombstone,
        "fields": new_fields,
    }
    # 5. a dry run stops here, having made every refusal the real run would
    if dry_run:
        return result

    # 6. the writes. Step 1, the tombstone, is already on disk on the completion path
    if not completing:
        step = "write the tombstone"
        try:
            _update_sidecar(reading.path, new_fields, primary)
        except Exception as e:
            now = read_sidecar(primary)
            raise RuntimeError(
                f'retire of store "{primary}" failed at step "{step}": {type(e).__name__}: {e}. '
                f"Nothing was deleted, and nothing was cleaned up; the sidecar now reads as "
                f"{'a tombstone in state retiring' if now.retired else 'the live store'}"
                + (
                    f', and "{tmp}" exists, left by the interrupted write, which a person '
                    f"removes"
                    if os.path.lexists(tmp)
                    else ""
                )
                + ". Once the failure is understood, run `store retire` again"
            ) from e

    deleted = []
    if planned:
        step = "delete the store's files"
        try:
            deleted = ShardedPool.delete_store(primary, resume=resume)
        except Exception as e:
            raise _retire_failure(primary, step, e, files, reading.path) from e

    step = "check that no listed file remains"
    remaining = _present(files, primary)
    unlisted = [str(p) for p in deleted if _repo_path(p) not in files]
    if remaining or unlisted:
        raise _retire_failure(
            primary,
            step,
            RuntimeError(
                f"after the deletion, listed files remain, {[str(p) for p in remaining]}, and "
                f"files not listed were deleted, {unlisted}"
            ),
            files,
            reading.path,
        )

    step = "mark the tombstone retired"
    new_fields = _copy.deepcopy(new_fields)
    new_fields["retired"]["state"] = "retired"
    new_fields["retired"]["completed"] = now_iso()
    try:
        _update_sidecar(reading.path, new_fields, primary)
    except Exception as e:
        raise _retire_failure(primary, step, e, files, reading.path) from e

    result.update(
        deleted=[_repo_path(p) for p in deleted],
        tombstone=new_fields["retired"],
        fields=new_fields,
    )
    return result


# =================================================================================================
# amend (store-retirement prompt 04)
#
# `amend_sidecar` replaces or removes exactly one **unknown** field of a registry sidecar, so that
# a present-tense claim it makes (the live A3 sidecar's `backup.retained`, once the backup is
# retired) can be corrected by a person, with nothing lost and no hand edit of a sidecar (README
# D5). It never opens the store's files, and it never touches a known field: those belong to the
# operation that owns them (`_FIELD_OWNERS`).

# the one value `value=` never equals: how `amend_sidecar` tells "no value was given" from a value
# of `None`, which is a JSON value (null) a field can genuinely hold
_NO_VALUE = object()


def _amend_slot(fields: dict, name: str) -> dict:
    """The `before` (or, on the destination side, the `after`) marker for ``name`` in ``fields``:
    ``{"present": True, "value": fields[name]}`` if it is there, ``{"present": False}`` if not.
    Deep-copied, so later mutation of ``fields`` cannot reach back into a written history entry.
    """
    if name in fields:
        return {"present": True, "value": _copy.deepcopy(fields[name])}
    return {"present": False}


def amend_sidecar(
    primary, field, reason, *, value=_NO_VALUE, remove=False, runs_root=None
) -> dict:
    """Replace or remove the **unknown** field ``field`` of the registry sidecar beside
    ``primary``, with a required ``reason``, and return what changed.

    Exactly one of ``value`` (the new value, any JSON-serialisable Python value) or
    ``remove=True`` is given. Every refusal is a `RuntimeError` naming the sidecar and the reason,
    and ending "Nothing was written". In order:

    1. a blank or missing ``reason``;
    2. both ``value`` and ``remove``, or neither;
    3. a **tombstone**, complete or not (`SidecarReading.retired`, `_tombstone_text`) — checked
       before the generic "not a problem-free registry sidecar" refusal, so the message says
       "tombstone";
    4. an absent, unreadable or legacy sidecar, or a registry sidecar with problems (`create` or
       `adopt` it first; amend never upgrades a sidecar);
    5. a **known** field, ``field in KNOWN_FIELDS``, `retired` included — the message names the
       operation(s) that own it (`_FIELD_OWNERS`);
    6. a store that a `running` run names, alive or stale, by path or by the sidecar's *recorded*
       `store_id` field (not `SidecarReading.store_id`, which the reader gives only for a
       problem-free sidecar — here that is already established, but the recorded field is read
       directly, as `retire_store` does, for the same reason: a completed tombstone's `store_id`
       would otherwise read as `None`);
    7. ``remove`` of a field that is absent;
    8. a ``value`` that is not JSON-serialisable;
    9. a ``value`` that is identical, after a JSON round trip, to the field's current value.

    The write, through `_update_sidecar` (its fifth use), replaces or deletes only ``field`` and
    appends one `amend` history entry: `HISTORY_KEYS`, with `from` and `to` both null (an amend has
    neither a source nor a destination), plus `field`, `reason`, and `before` and `after`, each
    `_amend_slot`'s tagged marker — never a sentinel string, so that no JSON value a field could
    hold, including one shaped like the marker itself, is ever mistaken for it. Every other field,
    known or unknown, is value-identical after the write.

    Returns `{"primary", "sidecar", "field", "before", "after", "entry", "fields"}`: `before` and
    `after` are the plain values (`None` when absent or removed — read `entry`'s markers to tell
    "removed" from "was null"), `entry` is the appended history entry, and `fields` is the
    sidecar's new fields.
    """
    primary = Path(os.path.abspath(primary))

    def refuse(why):
        return RuntimeError(f'Cannot amend "{primary}": {why}. Nothing was written')

    if not _nonempty_string(reason):
        raise refuse(
            "a reason is required, saying why the field is amended, which only a person can say"
        )
    given_value = value is not _NO_VALUE
    if given_value and remove:
        raise refuse(
            "both a value and --remove were given; amend replaces or removes a field, never both "
            "at once"
        )
    if not given_value and not remove:
        raise refuse("neither a value nor --remove was given; exactly one is required")

    reading = read_sidecar(primary)
    if reading.retired:
        raise refuse(_tombstone_text(reading))
    if not reading.ok:
        raise refuse(
            f'its sidecar "{reading.path}" is not a problem-free registry sidecar '
            f"({_sidecar_refusal(reading)}). Run `python -m RunRegistry store create` or `store "
            f"adopt` on it first; amend never upgrades a sidecar implicitly"
        )
    fields = reading.fields

    if field in KNOWN_FIELDS:
        owner = _FIELD_OWNERS.get(field, "another operation")
        raise refuse(
            f"{field!r} is a known field, owned by {owner}; amend replaces or removes only an "
            f"unknown field, one this registry carries and never interprets"
        )

    # the recorded field, read directly: reading.ok is already established, but the pattern
    # matches retire_store's, which must read it this way for a tombstone
    store_id = fields["store_id"]
    running = _running_runs_naming("amend", [primary], store_id, runs_root)
    if running:
        raise refuse(
            f"{_describe_running(running)}. A stale run is ended by a person, with Run.finish, "
            f"before its sidecar is amended; the registry does not decide that a run is dead"
        )

    if remove:
        if field not in fields:
            raise refuse(f"{field!r} is absent, so there is nothing to remove")
        new_value = _NO_VALUE
    else:
        try:
            normalised = json.loads(json.dumps(value))
        except (TypeError, ValueError) as e:
            raise refuse(
                f"the given value is not JSON-serialisable ({type(e).__name__}: {e})"
            )
        if field in fields and fields[field] == normalised:
            raise refuse(
                f"the given value is identical, after a JSON round trip, to {field!r}'s current "
                f"value; nothing would change"
            )
        new_value = normalised

    before = _amend_slot(fields, field)
    after = {"present": False} if remove else {"present": True, "value": new_value}

    provenance = git_provenance()
    entry = _history_entry(AMEND_OPERATION, None, None, provenance)
    entry.update(field=field, reason=reason, before=before, after=after)

    new_fields = _copy.deepcopy(fields)
    if remove:
        del new_fields[field]
    else:
        new_fields[field] = new_value
    new_fields["history"] = list(new_fields["history"]) + [entry]

    _update_sidecar(reading.path, new_fields, primary)

    return {
        "primary": str(primary),
        "sidecar": str(reading.path),
        "field": field,
        "before": before["value"] if before["present"] else None,
        "after": after["value"] if after["present"] else None,
        "entry": entry,
        "fields": new_fields,
    }
