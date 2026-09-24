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
                    "git_dirty"}

Every other field is **unknown**, and every write preserves it verbatim, meaning value-identical
after a JSON round trip (`write_json_atomic` re-indents and sorts keys). Unknown fields are carried
and never interpreted, even ones holding paths, which may be stale after a move.

The paths in `copied_from` and `history` say where something was **at the time**, written as
`_repo_path` writes paths. Nothing opens them or finds a store from them. Only `datastore` says
where the store is, and it says it by name, relative to the sidecar it sits in.

A *legacy* sidecar is any other JSON object, such as the two written by hand or by a script
before this module existed. It is read as it is and never rewritten except by an explicit
`adopt_sidecar`. A legacy `datastore` that is a path is read by its final component, never as a
path: that is prompt 01's rule for legacy shard records, applied to JSON.

**The operations** are `create_sidecar`, `adopt_sidecar`, `copy_store` and `move_store`. The last
two call `ShardedPool.copy_store` / `move_store` for the store's files, then carry the sidecar.
They refuse a store that any `running` run names, alive or stale; that check is what the bare
script cannot make. It is a check and not a lock: a run begun after it is not seen. Nothing here
deletes a file, overwrites one (except the two in-place updates named below), cleans up after a
failure, or decides that a run is over.

`ShardedPool` is imported inside `copy_store` and `move_store` only, so that importing this
module, like `import RunRegistry`, loads neither `ray` nor `sqlalchemy`.
"""

import copy as _copy
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import (
    DEFAULT_ROOT,
    DEFAULT_STALE_AFTER,
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
KNOWN_FIELDS = REQUIRED_FIELDS + ("copied_from",)

# the operations a history entry may record. The first entry assigned the store_id
IDENTITY_OPERATIONS = ("create", "adopt")
RELOCATION_OPERATIONS = ("copy", "move")
HISTORY_KEYS = ("operation", "from", "to", "when", "git_head", "git_dirty")

# a move updates the sidecar under this name, in the destination directory, and renames it onto
# the destination's sidecar name last
INCOMPLETE_MOVE_SUFFIX = ".incomplete-move"

_STORE_ID = re.compile(r"[0-9a-f]{32}")

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
    """

    primary: Path
    path: Path
    kind: str
    fields: Optional[dict] = None
    problems: List[str] = field(default_factory=list)
    legacy_path: bool = False

    @property
    def ok(self) -> bool:
        """A problem-free registry sidecar."""
        return self.kind == "registry" and not self.problems

    @property
    def store_id(self) -> Optional[str]:
        """The store_id of a problem-free registry sidecar, and `None` otherwise."""
        return self.fields["store_id"] if self.ok else None

    @property
    def unknown_fields(self) -> dict:
        return {k: v for k, v in (self.fields or {}).items() if k not in KNOWN_FIELDS}


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
        allowed = IDENTITY_OPERATIONS if index == 0 else RELOCATION_OPERATIONS
        if operation not in allowed:
            problems.append(
                f"{where} records operation {operation!r}, where only {allowed} can stand"
            )
        if index == 0 and entry["from"] is not None:
            problems.append(f"{where} ({operation}) has a from, {entry['from']!r}")
        if index > 0 and not _nonempty_string(entry["from"]):
            problems.append(f"{where} ({operation}) has no from")
        for key in ("to", "when", "git_head"):
            if not _nonempty_string(entry[key]):
                problems.append(f"{where} has a malformed {key}, {entry[key]!r}")
        if not isinstance(entry["git_dirty"], bool):
            problems.append(
                f"{where} has a malformed git_dirty, {entry['git_dirty']!r}"
            )
    return problems


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
    problems.extend(_primary_problems(primary))

    if "sidecar_format" in fields:
        problems.extend(_registry_problems(fields, primary))
        return SidecarReading(primary, path, "registry", fields, problems)

    # legacy: read as it is. A path in `datastore` is read by its name; a missing `datastore` or
    # `name` makes no claim, and the sidecar's own file name is what ties it to this primary
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
    """Replace an existing sidecar's contents in place. Used in exactly two places: adopt, and
    the move's temporary sidecar. Refuses if the `.tmp` name exists."""
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
        "to": _repo_path(destination),
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


def _refuse_if_in_use(operation, src, dst, store_id, runs_root) -> None:
    """Refuse if any run whose state is `running`, alive **or stale**, names the source or the
    destination. Liveness is evidence, not proof, and the registry does not decide that a run is
    dead: a stale run is ended by a person, with `Run.finish`, before its store is moved.
    """
    root = runs_root or DEFAULT_ROOT
    if not os.path.isdir(root):
        raise RuntimeError(
            f'Cannot {operation} "{src}": the runs root "{root}" is not a directory, so whether a '
            f"running run names this store cannot be checked. Nothing was written"
        )
    running = [
        entry
        for entry in runs_naming([src, dst], store_id, runs_root=root)
        if entry["state"] == "running"
    ]
    if running:
        described = "; ".join(
            f"run {entry['id']} is running ({entry['liveness']}, pid {entry['pid']}) and names "
            f"this store by its {' and '.join(entry['matched_by'])}"
            for entry in running
        )
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
