"""
Where a ShardedPool's shard files live, given what its primary's ``shards`` table says.

A ShardedPool datastore is one primary SQLite file plus *N* shard files beside it
(``foo.sqlite``, ``foo-shard0000.sqlite``, ...). The primary records each shard in its ``shards``
table, column ``filename``. Two forms of record exist:

* **relative (current)**: the shard's bare file name, e.g. ``foo-shard0000.sqlite``, meaning "the
  file of that name in the primary's directory". ``ShardedPool._write_shard_data`` writes this
  form. A store recorded this way can be moved, or its primary renamed, and still opens against
  its own shards;
* **absolute (legacy)**: the absolute path, symlinks resolved, that the store was created at.
  Every store created before this module existed holds this form, and so does any primary
  re-pointed by ``docs/handover/quadsource_atol_sweep.py`` ``prepare()``.

This module is the **one** definition of how a record is turned into a path. ``ShardedPool`` and
``tools/shard_key_audit.py`` both call it; do not copy it. Two definitions of where a shard lives
is the disagreement that made copied stores write into their originals.

It is also the one definition of what a shard is **called** (``shard_file_name``): the creator
names a new store's shards with it, and ``ShardedPool.copy_store`` / ``move_store`` name a
destination's shards with it. Do not write the pattern a second time.

**A legacy absolute record is read as a sibling by name, and the absolute path itself is never
used** -- not even when the sibling is missing and the absolute path exists. That fallback *is*
the failure this module exists to remove: a copied primary still names the original's shards,
which do exist, and a pool that used them read and wrote the original (2026-09-23, 54 rows into
the A3 baseline store). If the sibling is missing, the caller fails closed
(``shard_file_problem`` below; ``ShardedPool._check_shard_files``).

Nothing here rewrites a record. Old rows are interpreted when read; opening a store does not
change its ``shards`` table.

This module imports only the standard library, and lives outside the ``Datastore.SQL`` package
on purpose: importing it must not pull in ``ray`` or ``sqlalchemy``, so that the read-only audit
tool can use it and stay a standalone script.
"""

from pathlib import Path, PurePath
from typing import Optional, Union

# characters that make a record something other than a bare file name. Both separators are
# refused on every platform: the creator never writes either, so a record containing one was
# written by something else.
_SEPARATORS = ("/", "\\")


def _require_bare_name(name: str, stored: str) -> str:
    if not isinstance(name, str):
        raise ValueError(
            f"shard record {stored!r} is not a string (type {type(name).__name__})"
        )
    if name in ("", ".", ".."):
        raise ValueError(
            f'shard record {stored!r} does not name a file (name part "{name}")'
        )
    if any(sep in name for sep in _SEPARATORS) or "\x00" in name:
        raise ValueError(
            f"shard record {stored!r} is not a bare file name; ShardedPool only ever records "
            f"shards that are siblings of the primary, so this row was written by something else"
        )
    return name


def shard_file_name(primary: Union[str, Path], serial: int) -> str:
    """
    Return the bare file name of shard ``serial`` of the store whose primary is ``primary``:
    ``<stem>-shard<serial, 4 digits>`` with the primary's suffix, e.g. ``foo-shard0003.sqlite``
    for ``foo.sqlite``.

    Only the primary's file name is used; its directory plays no part, and the shard lives beside
    the primary (``primary.parent / shard_file_name(primary, serial)``). This function does no I/O.
    """
    primary = Path(primary)
    return primary.with_stem(f"{primary.stem}-shard{serial:04d}").name


def resolve_shard_path(primary: Union[str, Path], stored: str) -> Path:
    """
    Return the absolute path of the shard that ``stored`` (a ``shards.filename`` value) names,
    for the primary file ``primary``.

    The result is always ``primary.parent / <bare name>``: a file in the primary's own directory.

    * a bare file name (the current form) resolves to that name in the primary's directory;
    * an absolute path (the legacy form) resolves to **its final component** in the primary's
      directory. The absolute path itself is never returned, whether or not it exists;
    * anything else -- empty, ``.``, ``..``, a relative path with a directory part, a name with a
      path separator in it -- raises ``ValueError``.

    ``primary`` must be absolute (``ShardedPool`` passes ``Path(db_name).resolve()``), and the
    result is therefore absolute too. That is required, not cosmetic: the result is handed to Ray
    actors, whose working directory is not the driver's.

    This function does no I/O. Whether the file exists is ``shard_file_problem``'s question.
    """
    primary = Path(primary)
    if not primary.is_absolute():
        raise ValueError(
            f'primary path "{primary}" is not absolute; resolve it before resolving its shards'
        )

    if not isinstance(stored, str):
        raise ValueError(
            f"shard record {stored!r} is not a string (type {type(stored).__name__})"
        )

    if PurePath(stored).is_absolute():
        # legacy record: keep only the file name, taken from the raw string rather than from
        # PurePath, which would normalise '/x/A/.' to name 'A' and '/x/A/' to 'A'. The creator
        # wrote str(Path(...)) of a file, so its last component is always a bare name; '/x/..',
        # '/x/.', '/x/' and '/' all fail the same check a bare record does.
        name = _require_bare_name(stored.rsplit("/", 1)[-1], stored)
    else:
        name = _require_bare_name(stored, stored)

    return primary.parent / name


def is_legacy_record(stored: str) -> bool:
    """True if ``stored`` is an absolute (pre-relative) shard record."""
    return isinstance(stored, str) and PurePath(stored).is_absolute()


def shard_file_problem(path: Path) -> Optional[str]:
    """
    Return why the resolved shard path ``path`` cannot be used as a shard, or ``None`` if it can.

    A shard must be an existing regular file that is **not a symbolic link**. A missing file is
    refused because the ``Datastore`` actor would otherwise create an empty database there and a
    pool would open with nothing in it. A symlink is refused because the actor resolves it
    (``Datastore.py``: ``Path(db_name).resolve()``), which would put the shard outside the
    primary's directory; the creator never makes one.
    """
    if path.is_symlink():
        return f'is a symbolic link (to "{path.resolve()}")'
    if not path.exists():
        return "does not exist"
    if not path.is_file():
        return "is not a regular file"
    return None
