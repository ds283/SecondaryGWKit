"""
The exceptions of the replicated write (prompts/datastore-integrity, prompt 01).

A replicated object is written on a controlling shard and then copied to every other shard, each
in its own transaction (``ShardedPool._replicated_write``). Two things can go wrong that must be
loud rather than silent:

- a replica answers with something other than what the controller wrote: a different serial for
  the same key, a row under the controller's serial whose columns differ, or a different
  validation outcome. That is ``ReplicationMismatch``. It is raised by the driver when it checks
  the replicas' answers, and by a replica's own ``store()`` when it finds the key or the serial
  already taken by something else;
- a replicated write is attempted while the primary's ``replication_in_flight`` table already
  names one. That is ``ReplicationInFlight``. The store is then in the state an interrupted write
  left it, and no write may proceed over it until the store has been reconciled.

When a store is opened, ``ShardedPool`` compares its replicated tables across shards (prompt 02).
An interrupted replication is repaired from the record's controlling shard; any other difference
refuses the open with ``ReplicatedDivergence``.

A store can also be opened read-only (``ShardedPool(read_only=True)``, prompts/a3-v2-readiness
prompt 03). On it, a lookup that would insert raises ``ReadOnlyMiss``, and any other write raises
``ReadOnlyWrite``, each before anything is written.

This module lives at the top of the ``Datastore`` package, and imports nothing from it, so that the
factories (which raise ``ReplicationMismatch`` inside a Datastore actor), ``ShardedPool`` (which
raises both in the driver) and the tests can all import it without importing Ray's actor classes
or creating an import cycle through ``Datastore.SQL``.

The exceptions pickle with their fields intact, because under Ray an exception raised inside an
actor reaches the driver by pickling.
"""

from typing import Optional


class ReplicationMismatch(RuntimeError):
    """
    A replica's answer differs from the controlling shard's.

    ``shard`` names the replica (its shard id, or the Datastore actor's name when raised inside the
    actor), ``class_name`` the replicated class, ``controller_serial`` the serial the controller
    wrote and ``replica_serial`` the serial the replica holds or returned. ``detail`` says what
    differed when the serials alone do not (a column, a validation outcome).
    """

    def __init__(
        self,
        shard,
        class_name: str,
        controller_serial,
        replica_serial,
        detail: Optional[str] = None,
    ):
        self.shard = shard
        self.class_name = class_name
        self.controller_serial = controller_serial
        self.replica_serial = replica_serial
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        message = (
            f'Replicated write of "{self.class_name}" is inconsistent on shard '
            f"{self.shard}: the controlling shard wrote serial {self.controller_serial}, "
            f"the replica holds serial {self.replica_serial}"
        )
        if self.detail is not None:
            message += f" ({self.detail})"
        return message

    def with_shard(self, shard) -> "ReplicationMismatch":
        """Fill in the shard if the raiser could not name it (a factory does not know its shard)."""
        if self.shard is None:
            self.shard = shard
            self.args = (self._message(),)
        return self

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (
            type(self),
            (
                self.shard,
                self.class_name,
                self.controller_serial,
                self.replica_serial,
                self.detail,
            ),
        )


class ReplicationInFlight(RuntimeError):
    """
    A replicated write was attempted while the primary's ``replication_in_flight`` table names one.

    ``record`` is that row, as a dict: ``operation``, ``class_name``, ``controller_shard``,
    ``store_id`` and ``started``.
    """

    def __init__(self, primary, record: dict):
        self.primary = primary
        self.record = dict(record)
        super().__init__(self._message())

    def _message(self) -> str:
        r = self.record
        return (
            f'Sharded datastore "{self.primary}" records a replicated write in flight '
            f'(operation="{r.get("operation")}", class="{r.get("class_name")}", '
            f'controlling shard={r.get("controller_shard")}, store_id={r.get("store_id")}, '
            f'started={r.get("started")}). A write interrupted there may have left the shards '
            f"different. No replicated write proceeds over it; nothing was written"
        )

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (type(self), (self.primary, self.record))


class ReplicatedDivergence(RuntimeError):
    """
    The replicated tables of a sharded store differ across its shards in a way that an interrupted
    replication does not explain, so the store is not opened (prompts/datastore-integrity, prompt
    02). Raised by ``ShardedPool._reconcile_replicated_tables`` in the constructor, before any
    actor exists.

    ``differences`` is a list of dicts, one per difference, each naming ``class_name`` (the
    table), ``shard`` (the shard, or the shards, that deviate), ``key`` (the serial or key of the
    row) and ``detail`` (what differs). ``record`` is the primary's ``replication_in_flight`` row
    as a dict, or None if there is none. ``after_repair`` is True when the difference was found by
    the comparison made after a repair was attempted, which was then rolled back.
    """

    def __init__(
        self,
        primary,
        differences: list,
        record: Optional[dict] = None,
        after_repair: bool = False,
    ):
        self.primary = primary
        self.differences = [dict(d) for d in differences]
        self.record = dict(record) if record is not None else None
        self.after_repair = after_repair
        super().__init__(self._message())

    def _message(self) -> str:
        r = self.record
        if r is None:
            why = "there is no replication_in_flight record, so no interrupted replication explains them"
        else:
            why = (
                f'the replication_in_flight record (operation="{r.get("operation")}", '
                f'class="{r.get("class_name")}", controlling shard={r.get("controller_shard")}, '
                f'store_id={r.get("store_id")}, started={r.get("started")}) does not explain them'
            )
        lines = [
            f'Cannot open sharded datastore "{self.primary}": its replicated tables differ across '
            f"shards, and {why}. {len(self.differences)} difference(s)"
            + (
                ", found by the comparison made after a repair was attempted and rolled back"
                if self.after_repair
                else ""
            )
            + ":"
        ]
        for d in self.differences:
            lines.append(
                f'  - class "{d.get("class_name")}", shard {d.get("shard")}, '
                f'{d.get("key")}: {d.get("detail")}'
            )
        lines.append(
            "Nothing was repaired: no shard and not the primary was written, and the "
            "replication_in_flight record, if any, was left as it was. A store in this state "
            "cannot be put right by this tree: it needs a person, or regeneration."
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (
            type(self),
            (self.primary, self.differences, self.record, self.after_repair),
        )


class ReadOnlyMiss(RuntimeError):
    """
    A lookup on a store opened read-only (``ShardedPool(read_only=True)``, prompts/a3-v2-readiness
    prompt 03) matched no row, where a read-write store would have inserted one. Raised before any
    ``INSERT`` is issued: a read-only ``Datastore`` actor hands every factory's ``build`` an
    inserter that raises this instead of inserting, for every replicated class. Also raised by the
    pool when the version label it is opened under is held by no shard.

    ``class_name`` is the class whose row would have been inserted, ``payload`` the row as the
    factory would have inserted it (or the lookup's key), ``store`` the actor or primary that
    refused, and ``detail`` anything more the raiser knows (for the version row, the labels the
    store holds).
    """

    def __init__(
        self,
        class_name: str,
        payload,
        store=None,
        detail: Optional[str] = None,
    ):
        self.class_name = class_name
        self.payload = payload
        self.store = store
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        where = f' "{self.store}"' if self.store is not None else ""
        message = (
            f'Datastore{where} was opened read-only, and a lookup of "{self.class_name}" matched '
            f"no row; a read-write store would have inserted one with payload {self.payload!r}"
        )
        if self.detail is not None:
            message += f" ({self.detail})"
        return message + ". Nothing was written"

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (
            type(self),
            (self.class_name, self.payload, self.store, self.detail),
        )


class ReadOnlyWrite(RuntimeError):
    """
    A write was attempted on a store opened read-only (``ShardedPool(read_only=True)``,
    prompts/a3-v2-readiness prompt 03): a store, a validate, a prune, a drop, a repair, a new shard
    key, or an insert that is not a lookup's. Raised before anything is written. It is also the
    backstop: a write that reaches SQLite through a read-only actor is refused by the file, opened
    ``mode=ro``, and that refusal is re-raised as this exception, chained.

    ``store`` names the actor or primary that refused, ``operation`` what was attempted,
    ``class_name`` the class it was attempted on (if one), ``detail`` why it would write.
    """

    def __init__(
        self,
        store,
        operation: str,
        class_name: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        self.store = store
        self.operation = operation
        self.class_name = class_name
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        what = self.operation
        if self.class_name is not None:
            what += f' of "{self.class_name}"'
        message = f'Datastore "{self.store}" was opened read-only, and {what} would write to it'
        if self.detail is not None:
            message += f" ({self.detail})"
        return message + ". Nothing was written"

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (
            type(self),
            (self.store, self.operation, self.class_name, self.detail),
        )


def same_value(a, b) -> bool:
    """
    Equality of two column values as a replica compares them: exact, except that two NaN floats
    are the same value. Copies of one row are written from one pickled object, so every float is
    bit-identical and an approximate comparison has no place here.
    """
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def differing_columns(expected: dict, stored) -> list:
    """The names of the columns of ``expected`` whose value in ``stored`` (a row mapping) differs."""
    return [
        name for name, value in expected.items() if not same_value(value, stored[name])
    ]
