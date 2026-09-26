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

This module lives at the top of the ``Datastore`` package, and imports nothing from it, so that the
factories (which raise ``ReplicationMismatch`` inside a Datastore actor), ``ShardedPool`` (which
raises both in the driver) and the tests can all import it without importing Ray's actor classes
or creating an import cycle through ``Datastore.SQL``.

Both exceptions pickle with their fields intact, because under Ray an exception raised inside an
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


def same_value(a, b) -> bool:
    """
    Equality of two column values as a replica compares them: exact, except that two NaN floats
    are the same value. Copies of one row are written from one pickled object, so every float is
    bit-identical and no tolerance belongs here.
    """
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def differing_columns(expected: dict, stored) -> list:
    """The names of the columns of ``expected`` whose value in ``stored`` (a row mapping) differs."""
    return [
        name for name, value in expected.items() if not same_value(value, stored[name])
    ]
