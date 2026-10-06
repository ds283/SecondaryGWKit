"""Phase A: the rough edit of ShardedPool.py (run from the worktree root)."""
p = "Datastore/SQL/ShardedPool.py"
s = open(p).read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (old[:80], c)
    s = s.replace(old, new)


# imports
rep(
    "from Datastore.object import DatastoreObject\n",
    "from Datastore.contract import TAG_TABLE, VERSION_LABEL, VERSION_TABLE\n",
)
rep("from MetadataConcepts import store_tag\n", "")

# the version get in __init__
rep(
    '''                self._get_impl_replicated_table("version", {"label": version_label})''',
    '''                self._get_impl_replicated_table(
                    VERSION_TABLE, {VERSION_LABEL: version_label}
                )''',
)

# _find_version_row
rep(
    '''        The version row of ``version_label``, read from every shard file with a plain ``mode=ro``
        select and nothing written, as a ``store_tag`` with its serial and label; or None if no
        shard holds it. For an existing store the check at open has compared the ``version``
        table across the shards already, and a new store's shards are empty, so the shards agree;
        reading every one costs a select each, and makes a disagreement raise rather than be
        resolved by whichever shard was read. A label held twice on a shard raises, as the
        factory's own lookup does.
        """''',
    '''        The version row of ``version_label``, read from every shard file with a plain ``mode=ro``
        select and nothing written; or None if no shard holds it. For an existing store the check
        at open has compared the version table across the shards already, and a new store's shards
        are empty, so the shards agree; reading every one costs a select each, and makes a
        disagreement raise rather than be resolved by whichever shard was read. A label held twice
        on a shard raises, as the factory's own lookup does.

        The object returned is built by the version table's factory, as an actor's get builds it,
        on a ``mode=ro`` connection to the lowest-serial shard, with an inserter that refuses: the
        row exists, so the factory finds it and inserts nothing (README U4 of
        prompts/datastore-generic).
        """''',
)
rep(
    '''                rows = conn.execute(
                    "SELECT serial, label FROM version WHERE label = ?",
                    (version_label,),
                ).fetchall()''',
    '''                rows = conn.execute(
                    f'SELECT serial, "{VERSION_LABEL}" FROM "{VERSION_TABLE}" '
                    f'WHERE "{VERSION_LABEL}" = ?',
                    (version_label,),
                ).fetchall()''',
)
rep(
    '''        row = next(iter(found.values()), None)
        if row is None:
            return None
        version = store_tag(store_id=row[0], label=row[1])
        version._deserialized = True
        return version
''',
    '''        row = next(iter(found.values()), None)
        if row is None:
            return None

        # store_reader imports this module at module scope, so it is imported here
        from Datastore.store_reader import read_only_url

        def refuse_insert(conn, payload):
            raise RuntimeError(
                f'Sharded datastore "{str(self._primary_file)}": building the version row of '
                f'label "{version_label}" would insert {payload!r}, though every shard holds it'
            )

        built = build_schema(sqla.MetaData(), self._factories)
        sid = min(found)
        engine = sqla.create_engine(
            read_only_url(Path(self._shard_db_files[sid])),
            future=True,
            poolclass=sqla.pool.NullPool,
        )
        try:
            with engine.connect() as conn:
                try:
                    version = self._factories[VERSION_TABLE].build(
                        payload={VERSION_LABEL: version_label},
                        conn=conn,
                        table=built.tables[VERSION_TABLE],
                        inserter=refuse_insert,
                        tables=built.tables,
                        inserters={},
                    )
                finally:
                    conn.rollback()
        finally:
            engine.dispose()

        if version.store_id != row[0]:
            raise RuntimeError(
                f'Sharded datastore "{str(self._primary_file)}": the version row of label '
                f'"{version_label}" was read with serial {row[0]}, and its factory built serial '
                f"{version.store_id}"
            )
        return version
''',
)

# read-only miss
rep(
    '''            raise ReadOnlyMiss(
                "version",
                {"label": version_label},''',
    '''            raise ReadOnlyMiss(
                VERSION_TABLE,
                {VERSION_LABEL: version_label},''',
)

# _version_labels_present
rep(
    '''                rows = conn.execute(
                    "SELECT label FROM version ORDER BY serial"
                ).fetchall()''',
    '''                rows = conn.execute(
                    f'SELECT "{VERSION_LABEL}" FROM "{VERSION_TABLE}" ORDER BY serial'
                ).fetchall()''',
)

# reconcile comment
rep(
    '''    #   3. otherwise every replicated table, and the tag table of every replicated class, is read
    #      from every shard and compared by full row; the primary's shard_keys are checked against
    #      the wavenumbers every shard holds;''',
    '''    #   3. otherwise every replicated table, and the tag table of every replicated class, is read
    #      from every shard and compared by full row; the primary's shard_keys are checked against
    #      the shard-key rows every shard holds;''',
)
rep(
    '''    #      row of the record's class that the record's controlling shard holds and another shard
    #      lacks is copied to that shard verbatim (a BackgroundModel as a unit: row, tags and
    #      values), a flag the controller holds set is set, and after a validate
    #      BackgroundModel.validated is recomputed on each shard by the factory's own rule. Every''',
    '''    #      row of the record's class that the record's controlling shard holds and another shard
    #      lacks is copied to that shard verbatim (a class whose rows own others' as a unit: its
    #      row, its tag rows and the rows that declare it their owner), a monotone flag the
    #      controller holds set is set, and after a validate the class's declared validated column
    #      is recomputed on each shard by its factory's revalidate. Every''',
)

# _replicated_table_specs
rep(
    '''        What is compared, table by table: every replicated class that has a table, in the order
        the constructor was given them, then the tag table of every replicated class (a table
        that is neither replicated nor sharded, whose foreign keys other than store_tag's all
        name replicated tables: BackgroundModel_tags).

        Each spec gives the table, the key its rows are compared under and the columns compared.
        A table is keyed by its primary key, and every column is compared, ``timestamp``
        included: every replicated write stamps every shard's copy with its one timestamp.
        """''',
    '''        What is compared, table by table: every replicated class that has a table, in the order
        the constructor was given them, then the tag table of every replicated class (a table
        that is neither replicated nor sharded, whose foreign keys other than the tag table's all
        name replicated tables).

        Each spec gives the table, the key its rows are compared under and the columns compared.
        A table is keyed by its primary key, and every column is compared, ``timestamp``
        included: every replicated write stamps every shard's copy with its one timestamp. It also
        carries what the table's factory declares (``build_schema``'s record): its
        ``owner_column``, ``monotone_flags`` and ``validated_column``.
        """''',
)
rep(
    '''            parents = {fk.column.table.name for fk in table.foreign_keys} - {
                "store_tag"
            }''',
    '''            parents = {fk.column.table.name for fk in table.foreign_keys} - {TAG_TABLE}''',
)
rep(
    '''            table = built.tables[name]
            key = tuple(c.name for c in table.primary_key.columns)
            compared = [c.name for c in table.columns]
            specs.append(
                {
                    "name": name,
                    "table": table,
                    "key": key,
                    "compared": compared,
                    "serial_keyed": key == ("serial",),
                }
            )''',
    '''            table = built.tables[name]
            record = built.records[name]
            key = tuple(c.name for c in table.primary_key.columns)
            compared = [c.name for c in table.columns]
            specs.append(
                {
                    "name": name,
                    "table": table,
                    "key": key,
                    "compared": compared,
                    "serial_keyed": key == ("serial",),
                    "owner_column": record.get("owner_column"),
                    "monotone_flags": tuple(record.get("monotone_flags", ())),
                    "validated_column": record.get("validated_column"),
                }
            )''',
)

# _replicated_differences docstring and comment
rep(
    '''        the shards that hold it) and ``shard_key`` (a key_serial of the primary's shard_keys that
        is not a wavenumber serial on every shard).''',
    '''        the shards that hold it) and ``shard_key`` (a key_serial of the primary's shard_keys that
        is not a serial of the shard-key class on every shard).''',
)
rep(
    "        # the primary's shard keys name wavenumbers every shard holds\n",
    "        # the primary's shard keys name rows of the shard-key class every shard holds\n",
)
rep(
    '''                    "detail": f"the primary assigns wavenumber serial {d['key']} to shard "''',
    '''                    "detail": f"the primary assigns {self._ShardKeyType_name} serial "
                    f"{d['key']} to shard "''',
)

# _plan_replicated_repair docstring
rep(
    '''        The licence: only from the record's controlling shard, only the tables of the record's
        class (a BackgroundModel's row, tags and values), only rows the controller holds and
        another shard lacks, written by the recorded write (their timestamp is the record's
        start; under a store_id, that serial), a BackgroundModel as a whole unit, a source or
        response flag the controller holds set after a get, and a validated flag after a
        validate of the record's model, which is recomputed. Anything else is refused.''',
    '''        The licence: only from the record's controlling shard, only the tables of the record's
        class's unit (``_unit_tables``: its own, its tag table, and the tables that declare it
        their owner), only rows the controller holds and another shard lacks, written by the
        recorded write (their timestamp is the record's start; under a store_id, that serial), a
        unit of more than one table as a whole, a declared monotone flag the controller holds set
        after a get, and the declared validated column after a validate of the record's row, which
        is recomputed. Anything else is refused.''',
)
rep(
    '''        unit = [cls_name]
        if cls_name == "BackgroundModel":
            unit = [
                t
                for t in (
                    "BackgroundModel",
                    "BackgroundModel_tags",
                    "BackgroundModelValue",
                )
                if t in by_name
            ]
        flag_columns = {
            "redshift": ("source", "response"),
            "wavenumber": ("source", "response"),
        }
''',
    '''        unit = self._unit_tables(cls_name, specs)
        flag_columns = {
            spec["name"]: spec["monotone_flags"]
            for spec in specs
            if len(spec["monotone_flags"]) > 0
        }
        # the refusal names what a get or a validate may turn on, as the factories declare it
        flag_names = list(
            dict.fromkeys(f for spec in specs for f in spec["monotone_flags"])
        )
        validated_names = list(
            dict.fromkeys(
                spec["validated_column"]
                for spec in specs
                if spec["validated_column"] is not None
            )
        )
        allowed_text = (
            f"only a {' or '.join(flag_names)} flag after a get, or "
            f"{' or '.join(validated_names)} after a validate"
        )
''',
)
rep(
    '''                # compared again once the repair is written: a copied wavenumber can satisfy it''',
    '''                # compared again once the repair is written: a copied shard-key row can satisfy
                # it''',
)
rep(
    '''                elif (
                    name == "BackgroundModel"
                    and differing == ["validated"]
                    and turned_on''',
    '''                elif (
                    name == cls_name
                    and by_name[name]["validated_column"] is not None
                    and differing == [by_name[name]["validated_column"]]
                    and turned_on''',
)
rep(
    '''                        f"only a source or response flag after a get, or validated after a "
                        f"validate, can be turned on from the controlling shard {controller}; "
                        f"shard {sid} differs in {differing}",''',
    '''                        f"{allowed_text}, can be turned on from the controlling shard "
                        f"{controller}; shard {sid} differs in {differing}",''',
)
rep(
    '''        # a BackgroundModel is copied as a unit: a shard lacking the model row receives every
        # one of its tag and value rows, and a shard holding any part of it receives nothing
        if cls_name == "BackgroundModel":
            for sid in sorted(snapshots.keys()):
                models = {k[0] for k in copies.get(("BackgroundModel", sid), set())}
                for name in unit:
                    if name == "BackgroundModel":
                        continue
                    # a row's model is the column its foreign key to BackgroundModel names
                    snap = snapshots[controller][name]
                    model_column = snap["columns"].index(
                        next(
                            fk.parent.name
                            for fk in by_name[name]["table"].foreign_keys
                            if fk.column.table.name == "BackgroundModel"
                        )
                    )''',
    '''        # a unit of more than one table is copied as a unit: a shard lacking the class's row
        # receives every one of its tag and owned rows, and a shard holding any part of it
        # receives nothing
        if len(unit) > 1:
            for sid in sorted(snapshots.keys()):
                models = {k[0] for k in copies.get((cls_name, sid), set())}
                for name in unit:
                    if name == cls_name:
                        continue
                    # a member row's owner is the column naming the class's row: the declared
                    # owner_column, or a tag table's foreign key to the class
                    snap = snapshots[controller][name]
                    model_column = snap["columns"].index(
                        self._owner_column_of(by_name[name], cls_name)
                    )''',
)
rep(
    '''                            f"shard {sid} holds its model row, so this is not part of a "
                            "BackgroundModel it lacks",''',
    '''                            f"shard {sid} holds its model row, so this is not part of a "
                            f"{cls_name} it lacks",''',
)
rep(
    '''                                "table": "BackgroundModel",
                                "key": (model_of[key],),''',
    '''                                "table": cls_name,
                                "key": (model_of[key],),''',
)
rep(
    '''                            f"shard {sid} already holds part of this model ({name} {key}), "
                            "and a BackgroundModel is copied only as a unit",''',
    '''                            f"shard {sid} already holds part of this model ({name} {key}), "
                            f"and a {cls_name} is copied only as a unit",''',
)

# _apply_replicated_repair docstring
rep(
    '''        Write the plan, inside each shard's open transaction: the copies (in the specs' order,
        so that a model row precedes its tags and values), then the flags, then the recomputed
        validated flags. Return (actions, refusals).''',
    '''        Write the plan, inside each shard's open transaction: the copies (in the specs' order,
        so that a class's row precedes its tag and owned rows), then the flags, then the
        recomputed validated flags. Return (actions, refusals).''',
)

# _recompute_validated
rep(
    '''        After a validate of the record's BackgroundModel, recompute its ``validated`` flag on each
        shard by the factory's own rule (sqla_BackgroundModelFactory.validate: its value-row count
        against z_samples), from that shard's own value rows, each inside a savepoint. A result
        that turns the flag on is kept; any other is rolled back to the savepoint, so that no flag
        is written False. A shard holding True whose rule gives False refuses. Return (actions,
        refusals); whether the shards then agree is for the second comparison.
        """
        factory = self._factories["BackgroundModel"]
        table = built.tables["BackgroundModel"]
        serial = record["store_id"]

        class _StoredModel(DatastoreObject):
            """What the factory's validate reads of a model: its serial and its label."""

            def __init__(self, store_id: int, label):
                super().__init__(store_id)
                self.label = label

        actions = []
        refusals = []
        for sid in sorted(conns.keys()):
            snap = snapshots[sid]["BackgroundModel"]
            row = snap["rows"].get((serial,))
            if row is None:
                continue
            columns = snap["columns"]
            stored = row[columns.index("validated")] is True
            label = row[columns.index("label")]

            savepoint = conns[sid].begin_nested()
            result = factory.validate(
                _StoredModel(serial, label),
                conn=conns[sid],
                table=table,
                tables=built.tables,
            )''',
    '''        After a validate of the record's row, recompute the class's declared validated column on
        each shard by its factory's ``revalidate`` (the same rule as its ``validate``), from that
        shard's own rows, each inside a savepoint. A result that turns the flag on is kept; any
        other is rolled back to the savepoint, so that no flag is written False. A shard holding
        True whose rule gives False refuses. Return (actions, refusals); whether the shards then
        agree is for the second comparison.
        """
        cls_name = record["class_name"]
        factory = self._factories[cls_name]
        table = built.tables[cls_name]
        column = built.records[cls_name]["validated_column"]
        serial = record["store_id"]

        actions = []
        refusals = []
        for sid in sorted(conns.keys()):
            snap = snapshots[sid][cls_name]
            row = snap["rows"].get((serial,))
            if row is None:
                continue
            columns = snap["columns"]
            stored = row[columns.index(column)] is True

            savepoint = conns[sid].begin_nested()
            result = factory.revalidate(serial, conns[sid], table, built.tables)''',
)
rep(
    '''                        "action": "validated recomputed",
                        "class_name": "BackgroundModel",''',
    '''                        "action": "validated recomputed",
                        "class_name": cls_name,''',
)
rep(
    '''                            "class_name": "BackgroundModel",
                            "shard": sid,
                            "key": f"serial {serial}",
                            "detail": "the shard holds validated=True, and the factory's rule "
                            "gives False from its own value rows",''',
    '''                            "class_name": cls_name,
                            "shard": sid,
                            "key": f"serial {serial}",
                            "detail": f"the shard holds {column}=True, and the factory's rule "
                            "gives False from its own value rows",''',
)

# the prune comment
rep(
    '''    # validates at startup (today BackgroundModel, with its tags and values) are deleted by the''',
    '''    # validates at startup (with its unit's tag and owned rows) are deleted by the''',
)

# _prune_unit -> _unit_tables
rep(
    '''    @staticmethod
    def _prune_unit(cls_name: str, built, specs) -> List[str]:
        """The compared tables a prune of ``cls_name`` may delete from: its own, and every
        compared table with a foreign key to it (for BackgroundModel, its tags and values).
        """
        unit = [cls_name]
        for spec in specs:
            name = spec["name"]
            if name == cls_name:
                continue
            if any(
                fk.column.table.name == cls_name
                for fk in built.tables[name].foreign_keys
            ):
                unit.append(name)
        return unit
''',
    '''    @staticmethod
    def _is_tag_table_of(spec: dict, cls_name: str) -> bool:
        """Whether the compared table of ``spec`` is a tag table of ``cls_name``: a foreign key to
        the tag table, and every other foreign key to ``cls_name``."""
        targets = [fk.column.table.name for fk in spec["table"].foreign_keys]
        others = [t for t in targets if t != TAG_TABLE]
        return (
            TAG_TABLE in targets
            and len(others) > 0
            and all(t == cls_name for t in others)
        )

    @staticmethod
    def _declares_owner(spec: dict, cls_name: str) -> bool:
        """Whether the compared table of ``spec`` declares an ``owner_column`` naming
        ``cls_name``."""
        owner = spec["owner_column"]
        if owner is None:
            return False
        return any(
            fk.column.table.name == cls_name
            for fk in spec["table"].c[owner].foreign_keys
        )

    @staticmethod
    def _unit_tables(cls_name: str, specs) -> List[str]:
        """
        The tables of ``cls_name``'s unit: what a replicated write of it may write, what the
        check at open copies as a whole, and what a prune of it may delete from. The class's own
        table, then the compared tables that are its tag tables, then the compared tables whose
        declared ``owner_column`` names it, each group in the specs' order.
        """
        unit = [cls_name]
        for spec in specs:
            if spec["name"] != cls_name and ShardedPool._is_tag_table_of(spec, cls_name):
                unit.append(spec["name"])
        for spec in specs:
            if spec["name"] != cls_name and ShardedPool._declares_owner(spec, cls_name):
                unit.append(spec["name"])
        return unit

    @staticmethod
    def _owner_column_of(spec: dict, cls_name: str) -> str:
        """The column of a unit member's table naming its ``cls_name`` row: the declared
        ``owner_column``, or a tag table's foreign key to the class."""
        if ShardedPool._declares_owner(spec, cls_name):
            return spec["owner_column"]
        return next(
            fk.parent.name
            for fk in spec["table"].foreign_keys
            if fk.column.table.name == cls_name
        )
''',
)
rep(
    "        unit = self._prune_unit(cls_name, built, specs)\n",
    "        unit = self._unit_tables(cls_name, specs)\n",
)

# _value_serials -> owned_serials
rep(
    '''    @staticmethod
    def _value_serials(obj) -> Optional[List[Optional[int]]]:
        """The serials of an object's value rows, if it carries any (BackgroundModel), else None."""
        values = getattr(obj, "_values", None)
        if values is None:
            return None
        return [getattr(value, "_my_id", None) for value in values]

''',
    "",
)
rep(
    '''        # we push an initial 'store' to the controlling shard, and then push the object, complete
        # with its new store_id (and its value rows' serials), to all the other shards, each of
        # which inserts it under those serials or verifies that it already holds it''',
    '''        # we push an initial 'store' to the controlling shard, and then push the object, complete
        # with its new store_id (and the serials of the rows it owns, which its factory's
        # owned_serials names), to all the other shards, each of which inserts it under those
        # serials or verifies that it already holds it
        factory = self._factories[cls_name]
''',
)
rep(
    "            value_serials = ShardedPool._value_serials(obj)\n",
    "            value_serials = factory.owned_serials(obj)\n",
)
rep(
    "                got_values = ShardedPool._value_serials(answer)\n",
    "                got_values = factory.owned_serials(answer)\n",
)

# _assign_shard_keys comment
rep(
    '''        # a read-only pool assigns no shard key (prompts/a3-v2-readiness, prompt 03): the check at
        # open compared shard_keys with the wavenumbers every shard holds, so a wavenumber with
        # no key here is one the store's primary does not know''',
    '''        # a read-only pool assigns no shard key (prompts/a3-v2-readiness, prompt 03): the check at
        # open compared shard_keys with the shard-key rows every shard holds, so a shard-key row
        # with no key here is one the store's primary does not know''',
)

open(p, "w").write(s)
print("ok")
