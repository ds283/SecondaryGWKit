"""M11: production drops with enforcement off. For each drop group alone, on a copy of each
build_full_store shard: drop the group's tables (any order is accepted with enforcement off), then
count PRAGMA foreign_key_check rows, by referencing table -> referenced table. Run at dbc59f9."""
import shutil, sqlite3, sys, tempfile
from collections import Counter
from pathlib import Path
from Datastore.SQL.Datastore import _drop_actions
from Datastore.tests.real_store_fixtures import build_full_store

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
d = Path(tempfile.mkdtemp(dir=root))
store = build_full_store(d)
for sid, path in sorted(store.shard_files.items()):
    base = sqlite3.connect(path)
    assert base.execute("PRAGMA foreign_key_check").fetchall() == []
    base.close()
    for group, tables in _drop_actions.items():
        copy = d / f"s{sid}-{group}.sqlite"
        shutil.copy(path, copy)
        c = sqlite3.connect(copy)
        for t in tables:
            c.execute(f'DROP TABLE "{t}"')
        c.commit()
        rows = c.execute("PRAGMA foreign_key_check").fetchall()
        c.close()
        tally = Counter(f"{r[0]}->{r[2]}" for r in rows)
        print(f"shard {sid} {group}: {len(rows)} dangling reference(s) {dict(tally)}")
