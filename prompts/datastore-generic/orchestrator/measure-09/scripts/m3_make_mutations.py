"""M3, step 1: write each mutation as a unified diff against the rough tree (staged in the index),
then revert it. Usage: python m3_make_mutations.py <worktree> <outdir>"""
import subprocess, sys
from pathlib import Path
tree, out = Path(sys.argv[1]), Path(sys.argv[2])
def git(*a, **k):
    return subprocess.run(["git", "-C", str(tree), *a], check=True, capture_output=True, text=True, **k).stdout
MUTATIONS = {
    "i_table_in_clientpool_comment": [("Datastore/SQL/ClientPool.py",
        "        self._leased = set()\n",
        "        # serials leased for GkSourceValue rows\n        self._leased = set()\n")],
    "ii_column_as_local_in_schema": [("Datastore/SQL/schema.py",
        '    given: List[sqla.Table] = []\n    for table in tables:\n',
        '    given: List[sqla.Table] = []\n    model_serial = None\n    for table in tables:\n')],
    "iii_project_import_in_replication": [("Datastore/replication.py",
        "    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:\n",
        "    from CosmologyModels import model_ids\n\n    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:\n")],
    "iv_config_sharding_in_store_reader": [("Datastore/store_reader.py",
        "from Datastore.shard_paths import shard_file_problem\n",
        "from Datastore.shard_paths import shard_file_problem\nfrom config.sharding import replicated_tables\n")],
    "v_flag_phrase_in_shard_paths_docstring": [("Datastore/shard_paths.py",
        "Where a ShardedPool's shard files live, given what its primary's ``shards`` table says.\n",
        "Where a ShardedPool's shard files live, given what its primary's ``shards`` table says.\nNothing here reads the source and response columns.\n")],
    "vi_tags_fragment_in_store_reader_comment": [("Datastore/store_reader.py",
        '    return f"sqlite:///file:{path}?mode=ro&uri=true"\n',
        '    # the *_tags tables open read-only like any other\n    return f"sqlite:///file:{path}?mode=ro&uri=true"\n')],
    "vii_new_layer_file_naming_a_class": [("Datastore/scratch_module.py", None,
        '"""A module added to the layer."""\n\nKIND = "BackgroundModel"\n')],
    "viii_glob_narrowed_to_SQL": [("Datastore/tests/test_layer_is_generic.py",
        '    for path in (REPO_ROOT / "Datastore").rglob("*.py"):\n',
        '    for path in (REPO_ROOT / "Datastore" / "SQL").rglob("*.py"):\n'),
        ("Datastore/tests/test_layer_is_generic.py",
        '    for path in (REPO_ROOT / "RayTools").glob("*.py"):\n        found.add(path.relative_to(REPO_ROOT).as_posix())\n',
        '')],
}
assert git("diff") == "", "tree not clean against the index"
for name, edits in MUTATIONS.items():
    new_files = []
    for rel, old, new in edits:
        p = tree / rel
        if old is None:
            p.write_text(new); new_files.append(rel)
            git("add", "-N", rel)
        else:
            s = p.read_text(); assert s.count(old) == 1, (name, rel)
            p.write_text(s.replace(old, new))
    (out / f"{name}.diff").write_text(git("diff"))
    for rel in new_files:
        git("rm", "--cached", "-q", rel); (tree / rel).unlink()
    git("checkout", "--", ".")
    assert git("diff") == "" and git("status", "--porcelain", "--untracked-files=all").count("??") == 0, name
    print("wrote", name)
