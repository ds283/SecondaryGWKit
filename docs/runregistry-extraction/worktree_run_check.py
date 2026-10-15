"""
The worktree check (runregistry-extraction prompt 06, acceptance 4; README §6.1 Q3, §6.2 C15).

A run begun from a linked git worktree of this repository must be recorded in the **main
checkout's** `var/runs/`, where it outlives the worktree, while its provenance (`git_head`,
`git_dirty`, the script's hash) is the worktree's own. The registry this repository held until
prompt 06 recorded it in the worktree's `var/runs/`, and so lost it with the worktree
(`[00-a-run-launched-from-a-worktree-records-into-the-worktree]`).

Run from inside a linked worktree, with the main checkout's venv by absolute path (a worktree has
no `venv/`, and this script needs nothing but `runregistrykit` and the standard library)::

    /path/to/main/checkout/venv/bin/python docs/runregistry-extraction/worktree_run_check.py

It begins one run, `runregistry-extraction-06-<--slug>-<timestamp>`, with
`runregistrykit.begin(..., script=__file__)`, so that the client root is found from this script's
own directory, in the worktree. Then it checks, printing each check with its result:

1. the main checkout, the parent of `git rev-parse --git-common-dir` taken from this script's
   directory, is not the worktree's top level (so this is a linked worktree);
2. the run directory lies directly under the main checkout's `var/runs/`;
3. the manifest's `git_head` is the worktree's `HEAD`;
4. and is not the main checkout's `HEAD`.

The run is finished on every path, `done` with exit code 0 when every check holds and `failed`
with exit code 1 otherwise, a check that raises included; it is never left `running`. The script
prints the run's id and path, and exits 0 if and only if every check holds. **The run is a record,
meant to outlive the worktree: never delete it.**
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def git(*args, cwd=HERE) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Begin a run from a linked worktree, and check that it is recorded in the "
        "main checkout's var/runs/ with the worktree's provenance."
    )
    parser.add_argument("--slug", default="worktree-check")
    args = parser.parse_args(argv)

    import runregistrykit

    worktree_head = git("rev-parse", "HEAD")
    worktree_top = git("rev-parse", "--show-toplevel")
    common_dir = os.path.join(HERE, git("rev-parse", "--git-common-dir"))
    main_checkout = os.path.dirname(os.path.realpath(common_dir))
    main_head = git("rev-parse", "HEAD", cwd=main_checkout)
    expected_root = os.path.join(main_checkout, "var", "runs")

    run = runregistrykit.begin(
        campaign="runregistry-extraction",
        prompt="06",
        slug=args.slug,
        purpose=(
            "runregistry-extraction prompt 06, the worktree check (acceptance 4, Q3): a run "
            f"begun from a linked git worktree at {worktree_head[:12]}, the probe commit of "
            "06's staged tree, which no branch names; it is recorded in the main checkout's "
            "var/runs/, and this record is meant to outlive the worktree"
        ),
        script=__file__,
    )
    print(f"run {run.id}")
    print(f"path {run.path}")

    results = []
    try:

        def check(name, ok, detail=""):
            results.append(bool(ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

        manifest = runregistrykit.read_json(run.manifest_path) or {}
        print(f"  worktree top level  {worktree_top}")
        print(f"  main checkout       {main_checkout}")
        print(f"  worktree HEAD       {worktree_head}")
        print(f"  main checkout HEAD  {main_head}")
        print(
            f"  manifest git_head   {manifest.get('git_head')}, git_dirty "
            f"{manifest.get('git_dirty')}"
        )
        check(
            "this is a linked worktree, not the main checkout",
            os.path.realpath(worktree_top) != os.path.realpath(main_checkout),
            f"{worktree_top} and {main_checkout}",
        )
        check(
            "the run directory lies directly under the main checkout's var/runs/",
            os.path.realpath(os.path.dirname(run.path))
            == os.path.realpath(expected_root),
            f"{os.path.dirname(run.path)} and {expected_root}",
        )
        check(
            "the manifest's git_head is the worktree's HEAD",
            manifest.get("git_head") == worktree_head,
            f"{manifest.get('git_head')} and {worktree_head}",
        )
        check(
            "the manifest's git_head is not the main checkout's HEAD",
            manifest.get("git_head") != main_head,
            f"{manifest.get('git_head')} and {main_head}",
        )
    finally:
        ok = bool(results) and all(results)
        run.finish("done" if ok else "failed", exit_code=0 if ok else 1)
    print(
        f"=> {'every check holds' if ok else 'FAILED'} ({sum(results)} of {len(results)}); "
        f"run {run.id} finished {'done' if ok else 'failed'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
