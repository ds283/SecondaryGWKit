"""Run main.py's pipeline on a scoped wavenumber sample, against a locally bootstrapped Ray
instance and a fresh datastore.

Written for prompt 12 of prompts/source-remediation (Layer 2 of the verification pass). The
pipeline itself is main.py's: this script does not reimplement any stage. main.py cannot be
imported (it parses sys.argv, opens a Ray connection and a ShardedPool at module scope, and then
runs the pipeline), so this driver

  1. patches ray.init so that main.py's own `ray.init(address=...)` bootstraps a local instance
     with a chosen CPU count instead of attaching to a cluster that does not exist here;
  2. optionally filters config.model_list.build_model_list's return value, so that a run can be
     restricted to one cosmology;
  3. reads main.py's source, substitutes the two hardcoded 50-point wavenumber grids
     (`np.logspace(np.log10(1e5), np.log10(3e8), 50)`, main.py's source and response k samples)
     for a name supplied in the execution globals, and exec()s the result.

Only those three things change; every stage, tag, tolerance and work-queue parameter is main.py's.
The substitution is textual and exact, and the script prints what it replaced and how many times.

Usage (all main.py options after `--` are passed through verbatim):

    PYTHONPATH=. ./venv/bin/python docs/source-remediation-verification/scoped_pipeline_run.py \
        --k-min 1e5 --k-max 1e7 --k-count 7 --cpus 10 --models LambdaCDM \
        -- --database /path/to/fresh.sqlite --job-name label --shards 4 \
           --zend 1e4 --source-samples-log10z 100 --no-prune-unvalidated

The datastore path must not already exist; this script refuses to run otherwise, so that no
existing datastore can be overwritten or migrated.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"

# the exact text of both hardcoded wavenumber grids in main.py's driver block
K_GRID_LITERAL = "np.logspace(np.log10(1e5), np.log10(3e8), 50)"
K_GRID_NAME = "SCOPED_K_SAMPLE"


def main():
    parser = argparse.ArgumentParser(
        description="run main.py's pipeline on a scoped wavenumber sample"
    )
    parser.add_argument("--k-min", type=float, required=True)
    parser.add_argument("--k-max", type=float, required=True)
    parser.add_argument("--k-count", type=int, required=True)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="restrict config.model_list.build_model_list to these labels",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="permit an existing datastore (for a resume/lookup test); the pipeline only adds "
        "work that is missing, it never migrates or rewrites existing rows",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="arguments for main.py, after a bare --",
    )
    args = parser.parse_args()

    main_args = list(args.main_args)
    if main_args and main_args[0] == "--":
        main_args = main_args[1:]

    # refuse to touch an existing datastore
    if "--database" in main_args:
        db = Path(main_args[main_args.index("--database") + 1])
        if db.exists() and not args.allow_existing:
            raise RuntimeError(
                f"scoped_pipeline_run: datastore {db} already exists; refusing to reuse, "
                "overwrite or migrate it. Choose a path that does not exist."
            )
        if not db.parent.is_dir():
            raise RuntimeError(
                f"scoped_pipeline_run: {db.parent} is not a directory, so {db} is not writable"
            )

    k_sample = np.logspace(
        np.log10(args.k_min), np.log10(args.k_max), args.k_count
    ).tolist()
    print(f"** scoped wavenumber sample ({len(k_sample)} modes, 1/Mpc):")
    for k in k_sample:
        print(f"     {k:.6g}")

    # (1) bootstrap Ray locally rather than attaching to a cluster
    import ray

    _real_ray_init = ray.init

    def _local_ray_init(*_args, **_kwargs):
        print(
            f"** scoped_pipeline_run: bootstrapping a local Ray instance "
            f"(num_cpus={args.cpus}, include_dashboard=False)"
        )
        return _real_ray_init(
            num_cpus=args.cpus, include_dashboard=False, ignore_reinit_error=True
        )

    ray.init = _local_ray_init

    # (2) optionally restrict the model list
    if args.models is not None:
        import config.model_list as model_list_module

        _real_build_model_list = model_list_module.build_model_list

        def _filtered_build_model_list(pool, units):
            models = _real_build_model_list(pool, units)
            kept = [m for m in models if m["label"] in args.models]
            if len(kept) == 0:
                raise RuntimeError(
                    f"scoped_pipeline_run: no model in {[m['label'] for m in models]} "
                    f"matches {args.models}"
                )
            print(
                f"** scoped_pipeline_run: running models {[m['label'] for m in kept]}"
            )
            return kept

        model_list_module.build_model_list = _filtered_build_model_list

    # (3) substitute the wavenumber grids and run
    source = MAIN_PY.read_text()
    count = source.count(K_GRID_LITERAL)
    if count != 2:
        raise RuntimeError(
            f"scoped_pipeline_run: expected exactly 2 occurrences of "
            f"{K_GRID_LITERAL!r} in main.py, found {count}"
        )
    source = source.replace(K_GRID_LITERAL, K_GRID_NAME)
    print(
        f"** scoped_pipeline_run: replaced {count} occurrences of {K_GRID_LITERAL!r} "
        f"in main.py with {K_GRID_NAME}"
    )

    sys.argv = ["main.py"] + main_args
    print(f"** scoped_pipeline_run: main.py argv = {sys.argv[1:]}")

    namespace = {
        "__name__": "__main__",
        "__file__": str(MAIN_PY),
        K_GRID_NAME: k_sample,
    }
    exec(compile(source, str(MAIN_PY), "exec"), namespace)


if __name__ == "__main__":
    main()
