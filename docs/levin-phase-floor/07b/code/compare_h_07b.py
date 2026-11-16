"""
levin-phase-floor prompt 07b, Part B: the code compared with Part A's variant (h) (section 1.1, section 9.5 U8) on the
problems this Part runs on the laptop: every call of section 3's tests (a)-(e) in
AdaptiveLevin/tests/test_levin_rate_rules.py, and every cell of the three existing tests whose assertions section 3's
refusals change (test_levin_trust_yardstick test 2, test_levin_leaf_values test 1, test_levin_roundoff_exit test 1).
The LiouvilleGreen draws are compared by lg_draws_07b.py.

Three modules are loaded under names of their own: the commit's AdaptiveLevin/levin_quadrature.py; (h),
docs/levin-phase-floor/07b/measurement/scripts/ext/variants/levin_quadrature_07b_h.py, its SHA-256 checked; and the
parent, the dispatch commit c768c4ad's module read with `git show`. Each call is run under all three, and compared:
every result key but "elapsed" and the commit's two refusal counts, and every returned region field by field (each
float by its hex). Commit against (h) must be identical on every call; commit against the parent is reported.

Run from the repository root:  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/07b/code/compare_h_07b.py
Writes docs/levin-phase-floor/07b/code/out/compare_h_07b.txt and .json. No Ray and no datastore.
"""

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
H_PATH = (
    REPO
    / "docs/levin-phase-floor/07b/measurement/scripts/ext/variants/levin_quadrature_07b_h.py"
)
H_SHA256 = "475120223bd6bdd1a62c98708969c2aac700c7cacb0d2110154c09ebbc080af3"
PARENT_SHA = "c768c4ad3f249c2094040601b3b5d86c93659fc0"
COUNTS = ("num_edge_rule_refusals", "num_range_rule_refusals")


def _exec(name, source, label):
    module = types.ModuleType(name)
    module.__file__ = label
    sys.modules[name] = module
    exec(compile(source, label, "exec"), module.__dict__)
    return module


def canon(x):
    import numpy as np

    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.ndarray):
        return [canon(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {
            str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if x is None or isinstance(x, str):
        return x
    if hasattr(x, "__dict__"):
        return canon(vars(x))
    return repr(x)


def record(data):
    return canon({k: v for k, v in data.items() if k not in ("elapsed",) + COUNTS})


def main():
    import AdaptiveLevin.levin_quadrature as commit
    import AdaptiveLevin.tests.test_levin_rate_rules as rr
    import AdaptiveLevin.tests.test_levin_trust_yardstick as ty
    import AdaptiveLevin.tests.test_levin_leaf_values as lv
    import AdaptiveLevin.tests.test_levin_roundoff_exit as rx

    raw = H_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == H_SHA256
    hmod = _exec("levin_quadrature_07b_h", raw.decode(), str(H_PATH))
    source = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parent = _exec("levin_quadrature_parent_c768c4a", source, "c768c4ad")

    def sincos(span, f, theta, **kw):
        return lambda m: m.adaptive_levin_sincos(span, f, theta, **kw)

    def driver(span, f, theta, **kw):
        return lambda m: m._adaptive_levin(span, f, m._Basis_SinCos(theta), **kw)

    calls = [
        (
            "(b)",
            sincos(
                (0.0, 1.0),
                rr.Q_F,
                rr.Q_THETA,
                atol=rr.B_ATOL,
                rtol=rr.B_RTOL,
                chebyshev_order=rr.Q_ORDER,
            ),
        ),
        (
            "(c)",
            sincos(
                rr.C_SPAN,
                rr.Q_F,
                rr.Q_THETA,
                atol=rr.C_ATOL,
                rtol=rr.C_RTOL,
                chebyshev_order=rr.Q_ORDER,
            ),
        ),
        (
            "(d)",
            sincos(
                rr.D_SPAN,
                rr.D_F,
                rr.D_THETA,
                atol=rr.D_ATOL,
                rtol=rr.D_RTOL,
                depth_max=rr.D_DEPTH_MAX,
            ),
        ),
        (
            "(e1)",
            sincos((0.0, 1.0), rr.E1_F, rr.E1_THETA, atol=rr.E_ATOL, rtol=rr.E_RTOL),
        ),
        (
            "(e2)",
            sincos((0.0, 6.0), rr.F_23, rr.E2_THETA, atol=rr.E_ATOL, rtol=rr.E_RTOL),
        ),
    ]
    for name, span, f, theta, kwargs in ty.test2_cells():
        calls.append((f"yardstick test 2 {name}", sincos(span, f, theta, **kwargs)))
    for name, span, f, theta, kwargs, _, _ in lv.test1_cells():
        calls.append((f"leaf values test 1 {name}", driver(span, f, theta, **kwargs)))
    f, theta = rx._test1_problem()
    calls.append(
        (
            "roundoff exit test 1",
            driver(rx.TEST1_SPAN, f, theta, atol=rx.TEST1_ATOL, rtol=rx.TEST1_RTOL),
        )
    )

    lines, rows = [], []
    for name, call in calls:
        c, h, p = call(commit), call(hmod), call(parent)
        row = {
            "call": name,
            "edge": c["num_edge_rule_refusals"],
            "range": c["num_range_rule_refusals"],
            "identical_to_h": record(c) == record(h),
            "identical_to_parent": record(c) == record(p),
            "value_hex": float(c["value"]).hex(),
            "abserr_hex": float(c["abserr"]).hex(),
            "regions": c["num_regions"],
            "parent_regions": p["num_regions"],
        }
        rows.append(row)
        lines.append(
            f"{name}: refusals edge {row['edge']} range {row['range']}; regions {row['parent_regions']} -> "
            f"{row['regions']}; identical to (h) {row['identical_to_h']}; identical to the parent "
            f"{row['identical_to_parent']}; value {row['value_hex']} abserr {row['abserr_hex']}"
        )
    summary = (
        f"{len(rows)} calls; identical to (h): {sum(r['identical_to_h'] for r in rows)}; rules act in "
        f"{sum(1 for r in rows if r['edge'] + r['range'])}; of the calls where they do not, identical to the parent: "
        f"{sum(1 for r in rows if not (r['edge'] + r['range']) and r['identical_to_parent'])} of "
        f"{sum(1 for r in rows if not (r['edge'] + r['range']))}"
    )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    provenance = [
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"git HEAD {head}; commit module sha256 {hashlib.sha256(Path(commit.__file__).read_bytes()).hexdigest()}",
        f"(h) {H_PATH.relative_to(REPO)} sha256 {H_SHA256}; parent {PARENT_SHA}",
    ]
    text = "\n".join(provenance + [summary, ""] + lines) + "\n"
    (HERE / "out" / "compare_h_07b.txt").write_text(text)
    (HERE / "out" / "compare_h_07b.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
