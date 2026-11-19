"""
levin-phase-floor prompt 07c, Part B: an instrument on the commit's AdaptiveLevin/levin_quadrature.py, for the facts
section 3's tests (c), (d) and (e) assert through a call's regions and counts (AdaptiveLevin/tests/
test_levin_chase_stop.py): which leaves the stop reached at the stop and which after it, which leaves of the stopped
subtree were at the noise level, and, on (c3)'s call, how many bisections were made at the noise level below a leaf
the trust test refused with no rate-rule refusal above it.

The module's source is read from the working tree and executed under another name with log statements inserted (each
insertion a text replacement that must match exactly once); the instrument records and changes nothing, and each
instrumented call is checked equal to the commit's own (value, abserr, regions and counts) before its log is written.

  (b)'s call: every trust check (leaf, depth, span, Levin or Clenshaw-Curtis, trusted, refused by a rate rule, root,
              at the noise level, delta1 / nu_B, delta2 / nu_B); every bisection counted against a root; the stop; and
              every leaf stopped, with the bisection count at which it was stopped.
  (c3)'s call: every bisection of a leaf at the noise level whose nearest ancestor-or-self found untrusted by the
              trust check (not by a rate rule) exists, with no rate-rule root: the bisections a stop rooted at any
              untrusted leaf would count (section 3 breakage (vi)).

Run from the repository root:  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/07c/code/trace_07c.py
Writes docs/levin-phase-floor/07c/code/out/trace_07c.txt. No Ray and no datastore.
"""

import hashlib
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
LQ = REPO / "AdaptiveLevin/levin_quadrature.py"

INSERTIONS = [
    (
        "        if chase_root.get(leaf.seq) in chase_stopped and _leaf_at_chase_noise_level(\n",
        "        LOG.append(('check', leaf.seq, leaf.depth, min(a, b), max(a, b), 'CC' if leaf.is_direct else 'Levin',"
        " trusted, refused_by_rate_rule, chase_root.get(leaf.seq), _leaf_at_chase_noise_level(leaf.seq),"
        " delta1 / noise_yardstick if noise_yardstick > 0 else None,"
        " delta2 / noise_yardstick if noise_yardstick > 0 else None))\n"
        "        if not trusted and not refused_by_rate_rule and leaf.seq not in chase_root:\n"
        "            TRUST_ROOT.setdefault(leaf.seq, leaf.seq)\n"
        "        if chase_root.get(leaf.seq) in chase_stopped and _leaf_at_chase_noise_level(\n",
    ),
    (
        "        _floor_limit_stopped_window_leaf(leaf)\n        num_chase_stopped_leaves += 1\n",
        "        _floor_limit_stopped_window_leaf(leaf)\n        num_chase_stopped_leaves += 1\n"
        "        LOG.append(('stopped', leaf.seq, leaf.depth, leaf.start, leaf.end, 'CC' if leaf.is_direct else 'Levin',"
        " num_splits))\n",
    ),
    (
        "            num_chase_noise_bisections += 1\n",
        "            num_chase_noise_bisections += 1\n"
        "            LOG.append(('count', leaf.seq, leaf.depth, leaf.start, leaf.end, 'CC' if leaf.is_direct else 'Levin',"
        " num_splits, chase_root_seq, left.seq, right.seq))\n",
    ),
    (
        "                chase_stopped.add(chase_root_seq)\n",
        "                chase_stopped.add(chase_root_seq)\n"
        "                LOG.append(('STOP', chase_root_seq, num_splits))\n",
    ),
    (
        "        chase_root_seq = chase_root.get(leaf.seq)\n",
        "        chase_root_seq = chase_root.get(leaf.seq)\n"
        "        trust_root_seq = TRUST_ROOT.get(leaf.seq)\n"
        "        if trust_root_seq is not None and chase_root_seq is None:\n"
        "            TRUST_ROOT[left.seq] = trust_root_seq\n"
        "            TRUST_ROOT[right.seq] = trust_root_seq\n"
        "            if _leaf_at_chase_noise_level(leaf.seq):\n"
        "                LOG.append(('trust-noise-bisection', leaf.seq, leaf.depth, leaf.start, leaf.end,"
        " 'CC' if leaf.is_direct else 'Levin', num_splits, trust_root_seq))\n",
    ),
]


def load_instrumented():
    source = LQ.read_text()
    for old, new in INSERTIONS:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType("levin_quadrature_07c_trace")
    module.__file__ = "trace"
    module.LOG = []
    module.TRUST_ROOT = {}
    sys.modules[module.__name__] = module
    exec(compile(source, "trace", "exec"), module.__dict__)
    return module


def fmt(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def same(a, b):
    keys = (
        "value",
        "abserr",
        "converged",
        "num_regions",
        "num_chase_stops",
        "num_chase_stopped_leaves",
        "num_chase_noise_bisections",
    )
    return all(a[k] == b[k] for k in keys) and [
        (r.start, r.end, r.value, r.declared_resolution) for r in a["regions"]
    ] == [(r.start, r.end, r.value, r.declared_resolution) for r in b["regions"]]


def main():
    import AdaptiveLevin.levin_quadrature as commit
    import AdaptiveLevin.tests.test_levin_chase_stop as cs

    lines = [
        f"script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        "git HEAD "
        + subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        + f"; levin_quadrature.py sha256 {hashlib.sha256(LQ.read_bytes()).hexdigest()}",
        "",
    ]
    for label, (A, kappa) in (
        ("(b)", (cs.B_A, cs.B_KAPPA)),
        ("(c3)", (cs.C3_A, cs.C3_KAPPA)),
    ):
        m = load_instrumented()
        f, theta = cs.problem(A, kappa, cs.ETA)
        kwargs = dict(atol=cs.B_ATOL, rtol=cs.B_RTOL)
        data = m.adaptive_levin_sincos(cs.SPAN, f, theta, **kwargs)
        own = commit.adaptive_levin_sincos(cs.SPAN, f, theta, **kwargs)
        assert same(data, own), label
        lines.append(
            f"{label} A = {A}, kappa = {kappa}, eta = {cs.ETA}: the instrumented call equals the commit's; value "
            f"{data['value'].hex()}, abserr {data['abserr'].hex()}, converged {data['converged']}, regions "
            f"{data['num_regions']}, max depth {data['max_depth']}, refusals range {data['num_range_rule_refusals']} "
            f"edge {data['num_edge_rule_refusals']}; stops {data['num_chase_stops']}, stopped leaves "
            f"{data['num_chase_stopped_leaves']}, bisections counted {data['num_chase_noise_bisections']}"
        )
        lines.append(
            "  check: seq depth lo hi route trusted refused_by_rate_rule root at_noise_level delta1/nu_B delta2/nu_B"
        )
        lines.append(
            "  count: seq depth lo hi route split root left right; STOP: root split; stopped: seq depth lo hi route "
            "split; trust-noise-bisection: seq depth lo hi route split trust_root"
        )
        stop_split = None
        for event in m.LOG:
            if event[0] == "STOP":
                stop_split = event[2]
            lines.append("  " + " ".join(fmt(x) for x in event))
        if label == "(c3)":
            n = sum(1 for e in m.LOG if e[0] == "trust-noise-bisection")
            lines.append(
                f"  bisections at the noise level below a leaf the trust test refused, with no rate-rule root: {n}"
            )
        else:
            start = next(i for i, x in enumerate(m.LOG) if x[0] == "STOP")
            at, after, in_loop = [], [], True
            for e in m.LOG[start + 1 :]:
                if e[0] != "stopped":
                    in_loop = False
                    continue
                (at if in_loop else after).append(e)
            lines.append(
                f"  stopped by the stop itself (split {stop_split}): {len(at)}; stopped by their own check after the stop: "
                f"{len(after)}: "
                + ", ".join(f"[{e[3]!r}, {e[4]!r}] depth {e[2]}" for e in after)
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    (HERE / "out" / "trace_07c.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
