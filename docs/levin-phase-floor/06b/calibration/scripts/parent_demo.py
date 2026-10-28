"""levin-phase-floor prompt 06b, section 2 tests 1-4: the parent behaviour, demonstrated against the
parent commit's own module.

The parent's AdaptiveLevin/levin_quadrature.py is read with ``git show <parent>:...`` into a
temporary directory and imported under its own name, beside the working tree's module. Every
problem of AdaptiveLevin/tests/test_levin_roundoff_exit.py -- tests 1, 2 (a, b) and 3, and test
4's restatement of every driver problem in test_levin_quadrature.py -- is run three ways:

  parent   the parent commit's _adaptive_levin()
  bypass   the working tree's _adaptive_levin(..., _test_only_bypass_roundoff_premise=True)
  new      the working tree's _adaptive_levin()

and the output records, per problem, whether parent and bypass agree bit for bit (value, abserr,
evaluations, region count, phase-limited count, max depth, and every field of every region), and
both results in full where the parent has a phase-limited region (prompt section 2 test 4: "Where
it has one, the log tabulates both results"). Test 1's parent numbers come from here.

Writes ../out/parent_demo.txt. No Ray and no datastore.
"""

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

REPO = prov.REPO
sys.path.insert(0, str(REPO))

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
import AdaptiveLevin.tests.test_levin_roundoff_exit as T  # noqa: E402

OUT = prov.OUT / "parent_demo.txt"
TEST_MODULE = REPO / "AdaptiveLevin" / "tests" / "test_levin_roundoff_exit.py"


def load_parent(sha, directory):
    source = prov.git("show", f"{sha}:AdaptiveLevin/levin_quadrature.py")
    path = Path(directory) / "levin_quadrature_parent.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("levin_quadrature_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, prov.sha256(path)


def problems():
    """(name, span, f, theta, kwargs, basis_factory, raises, reference)"""
    out = []
    f, theta = T._test1_problem()
    out.append(
        (
            "test 1 (Levin, theta = 100 e^x on [0, 20])",
            T.TEST1_SPAN,
            f,
            theta,
            dict(atol=T.TEST1_ATOL, rtol=T.TEST1_RTOL),
            None,
            None,
            T._test1_reference(),
        )
    )
    for label, problem, span in (
        ("test 2a (constant theta')", T._test2a_problem(), T.TEST2A_SPAN),
        (
            "test 2b (theta' = 5e5 e^{x/2} on [0, 2])",
            T._test2b_problem(),
            T.TEST2B_SPAN,
        ),
    ):
        f, theta, kwargs, _, expression = problem
        out.append(
            (label, span, f, theta, kwargs, None, None, T._reference(expression))
        )
    f, theta = T._test3_problem()
    out.append(
        (
            "test 3 (fallback, theta = 1e-6 e^x on [0, 6])",
            T.TEST3_SPAN,
            f,
            theta,
            dict(atol=T.TEST3_ATOL, rtol=T.TEST3_RTOL),
            None,
            None,
            T._test3_reference(),
        )
    )
    for name, span, f, theta, kwargs, basis, raises in T.levin_quadrature_problems():
        factory = None
        if basis is not None:
            factory = lambda b=basis: b  # noqa: E731
        out.append((f"test 4: {name}", span, f, theta, kwargs, factory, raises, None))
    return out


def call(module, span, f, theta, kwargs, factory, **extra):
    basis = factory() if factory is not None else module._Basis_SinCos(theta)
    return module._adaptive_levin(span, f, basis, **kwargs, **extra)


def describe(data):
    return (
        f"value {data['value']!r} abserr {data['abserr']!r} regions {data['num_regions']}"
        f" phase-limited {data['num_phase_limited_regions']} max depth {data['max_depth']}"
        f" evaluations {data['evaluations']} converged {data['converged']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parent", default="da5c24d")
    args = parser.parse_args()

    lines = prov.header(
        __file__,
        "section 2 tests 1-4 against the parent commit's own module",
        (TEST_MODULE, REPO / "AdaptiveLevin" / "tests" / "test_levin_quadrature.py"),
    )
    with tempfile.TemporaryDirectory() as directory:
        parent, parent_sha = load_parent(args.parent, directory)
        full = prov.git("rev-parse", args.parent).strip()
        lines.append(
            f"# parent module: git show {full}:AdaptiveLevin/levin_quadrature.py,"
            f" sha256 {parent_sha}"
        )
        lines.append("")

        mismatches = []
        for name, span, f, theta, kwargs, factory, raises, reference in problems():
            lines.append(f"== {name}")
            lines.append(f"   span {span} kwargs {kwargs}")
            if raises is not None:
                outcomes = []
                for label, module, extra in (
                    ("parent", parent, {}),
                    ("bypass", lq, {"_test_only_bypass_roundoff_premise": True}),
                    ("new", lq, {}),
                ):
                    try:
                        call(module, span, f, theta, kwargs, factory, **extra)
                        outcomes.append(f"{label}: no exception")
                    except raises as error:
                        outcomes.append(f"{label}: {type(error).__name__}")
                lines.append("   " + "; ".join(outcomes))
                if any("no exception" in o for o in outcomes):
                    mismatches.append(name)
                continue

            p = call(parent, span, f, theta, kwargs, factory)
            b = call(
                lq,
                span,
                f,
                theta,
                kwargs,
                factory,
                _test_only_bypass_roundoff_premise=True,
            )
            n = call(lq, span, f, theta, kwargs, factory)
            same_pb = T._call_signature(p) == T._call_signature(b)
            same_bn = T._call_signature(b) == T._call_signature(n)
            if not same_pb:
                mismatches.append(name)
            lines.append(f"   parent == bypass, bit for bit: {same_pb}")
            lines.append(f"   bypass == new, bit for bit:    {same_bn}")
            lines.append(f"   parent  {describe(p)}")
            lines.append(f"   new     {describe(n)}")
            if reference is not None:
                ref, ref_err = reference
                lines.append(
                    f"   reference {ref!r} (its own error estimate {ref_err:.2e});"
                    f" true error parent {abs(p['value'] - ref):.4e}"
                    f" (relative {abs(p['value'] - ref) / abs(ref):.4e}),"
                    f" new {abs(n['value'] - ref):.4e}"
                    f" (relative {abs(n['value'] - ref) / abs(ref):.4e})"
                )
            if p["num_phase_limited_regions"] > 0 or not same_bn:
                for label, data in (("parent", p), ("new", n)):
                    for region in data["regions"]:
                        lines.append(f"   {label} region {region}")

        lines.append("")
        lines.append(
            f"== parent vs bypass: problems that differ {mismatches if mismatches else 'none'}"
        )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}; parent/bypass mismatches: {mismatches}")


if __name__ == "__main__":
    main()
