"""levin-phase-floor prompt 06k section 1 ("The loading must be transparent"): load a variant of
``AdaptiveLevin/levin_quadrature.py`` in place of the tree's module, in this process. New in 06k.

``install(path, sha256)`` must run before anything imports ``AdaptiveLevin``. It executes the
variant file as the module ``AdaptiveLevin.levin_quadrature`` and registers it in ``sys.modules``,
then imports the package ``AdaptiveLevin`` from the tree, whose ``__init__``
(``from .levin_quadrature import adaptive_levin_sincos``) finds the registered module and binds the
variant's function; the package's attribute ``levin_quadrature`` is set to the variant too. Every
production import (``from AdaptiveLevin import adaptive_levin_sincos`` in
``ComputeTargets.QuadSourceIntegral`` and ``LiouvilleGreen.three_bessel_integrals``) and every
``importlib.import_module("AdaptiveLevin.levin_quadrature")`` then sees the variant, and every
wrapper the harness installs at a module attribute of it is seen by the variant's own functions,
which look those names up in their module's globals at call time. The tree's file is never
executed in the process. Nothing is written.

``evidence()`` reports what was installed, for the worker's tree evidence.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

MODULE = "AdaptiveLevin.levin_quadrature"
_STATE = {}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(path, expected_sha256, name):
    path = Path(path).resolve()
    if MODULE in sys.modules or "AdaptiveLevin" in sys.modules:
        raise RuntimeError("variant_loader: AdaptiveLevin is already imported; the variant must be installed first")
    got = _sha256(path)
    if got != expected_sha256:
        raise RuntimeError(f"variant_loader: {path} has sha256 {got}, not {expected_sha256}")
    spec = importlib.util.spec_from_file_location(MODULE, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(MODULE, None)
        raise
    import AdaptiveLevin  # the tree's package; its __init__ binds the registered module's function

    AdaptiveLevin.levin_quadrature = module
    if AdaptiveLevin.adaptive_levin_sincos is not module.adaptive_levin_sincos:
        raise RuntimeError("variant_loader: AdaptiveLevin.adaptive_levin_sincos is not the variant's")
    _STATE.update(name=name, path=str(path), sha256=got, module=module, package_file=AdaptiveLevin.__file__)
    return module


def installed():
    return bool(_STATE)


def evidence():
    if not _STATE:
        return None
    import AdaptiveLevin

    m = sys.modules.get(MODULE)
    return {
        "name": _STATE["name"],
        "path": _STATE["path"],
        "sha256": _STATE["sha256"],
        "sha256_now": _sha256(_STATE["path"]),
        "sys_modules_is_variant": m is _STATE["module"],
        "module_file": getattr(m, "__file__", None),
        "package_file": _STATE["package_file"],
        "package_attribute_is_variant": getattr(AdaptiveLevin, "levin_quadrature", None) is _STATE["module"],
        "package_function_is_variants": AdaptiveLevin.adaptive_levin_sincos is _STATE["module"].adaptive_levin_sincos,
    }
