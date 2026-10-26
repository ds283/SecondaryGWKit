"""
levin-phase-floor prompt 05 section 6: the dispatcher at ``PrimitivePhase.theta_mod_2pi``, cut down
from prompt 04's ``scripts/phases.py`` (docs/levin-phase-floor/04/scripts/phases.py, sha256
9f046157..., a record that is not edited; copied here as dispatch.py and then reduced).

What is kept from 04 is only its pass-through arm P and its budget: ``install()`` replaces the
class attribute ``ComputeTargets.primitive_phase.PrimitivePhase.theta_mod_2pi`` for the duration of
a ``with`` block with a dispatcher that
- raises ``stop_exc`` (the run's ``OverBudget``, a ``BaseException``) at its entry once
  ``time.perf_counter()`` is past ``deadline``, before calling anything (prompt 05 section 6:
  "raise from the dispatcher, not by a signal inside ``mpmath``", 04's I9);
- counts every call by the kind of phase (G: sign -1, leading ``tau``; T: sign +1, leading
  ``cs_tau``; anything else ``other``) and by the phase's ``_exact_remainder`` flag (``None`` where
  the tree has no such attribute, which is the parent tree ``e648951``);
- otherwise calls the tree's own ``PrimitivePhase.theta_mod_2pi``, captured at import as
  ``ORIG_THETA_MOD_2PI``, with exactly its arguments, and returns its value. So the path is
  production's in each tree: at ``e648951`` the default remainder for G and T alike; at the code
  commit the node-anchored remainder for G (``exact_remainder=True``, set by
  ``GkSourcePolicyData._build_phase``) and the default for T.

04's arms A and X, Delta^ext, Phi and F are not used by prompt 05 and are not carried over.
Nothing here changes production code; the original is restored in a ``finally``.
"""

import contextlib
import time

from ComputeTargets.primitive_phase import PrimitivePhase

ORIG_THETA_MOD_2PI = PrimitivePhase.theta_mod_2pi
ORIG_FILE = ORIG_THETA_MOD_2PI.__code__.co_filename


def kind_of(phase) -> str:
    if type(phase) is not PrimitivePhase:
        return "other"
    label = getattr(phase._leading, "label", None)
    if phase._sign == -1 and label == "tau":
        return "G"
    if phase._sign == +1 and label == "cs_tau":
        return "T"
    return "other"


class Dispatcher:
    def __init__(self, deadline=None, stop_exc=None):
        self.deadline = deadline
        self.stop_exc = stop_exc
        self.counts = {"G": 0, "T": 0, "other": 0}
        self.flags = {}  # "G exact_remainder=True" -> count

    def __call__(self, phase, x, x_is_log=False):
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise self.stop_exc()
        kind = kind_of(phase)
        self.counts[kind] += 1
        key = f"{kind} exact_remainder={getattr(phase, '_exact_remainder', None)}"
        self.flags[key] = self.flags.get(key, 0) + 1
        return ORIG_THETA_MOD_2PI(phase, x, x_is_log)


@contextlib.contextmanager
def install(deadline=None, stop_exc=None):
    """Patch PrimitivePhase.theta_mod_2pi at the class attribute with the dispatcher, and restore
    the original in a finally. Yields the Dispatcher (its .counts and .flags)."""
    if PrimitivePhase.__dict__["theta_mod_2pi"] is not ORIG_THETA_MOD_2PI:
        raise RuntimeError("dispatch: PrimitivePhase.theta_mod_2pi is already patched")
    dispatcher = Dispatcher(deadline=deadline, stop_exc=stop_exc)

    def theta_mod_2pi(self, x, x_is_log=False):
        return dispatcher(self, x, x_is_log)

    theta_mod_2pi.__name__ = "theta_mod_2pi"
    theta_mod_2pi.__qualname__ = "PrimitivePhase.theta_mod_2pi[dispatcher]"
    PrimitivePhase.theta_mod_2pi = theta_mod_2pi
    try:
        yield dispatcher
    finally:
        PrimitivePhase.theta_mod_2pi = ORIG_THETA_MOD_2PI
    if PrimitivePhase.__dict__["theta_mod_2pi"] is not ORIG_THETA_MOD_2PI:
        raise RuntimeError("dispatch: PrimitivePhase.theta_mod_2pi was not restored")
