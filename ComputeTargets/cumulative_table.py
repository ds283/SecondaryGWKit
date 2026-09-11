"""
A cumulative-integral table on a fixed redshift grid, held in double-double precision, with a
pointwise accessor and an *interval* accessor.

This is the numerical kernel behind ``BackgroundModel.functions.tau`` (and, from prompt 04 of the
``prompts/GkTk-remedial`` campaign, its siblings ``cs_tau`` and ``friction_F``). It knows nothing
about cosmology: it is handed a grid of nodes, an integrand ``f(z)`` and a Gauss-Legendre order,
and it accumulates

    T(z) = int_z^{z_top} f(z') dz'

on the nodes, integrating in ``u = log(1+z)`` (so the integrand it actually samples is
``f(z) (1+z)``), one Gauss-Legendre panel per grid interval, split further at any *break points*
of the integrand the caller declares (review §11; ``docs/gktk-remedial/RESIDUAL-CONVERGENCE.md``).

**Why the table is double-double** (review §13.3). The phase wants ``k [tau(z_a) - tau(z_b)]`` over
baselines that can be a single grid interval, while ``tau`` itself is ~1.4e4 Mpc at low redshift.
A correctly rounded double node value carries half an ulp of *tau*, ~1.5e-12 Mpc, and the
difference of two such values inherits it: at ``k = 3e8/Mpc`` that is ~9e-4 rad of phase however
short the baseline. Holding every node as a ``(hi, lo)`` pair -- ``hi = fsum(terms)``,
``lo = fsum([*terms, -hi])``, the exact residual correctly rounded -- and differencing hi-with-hi
and lo-with-lo makes the error of the difference relative to the *difference*, not to the
primitive. Only the stored table is wide; both accessors return plain floats.

**The interval accessor is never a difference of two pointwise values.** With ``n(a)``, ``n(b)``
the nodes nearest the endpoints,

    delta(z_a, z_b) = T(z_b) - T(z_a)
                    = partial(a -> n(a)) + [(hi_b - hi_a) + (lo_b - lo_a)] + partial(n(b) -> b)

where each partial is a local Gauss panel over at most half a grid interval (zero, and free, for
an on-grid endpoint). Node lookup is by **exact** ``z`` equality -- the production grids are the
tables' own nodes -- never through a ``log(1+z) -> z`` round trip, which is lossy at large ``z``
(``CLAUDE.md``, "Redshift arithmetic").

Sign convention (``prompts/GkTk-remedial/README.md`` §2 (c)): ``delta(z_a, z_b) = T(z_b) - T(z_a)
= int_{z_b}^{z_a} f dz``, positive when ``z_b < z_a`` and ``f >= 0``.

No Ray, no datastore, and no other ``ComputeTargets`` object is imported here.
"""

from math import fsum, log1p, expm1
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def two_sum(a: float, b: float) -> Tuple[float, float]:
    """
    Knuth's error-free transformation: ``s = fl(a + b)`` and the exact rounding error ``e`` such
    that ``a + b = s + e`` in exact arithmetic. Works elementwise on numpy arrays.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def quick_two_sum(a: float, b: float) -> Tuple[float, float]:
    """Dekker's fast two-sum, valid when ``|a| >= |b|``. Works elementwise on numpy arrays."""
    s = a + b
    e = b - (s - a)
    return s, e


class CumulativeTable:
    """
    ``T(z) = int_z^{z_top} f(z') dz'`` accumulated on a fixed grid of nodes
    ``z_0 > z_1 > ... > z_{n-1}`` by per-interval Gauss-Legendre in ``u = log(1+z)`` (integrand
    ``f(z) (1+z)``), with the prefix sums held as double-double ``(hi, lo)`` pairs. ``T`` is
    non-decreasing as ``z`` falls when ``f >= 0``; ``T(z_0) = 0`` unless the table has been
    ``shifted``.

    Indices follow the node order given: index ``0`` is the *top* (largest ``z``) node.

    :param z_nodes: the grid, strictly descending in ``z``
    :param f: the integrand in ``z`` (called as ``f(z) -> float``); required to build the table,
        and again to evaluate off-grid partials on a reconstructed one. May be ``None`` for a
        reconstructed table that will only ever be evaluated on-grid.
    :param order: Gauss-Legendre order per panel (build *and* partials)
    :param hi: (keyword) persisted high limbs -- supply together with ``lo`` to reconstruct the
        table with no quadrature
    :param lo: (keyword) persisted low limbs
    :param break_points: (keyword) ascending ``u = log(1+z)`` values at which the integrand loses
        smoothness; every panel -- build or partial -- is split at the break points strictly
        inside it and each piece gets its own Gauss rule. Empty for a smooth integrand.
    :param label: a name for error messages
    """

    def __init__(
        self,
        z_nodes: Sequence[float],
        f: Optional[Callable[[float], float]],
        order: int,
        *,
        hi: Optional[Sequence[float]] = None,
        lo: Optional[Sequence[float]] = None,
        break_points: Optional[Sequence[float]] = None,
        label: str = "",
    ):
        self._label = label

        z = np.array([float(v) for v in z_nodes], dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError(
                f"CumulativeTable[{label}]: at least two nodes are required (got {z.size})"
            )
        if not np.all(np.isfinite(z)):
            raise ValueError(f"CumulativeTable[{label}]: nodes must be finite")
        if not np.all(np.diff(z) < 0.0):
            raise ValueError(
                f"CumulativeTable[{label}]: nodes must be strictly descending in z"
            )

        self._z = z
        self._z_list = z.tolist()
        self._n = int(z.size)
        # u = log(1+z) is the integration variable; z -> u is safe to ~1 ulp (CLAUDE.md)
        self._u = np.log1p(z)
        # ascending copy for searchsorted; ascending index k <-> table index n-1-k
        self._u_asc = self._u[::-1].copy()

        # exact-node lookup, keyed on the float itself (README §5 rule 9)
        self._index = {float(v): j for j, v in enumerate(z)}

        order = int(order)
        if order < 1:
            raise ValueError(
                f"CumulativeTable[{label}]: order must be >= 1 (got {order})"
            )
        self._order = order
        gx, gw = np.polynomial.legendre.leggauss(order)
        self._gauss = list(zip(gx.tolist(), gw.tolist()))

        if break_points is None or len(break_points) == 0:
            self._breaks = np.empty(0, dtype=float)
        else:
            self._breaks = np.unique(np.asarray(break_points, dtype=float))
            if not np.all(np.isfinite(self._breaks)):
                raise ValueError(
                    f"CumulativeTable[{label}]: break points must be finite"
                )
        self._breaks_list = self._breaks.tolist()

        self._f = f
        self._evaluations = 0
        self._build_evaluations = 0

        if (hi is None) != (lo is None):
            raise ValueError(
                f"CumulativeTable[{label}]: hi and lo must be supplied together"
            )

        if hi is not None:
            hi_arr = np.array([float(v) for v in hi], dtype=float)
            lo_arr = np.array([float(v) for v in lo], dtype=float)
            if hi_arr.shape != z.shape or lo_arr.shape != z.shape:
                raise ValueError(
                    f"CumulativeTable[{label}]: hi/lo must have one entry per node "
                    f"(nodes={z.size}, hi={hi_arr.size}, lo={lo_arr.size})"
                )
            self._hi = hi_arr
            self._lo = lo_arr
        else:
            if f is None:
                raise ValueError(
                    f"CumulativeTable[{label}]: an integrand is required to build the table"
                )
            self._hi, self._lo = self._accumulate()
            self._build_evaluations = self._evaluations

        self._hi_list = self._hi.tolist()
        self._lo_list = self._lo.tolist()

    # -----------------------------------------------------------------------------------------
    # construction
    # -----------------------------------------------------------------------------------------

    def _accumulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-interval increments from the Gauss rule, then prefix sums as (hi, lo) pairs:
        ``hi[j] = fsum(inc[:j])``, ``lo[j] = fsum([*inc[:j], -hi[j]])`` (README §2 (c)). The
        O(n^2) ``fsum`` work is ~0.05 s at n = 1732 and buys a correctly rounded pair at every
        node; a compensated running sum would be O(n) but is not bit-reproducible against this.
        """
        z = self._z.tolist()
        # interval i lies between nodes i+1 (lower z) and i (higher z)
        increments = [self._increment(z[i + 1], z[i]) for i in range(self._n - 1)]

        hi = np.empty(self._n, dtype=float)
        lo = np.empty(self._n, dtype=float)
        hi[0] = 0.0
        lo[0] = 0.0
        for j in range(1, self._n):
            terms = increments[:j]
            h = fsum(terms)
            hi[j] = h
            lo[j] = fsum([*terms, -h])
        return hi, lo

    def shifted(self, offset: float) -> "CumulativeTable":
        """
        A new table whose values are ``T(z) + offset``, the offset added in double-double:
        ``hi', e = two_sum(hi, offset)``; ``lo' = lo + e``; then renormalised so that ``|lo'|``
        is below an ulp of ``hi'``. Used to attach an absolute constant (``compute_background``'s
        ``tau_init``) to a table anchored at zero without discarding the low limb.
        """
        offset = float(offset)
        hi, e = two_sum(self._hi, offset)
        lo = self._lo + e
        hi, lo = quick_two_sum(hi, lo)
        table = CumulativeTable(
            self._z,
            self._f,
            self._order,
            hi=hi,
            lo=lo,
            break_points=self._breaks,
            label=self._label,
        )
        table._evaluations = self._evaluations
        table._build_evaluations = self._build_evaluations
        return table

    # -----------------------------------------------------------------------------------------
    # quadrature
    # -----------------------------------------------------------------------------------------
    #
    # Every panel is parametrised by its *lower* redshift endpoint and its width in u, and the
    # width is formed from the redshifts, W = log1p((z_hi - z_lo)/(1 + z_lo)), not as a
    # difference of two rounded u values. A node's u = log1p(z) carries half an ulp of u --
    # ~2e-15 absolute at u ~ 30 -- and a panel width formed as u_hi - u_lo would inherit that
    # as a *relative* error of ~1e-13 on a production interval (du ~ 0.023); the width from the
    # redshifts is accurate to a few ulps of itself. The Gauss abscissae are likewise placed as
    # 1 + z_x = (1 + z_lo) e^t with t in [0, W], so no abscissa passes through a rounded u.
    # Interior prefix sums do not care (a shared edge cancels between neighbouring panels), but
    # the first increment below the top node, and every off-grid partial, do.

    def _g(self, z: float) -> float:
        """The integrand in ``u``: ``f(z) (1+z)``, at the redshift ``z``."""
        self._evaluations += 1
        return self._f(z) * (1.0 + z)

    def _panel(self, z_lo: float, width: float) -> float:
        """One Gauss-Legendre panel of the build order over ``[u(z_lo), u(z_lo) + width]``."""
        half = 0.5 * width
        opz = 1.0 + z_lo
        g = self._g
        return half * fsum(
            w * g(z_lo + opz * expm1(half * (1.0 + x))) for x, w in self._gauss
        )

    def _increment(self, z_lo: float, z_hi: float) -> float:
        """
        ``int_{z_lo}^{z_hi} f dz = int_{u_lo}^{u_hi} f(z)(1+z) du`` for ``z_lo < z_hi``, split at
        every declared break point strictly inside the range. With breaks the sub-panel widths
        are taken from the break positions in ``u`` except the last, which is what remains of the
        exact total width, so a break's own rounding moves only the boundary between two
        sub-panels (which costs ``delta_u`` times the jump of the integrand there, not
        ``delta_u`` times the integrand).
        """
        if self._f is None:
            raise RuntimeError(
                f"CumulativeTable[{self._label}]: an off-grid evaluation needs the integrand, "
                "but this table was reconstructed without one"
            )
        width = log1p((z_hi - z_lo) / (1.0 + z_lo))

        if len(self._breaks_list) == 0:
            return self._panel(z_lo, width)

        u_lo = log1p(z_lo)
        u_hi = log1p(z_hi)
        i0 = int(np.searchsorted(self._breaks, u_lo, side="right"))
        i1 = int(np.searchsorted(self._breaks, u_hi, side="left"))
        if i1 <= i0:
            return self._panel(z_lo, width)

        interior = self._breaks_list[i0:i1]
        edges = [u_lo, *interior]
        parts = []
        used = 0.0
        for i, start_u in enumerate(edges):
            start_z = z_lo if i == 0 else expm1(start_u)
            if i + 1 < len(edges):
                w = edges[i + 1] - start_u
                used += w
            else:
                w = width - used
            parts.append(self._panel(start_z, w))
        return fsum(parts)

    def _integral(self, z_from: float, z_to: float) -> float:
        """Signed ``int_{z_from}^{z_to} f dz``; exactly ``0.0`` with no integrand call when equal."""
        if z_from == z_to:
            return 0.0
        if z_from < z_to:
            return self._increment(z_from, z_to)
        return -self._increment(z_to, z_from)

    # -----------------------------------------------------------------------------------------
    # lookup
    # -----------------------------------------------------------------------------------------

    def node_index(self, z: float) -> Optional[int]:
        """The table index of an exact node, or ``None`` for any other ``z``."""
        return self._index.get(float(z))

    def _nearest_node(self, u: float) -> int:
        """
        Table index of the node nearest to ``u`` (off-grid ``u`` only). Beyond either end of
        the grid by more than one grid interval the table refuses: a single Gauss panel is
        accurate over about one interval, and nothing in production evaluates there.
        """
        asc = self._u_asc
        n = self._n
        i = int(np.searchsorted(asc, u))
        if i <= 0:
            if u < asc[0] - (asc[1] - asc[0]):
                raise RuntimeError(
                    f"CumulativeTable[{self._label}]: z={expm1(u):.6g} lies more than one grid "
                    f"interval below the table (min node z={self._z[-1]:.6g})"
                )
            k = 0
        elif i >= n:
            if u > asc[-1] + (asc[-1] - asc[-2]):
                raise RuntimeError(
                    f"CumulativeTable[{self._label}]: z={expm1(u):.6g} lies more than one grid "
                    f"interval above the table (max node z={self._z[0]:.6g})"
                )
            k = n - 1
        else:
            k = i if (asc[i] - u) < (u - asc[i - 1]) else i - 1
        return n - 1 - k

    def _locate(self, z: float) -> Tuple[int, bool]:
        """``(table index, on_grid)`` for ``z``: exact node lookup first, nearest node otherwise."""
        j = self._index.get(z)
        if j is not None:
            return j, True
        return self._nearest_node(log1p(z)), False

    # -----------------------------------------------------------------------------------------
    # accessors
    # -----------------------------------------------------------------------------------------

    def value(self, z: float) -> float:
        """
        Pointwise ``T(z)``: the node's ``hi + lo`` plus, off-grid, one local Gauss partial from
        the nearest node (at most half a grid interval, ``order`` integrand calls; zero calls
        on-grid). Carries half an ulp of ``T`` -- use ``delta`` for differences.
        """
        z = float(z)
        j, on_grid = self._locate(z)
        base = self._hi_list[j] + self._lo_list[j]
        if on_grid:
            return base
        return base + self._integral(z, self._z_list[j])

    def delta(self, z_a: float, z_b: float) -> float:
        """
        ``T(z_b) - T(z_a) = int_{z_b}^{z_a} f dz``, positive when ``z_b < z_a`` and ``f >= 0``
        (README §2 (c)).

        Formed as

            partial(a -> n(a)) + [(hi_b - hi_a) + (lo_b - lo_a)] + partial(n(b) -> b)

        and **never** as ``value(z_b) - value(z_a)``. A pointwise value is correctly rounded to
        half an ulp of ``T`` -- ~1.5e-12 Mpc for the conformal time at low redshift -- and the
        difference of two of them keeps that absolute error however short the baseline: 9e-4 rad
        at k = 3e8/Mpc (review §13.3). Differencing the two limbs separately makes the table
        term exact whenever the two nodes are within a factor of two of each other (Sterbenz) and
        otherwise rounds relative to the difference itself; the partials are local panels whose
        error is relative to their own (small) span. So the result is accurate relative to
        ``delta``, not to ``T``.

        On-grid endpoints cost no integrand call; an off-grid endpoint costs ``order`` calls per
        break-free panel. ``delta(z, z) == 0.0`` exactly and ``delta(a, b) == -delta(b, a)`` to
        the bit: the two terms are formed so that reversing the arguments negates each exactly.
        """
        z_a = float(z_a)
        z_b = float(z_b)
        j_a, on_a = self._locate(z_a)
        j_b, on_b = self._locate(z_b)

        table = (self._hi_list[j_b] - self._hi_list[j_a]) + (
            self._lo_list[j_b] - self._lo_list[j_a]
        )

        # partial(n(b) -> b) = int_{z_b}^{z_{n(b)}}; partial(a -> n(a)) = -int_{z_a}^{z_{n(a)}}
        p_b = 0.0 if on_b else self._integral(z_b, self._z_list[j_b])
        p_a = 0.0 if on_a else self._integral(z_a, self._z_list[j_a])
        ends = p_b - p_a

        return table + ends

    # -----------------------------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------------------------

    @property
    def z_nodes(self) -> np.ndarray:
        """The nodes, descending in ``z`` (index 0 is the top node)."""
        return self._z

    @property
    def u_nodes(self) -> np.ndarray:
        """``log(1+z)`` of the nodes, in node order."""
        return self._u

    @property
    def hi(self) -> np.ndarray:
        return self._hi

    @property
    def lo(self) -> np.ndarray:
        return self._lo

    @property
    def order(self) -> int:
        return self._order

    @property
    def break_points(self) -> np.ndarray:
        """Ascending ``u`` values the panels are split at."""
        return self._breaks

    @property
    def evaluations(self) -> int:
        """Integrand evaluations spent building the table (zero for a reconstructed one)."""
        return self._build_evaluations

    @property
    def total_evaluations(self) -> int:
        """Integrand evaluations spent so far: the build plus every off-grid partial since."""
        return self._evaluations

    @property
    def label(self) -> str:
        return self._label

    def __len__(self) -> int:
        return self._n
