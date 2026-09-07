import os
"""QI_04: recompute the F3 grid arithmetic from main.py:2669-2685 and main.py:2488-2492."""
import itertools
import numpy as np

# main.py:2671 and main.py:2682 -- identical 50-point log-spaced grids
source_k = np.logspace(np.log10(1e5), np.log10(3e8), 50)
response_k = np.logspace(np.log10(1e5), np.log10(3e8), 50)

pairs = list(itertools.combinations_with_replacement(range(len(source_k)), 2))
print("n source k =", len(source_k), " n response k =", len(response_k))
print("combinations_with_replacement pairs =", len(pairs))
triples = len(pairs) * len(response_k)
print("(k,q,r) triples =", triples)

closing = 0
for (i, j) in pairs:
    q, r = source_k[i], source_k[j]
    for k in response_k:
        if abs(q - r) <= k <= q + r:
            closing += 1
print("triples satisfying |q-r| <= k <= q+r :", closing,
      f"({100.0*closing/triples:.2f}%)")

# per-k breakdown for the mid-grid k quoted in the reconciliation doc
kmid = response_k[len(response_k) // 2]
surv = [(source_k[i], source_k[j]) for (i, j) in pairs
        if abs(source_k[i] - source_k[j]) <= kmid <= source_k[i] + source_k[j]]
s_vals = sorted((q + r) / kmid for q, r in surv)
print(f"\nmid-grid k = {kmid:.3g}/Mpc: {len(surv)} surviving pairs, "
      f"s=(q+r)/k in [{s_vals[0]:.4g}, {s_vals[-1]:.4g}]")
near = [s for s in s_vals if abs(s - 3 ** 0.5) < 0.05]
print(f"  nodes within +-0.05 of s=sqrt(3): {len(near)} -> {[round(x,4) for x in near]}")
