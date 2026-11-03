# The slow leaves behind V8's lost calls (descriptive; the census brief's addendum)

Computed by `scripts/v8_slow_leaves.py` from `out/v8_calls.json` (read only, sha256 `1e9a62b0…`) and the census's region pickles (`checked_leaves_06g`), every record SHA-checked. Nothing was rerun. The scope, the items and the clustering rule are the script's docstring, fixed before its result was looked at.

- Lost calls outside the 720: 1631 (946 units).
- 1e-7: scoped leaves (returned, untrusted, 1/4 <= r < 1, delta1 > phi_P) 4342; returned (b)-class leaves exempted floor-limited after their check (outside the scope) 20848; (b)-class returned leaves flagged trusted 0 (expected 0).
- 1e-8: scoped leaves (returned, untrusted, 1/4 <= r < 1, delta1 > phi_P) 4478; returned (b)-class leaves exempted floor-limited after their check (outside the scope) 22349; (b)-class returned leaves flagged trusted 0 (expected 0).

## rtol 1e-7

### all groups (4342 leaves, 578 calls)

**1. r.** [0.25, 0.5): 1980; [0.5, 0.75): 1534; [0.75, 1): 828. Quantiles: p0 0.25, p10 0.4208, p25 0.4944, p50 0.501, p75 0.6741, p90 0.8521, p99 0.9817, p100 0.9991.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 77, 0.2812: 67, 0.3125: 79, 0.3438: 68, 0.375: 106, 0.4062: 91, 0.4375: 112, 0.4688: 1380, 0.5: 715, 0.5312: 135, 0.5625: 120, 0.5938: 112, 0.625: 129, 0.6562: 111, 0.6875: 106, 0.7188: 106, 0.75: 120, 0.7812: 127, 0.8125: 124, 0.8438: 94, 0.875: 99, 0.9062: 93, 0.9375: 97, 0.9688: 74
  Within 0.01 of 1/4, 1/2, 3/4: 30, 1763, 70.

**2. delta1/phi_P.** p50 2.269, p75 6.305, p90 10.44, p99 23.01, p100 44.58; min 1.004.
  - r in [0.25, 0.5): 1980 leaves; delta1/phi_P p0 1.01, p50 1.497, p90 8.315, p100 35.51; at most 2: 1373, at most 10: 1835
  - r in [0.5, 0.75): 1534 leaves; delta1/phi_P p0 1.023, p50 2.936, p90 10.71, p100 44.58; at most 2: 686, at most 10: 1342
  - r in [0.75, 1): 828 leaves; delta1/phi_P p0 1.004, p50 6.038, p90 12.16, p100 41.15; at most 2: 66, at most 10: 675

**3. r against depth.** Depth of the scoped leaves: {20: 4342}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 4342; max |log2(r_ancestor / r_leaf)| over them: p0 0.8277, p25 3.221, p50 4.466, p75 5.882, p90 7.322, p100 11.64; within a factor 2 throughout (<= 1): 7.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 4342; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.256, p50 1.675, p75 2.927, p90 4.107, p100 10.76; within a factor 2 throughout: 651; within 10 %: 77.
  r by depth over every checked leaf of these calls (returned or bisected), median [p10, p90] (n):
    - depth 0: 1.99 [0.435, 22.5] (362; not finite 0)
    - depth 1: 1.69 [0.183, 82.5] (1475; not finite 1)
    - depth 2: 1.18 [0.13, 7.51] (2605; not finite 25)
    - depth 3: 1.04 [0.123, 5.66] (4180; not finite 39)
    - depth 4: 0.832 [0.0971, 4.3] (6410; not finite 39)
    - depth 5: 0.488 [0.0267, 2.79] (9316; not finite 14)
    - depth 6: 0.327 [0.0177, 2.15] (11921; not finite 2)
    - depth 7: 0.24 [0.0134, 1.79] (13489; not finite 0)
    - depth 8: 0.263 [0.0164, 2.18] (12880; not finite 1)
    - depth 9: 0.467 [0.0292, 5.45] (12510; not finite 5)
    - depth 10: 1.14 [0.134, 19.5] (13338; not finite 10)
    - depth 11: 1.48 [0.268, 23.6] (16138; not finite 7)
    - depth 12: 1.18 [0.223, 6.68] (18830; not finite 0)
    - depth 13: 1.12 [0.227, 4.49] (19514; not finite 0)
    - depth 14: 0.807 [0.241, 3.82] (18496; not finite 0)
    - depth 15: 0.846 [0.23, 3.77] (16224; not finite 0)
    - depth 16: 0.745 [0.218, 2.74] (14794; not finite 0)
    - depth 17: 0.657 [0.163, 1.8] (13510; not finite 0)
    - depth 18: 0.794 [0.167, 1.88] (12548; not finite 0)
    - depth 19: 0.503 [0.164, 1.68] (13198; not finite 6)
    - depth 20: 0.503 [0.163, 1.56] (12900; not finite 7)
  Sample chains (the leftmost scoped leaf of each of the first eight lost calls in (row, index) order): depth: r of each recorded ancestor, then the leaf.
    - 0-12576 call 1 G+q+r, leaf depth 20 r 0.4911 d1/phi 1.48: 0: 23.7, 1: 1.26, 2: 3.81, 3: 2.74, 4: 2.07, 5: 0.909, 6: 0.323, 7: 2.44, 8: 1.54, 9: 2.53, 10: 9.19, 11: 1.12, 12: 1.02, 13: 0.668, 14: 103, 15: 1.21, 16: 0.414, 17: 0.453, 18: 0.476, 19: 0.488
    - 0-12576 call 2 G+q-r, leaf depth 20 r 0.4951 d1/phi 1.49: 1: 0.365, 2: 4.02, 3: 6.21, 4: 0.871, 5: 1.75, 6: 0.48, 7: 0.391, 8: 0.798, 9: 0.916, 10: 3.16, 11: 1.23, 12: 13.5, 13: 1.64, 14: 11.2, 15: 1.22, 16: 0.486, 17: 0.489, 18: 0.502, 19: 0.501
    - 0-12576 call 3 G-q+r, leaf depth 20 r 0.4951 d1/phi 1.49: 1: 0.365, 2: 4.02, 3: 6.21, 4: 0.871, 5: 1.75, 6: 0.48, 7: 0.391, 8: 0.798, 9: 0.916, 10: 3.16, 11: 1.23, 12: 13.5, 13: 1.64, 14: 11.2, 15: 1.22, 16: 0.486, 17: 0.489, 18: 0.502, 19: 0.501
    - 0-12578 call 1 G+q+r, leaf depth 20 r 0.4941 d1/phi 1.48: 0: 0.63, 1: 1.79, 2: 2.26, 3: 0.267, 4: 7.97, 5: 0.86, 6: 0.293, 7: 0.464, 8: 0.442, 9: 1.03, 10: 49.6, 11: 2.26, 12: 1.34, 13: 3.44, 14: 3.92, 15: 1.25, 16: 0.423, 17: 0.447, 18: 0.712, 19: 1.43
    - 0-12596 call 2 G+q-r, leaf depth 20 r 0.485 d1/phi 1.46: 1: 4.93, 2: 2.15, 3: 1.61, 4: 1.19, 5: 0.612, 6: 0.276, 7: 1.45, 8: 1.1, 9: 0.522, 10: 1.75, 11: 6.67, 12: 10, 13: 1.75, 14: 2.9, 15: 1.22, 16: 0.319, 17: 0.282, 18: 0.737, 19: 1.25
    - 0-12596 call 3 G-q+r, leaf depth 20 r 0.485 d1/phi 1.46: 1: 4.93, 2: 2.15, 3: 1.61, 4: 1.19, 5: 0.612, 6: 0.276, 7: 1.45, 8: 1.1, 9: 0.522, 10: 1.75, 11: 6.67, 12: 10, 13: 1.75, 14: 2.9, 15: 1.22, 16: 0.319, 17: 0.282, 18: 0.737, 19: 1.25
    - 0-12610 call 3 G+q+r, leaf depth 20 r 0.4976 d1/phi 1.5: 0: 7.2, 1: 0.518, 2: 1.45, 3: 0.876, 4: 0.802, 5: 1.01, 6: 0.669, 7: 2.69, 8: 1.1, 9: 8.47, 10: 17.5, 11: 1.06, 12: 62.8, 13: 1.61, 14: 1.4, 15: 1.27, 16: 0.483, 17: 0.887, 18: 1.48, 19: 0.499
    - 0-12610 call 4 G+q-r, leaf depth 20 r 0.5026 d1/phi 1.49: 0: 3.44, 1: 0.315, 2: 0.323, 3: 2.21, 4: 109, 5: 1.31, 6: 0.301, 7: 0.294, 8: 0.454, 9: 0.395, 10: 2.25, 11: 40.1, 12: 0.633, 13: 1.04, 14: 6.28, 15: 1.26, 16: 0.466, 17: 0.639, 18: 1.42, 19: 0.499

**4. Position (depth-20 scoped leaves, 4342 in 578 calls).** Clusters per call: median 2, max 29, distribution {1: 264, 2: 119, 3: 65, 4: 36, 5: 26, 6: 22, 7: 12, 8: 10, 9: 4, 10: 9, 11: 1, 12: 3, 13: 4, 14: 1, 21: 1, 29: 1}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 1556.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 44, [0.1, 0.2): 55, [0.2, 0.3): 62, [0.3, 0.4): 74, [0.4, 0.5): 129, [0.5, 0.6): 203, [0.6, 0.7): 254, [0.7, 0.8): 345, [0.8, 0.9): 314, [0.9, 1.0): 76.
  Example calls (the first eight in (row, index) order): x_span (a, b) and clusters:
    - 0-12576 call 1 G+q+r, x_span (17.3206916, 29.2743073): u=0.611312 (1 leaves, distance to an end 3.89e-01); u=0.611557 (1 leaves, distance to an end 3.88e-01); u=0.612810 (3 leaves, distance to an end 3.87e-01); u=0.613724 (1 leaves, distance to an end 3.86e-01); u=0.614823 (1 leaves, distance to an end 3.85e-01); u=0.615217 (3 leaves, distance to an end 3.85e-01); u=0.615643 (2 leaves, distance to an end 3.84e-01); u=0.616256 (1 leaves, distance to an end 3.84e-01) ...
    - 0-12576 call 2 G+q-r, x_span (17.3206916, 29.2743073): u=0.009544 (5 leaves, distance to an end 9.54e-03); u=0.009720 (1 leaves, distance to an end 9.72e-03); u=0.011347 (26 leaves, distance to an end 1.13e-02); u=0.012912 (9 leaves, distance to an end 1.29e-02); u=0.013288 (3 leaves, distance to an end 1.33e-02); u=0.015410 (9 leaves, distance to an end 1.54e-02); u=0.033615 (1 leaves, distance to an end 3.36e-02); u=0.088326 (12 leaves, distance to an end 8.83e-02)
    - 0-12576 call 3 G-q+r, x_span (17.3206916, 29.2743073): u=0.009544 (5 leaves, distance to an end 9.54e-03); u=0.009720 (1 leaves, distance to an end 9.72e-03); u=0.011347 (26 leaves, distance to an end 1.13e-02); u=0.012912 (9 leaves, distance to an end 1.29e-02); u=0.013288 (3 leaves, distance to an end 1.33e-02); u=0.015410 (9 leaves, distance to an end 1.54e-02); u=0.033615 (1 leaves, distance to an end 3.36e-02); u=0.088326 (12 leaves, distance to an end 8.83e-02)
    - 0-12578 call 1 G+q+r, x_span (15.3860026, 28.1227066): u=0.562103 (2 leaves, distance to an end 4.38e-01)
    - 0-12596 call 2 G+q-r, x_span (15.3860026, 28.1227066): u=0.160263 (2 leaves, distance to an end 1.60e-01); u=0.160440 (1 leaves, distance to an end 1.60e-01); u=0.160775 (15 leaves, distance to an end 1.61e-01); u=0.161093 (1 leaves, distance to an end 1.61e-01); u=0.164915 (8 leaves, distance to an end 1.65e-01); u=0.168417 (8 leaves, distance to an end 1.68e-01); u=0.168540 (2 leaves, distance to an end 1.69e-01)
    - 0-12596 call 3 G-q+r, x_span (15.3860026, 28.1227066): u=0.160263 (2 leaves, distance to an end 1.60e-01); u=0.160440 (1 leaves, distance to an end 1.60e-01); u=0.160775 (15 leaves, distance to an end 1.61e-01); u=0.161093 (1 leaves, distance to an end 1.61e-01); u=0.164915 (8 leaves, distance to an end 1.65e-01); u=0.168417 (8 leaves, distance to an end 1.68e-01); u=0.168540 (2 leaves, distance to an end 1.69e-01)
    - 0-12610 call 3 G+q+r, x_span (13.1749308, 24.6909364): u=0.646245 (1 leaves, distance to an end 3.54e-01); u=0.653566 (2 leaves, distance to an end 3.46e-01); u=0.705807 (1 leaves, distance to an end 2.94e-01); u=0.707278 (2 leaves, distance to an end 2.93e-01); u=0.708743 (2 leaves, distance to an end 2.91e-01); u=0.709180 (2 leaves, distance to an end 2.91e-01)
    - 0-12610 call 4 G+q-r, x_span (13.1749308, 24.6909364): u=0.851468 (2 leaves, distance to an end 1.49e-01); u=0.855884 (2 leaves, distance to an end 1.44e-01); u=0.857958 (3 leaves, distance to an end 1.42e-01); u=0.858960 (2 leaves, distance to an end 1.41e-01); u=0.859082 (1 leaves, distance to an end 1.41e-01); u=0.872418 (3 leaves, distance to an end 1.28e-01)

### group G+q+r (1041 leaves, 148 calls)

**1. r.** [0.25, 0.5): 457; [0.5, 0.75): 370; [0.75, 1): 214. Quantiles: p0 0.25, p10 0.4233, p25 0.4951, p50 0.5014, p75 0.6941, p90 0.8493, p99 0.9734, p100 0.9984.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 22, 0.2812: 14, 0.3125: 21, 0.3438: 12, 0.375: 22, 0.4062: 26, 0.4375: 23, 0.4688: 317, 0.5: 157, 0.5312: 33, 0.5625: 29, 0.5938: 23, 0.625: 45, 0.6562: 29, 0.6875: 27, 0.7188: 27, 0.75: 41, 0.7812: 27, 0.8125: 37, 0.8438: 27, 0.875: 35, 0.9062: 18, 0.9375: 16, 0.9688: 13
  Within 0.01 of 1/4, 1/2, 3/4: 9, 406, 28.

**2. delta1/phi_P.** p50 2.674, p75 6.845, p90 10.35, p99 19.14, p100 44.58; min 1.018.
  - r in [0.25, 0.5): 457 leaves; delta1/phi_P p0 1.018, p50 1.499, p90 8.114, p100 17.55; at most 2: 314, at most 10: 430
  - r in [0.5, 0.75): 370 leaves; delta1/phi_P p0 1.271, p50 3.335, p90 10.81, p100 44.58; at most 2: 150, at most 10: 325
  - r in [0.75, 1): 214 leaves; delta1/phi_P p0 1.12, p50 6.061, p90 12.26, p100 41.15; at most 2: 14, at most 10: 172

**3. r against depth.** Depth of the scoped leaves: {20: 1041}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 18, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1041; max |log2(r_ancestor / r_leaf)| over them: p0 0.9248, p25 3.414, p50 4.627, p75 6.215, p90 7.637, p100 11.64; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1041; max |log2(r_ancestor / r_leaf)|: p0 0.05027, p25 1.171, p50 1.58, p75 2.697, p90 4.355, p100 9.663; within a factor 2 throughout: 183; within 10 %: 16.

**4. Position (depth-20 scoped leaves, 1041 in 148 calls).** Clusters per call: median 2, max 14, distribution {1: 72, 2: 27, 3: 14, 4: 8, 5: 8, 6: 7, 8: 4, 9: 3, 10: 2, 12: 1, 13: 1, 14: 1}; leaves per cluster p0 1, p50 2, p90 5, p100 17; clusters within 2^-10 of an end of the call's x_span: 0 of 400.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 2, [0.1, 0.2): 9, [0.2, 0.3): 15, [0.3, 0.4): 23, [0.4, 0.5): 23, [0.5, 0.6): 28, [0.6, 0.7): 46, [0.7, 0.8): 136, [0.8, 0.9): 92, [0.9, 1.0): 26.

### group G+q-r (1094 leaves, 142 calls)

**1. r.** [0.25, 0.5): 506; [0.5, 0.75): 375; [0.75, 1): 213. Quantiles: p0 0.2516, p10 0.4174, p25 0.4943, p50 0.5007, p75 0.6713, p90 0.8686, p99 0.9832, p100 0.9957.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 19, 0.2812: 19, 0.3125: 15, 0.3438: 22, 0.375: 26, 0.4062: 25, 0.4375: 33, 0.4688: 347, 0.5: 187, 0.5312: 35, 0.5625: 28, 0.5938: 28, 0.625: 24, 0.6562: 23, 0.6875: 24, 0.7188: 26, 0.75: 24, 0.7812: 35, 0.8125: 26, 0.8438: 25, 0.875: 17, 0.9062: 32, 0.9375: 31, 0.9688: 23
  Within 0.01 of 1/4, 1/2, 3/4: 8, 449, 14.

**2. delta1/phi_P.** p50 1.926, p75 6.239, p90 10.77, p99 24.39, p100 38.16; min 1.01.
  - r in [0.25, 0.5): 506 leaves; delta1/phi_P p0 1.01, p50 1.498, p90 8.923, p100 35.51; at most 2: 350, at most 10: 459
  - r in [0.5, 0.75): 375 leaves; delta1/phi_P p0 1.023, p50 2.26, p90 10.82, p100 38.16; at most 2: 184, at most 10: 323
  - r in [0.75, 1): 213 leaves; delta1/phi_P p0 1.031, p50 5.834, p90 12.25, p100 23.01; at most 2: 19, at most 10: 173

**3. r against depth.** Depth of the scoped leaves: {20: 1094}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 18, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1094; max |log2(r_ancestor / r_leaf)| over them: p0 0.85, p25 3.233, p50 4.426, p75 5.512, p90 6.68, p100 10.93; within a factor 2 throughout (<= 1): 1.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1094; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.286, p50 1.646, p75 2.741, p90 4.057, p100 10.76; within a factor 2 throughout: 151; within 10 %: 19.

**4. Position (depth-20 scoped leaves, 1094 in 142 calls).** Clusters per call: median 2, max 13, distribution {1: 58, 2: 32, 3: 20, 4: 7, 5: 4, 6: 7, 7: 7, 8: 2, 10: 2, 12: 1, 13: 2}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 395.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 19, [0.1, 0.2): 19, [0.2, 0.3): 25, [0.3, 0.4): 23, [0.4, 0.5): 43, [0.5, 0.6): 66, [0.6, 0.7): 77, [0.7, 0.8): 47, [0.8, 0.9): 67, [0.9, 1.0): 9.

### group G-q+r (1317 leaves, 169 calls)

**1. r.** [0.25, 0.5): 591; [0.5, 0.75): 484; [0.75, 1): 242. Quantiles: p0 0.2504, p10 0.3996, p25 0.4943, p50 0.5012, p75 0.6748, p90 0.8417, p99 0.975, p100 0.9981.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 30, 0.2812: 24, 0.3125: 21, 0.3438: 26, 0.375: 42, 0.4062: 28, 0.4375: 35, 0.4688: 385, 0.5: 223, 0.5312: 43, 0.5625: 39, 0.5938: 37, 0.625: 39, 0.6562: 31, 0.6875: 31, 0.7188: 41, 0.75: 34, 0.7812: 45, 0.8125: 36, 0.8438: 24, 0.875: 24, 0.9062: 31, 0.9375: 29, 0.9688: 19
  Within 0.01 of 1/4, 1/2, 3/4: 9, 515, 15.

**2. delta1/phi_P.** p50 2.684, p75 6.239, p90 10.37, p99 24.62, p100 38.16; min 1.004.
  - r in [0.25, 0.5): 591 leaves; delta1/phi_P p0 1.01, p50 1.499, p90 8.531, p100 35.51; at most 2: 380, at most 10: 542
  - r in [0.5, 0.75): 484 leaves; delta1/phi_P p0 1.023, p50 2.914, p90 10.6, p100 38.16; at most 2: 213, at most 10: 426
  - r in [0.75, 1): 242 leaves; delta1/phi_P p0 1.004, p50 5.653, p90 11.5, p100 29.35; at most 2: 21, at most 10: 202

**3. r against depth.** Depth of the scoped leaves: {20: 1317}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1317; max |log2(r_ancestor / r_leaf)| over them: p0 0.8277, p25 3.148, p50 4.514, p75 5.709, p90 6.693, p100 10.93; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1317; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.265, p50 1.603, p75 2.595, p90 3.924, p100 10.11; within a factor 2 throughout: 198; within 10 %: 22.

**4. Position (depth-20 scoped leaves, 1317 in 169 calls).** Clusters per call: median 2, max 29, distribution {1: 73, 2: 33, 3: 22, 4: 10, 5: 7, 6: 6, 7: 5, 8: 4, 9: 1, 10: 5, 12: 1, 13: 1, 29: 1}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 496.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 21, [0.1, 0.2): 17, [0.2, 0.3): 20, [0.3, 0.4): 19, [0.4, 0.5): 50, [0.5, 0.6): 87, [0.6, 0.7): 101, [0.7, 0.8): 78, [0.8, 0.9): 86, [0.9, 1.0): 17.

### group G-q-r (889 leaves, 118 calls)

**1. r.** [0.25, 0.5): 425; [0.5, 0.75): 305; [0.75, 1): 159. Quantiles: p0 0.2535, p10 0.4601, p25 0.4942, p50 0.5005, p75 0.6569, p90 0.8567, p99 0.9878, p100 0.9991.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 6, 0.2812: 10, 0.3125: 22, 0.3438: 8, 0.375: 16, 0.4062: 12, 0.4375: 21, 0.4688: 330, 0.5: 148, 0.5312: 24, 0.5625: 24, 0.5938: 24, 0.625: 21, 0.6562: 28, 0.6875: 24, 0.7188: 12, 0.75: 21, 0.7812: 20, 0.8125: 25, 0.8438: 18, 0.875: 23, 0.9062: 12, 0.9375: 21, 0.9688: 19
  Within 0.01 of 1/4, 1/2, 3/4: 4, 393, 13.

**2. delta1/phi_P.** p50 1.507, p75 6.227, p90 10.04, p99 17, p100 25.48; min 1.042.
  - r in [0.25, 0.5): 425 leaves; delta1/phi_P p0 1.042, p50 1.493, p90 7.532, p100 25.48; at most 2: 329, at most 10: 403
  - r in [0.5, 0.75): 305 leaves; delta1/phi_P p0 1.125, p50 2.919, p90 11.01, p100 24.83; at most 2: 139, at most 10: 268
  - r in [0.75, 1): 159 leaves; delta1/phi_P p0 1.173, p50 6.604, p90 11.68, p100 17.92; at most 2: 12, at most 10: 128

**3. r against depth.** Depth of the scoped leaves: {20: 889}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 19, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 889; max |log2(r_ancestor / r_leaf)| over them: p0 0.9036, p25 3.071, p50 4.394, p75 6.307, p90 8.208, p100 11.54; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 889; max |log2(r_ancestor / r_leaf)|: p0 0.02336, p25 1.342, p50 2.174, p75 3.443, p90 4.252, p100 10.39; within a factor 2 throughout: 119; within 10 %: 20.

**4. Position (depth-20 scoped leaves, 889 in 118 calls).** Clusters per call: median 1, max 21, distribution {1: 60, 2: 27, 3: 9, 4: 11, 5: 7, 6: 2, 11: 1, 21: 1}; leaves per cluster p0 1, p50 2, p90 8, p100 24; clusters within 2^-10 of an end of the call's x_span: 0 of 264.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 2, [0.1, 0.2): 9, [0.2, 0.3): 2, [0.3, 0.4): 9, [0.4, 0.5): 13, [0.5, 0.6): 22, [0.6, 0.7): 30, [0.7, 0.8): 84, [0.8, 0.9): 69, [0.9, 1.0): 24.

### group q+r (1 leaves, 1 calls)

**1. r.** [0.25, 0.5): 1; [0.5, 0.75): 0; [0.75, 1): 0. Quantiles: p0 0.476, p10 0.476, p25 0.476, p50 0.476, p75 0.476, p90 0.476, p99 0.476, p100 0.476.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.4688: 1
  Within 0.01 of 1/4, 1/2, 3/4: 0, 0, 0.

**2. delta1/phi_P.** p50 6.699, p75 6.699, p90 6.699, p99 6.699, p100 6.699; min 6.699.
  - r in [0.25, 0.5): 1 leaves; delta1/phi_P p0 6.699, p50 6.699, p90 6.699, p100 6.699; at most 2: 0, at most 10: 1
  - r in [0.5, 0.75): 0 leaves; delta1/phi_P none; at most 2: 0, at most 10: 0
  - r in [0.75, 1): 0 leaves; delta1/phi_P none; at most 2: 0, at most 10: 0

**3. r against depth.** Depth of the scoped leaves: {20: 1}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 20, p50 20, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1; max |log2(r_ancestor / r_leaf)| over them: p0 5.508, p25 5.508, p50 5.508, p75 5.508, p90 5.508, p100 5.508; within a factor 2 throughout (<= 1): 0.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1; max |log2(r_ancestor / r_leaf)|: p0 5.508, p25 5.508, p50 5.508, p75 5.508, p90 5.508, p100 5.508; within a factor 2 throughout: 0; within 10 %: 0.

**4. Position (depth-20 scoped leaves, 1 in 1 calls).** Clusters per call: median 1, max 1, distribution {1: 1}; leaves per cluster p0 1, p50 1, p90 1, p100 1; clusters within 2^-10 of an end of the call's x_span: 0 of 1.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 0, [0.1, 0.2): 1, [0.2, 0.3): 0, [0.3, 0.4): 0, [0.4, 0.5): 0, [0.5, 0.6): 0, [0.6, 0.7): 0, [0.7, 0.8): 0, [0.8, 0.9): 0, [0.9, 1.0): 0.

## rtol 1e-8

### all groups (4478 leaves, 600 calls)

**1. r.** [0.25, 0.5): 2047; [0.5, 0.75): 1583; [0.75, 1): 848. Quantiles: p0 0.25, p10 0.4224, p25 0.4944, p50 0.501, p75 0.6721, p90 0.8521, p99 0.9819, p100 0.9991.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 77, 0.2812: 68, 0.3125: 79, 0.3438: 70, 0.375: 110, 0.4062: 94, 0.4375: 122, 0.4688: 1427, 0.5: 741, 0.5312: 138, 0.5625: 122, 0.5938: 112, 0.625: 135, 0.6562: 117, 0.6875: 109, 0.7188: 109, 0.75: 125, 0.7812: 128, 0.8125: 125, 0.8438: 96, 0.875: 101, 0.9062: 96, 0.9375: 99, 0.9688: 78
  Within 0.01 of 1/4, 1/2, 3/4: 30, 1820, 74.

**2. delta1/phi_P.** p50 2.203, p75 6.227, p90 10.34, p99 21.63, p100 44.58; min 1.004.
  - r in [0.25, 0.5): 2047 leaves; delta1/phi_P p0 1.01, p50 1.497, p90 8.164, p100 35.51; at most 2: 1423, at most 10: 1902
  - r in [0.5, 0.75): 1583 leaves; delta1/phi_P p0 1.023, p50 2.919, p90 10.69, p100 44.58; at most 2: 709, at most 10: 1390
  - r in [0.75, 1): 848 leaves; delta1/phi_P p0 1.004, p50 5.951, p90 12.12, p100 41.15; at most 2: 67, at most 10: 695

**3. r against depth.** Depth of the scoped leaves: {20: 4478}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 4478; max |log2(r_ancestor / r_leaf)| over them: p0 0.8277, p25 3.147, p50 4.426, p75 5.831, p90 7.285, p100 11.64; within a factor 2 throughout (<= 1): 7.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 4478; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.249, p50 1.654, p75 2.885, p90 4.087, p100 10.76; within a factor 2 throughout: 678; within 10 %: 87.
  r by depth over every checked leaf of these calls (returned or bisected), median [p10, p90] (n):
    - depth 0: 2.34 [0.441, 23.7] (319; not finite 0)
    - depth 1: 1.93 [0.254, 30.2] (1442; not finite 1)
    - depth 2: 1.21 [0.202, 6.17] (2622; not finite 10)
    - depth 3: 1.09 [0.169, 5.66] (4224; not finite 14)
    - depth 4: 0.838 [0.113, 4.45] (6498; not finite 21)
    - depth 5: 0.476 [0.0259, 2.78] (9706; not finite 1)
    - depth 6: 0.297 [0.0119, 2.05] (13349; not finite 0)
    - depth 7: 0.223 [0.0105, 1.73] (15261; not finite 0)
    - depth 8: 0.273 [0.016, 2.23] (14392; not finite 1)
    - depth 9: 0.489 [0.0313, 5.07] (13940; not finite 5)
    - depth 10: 1.12 [0.139, 16.9] (14700; not finite 11)
    - depth 11: 1.43 [0.255, 21.1] (17414; not finite 7)
    - depth 12: 1.15 [0.217, 6.47] (19848; not finite 0)
    - depth 13: 1.13 [0.226, 4.49] (20328; not finite 0)
    - depth 14: 0.789 [0.243, 3.76] (19440; not finite 1)
    - depth 15: 0.851 [0.226, 3.74] (16926; not finite 0)
    - depth 16: 0.754 [0.223, 2.74] (15278; not finite 0)
    - depth 17: 0.657 [0.163, 1.8] (13996; not finite 0)
    - depth 18: 0.784 [0.167, 1.87] (12960; not finite 0)
    - depth 19: 0.504 [0.164, 1.66] (13576; not finite 6)
    - depth 20: 0.502 [0.163, 1.56] (13306; not finite 7)
  Sample chains (the leftmost scoped leaf of each of the first eight lost calls in (row, index) order): depth: r of each recorded ancestor, then the leaf.
    - 0-12576 call 1 G+q+r, leaf depth 20 r 0.4911 d1/phi 1.48: 0: 23.7, 1: 1.26, 2: 3.81, 3: 2.74, 4: 2.07, 5: 0.909, 6: 0.323, 7: 2.44, 8: 1.54, 9: 2.53, 10: 9.19, 11: 1.12, 12: 1.02, 13: 0.668, 14: 103, 15: 1.21, 16: 0.414, 17: 0.453, 18: 0.476, 19: 0.488
    - 0-12576 call 2 G+q-r, leaf depth 20 r 0.4951 d1/phi 1.49: 1: 0.365, 2: 4.02, 3: 6.21, 4: 0.871, 5: 1.75, 6: 0.48, 7: 0.391, 8: 0.798, 9: 0.916, 10: 3.16, 11: 1.23, 12: 13.5, 13: 1.64, 14: 11.2, 15: 1.22, 16: 0.486, 17: 0.489, 18: 0.502, 19: 0.501
    - 0-12576 call 3 G-q+r, leaf depth 20 r 0.4951 d1/phi 1.49: 1: 0.365, 2: 4.02, 3: 6.21, 4: 0.871, 5: 1.75, 6: 0.48, 7: 0.391, 8: 0.798, 9: 0.916, 10: 3.16, 11: 1.23, 12: 13.5, 13: 1.64, 14: 11.2, 15: 1.22, 16: 0.486, 17: 0.489, 18: 0.502, 19: 0.501
    - 0-12578 call 1 G+q+r, leaf depth 20 r 0.4941 d1/phi 1.48: 0: 0.63, 1: 1.79, 2: 2.26, 3: 0.267, 4: 7.97, 5: 0.86, 6: 0.293, 7: 0.464, 8: 0.442, 9: 1.03, 10: 49.6, 11: 2.26, 12: 1.34, 13: 3.44, 14: 3.92, 15: 1.25, 16: 0.423, 17: 0.447, 18: 0.712, 19: 1.43
    - 0-12596 call 2 G+q-r, leaf depth 20 r 0.485 d1/phi 1.46: 1: 4.93, 2: 2.15, 3: 1.61, 4: 1.19, 5: 0.612, 6: 0.276, 7: 1.45, 8: 1.1, 9: 0.522, 10: 1.75, 11: 6.67, 12: 10, 13: 1.75, 14: 2.9, 15: 1.22, 16: 0.319, 17: 0.282, 18: 0.737, 19: 1.25
    - 0-12596 call 3 G-q+r, leaf depth 20 r 0.485 d1/phi 1.46: 1: 4.93, 2: 2.15, 3: 1.61, 4: 1.19, 5: 0.612, 6: 0.276, 7: 1.45, 8: 1.1, 9: 0.522, 10: 1.75, 11: 6.67, 12: 10, 13: 1.75, 14: 2.9, 15: 1.22, 16: 0.319, 17: 0.282, 18: 0.737, 19: 1.25
    - 0-12610 call 3 G+q+r, leaf depth 20 r 0.4976 d1/phi 1.5: 0: 7.2, 1: 0.518, 2: 1.45, 3: 0.876, 4: 0.802, 5: 1.01, 6: 0.669, 7: 2.69, 8: 1.1, 9: 8.47, 10: 17.5, 11: 1.06, 12: 62.8, 13: 1.61, 14: 1.4, 15: 1.27, 16: 0.483, 17: 0.887, 18: 1.48, 19: 0.499
    - 0-12610 call 4 G+q-r, leaf depth 20 r 0.5026 d1/phi 1.49: 0: 3.44, 1: 0.315, 2: 0.323, 3: 2.21, 4: 109, 5: 1.31, 6: 0.301, 7: 0.294, 8: 0.454, 9: 0.395, 10: 2.25, 11: 40.1, 12: 0.633, 13: 1.04, 14: 6.28, 15: 1.26, 16: 0.466, 17: 0.639, 18: 1.42, 19: 0.499

**4. Position (depth-20 scoped leaves, 4478 in 600 calls).** Clusters per call: median 2, max 29, distribution {1: 274, 2: 122, 3: 68, 4: 36, 5: 28, 6: 23, 7: 13, 8: 11, 9: 4, 10: 9, 11: 1, 12: 3, 13: 5, 14: 1, 21: 1, 29: 1}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 1625.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 44, [0.1, 0.2): 58, [0.2, 0.3): 62, [0.3, 0.4): 80, [0.4, 0.5): 134, [0.5, 0.6): 228, [0.6, 0.7): 256, [0.7, 0.8): 356, [0.8, 0.9): 331, [0.9, 1.0): 76.
  Example calls (the first eight in (row, index) order): x_span (a, b) and clusters:
    - 0-12576 call 1 G+q+r, x_span (17.3206916, 29.2743073): u=0.611312 (1 leaves, distance to an end 3.89e-01); u=0.611557 (1 leaves, distance to an end 3.88e-01); u=0.612810 (3 leaves, distance to an end 3.87e-01); u=0.613724 (1 leaves, distance to an end 3.86e-01); u=0.614823 (1 leaves, distance to an end 3.85e-01); u=0.615217 (3 leaves, distance to an end 3.85e-01); u=0.615643 (2 leaves, distance to an end 3.84e-01); u=0.616256 (1 leaves, distance to an end 3.84e-01) ...
    - 0-12576 call 2 G+q-r, x_span (17.3206916, 29.2743073): u=0.009544 (5 leaves, distance to an end 9.54e-03); u=0.009720 (1 leaves, distance to an end 9.72e-03); u=0.011347 (26 leaves, distance to an end 1.13e-02); u=0.012912 (9 leaves, distance to an end 1.29e-02); u=0.013288 (3 leaves, distance to an end 1.33e-02); u=0.015410 (9 leaves, distance to an end 1.54e-02); u=0.033615 (1 leaves, distance to an end 3.36e-02); u=0.088326 (12 leaves, distance to an end 8.83e-02)
    - 0-12576 call 3 G-q+r, x_span (17.3206916, 29.2743073): u=0.009544 (5 leaves, distance to an end 9.54e-03); u=0.009720 (1 leaves, distance to an end 9.72e-03); u=0.011347 (26 leaves, distance to an end 1.13e-02); u=0.012912 (9 leaves, distance to an end 1.29e-02); u=0.013288 (3 leaves, distance to an end 1.33e-02); u=0.015410 (9 leaves, distance to an end 1.54e-02); u=0.033615 (1 leaves, distance to an end 3.36e-02); u=0.088326 (12 leaves, distance to an end 8.83e-02)
    - 0-12578 call 1 G+q+r, x_span (15.3860026, 28.1227066): u=0.562103 (2 leaves, distance to an end 4.38e-01)
    - 0-12596 call 2 G+q-r, x_span (15.3860026, 28.1227066): u=0.160263 (2 leaves, distance to an end 1.60e-01); u=0.160440 (1 leaves, distance to an end 1.60e-01); u=0.160775 (15 leaves, distance to an end 1.61e-01); u=0.161093 (1 leaves, distance to an end 1.61e-01); u=0.164915 (8 leaves, distance to an end 1.65e-01); u=0.168417 (8 leaves, distance to an end 1.68e-01); u=0.168540 (2 leaves, distance to an end 1.69e-01)
    - 0-12596 call 3 G-q+r, x_span (15.3860026, 28.1227066): u=0.160263 (2 leaves, distance to an end 1.60e-01); u=0.160440 (1 leaves, distance to an end 1.60e-01); u=0.160775 (15 leaves, distance to an end 1.61e-01); u=0.161093 (1 leaves, distance to an end 1.61e-01); u=0.164915 (8 leaves, distance to an end 1.65e-01); u=0.168417 (8 leaves, distance to an end 1.68e-01); u=0.168540 (2 leaves, distance to an end 1.69e-01)
    - 0-12610 call 3 G+q+r, x_span (13.1749308, 24.6909364): u=0.646245 (1 leaves, distance to an end 3.54e-01); u=0.653566 (2 leaves, distance to an end 3.46e-01); u=0.705807 (1 leaves, distance to an end 2.94e-01); u=0.707278 (2 leaves, distance to an end 2.93e-01); u=0.708743 (2 leaves, distance to an end 2.91e-01); u=0.709180 (2 leaves, distance to an end 2.91e-01)
    - 0-12610 call 4 G+q-r, x_span (13.1749308, 24.6909364): u=0.851468 (2 leaves, distance to an end 1.49e-01); u=0.855884 (2 leaves, distance to an end 1.44e-01); u=0.857958 (3 leaves, distance to an end 1.42e-01); u=0.858960 (2 leaves, distance to an end 1.41e-01); u=0.859082 (1 leaves, distance to an end 1.41e-01); u=0.872418 (3 leaves, distance to an end 1.28e-01)

### group G+q+r (1060 leaves, 151 calls)

**1. r.** [0.25, 0.5): 461; [0.5, 0.75): 381; [0.75, 1): 218. Quantiles: p0 0.25, p10 0.4268, p25 0.4952, p50 0.5014, p75 0.6964, p90 0.8493, p99 0.9799, p100 0.9984.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 22, 0.2812: 14, 0.3125: 21, 0.3438: 12, 0.375: 22, 0.4062: 26, 0.4375: 23, 0.4688: 321, 0.5: 160, 0.5312: 34, 0.5625: 30, 0.5938: 23, 0.625: 47, 0.6562: 30, 0.6875: 28, 0.7188: 29, 0.75: 42, 0.7812: 27, 0.8125: 38, 0.8438: 27, 0.875: 35, 0.9062: 18, 0.9375: 17, 0.9688: 14
  Within 0.01 of 1/4, 1/2, 3/4: 9, 411, 29.

**2. delta1/phi_P.** p50 2.683, p75 6.742, p90 10.34, p99 19.14, p100 44.58; min 1.018.
  - r in [0.25, 0.5): 461 leaves; delta1/phi_P p0 1.018, p50 1.499, p90 8.114, p100 17.55; at most 2: 317, at most 10: 434
  - r in [0.5, 0.75): 381 leaves; delta1/phi_P p0 1.271, p50 3.349, p90 10.68, p100 44.58; at most 2: 152, at most 10: 335
  - r in [0.75, 1): 218 leaves; delta1/phi_P p0 1.12, p50 6.033, p90 12.26, p100 41.15; at most 2: 14, at most 10: 176

**3. r against depth.** Depth of the scoped leaves: {20: 1060}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 17, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1060; max |log2(r_ancestor / r_leaf)| over them: p0 0.9248, p25 3.382, p50 4.586, p75 6.176, p90 7.63, p100 11.64; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1060; max |log2(r_ancestor / r_leaf)|: p0 0.05027, p25 1.171, p50 1.58, p75 2.685, p90 4.343, p100 9.663; within a factor 2 throughout: 184; within 10 %: 16.

**4. Position (depth-20 scoped leaves, 1060 in 151 calls).** Clusters per call: median 2, max 14, distribution {1: 73, 2: 27, 3: 15, 4: 8, 5: 8, 6: 7, 7: 1, 8: 4, 9: 3, 10: 2, 12: 1, 13: 1, 14: 1}; leaves per cluster p0 1, p50 2, p90 5, p100 17; clusters within 2^-10 of an end of the call's x_span: 0 of 411.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 2, [0.1, 0.2): 9, [0.2, 0.3): 15, [0.3, 0.4): 23, [0.4, 0.5): 23, [0.5, 0.6): 36, [0.6, 0.7): 48, [0.7, 0.8): 137, [0.8, 0.9): 92, [0.9, 1.0): 26.

### group G+q-r (1123 leaves, 147 calls)

**1. r.** [0.25, 0.5): 525; [0.5, 0.75): 382; [0.75, 1): 216. Quantiles: p0 0.2516, p10 0.4192, p25 0.4942, p50 0.5006, p75 0.6667, p90 0.8686, p99 0.9832, p100 0.9957.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 19, 0.2812: 19, 0.3125: 15, 0.3438: 22, 0.375: 27, 0.4062: 26, 0.4375: 35, 0.4688: 362, 0.5: 194, 0.5312: 35, 0.5625: 28, 0.5938: 28, 0.625: 24, 0.6562: 23, 0.6875: 24, 0.7188: 26, 0.75: 24, 0.7812: 35, 0.8125: 26, 0.8438: 25, 0.875: 18, 0.9062: 33, 0.9375: 31, 0.9688: 24
  Within 0.01 of 1/4, 1/2, 3/4: 8, 468, 14.

**2. delta1/phi_P.** p50 1.817, p75 6.186, p90 10.73, p99 24.07, p100 38.16; min 1.01.
  - r in [0.25, 0.5): 525 leaves; delta1/phi_P p0 1.01, p50 1.498, p90 8.857, p100 35.51; at most 2: 367, at most 10: 478
  - r in [0.5, 0.75): 382 leaves; delta1/phi_P p0 1.023, p50 2.064, p90 10.73, p100 38.16; at most 2: 190, at most 10: 330
  - r in [0.75, 1): 216 leaves; delta1/phi_P p0 1.031, p50 5.834, p90 12.25, p100 23.01; at most 2: 19, at most 10: 176

**3. r against depth.** Depth of the scoped leaves: {20: 1123}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1123; max |log2(r_ancestor / r_leaf)| over them: p0 0.85, p25 3.174, p50 4.419, p75 5.52, p90 6.678, p100 10.93; within a factor 2 throughout (<= 1): 1.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1123; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.285, p50 1.614, p75 2.699, p90 4.034, p100 10.76; within a factor 2 throughout: 158; within 10 %: 22.

**4. Position (depth-20 scoped leaves, 1123 in 147 calls).** Clusters per call: median 2, max 13, distribution {1: 62, 2: 32, 3: 20, 4: 7, 5: 5, 6: 7, 7: 7, 8: 2, 10: 2, 12: 1, 13: 2}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 404.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 19, [0.1, 0.2): 19, [0.2, 0.3): 25, [0.3, 0.4): 23, [0.4, 0.5): 44, [0.5, 0.6): 73, [0.6, 0.7): 77, [0.7, 0.8): 48, [0.8, 0.9): 67, [0.9, 1.0): 9.

### group G-q+r (1369 leaves, 175 calls)

**1. r.** [0.25, 0.5): 622; [0.5, 0.75): 499; [0.75, 1): 248. Quantiles: p0 0.2504, p10 0.4011, p25 0.4941, p50 0.5011, p75 0.6719, p90 0.8414, p99 0.9773, p100 0.9981.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 30, 0.2812: 24, 0.3125: 21, 0.3438: 26, 0.375: 43, 0.4062: 29, 0.4375: 40, 0.4688: 409, 0.5: 235, 0.5312: 44, 0.5625: 38, 0.5938: 37, 0.625: 40, 0.6562: 32, 0.6875: 32, 0.7188: 41, 0.75: 36, 0.7812: 45, 0.8125: 36, 0.8438: 25, 0.875: 24, 0.9062: 32, 0.9375: 30, 0.9688: 20
  Within 0.01 of 1/4, 1/2, 3/4: 9, 542, 17.

**2. delta1/phi_P.** p50 2.575, p75 6.086, p90 10.3, p99 24.62, p100 38.16; min 1.004.
  - r in [0.25, 0.5): 622 leaves; delta1/phi_P p0 1.01, p50 1.499, p90 8.38, p100 35.51; at most 2: 406, at most 10: 573
  - r in [0.5, 0.75): 499 leaves; delta1/phi_P p0 1.023, p50 2.897, p90 10.56, p100 38.16; at most 2: 224, at most 10: 441
  - r in [0.75, 1): 248 leaves; delta1/phi_P p0 1.004, p50 5.345, p90 11.5, p100 29.35; at most 2: 21, at most 10: 208

**3. r against depth.** Depth of the scoped leaves: {20: 1369}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1369; max |log2(r_ancestor / r_leaf)| over them: p0 0.8277, p25 3.044, p50 4.447, p75 5.644, p90 6.685, p100 10.93; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1369; max |log2(r_ancestor / r_leaf)|: p0 0.01665, p25 1.254, p50 1.595, p75 2.572, p90 3.905, p100 10.11; within a factor 2 throughout: 210; within 10 %: 27.

**4. Position (depth-20 scoped leaves, 1369 in 175 calls).** Clusters per call: median 2, max 29, distribution {1: 76, 2: 34, 3: 22, 4: 10, 5: 8, 6: 6, 7: 5, 8: 5, 9: 1, 10: 5, 12: 1, 13: 1, 29: 1}; leaves per cluster p0 1, p50 2, p90 6, p100 26; clusters within 2^-10 of an end of the call's x_span: 0 of 514.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 21, [0.1, 0.2): 17, [0.2, 0.3): 20, [0.3, 0.4): 16, [0.4, 0.5): 50, [0.5, 0.6): 95, [0.6, 0.7): 101, [0.7, 0.8): 84, [0.8, 0.9): 93, [0.9, 1.0): 17.

### group G-q-r (925 leaves, 126 calls)

**1. r.** [0.25, 0.5): 438; [0.5, 0.75): 321; [0.75, 1): 166. Quantiles: p0 0.2535, p10 0.4572, p25 0.4942, p50 0.5007, p75 0.6605, p90 0.8567, p99 0.9876, p100 0.9991.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.25: 6, 0.2812: 11, 0.3125: 22, 0.3438: 10, 0.375: 18, 0.4062: 13, 0.4375: 24, 0.4688: 334, 0.5: 152, 0.5312: 25, 0.5625: 26, 0.5938: 24, 0.625: 24, 0.6562: 32, 0.6875: 25, 0.7188: 13, 0.75: 23, 0.7812: 21, 0.8125: 25, 0.8438: 19, 0.875: 24, 0.9062: 13, 0.9375: 21, 0.9688: 20
  Within 0.01 of 1/4, 1/2, 3/4: 4, 399, 14.

**2. delta1/phi_P.** p50 1.514, p75 6.075, p90 9.928, p99 16.84, p100 25.48; min 1.042.
  - r in [0.25, 0.5): 438 leaves; delta1/phi_P p0 1.042, p50 1.493, p90 7.525, p100 25.48; at most 2: 333, at most 10: 416
  - r in [0.5, 0.75): 321 leaves; delta1/phi_P p0 1.125, p50 2.919, p90 10.56, p100 24.83; at most 2: 143, at most 10: 284
  - r in [0.75, 1): 166 leaves; delta1/phi_P p0 1.173, p50 6.485, p90 11.67, p100 17.92; at most 2: 13, at most 10: 135

**3. r against depth.** Depth of the scoped leaves: {20: 925}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 16, p50 19, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 925; max |log2(r_ancestor / r_leaf)| over them: p0 0.9036, p25 3.048, p50 4.351, p75 6.175, p90 8.165, p100 11.54; within a factor 2 throughout (<= 1): 2.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 925; max |log2(r_ancestor / r_leaf)|: p0 0.02336, p25 1.331, p50 2.104, p75 3.424, p90 4.207, p100 10.39; within a factor 2 throughout: 126; within 10 %: 22.

**4. Position (depth-20 scoped leaves, 925 in 126 calls).** Clusters per call: median 2, max 21, distribution {1: 62, 2: 29, 3: 11, 4: 11, 5: 7, 6: 3, 11: 1, 13: 1, 21: 1}; leaves per cluster p0 1, p50 2, p90 8, p100 24; clusters within 2^-10 of an end of the call's x_span: 0 of 295.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 2, [0.1, 0.2): 12, [0.2, 0.3): 2, [0.3, 0.4): 18, [0.4, 0.5): 17, [0.5, 0.6): 24, [0.6, 0.7): 30, [0.7, 0.8): 87, [0.8, 0.9): 79, [0.9, 1.0): 24.

### group q+r (1 leaves, 1 calls)

**1. r.** [0.25, 0.5): 1; [0.5, 0.75): 0; [0.75, 1): 0. Quantiles: p0 0.476, p10 0.476, p25 0.476, p50 0.476, p75 0.476, p90 0.476, p99 0.476, p100 0.476.
  Histogram of r in bins of 1/32 (bin lower edge: count): 0.4688: 1
  Within 0.01 of 1/4, 1/2, 3/4: 0, 0, 0.

**2. delta1/phi_P.** p50 6.699, p75 6.699, p90 6.699, p99 6.699, p100 6.699; min 6.699.
  - r in [0.25, 0.5): 1 leaves; delta1/phi_P p0 6.699, p50 6.699, p90 6.699, p100 6.699; at most 2: 0, at most 10: 1
  - r in [0.5, 0.75): 0 leaves; delta1/phi_P none; at most 2: 0, at most 10: 0
  - r in [0.75, 1): 0 leaves; delta1/phi_P none; at most 2: 0, at most 10: 0

**3. r against depth.** Depth of the scoped leaves: {20: 1}.
  Chains are partly recorded (a bisected leaf is recorded only if it was checked before its split): recorded ancestors per leaf p0 20, p50 20, p90 20, p100 20 of 20 levels for a depth-20 leaf; recorded at depth >= 10: p0 10, p50 10, p90 10, p100 10.
  Chains with >= 3 recorded ancestors at depth >= 10: 1; max |log2(r_ancestor / r_leaf)| over them: p0 5.508, p25 5.508, p50 5.508, p75 5.508, p90 5.508, p100 5.508; within a factor 2 throughout (<= 1): 0.
  The same over the last levels only (recorded ancestors at depth >= 16, chains with >= 3): 1; max |log2(r_ancestor / r_leaf)|: p0 5.508, p25 5.508, p50 5.508, p75 5.508, p90 5.508, p100 5.508; within a factor 2 throughout: 0; within 10 %: 0.

**4. Position (depth-20 scoped leaves, 1 in 1 calls).** Clusters per call: median 1, max 1, distribution {1: 1}; leaves per cluster p0 1, p50 1, p90 1, p100 1; clusters within 2^-10 of an end of the call's x_span: 0 of 1.
  Cluster centres' position u in tenths of the interval: [0.0, 0.1): 0, [0.1, 0.2): 1, [0.2, 0.3): 0, [0.3, 0.4): 0, [0.4, 0.5): 0, [0.5, 0.6): 0, [0.6, 0.7): 0, [0.7, 0.8): 0, [0.8, 0.9): 0, [0.9, 1.0): 0.


## Coverage (out/cmd/57, `scripts/v8_slow_leaves_coverage.py`; appended after the run of 56b)

Not every lost call returns a leaf in this scope. 1e-7: 578 of 801 lost calls do; the other 223 each return
untrusted leaves (all at depth 20) none of which is in the (b) class, so with r ≥ 1 or not finite; 221 of the 223
return (b)-class leaves that were exempted floor-limited after their check. 1e-8: 600 of 830 do; of the other 230,
3 are the node-limited calls and 227 return untrusted leaves outside the (b) class (2 with no returned (b)-class
leaf at all). V8's classification (state/predictions.md Part B rule 3) put all 1,628 non-node-limited calls in
"(b), refined to the depth cap", because its (b) test counts (b)-class leaves among every checked leaf, returned or
bisected; this note does not change it.
