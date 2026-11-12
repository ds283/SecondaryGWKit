# levin-phase-floor 07a Part A: the O9 check on the H3 leaves of the five failures

script o9_table_07a.py sha256 caffd1f29a1827a95d23e24e7ff64926f58c960d1f4b007166c3adf6b14b4aee. A measurement at 07's code (no variant), added at the user's word on 2026-11-11 (log deviations 21-22). 'route' is the gate's: Clenshaw-Curtis below SIX_PI; Levin, LU tried first, above _LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20 pi; Levin, least squares, between.

| call | N | H3 leaves / carrying excess (share of H3 true error) | regions checked | route changes | theta' sign changes | true / estimate (min-max) | main leaf: depth, span, true/declared, share | its phase_span (07) / true variation | route (07 = true) | parent phase_span / route | sibling phase_span / route | input sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v08 | 24 | 16 / 4 (1.001) | 16 | 0 | 0 | 0.730-0.999 | d0, [9.23151, 28.1227], 3.63, 1.001 | 4.061e+08 / 2.965e+08 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | be977557779b |
| v09 | 24 | 20 / 8 (1.000) | 30 | 0 | 0 | 0.466-1.389 | d1, [12.8372, 24.6909], 2.66, 1.000 | 9.146e+06 / 4.258e+06 | Levin (LU tried first) | 2.227e+09 / Levin (LU tried first) | 1.818e+09 / Levin (LU tried first) | 7365c0d45f57 |
| v12 | 24 | 14 / 3 (1.008) | 13 | 0 | 0 | 0.752-0.998 | d0, [9.23164, 26.9711], 7.44, 1.008 | 1.257e+08 / 9.448e+07 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | d651bfa7dcd3 |
| v17 | 24 | 8 / 4 (1.000) | 16 | 0 | 0 | 0.752-0.999 | d0, [9.23164, 26.9711], 2.94, 1.000 | 1.257e+08 / 9.448e+07 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | 15aea2d1afa3 |
| v22 | 24 | 109 / 14 (1.000) | 45 | 0 | 0 | 0.915-1.121 | d5, [9.23186, 9.75095], 4.67, 0.999 | 4.033e+06 / 4.275e+06 | Levin (LU tried first) | 1.026e+07 / Levin (LU tried first) | 7.139e+06 / Levin (LU tried first) | b79bff109063 |

## v08: O9 check on its H3 leaves

N = 24 nodes; H3 true error +6.131e-36 over 16 leaves; 4 leaves carry excess error, together 1.001 of it. Script sha256 f97b2de60c1f, module sha256 4e5da371f51e.

| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 (d7, true/decl 0.839) | top-3 |true error| in H3 | leaf | [8.99063, 9.05077] | 7.634e+06 | 7.626e+06 | no | 7.626e+06 | 0.999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 9 (d7, true/decl 0.839) | top-3 |true error| in H3 | parent | [8.99063, 9.11091] | 1.277e+07 | 1.271e+07 | no | 1.271e+07 | 0.9953 | Levin (LU tried first) | Levin (LU tried first) | no |
| 9 (d7, true/decl 0.839) | top-3 |true error| in H3 | sibling | [9.05077, 9.11091] | 5.093e+06 | 5.086e+06 | no | 5.086e+06 | 0.9986 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 1.93) | carries excess error | leaf | [9.11091, 9.17105] | 2.858e+06 | 2.851e+06 | no | 2.851e+06 | 0.9977 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 1.93) | carries excess error | parent | [9.11091, 9.23119] | 3.797e+06 | 3.747e+06 | no | 3.747e+06 | 0.9868 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 1.93) | carries excess error | sibling | [9.17105, 9.23119] | 9.015e+05 | 8.956e+05 | no | 8.956e+05 | 0.9934 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d8, true/decl 1.82) | carries excess error | leaf | [9.17105, 9.20112] | 6.765e+05 | 6.758e+05 | no | 6.758e+05 | 0.9989 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d8, true/decl 1.82) | carries excess error | parent | [9.17105, 9.23119] | 9.015e+05 | 8.956e+05 | no | 8.956e+05 | 0.9934 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d8, true/decl 1.82) | carries excess error | sibling | [9.20112, 9.23119] | 2.205e+05 | 2.198e+05 | no | 2.198e+05 | 0.9967 | Levin (LU tried first) | Levin (LU tried first) | no |
| 13 (d9, true/decl 7.45) | carries excess error | leaf | [9.20112, 9.21615] | 1.65e+05 | 1.649e+05 | no | 1.649e+05 | 0.9994 | Levin (LU tried first) | Levin (LU tried first) | no |
| 13 (d9, true/decl 7.45) | carries excess error | parent | [9.20112, 9.23119] | 2.205e+05 | 2.198e+05 | no | 2.198e+05 | 0.9967 | Levin (LU tried first) | Levin (LU tried first) | no |
| 13 (d9, true/decl 7.45) | carries excess error | sibling | [9.21615, 9.23119] | 5.498e+04 | 5.489e+04 | no | 5.489e+04 | 0.9984 | Levin (LU tried first) | Levin (LU tried first) | no |
| 14 (d9, true/decl 0.0264) | top-3 |true error| in H3 | leaf | [9.21615, 9.23119] | 5.498e+04 | 5.489e+04 | no | 5.489e+04 | 0.9984 | Levin (LU tried first) | Levin (LU tried first) | no |
| 14 (d9, true/decl 0.0264) | top-3 |true error| in H3 | parent | [9.20112, 9.23119] | 2.205e+05 | 2.198e+05 | no | 2.198e+05 | 0.9967 | Levin (LU tried first) | Levin (LU tried first) | no |
| 14 (d9, true/decl 0.0264) | top-3 |true error| in H3 | sibling | [9.20112, 9.21615] | 1.65e+05 | 1.649e+05 | no | 1.649e+05 | 0.9994 | Levin (LU tried first) | Levin (LU tried first) | no |
| 30 (d0, true/decl 3.63) | carries excess error ; top-3 |true error| in H3 | leaf | [9.23151, 28.1227] | 4.061e+08 | 2.965e+08 | no | 2.965e+08 | 0.7301 | Levin (LU tried first) | Levin (LU tried first) | no |
| 30 | carries excess error ; top-3 |true error| in H3 | parent | - | | | | | | | | |
| 30 | carries excess error ; top-3 |true error| in H3 | sibling | - | | | | | | | | |


## v09: O9 check on its H3 leaves

N = 24 nodes; H3 true error +3.640e-33 over 20 leaves; 8 leaves carry excess error, together 1.000 of it. Script sha256 f97b2de60c1f, module sha256 4e5da371f51e.

| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 (d3, true/decl 1.4) | carries excess error | leaf | [0.821061, 0.902033] | 2.283e+06 | 2.283e+06 | no | 2.283e+06 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 2 (d3, true/decl 1.4) | carries excess error | parent | [0.821061, 0.983006] | 3.045e+06 | 3.043e+06 | no | 3.043e+06 | 0.9994 | Levin (LU tried first) | Levin (LU tried first) | no |
| 2 (d3, true/decl 1.4) | carries excess error | sibling | [0.902033, 0.983006] | 7.612e+05 | 7.608e+05 | no | 7.608e+05 | 0.9995 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 4.41) | carries excess error | leaf | [0.902033, 0.94252] | 5.699e+05 | 5.699e+05 | no | 5.699e+05 | 0.9999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 4.41) | carries excess error | parent | [0.902033, 0.983006] | 7.612e+05 | 7.608e+05 | no | 7.608e+05 | 0.9995 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 4.41) | carries excess error | sibling | [0.94252, 0.983006] | 1.91e+05 | 1.909e+05 | no | 1.909e+05 | 0.9997 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 13.1) | carries excess error | leaf | [0.94252, 0.962763] | 1.427e+05 | 1.427e+05 | no | 1.427e+05 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 13.1) | carries excess error | parent | [0.94252, 0.983006] | 1.91e+05 | 1.909e+05 | no | 1.909e+05 | 0.9997 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 13.1) | carries excess error | sibling | [0.962763, 0.983006] | 4.822e+04 | 4.822e+04 | no | 4.822e+04 | 0.9998 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 28.9) | carries excess error | leaf | [0.962763, 0.972884] | 3.59e+04 | 3.59e+04 | no | 3.59e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 28.9) | carries excess error | parent | [0.962763, 0.983006] | 4.822e+04 | 4.822e+04 | no | 4.822e+04 | 0.9998 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 28.9) | carries excess error | sibling | [0.972884, 0.983006] | 1.232e+04 | 1.232e+04 | no | 1.232e+04 | 0.9999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.000931) | top-3 |true error| in H3 | leaf | [0.972884, 0.983006] | 1.232e+04 | 1.232e+04 | no | 1.232e+04 | 0.9999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.000931) | top-3 |true error| in H3 | parent | [0.962763, 0.983006] | 4.822e+04 | 4.822e+04 | no | 4.822e+04 | 0.9998 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.000931) | top-3 |true error| in H3 | sibling | [0.962763, 0.972884] | 3.59e+04 | 3.59e+04 | no | 3.59e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d11, true/decl 0.00246) | top-3 |true error| in H3 | leaf | [0.983474, 0.99505] | 1.6e+04 | 1.6e+04 | no | 1.6e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d11, true/decl 0.00246) | top-3 |true error| in H3 | parent | [0.983474, 1.00663] | 6.27e+04 | 6.271e+04 | no | 6.271e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d11, true/decl 0.00246) | top-3 |true error| in H3 | sibling | [0.99505, 1.00663] | 4.671e+04 | 4.671e+04 | no | 4.671e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 27 (d11, true/decl 13.7) | carries excess error | leaf | [0.99505, 1.00663] | 4.671e+04 | 4.671e+04 | no | 4.671e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 27 (d11, true/decl 13.7) | carries excess error | parent | [0.983474, 1.00663] | 6.27e+04 | 6.271e+04 | no | 6.271e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 27 (d11, true/decl 13.7) | carries excess error | sibling | [0.983474, 0.99505] | 1.6e+04 | 1.6e+04 | no | 1.6e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 28 (d10, true/decl 8.62) | carries excess error | leaf | [1.00663, 1.02978] | 1.852e+05 | 1.852e+05 | no | 1.852e+05 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 28 (d10, true/decl 8.62) | carries excess error | parent | [0.983474, 1.02978] | 2.478e+05 | 2.48e+05 | no | 2.48e+05 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 28 (d10, true/decl 8.62) | carries excess error | sibling | [0.983474, 1.00663] | 6.27e+04 | 6.271e+04 | no | 6.271e+04 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 31 (d7, true/decl 1.08) | carries excess error | leaf | [1.16869, 1.3539] | 1.135e+07 | 1.136e+07 | no | 1.136e+07 | 1.001 | Levin (LU tried first) | Levin (LU tried first) | no |
| 31 (d7, true/decl 1.08) | carries excess error | parent | [0.983474, 1.3539] | 1.516e+07 | 1.525e+07 | no | 1.525e+07 | 1.006 | Levin (LU tried first) | Levin (LU tried first) | no |
| 31 (d7, true/decl 1.08) | carries excess error | sibling | [0.983474, 1.16869] | 3.883e+06 | 3.892e+06 | no | 3.892e+06 | 1.002 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d1, true/decl 2.66) | carries excess error ; top-3 |true error| in H3 | leaf | [12.8372, 24.6909] | 9.146e+06 | 4.258e+06 | no | 4.258e+06 | 0.4656 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d1, true/decl 2.66) | carries excess error ; top-3 |true error| in H3 | parent | [0.983474, 24.6909] | 2.227e+09 | 2.529e+09 | no | 2.529e+09 | 1.136 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d1, true/decl 2.66) | carries excess error ; top-3 |true error| in H3 | sibling | [0.983474, 12.8372] | 1.818e+09 | 2.525e+09 | no | 2.525e+09 | 1.389 | Levin (LU tried first) | Levin (LU tried first) | no |


## v12: O9 check on its H3 leaves

N = 24 nodes; H3 true error +3.029e-34 over 14 leaves; 3 leaves carry excess error, together 1.008 of it. Script sha256 f97b2de60c1f, module sha256 4e5da371f51e.

| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 (d4, true/decl 0.648) | top-3 |true error| in H3 | leaf | [7.87339, 8.32595] | 2.668e+08 | 2.639e+08 | no | 2.639e+08 | 0.9893 | Levin (LU tried first) | Levin (LU tried first) | no |
| 7 (d4, true/decl 0.648) | top-3 |true error| in H3 | parent | [7.42083, 8.32595] | 8.383e+08 | 8.115e+08 | no | 8.115e+08 | 0.9681 | Levin (LU tried first) | Levin (LU tried first) | no |
| 7 (d4, true/decl 0.648) | top-3 |true error| in H3 | sibling | [7.42083, 7.87339] | 5.515e+08 | 5.476e+08 | no | 5.476e+08 | 0.9931 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d6, true/decl 1.98) | carries excess error | leaf | [9.00478, 9.11792] | 3.557e+06 | 3.541e+06 | no | 3.541e+06 | 0.9956 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d6, true/decl 1.98) | carries excess error | parent | [9.00478, 9.23106] | 4.709e+06 | 4.594e+06 | no | 4.594e+06 | 0.9756 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d6, true/decl 1.98) | carries excess error | sibling | [9.11792, 9.23106] | 1.066e+06 | 1.053e+06 | no | 1.053e+06 | 0.9876 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 5.01) | carries excess error | leaf | [9.11792, 9.17449] | 8.017e+05 | 8e+05 | no | 8e+05 | 0.9979 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 5.01) | carries excess error | parent | [9.11792, 9.23106] | 1.066e+06 | 1.053e+06 | no | 1.053e+06 | 0.9876 | Levin (LU tried first) | Levin (LU tried first) | no |
| 11 (d7, true/decl 5.01) | carries excess error | sibling | [9.17449, 9.23106] | 2.546e+05 | 2.53e+05 | no | 2.53e+05 | 0.9938 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d7, true/decl 0.0421) | top-3 |true error| in H3 | leaf | [9.17449, 9.23106] | 2.546e+05 | 2.53e+05 | no | 2.53e+05 | 0.9938 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d7, true/decl 0.0421) | top-3 |true error| in H3 | parent | [9.11792, 9.23106] | 1.066e+06 | 1.053e+06 | no | 1.053e+06 | 0.9876 | Levin (LU tried first) | Levin (LU tried first) | no |
| 12 (d7, true/decl 0.0421) | top-3 |true error| in H3 | sibling | [9.11792, 9.17449] | 8.017e+05 | 8e+05 | no | 8e+05 | 0.9979 | Levin (LU tried first) | Levin (LU tried first) | no |
| 24 (d0, true/decl 7.44) | carries excess error ; top-3 |true error| in H3 | leaf | [9.23164, 26.9711] | 1.257e+08 | 9.448e+07 | no | 9.448e+07 | 0.7515 | Levin (LU tried first) | Levin (LU tried first) | no |
| 24 | carries excess error ; top-3 |true error| in H3 | parent | - | | | | | | | | |
| 24 | carries excess error ; top-3 |true error| in H3 | sibling | - | | | | | | | | |


## v17: O9 check on its H3 leaves

N = 24 nodes; H3 true error -9.988e-35 over 8 leaves; 4 leaves carry excess error, together 1.000 of it. Script sha256 f97b2de60c1f, module sha256 4e5da371f51e.

| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (d2, true/decl 0.136) | top-3 |true error| in H3 | leaf | [8.71567, 8.97337] | 2.4e+07 | 2.375e+07 | no | 2.375e+07 | 0.9897 | Levin (LU tried first) | Levin (LU tried first) | no |
| 1 (d2, true/decl 0.136) | top-3 |true error| in H3 | parent | [8.71567, 9.23106] | 3.154e+07 | 2.986e+07 | no | 2.986e+07 | 0.9467 | Levin (LU tried first) | Levin (LU tried first) | no |
| 1 (d2, true/decl 0.136) | top-3 |true error| in H3 | sibling | [8.97337, 9.23106] | 6.28e+06 | 6.106e+06 | no | 6.106e+06 | 0.9723 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 1.52) | carries excess error | leaf | [9.10221, 9.16664] | 1.055e+06 | 1.052e+06 | no | 1.052e+06 | 0.9975 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 1.52) | carries excess error | parent | [9.10221, 9.23106] | 1.402e+06 | 1.382e+06 | no | 1.382e+06 | 0.9859 | Levin (LU tried first) | Levin (LU tried first) | no |
| 3 (d4, true/decl 1.52) | carries excess error | sibling | [9.16664, 9.23106] | 3.321e+05 | 3.298e+05 | no | 3.298e+05 | 0.9929 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 7.06) | carries excess error | leaf | [9.16664, 9.19885] | 2.49e+05 | 2.487e+05 | no | 2.487e+05 | 0.9988 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 7.06) | carries excess error | parent | [9.16664, 9.23106] | 3.321e+05 | 3.298e+05 | no | 3.298e+05 | 0.9929 | Levin (LU tried first) | Levin (LU tried first) | no |
| 4 (d5, true/decl 7.06) | carries excess error | sibling | [9.19885, 9.23106] | 8.137e+04 | 8.108e+04 | no | 8.108e+04 | 0.9965 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 16.8) | carries excess error | leaf | [9.19885, 9.21495] | 6.075e+04 | 6.072e+04 | no | 6.072e+04 | 0.9994 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 16.8) | carries excess error | parent | [9.19885, 9.23106] | 8.137e+04 | 8.108e+04 | no | 8.108e+04 | 0.9965 | Levin (LU tried first) | Levin (LU tried first) | no |
| 5 (d6, true/decl 16.8) | carries excess error | sibling | [9.21495, 9.23106] | 2.04e+04 | 2.037e+04 | no | 2.037e+04 | 0.9983 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.0036) | top-3 |true error| in H3 | leaf | [9.21495, 9.23106] | 2.04e+04 | 2.037e+04 | no | 2.037e+04 | 0.9983 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.0036) | top-3 |true error| in H3 | parent | [9.19885, 9.23106] | 8.137e+04 | 8.108e+04 | no | 8.108e+04 | 0.9965 | Levin (LU tried first) | Levin (LU tried first) | no |
| 6 (d6, true/decl 0.0036) | top-3 |true error| in H3 | sibling | [9.19885, 9.21495] | 6.075e+04 | 6.072e+04 | no | 6.072e+04 | 0.9994 | Levin (LU tried first) | Levin (LU tried first) | no |
| 21 (d0, true/decl 2.94) | carries excess error ; top-3 |true error| in H3 | leaf | [9.23164, 26.9711] | 1.257e+08 | 9.448e+07 | no | 9.448e+07 | 0.7515 | Levin (LU tried first) | Levin (LU tried first) | no |
| 21 | carries excess error ; top-3 |true error| in H3 | parent | - | | | | | | | | |
| 21 | carries excess error ; top-3 |true error| in H3 | sibling | - | | | | | | | | |


## v22: O9 check on its H3 leaves

N = 24 nodes; H3 true error +2.043e-33 over 109 leaves; 14 leaves carry excess error, together 1.000 of it. Script sha256 f97b2de60c1f, module sha256 4e5da371f51e.

| leaf i | why | region | span | phase_span (07) | |theta(b)-theta(a)| | theta' sign change | node-partition sum | true / est. | route (07) | route (true) | changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 (d6, true/decl 2.23) | carries excess error | leaf | [9.01997, 9.1254] | 9.719e+05 | 9.679e+05 | no | 9.679e+05 | 0.9959 | Levin (LU tried first) | Levin (LU tried first) | no |
| 7 (d6, true/decl 2.23) | carries excess error | parent | [9.01997, 9.23084] | 1.288e+06 | 1.259e+06 | no | 1.259e+06 | 0.9772 | Levin (LU tried first) | Levin (LU tried first) | no |
| 7 (d6, true/decl 2.23) | carries excess error | sibling | [9.1254, 9.23084] | 2.943e+05 | 2.91e+05 | no | 2.91e+05 | 0.9885 | Levin (LU tried first) | Levin (LU tried first) | no |
| 8 (d7, true/decl 3.82) | carries excess error | leaf | [9.1254, 9.17812] | 2.21e+05 | 2.205e+05 | no | 2.205e+05 | 0.998 | Levin (LU tried first) | Levin (LU tried first) | no |
| 8 (d7, true/decl 3.82) | carries excess error | parent | [9.1254, 9.23084] | 2.943e+05 | 2.91e+05 | no | 2.91e+05 | 0.9885 | Levin (LU tried first) | Levin (LU tried first) | no |
| 8 (d7, true/decl 3.82) | carries excess error | sibling | [9.17812, 9.23084] | 7.085e+04 | 7.044e+04 | no | 7.044e+04 | 0.9943 | Levin (LU tried first) | Levin (LU tried first) | no |
| 9 (d8, true/decl 39.7) | carries excess error | leaf | [9.17812, 9.20448] | 5.292e+04 | 5.286e+04 | no | 5.286e+04 | 0.999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 9 (d8, true/decl 39.7) | carries excess error | parent | [9.17812, 9.23084] | 7.085e+04 | 7.044e+04 | no | 7.044e+04 | 0.9943 | Levin (LU tried first) | Levin (LU tried first) | no |
| 9 (d8, true/decl 39.7) | carries excess error | sibling | [9.20448, 9.23084] | 1.763e+04 | 1.758e+04 | no | 1.758e+04 | 0.9972 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d8, true/decl 0.00276) | top-3 |true error| in H3 | leaf | [9.20448, 9.23084] | 1.763e+04 | 1.758e+04 | no | 1.758e+04 | 0.9972 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d8, true/decl 0.00276) | top-3 |true error| in H3 | parent | [9.17812, 9.23084] | 7.085e+04 | 7.044e+04 | no | 7.044e+04 | 0.9943 | Levin (LU tried first) | Levin (LU tried first) | no |
| 10 (d8, true/decl 0.00276) | top-3 |true error| in H3 | sibling | [9.17812, 9.20448] | 5.292e+04 | 5.286e+04 | no | 5.286e+04 | 0.999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d5, true/decl 4.67) | carries excess error ; top-3 |true error| in H3 | leaf | [9.23186, 9.75095] | 4.033e+06 | 4.275e+06 | no | 4.275e+06 | 1.06 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d5, true/decl 4.67) | carries excess error ; top-3 |true error| in H3 | parent | [9.23186, 10.27] | 1.026e+07 | 1.15e+07 | no | 1.15e+07 | 1.121 | Levin (LU tried first) | Levin (LU tried first) | no |
| 26 (d5, true/decl 4.67) | carries excess error ; top-3 |true error| in H3 | sibling | [9.75095, 10.27] | 7.139e+06 | 7.221e+06 | no | 7.221e+06 | 1.011 | Levin (LU tried first) | Levin (LU tried first) | no |
| 32 (d10, true/decl 2.47) | carries excess error | leaf | [17.6021, 17.6183] | 253.7 | 253.7 | no | 253.7 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 32 (d10, true/decl 2.47) | carries excess error | parent | [17.6021, 17.6345] | 503.4 | 503.4 | no | 503.4 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 32 (d10, true/decl 2.47) | carries excess error | sibling | [17.6183, 17.6345] | 249.7 | 249.7 | no | 249.7 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 33 (d10, true/decl 6.42) | carries excess error | leaf | [17.6183, 17.6345] | 249.7 | 249.7 | no | 249.7 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 33 (d10, true/decl 6.42) | carries excess error | parent | [17.6021, 17.6345] | 503.4 | 503.4 | no | 503.4 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 33 (d10, true/decl 6.42) | carries excess error | sibling | [17.6021, 17.6183] | 253.7 | 253.7 | no | 253.7 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d9, true/decl 19.8) | carries excess error | leaf | [18.0887, 18.1212] | 309.5 | 309.5 | no | 309.5 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d9, true/decl 19.8) | carries excess error | parent | [18.0563, 18.1212] | 629.2 | 629.2 | no | 629.2 | 0.9999 | Levin (LU tried first) | Levin (LU tried first) | no |
| 38 (d9, true/decl 19.8) | carries excess error | sibling | [18.0563, 18.0887] | 319.7 | 319.7 | no | 319.7 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 43 (d11, true/decl 2.19) | carries excess error | leaf | [18.5754, 18.5835] | 48.14 | 48.14 | no | 48.14 | 1 | Levin (least squares) | Levin (least squares) | no |
| 43 (d11, true/decl 2.19) | carries excess error | parent | [18.5754, 18.5916] | 95.9 | 95.89 | no | 95.89 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 43 (d11, true/decl 2.19) | carries excess error | sibling | [18.5835, 18.5916] | 47.75 | 47.75 | no | 47.75 | 1 | Levin (least squares) | Levin (least squares) | no |
| 49 (d10, true/decl 8.04) | carries excess error | leaf | [18.5916, 18.6078] | 94.35 | 94.35 | no | 94.35 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 49 (d10, true/decl 8.04) | carries excess error | parent | [18.5754, 18.6078] | 190.3 | 190.2 | no | 190.2 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 49 (d10, true/decl 8.04) | carries excess error | sibling | [18.5754, 18.5916] | 95.9 | 95.89 | no | 95.89 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 51 (d11, true/decl 2.08) | carries excess error | leaf | [18.624, 18.6321] | 45.86 | 45.86 | no | 45.86 | 1 | Levin (least squares) | Levin (least squares) | no |
| 51 (d11, true/decl 2.08) | carries excess error | parent | [18.624, 18.6403] | 91.34 | 91.34 | no | 91.34 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 51 (d11, true/decl 2.08) | carries excess error | sibling | [18.6321, 18.6403] | 45.49 | 45.49 | no | 45.49 | 1 | Levin (least squares) | Levin (least squares) | no |
| 61 (d10, true/decl 4.61) | carries excess error | leaf | [18.8674, 18.8836] | 71.62 | 71.61 | no | 71.61 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 61 (d10, true/decl 4.61) | carries excess error | parent | [18.8674, 18.8998] | 142.1 | 142.1 | no | 142.1 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 61 (d10, true/decl 4.61) | carries excess error | sibling | [18.8836, 18.8998] | 70.46 | 70.46 | no | 70.46 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 69 (d13, true/decl 1) | carries excess error | leaf | [18.9079, 18.9099] | 8.657 | 8.657 | no | 8.657 | 1 | Clenshaw-Curtis | Clenshaw-Curtis | no |
| 69 (d13, true/decl 1) | carries excess error | parent | [18.9079, 18.912] | 17.3 | 17.3 | no | 17.3 | 1 | Clenshaw-Curtis | Clenshaw-Curtis | no |
| 69 (d13, true/decl 1) | carries excess error | sibling | [18.9099, 18.912] | 8.64 | 8.64 | no | 8.64 | 1 | Clenshaw-Curtis | Clenshaw-Curtis | no |
| 73 (d10, true/decl 7.81) | carries excess error | leaf | [18.916, 18.9322] | 68.21 | 68.21 | no | 68.21 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 73 (d10, true/decl 7.81) | carries excess error | parent | [18.8998, 18.9322] | 137.5 | 137.5 | no | 137.5 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 73 (d10, true/decl 7.81) | carries excess error | sibling | [18.8998, 18.916] | 69.33 | 69.33 | no | 69.33 | 1 | Levin (LU tried first) | Levin (LU tried first) | no |
| 121 (d4, true/decl 1.21) | carries excess error ; top-3 |true error| in H3 | leaf | [22.728, 23.7662] | 62.02 | 60.52 | no | 60.52 | 0.9758 | Levin (least squares) | Levin (least squares) | no |
| 121 (d4, true/decl 1.21) | carries excess error ; top-3 |true error| in H3 | parent | [21.6899, 23.7662] | 253.1 | 231.4 | no | 231.4 | 0.9146 | Levin (LU tried first) | Levin (LU tried first) | no |
| 121 (d4, true/decl 1.21) | carries excess error ; top-3 |true error| in H3 | sibling | [21.6899, 22.728] | 175.2 | 170.9 | no | 170.9 | 0.9759 | Levin (LU tried first) | Levin (LU tried first) | no |


