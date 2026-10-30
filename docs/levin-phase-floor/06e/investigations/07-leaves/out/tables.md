# 07's driver: where the declared error sits, leaf by leaf

tables.py run at 2026-10-29T20:01:57.668346+00:00 on mps026088.phys.susx.ac.uk; base e5764a6708757cb11704b95bcd66825baab4ba29, patch sha256 722c7a477f8038946dc3bfb18ed3bbc41b3a574317022012aecc28203207e365, levin_quadrature.py sha256 4e4c22530b91f46ba4f68ec8e534261f89926614dfa4d04757d8496456916495; numpy 2.2.4, scipy 1.16.3, mpmath 1.3.0.

Script sha256: endnodes.py `b85a78b890886ead5aa98ffa9b5c5738d6a68129ed9abd90edb223cdf0ddb5fa`, lc.py `0cb32bb4e1d62d37f8875e9355d270cbf365a7ba34c1370329ac5f05cc0e01a7`, lg_capture.py `45e3bc2ce89b7792d9752b973ca2ea388ce4d56fd904427488d38dc85217a420`, q12_outside.py `28b85dbe730d265da906a4707154e18355bf910f71935c02caa2b85ecc005d67`, q3_decompose.py `2a4106d93df518d07b4c9e381cd1e4ffecbed43267c8c154b81e0ec66a3d725c`, q4_window.py `32887fd8f477296790c1ce8f7b50684bd03feec391fdc066abdfb8b2197b540a`, tables.py `93b5da8c87234210460468eeabd6ddf083b1a69f9f56649c95f8b325cc3abe2e`

Every JSON in out/ carries its own header (script, its sha256, every script's, base, patch, levin_quadrature.py, host, UTC, versions).

## c1e6w0: {'c': 1000000.0, 'x_s': 0.723606797749979, 'omega': 0.0}

Inputs: `q12_c1e6w0.json` (2026-10-29T19:28:40.884513+00:00), `q12_c1e6w0_512a.json` (2026-10-29T19:41:36.477472+00:00), `q12_c1e6w0_512d12.json` (2026-10-29T19:42:45.844605+00:00), `q12_c1e6w0_extra.json` (2026-10-29T19:38:39.273060+00:00)

### Per run

| Phi_w | d_max | abserr | true error | request | window declared | outside declared | outside true (sum) | outside declared by component |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | 1.67e-11 | +1.7e-11 | 1.43e-14 | 5.42e-15 | 1.67e-11 | +1.7e-11 | residual 1.7e-11, floor(phase_limited) 2.3e-16, floor 3.9e-18 |
| 2pi | 9 | 4.55e-14 | +1.7e-16 | 1.43e-14 | 5.34e-15 | 4.02e-14 | +1.7e-16 | floor 2.9e-14, floor(phase_limited) 8.1e-15, residual 3.0e-15 |
| 2pi | 12 | 8.57e-11 | -9.3e-16 | 1.43e-14 | 5.33e-15 | 8.57e-11 | -9.3e-16 | nested_pair 8.5e-11, floor 1.6e-13, floor(phase_limited) 1.1e-13 |
| 4pi | 6 | 7.55e-13 | +7.4e-13 | 1.43e-14 | 1.07e-14 | 7.45e-13 | +7.4e-13 | residual 7.4e-13, floor(phase_limited) 2.3e-16 |
| 4pi | 9 | 7.19e-10 | -1.5e-17 | 1.43e-14 | 1.07e-14 | 7.19e-10 | -1.5e-17 | nested_pair 7.2e-10, floor 3.3e-14, floor(phase_limited) 6.3e-15, residual 5.6e-16 |
| 4pi | 12 | 1.86e-13 | +1.2e-16 | 1.43e-14 | 1.06e-14 | 1.76e-13 | +1.2e-16 | floor(phase_limited) 1.6e-13, floor 1.8e-14 |
| 8pi | 6 | 4.64e-14 | -2.3e-14 | 1.43e-14 | 2.14e-14 | 2.50e-14 | -2.3e-14 | residual 2.5e-14, floor(phase_limited) 2.2e-16, floor 5.1e-18 |
| 8pi | 9 | 9.84e-10 | +5.1e-16 | 1.43e-14 | 2.13e-14 | 9.84e-10 | +5.1e-16 | nested_pair 9.8e-10, floor 4.3e-14, residual 2.9e-15, floor(phase_limited) 2.4e-16 |
| 8pi | 12 | 5.16e-10 | +5.2e-16 | 1.43e-14 | 2.13e-14 | 5.16e-10 | +5.2e-16 | nested_pair 5.2e-10, floor(phase_limited) 2.3e-13, floor 1.1e-13, residual 3.7e-15 |
| 16pi | 6 | 4.38e-14 | -2.5e-16 | 1.43e-14 | 4.26e-14 | 1.20e-15 | -2.5e-16 | residual 9.8e-16, floor(phase_limited) 2.2e-16 |
| 16pi | 9 | 1.25e-13 | +6.0e-16 | 1.43e-14 | 4.25e-14 | 8.27e-14 | +6.0e-16 | nested_pair 4.5e-14, floor 3.7e-14, floor(phase_limited) 2.3e-16 |
| 16pi | 12 | 9.59e-14 | +5.4e-16 | 1.43e-14 | 4.25e-14 | 5.34e-14 | +5.4e-16 | floor(phase_limited) 5.3e-14 |
| 24pi | 6 | 6.46e-14 | -9.7e-17 | 1.43e-14 | 6.39e-14 | 6.84e-16 | -9.3e-17 | residual 4.4e-16, floor(phase_limited) 2.4e-16 |
| 24pi | 9 | 2.15e-09 | +4.9e-18 | 1.43e-14 | 6.38e-14 | 2.15e-09 | +2.9e-18 | nested_pair 2.2e-09, floor 2.7e-14, residual 1.7e-15, floor(phase_limited) 2.4e-16 |
| 24pi | 12 | 4.78e-13 | -8.7e-17 | 1.43e-14 | 6.37e-14 | 4.14e-13 | -8.8e-17 | floor(phase_limited) 4.0e-13, floor 1.0e-14 |
| 32pi | 6 | 8.54e-14 | -2.5e-16 | 1.43e-14 | 8.51e-14 | 2.68e-16 | -2.7e-16 | floor(phase_limited) 2.4e-16, residual 2.2e-17, floor 7.7e-18 |
| 32pi | 9 | 8.53e-14 | -1.7e-16 | 1.43e-14 | 8.50e-14 | 2.48e-16 | -1.7e-16 | floor(phase_limited) 2.5e-16 |
| 32pi | 12 | 8.52e-14 | -1.7e-16 | 1.43e-14 | 8.50e-14 | 2.48e-16 | -1.7e-16 | floor(phase_limited) 2.5e-16 |
| 48pi | 6 | 1.29e-13 | +3.1e-16 | 1.43e-14 | 1.28e-13 | 1.68e-15 | +3.0e-16 | residual 1.5e-15, floor(phase_limited) 1.8e-16, floor 1.7e-17 |
| 48pi | 9 | 2.73e-10 | -1.0e-15 | 1.43e-14 | 1.28e-13 | 2.73e-10 | -1.0e-15 | nested_pair 2.7e-10, floor 4.0e-15, floor(phase_limited) 2.1e-16 |
| 48pi | 12 | 2.23e-13 | -6.1e-16 | 1.43e-14 | 1.28e-13 | 9.56e-14 | -6.1e-16 | floor(phase_limited) 9.6e-14 |
| 64pi | 6 | 1.71e-13 | +2.9e-16 | 1.43e-14 | 1.70e-13 | 4.68e-16 | +3.2e-16 | floor(phase_limited) 2.3e-16, residual 2.3e-16 |
| 64pi | 9 | 1.70e-13 | +4.2e-16 | 1.43e-14 | 1.70e-13 | 4.29e-16 | +4.2e-16 | floor(phase_limited) 2.4e-16, residual 1.9e-16, floor 7.3e-19 |
| 64pi | 12 | 2.15e-13 | +2.8e-16 | 1.43e-14 | 1.70e-13 | 4.49e-14 | +2.8e-16 | floor 4.1e-14, floor(phase_limited) 3.7e-15 |
| 128pi | 6 | 3.41e-13 | +9.0e-17 | 1.43e-14 | 3.40e-13 | 7.58e-16 | +1.1e-16 | residual 5.4e-16, floor(phase_limited) 2.2e-16, floor 2.9e-18 |
| 128pi | 9 | 3.42e-13 | +1.5e-16 | 1.43e-14 | 3.40e-13 | 1.76e-15 | +1.5e-16 | residual 1.5e-15, floor(phase_limited) 2.2e-16, floor 1.2e-17 |
| 128pi | 12 | 1.56e-10 | +1.8e-16 | 1.43e-14 | 3.40e-13 | 1.56e-10 | +1.8e-16 | nested_pair 1.6e-10, floor 1.6e-13, floor(phase_limited) 2.3e-16 |
| 512pi | 6 | 1.37e-12 | -5.2e-16 | 1.43e-14 | 1.37e-12 | 2.67e-15 | -5.2e-16 | residual 2.5e-15, floor(phase_limited) 1.9e-16 |
| 512pi | 9 | 1.37e-12 | +5.1e-17 | 1.43e-14 | 1.36e-12 | 6.07e-15 | +5.4e-17 | residual 5.9e-15, floor(phase_limited) 1.9e-16, floor 7.1e-18 |
| 512pi | 12 | 3.81e-11 | -2.2e-16 | 1.43e-14 | 1.36e-12 | 3.68e-11 | -2.1e-16 | nested_pair 3.6e-11, floor 8.0e-13, floor(phase_limited) 1.5e-13 |

### Outside leaves carrying >= 1% of the outside declared error, grouped by component and type

| Phi_w | d_max | component | type | leaves | depth | dtheta_near (rad) | abs theta' near | declared (sum) | abs true (sum) | max abs true | true (signed sum) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | residual | Levin | 7 | 6-6 | 6.28-2.52e+03 | 3.5e+03-7.1e+04 | 1.67e-11 | 1.68e-11 | 1.68e-11 | +1.7e-11 |
| 2pi | 9 | floor | direct | 12 | 9-9 | 6.28-66.6 | 3.5e+03-1.2e+04 | 2.91e-14 | 1.77e-16 | 3.78e-17 | +8.4e-17 |
| 2pi | 9 | floor(phase_limited) | direct | 3 | 8-8 | 6.28-16.1 | 3.5e+03-5.7e+03 | 7.82e-15 | 2.45e-17 | 1.50e-17 | -2.5e-17 |
| 2pi | 9 | residual | Levin | 32 | 9-9 | 109-3.27e+03 | 1.5e+04-8.1e+04 | 3.00e-15 | 1.26e-15 | 7.98e-17 | +6.4e-16 |
| 2pi | 12 | nested_pair | direct | 125 | 12-12 | 1.79e+03-3.37e+03 | 6e+04-8.2e+04 | 8.54e-11 | 2.78e-15 | 6.91e-17 | -1.9e-16 |
| 4pi | 6 | residual | Levin | 4 | 6-6 | 12.6-132 | 5e+03-1.6e+04 | 7.45e-13 | 7.41e-13 | 7.41e-13 | +7.4e-13 |
| 4pi | 9 | nested_pair | direct | 5 | 9-9 | 20.6-72.4 | 6.4e+03-1.2e+04 | 7.19e-10 | 3.14e-16 | 1.12e-16 | +1.9e-16 |
| 4pi | 12 | floor(phase_limited) | direct | 59 | 8-11 | 12.6-304 | 5e+03-2.5e+04 | 1.58e-13 | 1.42e-15 | 7.40e-17 | +2.9e-16 |
| 4pi | 12 | floor | direct | 10 | 12-12 | 313-354 | 2.5e+04-2.7e+04 | 1.77e-14 | 6.73e-17 | 4.02e-17 | +4.8e-17 |
| 8pi | 6 | residual | Levin | 16 | 6-6 | 25.1-1.53e+04 | 7.1e+03-1.8e+05 | 2.48e-14 | 2.35e-14 | 2.32e-14 | -2.3e-14 |
| 8pi | 9 | nested_pair | direct | 13 | 9-9 | 25.1-381 | 7.1e+03-2.8e+04 | 9.84e-10 | 5.98e-16 | 1.28e-16 | -6.5e-17 |
| 8pi | 12 | nested_pair | direct | 8 | 12-12 | 5.23e+03-5.5e+03 | 1e+05-1e+05 | 5.16e-10 | 4.46e-16 | 9.41e-17 | +1.0e-16 |
| 16pi | 6 | residual | Levin | 6 | 6-6 | 50.3-945 | 1e+04-4.3e+04 | 9.84e-16 | 3.33e-16 | 2.06e-16 | -3.3e-16 |
| 16pi | 6 | floor(phase_limited) | Levin | 15 | 1-5 | 168-6.73e+04 | 1.8e+04-3.7e+05 | 2.15e-16 | 4.97e-16 | 9.89e-17 | +7.9e-17 |
| 16pi | 9 | nested_pair | direct | 4 | 9-9 | 124-159 | 1.6e+04-1.8e+04 | 4.54e-14 | 8.21e-18 | 6.55e-18 | +4.9e-18 |
| 16pi | 9 | floor | direct | 12 | 9-9 | 50.3-132 | 1e+04-1.6e+04 | 3.71e-14 | 2.36e-16 | 6.12e-17 | +1.4e-16 |
| 16pi | 12 | floor(phase_limited) | direct | 20 | 9-10 | 50.3-164 | 1e+04-1.8e+04 | 5.32e-14 | 3.67e-16 | 6.12e-17 | +9.0e-17 |
| 24pi | 6 | residual | Levin | 8 | 6-6 | 75.4-847 | 1.2e+04-4.1e+04 | 4.43e-16 | 1.66e-16 | 6.62e-17 | -9.6e-17 |
| 24pi | 6 | floor(phase_limited) | Levin | 9 | 1-5 | 211-6.77e+04 | 2.1e+04-3.7e+05 | 2.40e-16 | 2.02e-17 | 4.90e-18 | +3.4e-18 |
| 24pi | 9 | nested_pair | direct | 24 | 9-9 | 75.4-648 | 1.2e+04-3.6e+04 | 2.15e-09 | 1.86e-15 | 1.85e-16 | +2.1e-16 |
| 24pi | 12 | floor(phase_limited) | direct | 148 | 9-11 | 75.4-1.02e+03 | 1.2e+04-4.5e+04 | 4.04e-13 | 4.46e-15 | 9.00e-17 | -6.6e-17 |
| 24pi | 12 | floor | direct | 6 | 12-12 | 316-338 | 2.5e+04-2.6e+04 | 1.04e-14 | 1.33e-16 | 3.71e-17 | +3.6e-17 |
| 32pi | 6 | floor(phase_limited) | Levin | 11 | 1-5 | 101-6.8e+04 | 1.4e+04-3.7e+05 | 2.38e-16 | 2.38e-16 | 7.50e-17 | -9.7e-17 |
| 32pi | 6 | residual | Levin | 1 | 6-6 | 319-319 | 2.5e+04-2.5e+04 | 2.23e-17 | 1.20e-17 | 1.20e-17 | -1.2e-17 |
| 32pi | 6 | floor | Levin | 7 | 6-6 | 101-2.86e+03 | 1.4e+04-7.6e+04 | 7.69e-18 | 3.90e-16 | 8.93e-17 | -1.6e-16 |
| 32pi | 9 | floor(phase_limited) | Levin | 21 | 1-8 | 101-6.8e+04 | 1.4e+04-3.7e+05 | 2.48e-16 | 7.10e-16 | 8.93e-17 | -1.7e-16 |
| 32pi | 12 | floor(phase_limited) | Levin | 21 | 1-8 | 101-6.8e+04 | 1.4e+04-3.7e+05 | 2.48e-16 | 7.10e-16 | 8.93e-17 | -1.7e-16 |
| 48pi | 6 | residual | Levin | 14 | 6-6 | 151-6.46e+04 | 1.7e+04-3.6e+05 | 1.48e-15 | 4.43e-16 | 7.23e-17 | +2.7e-16 |
| 48pi | 6 | floor(phase_limited) | Levin | 8 | 1-4 | 563-6.86e+04 | 3.4e+04-3.7e+05 | 1.81e-16 | 2.49e-16 | 6.67e-17 | -1.9e-16 |
| 48pi | 9 | nested_pair | direct | 15 | 9-9 | 160-546 | 1.8e+04-3.3e+04 | 2.73e-10 | 9.07e-16 | 2.02e-16 | -3.2e-16 |
| 48pi | 12 | floor(phase_limited) | direct | 31 | 9-10 | 151-555 | 1.7e+04-3.3e+04 | 9.54e-14 | 7.48e-16 | 9.10e-17 | +9.4e-17 |
| 64pi | 6 | floor(phase_limited) | Levin | 11 | 1-5 | 201-6.91e+04 | 2e+04-3.7e+05 | 2.34e-16 | 2.82e-16 | 6.94e-17 | +2.7e-16 |
| 64pi | 6 | residual | Levin | 2 | 6-6 | 201-482 | 2e+04-3.1e+04 | 2.34e-16 | 5.66e-17 | 5.64e-17 | +5.7e-17 |
| 64pi | 9 | floor(phase_limited) | Levin | 14 | 1-7 | 201-6.91e+04 | 2e+04-3.7e+05 | 2.38e-16 | 4.59e-16 | 6.94e-17 | +3.3e-16 |
| 64pi | 9 | residual | Levin | 3 | 9-9 | 326-440 | 2.6e+04-3e+04 | 1.91e-16 | 4.58e-17 | 4.53e-17 | +4.5e-17 |
| 64pi | 12 | floor | direct | 22 | 12-12 | 335-477 | 2.6e+04-3.1e+04 | 4.12e-14 | 5.78e-16 | 5.97e-17 | -8.8e-17 |
| 64pi | 12 | floor(phase_limited) | direct | 1 | 11-11 | 326-326 | 2.6e+04-2.6e+04 | 3.45e-15 | 1.05e-17 | 1.05e-17 | -1.1e-17 |
| 128pi | 6 | residual | Levin | 6 | 6-6 | 402-3.42e+03 | 2.8e+04-8.3e+04 | 5.36e-16 | 2.67e-16 | 7.26e-17 | +2.2e-17 |
| 128pi | 6 | floor(phase_limited) | Levin | 12 | 1-5 | 402-7.07e+04 | 2.8e+04-3.8e+05 | 2.19e-16 | 4.77e-16 | 9.21e-17 | +5.4e-17 |
| 128pi | 9 | residual | Levin | 18 | 9-9 | 402-4.25e+03 | 2.8e+04-9.2e+04 | 1.53e-15 | 1.56e-17 | 3.06e-18 | +6.8e-19 |
| 128pi | 9 | floor(phase_limited) | Levin | 16 | 1-6 | 402-7.07e+04 | 2.8e+04-3.8e+05 | 2.22e-16 | 7.86e-16 | 9.21e-17 | +8.5e-17 |
| 128pi | 12 | nested_pair | direct | 82 | 12-12 | 1.51e+03-4.36e+03 | 5.5e+04-9.3e+04 | 1.56e-10 | 4.02e-15 | 8.92e-17 | -3.9e-16 |
| 512pi | 6 | residual | Levin | 20 | 6-6 | 2.48e+03-3.36e+04 | 7e+04-2.6e+05 | 2.49e-15 | 1.24e-15 | 1.10e-16 | -2.9e-16 |
| 512pi | 6 | floor(phase_limited) | Levin | 30 | 1-5 | 1.61e+03-7.61e+04 | 5.7e+04-3.9e+05 | 1.86e-16 | 1.80e-15 | 1.33e-16 | -2.3e-16 |
| 512pi | 9 | residual | Levin | 44 | 9-9 | 2.48e+03-3.44e+04 | 7e+04-2.6e+05 | 5.87e-15 | 2.59e-16 | 3.25e-17 | -4.8e-17 |
| 512pi | 9 | floor(phase_limited) | Levin | 50 | 1-7 | 1.61e+03-7.61e+04 | 5.7e+04-3.9e+05 | 1.92e-16 | 3.04e-15 | 1.33e-16 | +8.4e-17 |
| 512pi | 12 | nested_pair | direct | 117 | 12-12 | 1.88e+04-3.45e+04 | 1.9e+05-2.6e+05 | 3.58e-11 | 5.03e-15 | 1.57e-16 | -5.7e-17 |
| 512pi | 12 | floor | direct | 167 | 12-12 | 4.14e+03-3.28e+04 | 9.1e+04-2.6e+05 | 8.01e-13 | 7.05e-15 | 1.21e-16 | -4.2e-16 |

### The leaves adjacent to the window

| Phi_w | d_max | side | span | depth | type | component | declared | true |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | left | [0.708811, 0.720062] | 6 | Levin | residual | 1.67e-11 | +1.7e-11 |
| 2pi | 6 | right | [0.727152, 0.731415] | 6 | Levin | residual | 1.40e-15 | +1.4e-15 |
| 2pi | 9 | left | [0.718656, 0.720062] | 9 | direct | floor | 2.91e-15 | +3.8e-17 |
| 2pi | 9 | right | [0.727152, 0.728218] | 8 | direct | floor(phase_limited) | 2.11e-15 | -8.2e-18 |
| 2pi | 12 | left | [0.718656, 0.720062] | 9 | direct | floor(phase_limited) | 2.91e-15 | +3.8e-17 |
| 2pi | 12 | right | [0.727152, 0.728218] | 8 | direct | floor(phase_limited) | 2.11e-15 | -8.2e-18 |
| 4pi | 6 | left | [0.707366, 0.718594] | 6 | Levin | residual | 7.44e-13 | +7.4e-13 |
| 4pi | 6 | right | [0.728620, 0.732860] | 6 | Levin | residual | 1.83e-16 | +1.9e-17 |
| 4pi | 9 | left | [0.717190, 0.718594] | 9 | direct | floor | 3.74e-15 | -7.4e-17 |
| 4pi | 9 | right | [0.728620, 0.729680] | 8 | direct | floor(phase_limited) | 2.78e-15 | +5.2e-18 |
| 4pi | 12 | left | [0.717190, 0.718594] | 9 | direct | floor(phase_limited) | 3.74e-15 | -7.4e-17 |
| 4pi | 12 | right | [0.728620, 0.729680] | 8 | direct | floor(phase_limited) | 2.78e-15 | +5.2e-18 |
| 8pi | 6 | left | [0.705321, 0.716517] | 6 | Levin | residual | 2.31e-14 | -2.3e-14 |
| 8pi | 6 | right | [0.730697, 0.734904] | 6 | Levin | residual | 1.36e-16 | +6.5e-17 |
| 8pi | 9 | left | [0.715118, 0.716517] | 9 | direct | nested_pair | 3.08e-13 | +1.7e-17 |
| 8pi | 9 | right | [0.730697, 0.731223] | 9 | direct | floor | 1.74e-15 | +3.8e-17 |
| 8pi | 12 | left | [0.715817, 0.716517] | 10 | direct | floor(phase_limited) | 2.25e-15 | +1.1e-17 |
| 8pi | 12 | right | [0.730697, 0.731223] | 9 | direct | floor(phase_limited) | 1.74e-15 | +3.8e-17 |
| 16pi | 6 | left | [0.702431, 0.713580] | 6 | Levin | residual | 3.65e-16 | -2.1e-16 |
| 16pi | 6 | right | [0.733633, 0.737795] | 6 | Levin | residual | 1.30e-16 | -6.5e-17 |
| 16pi | 9 | left | [0.708005, 0.713580] | 7 | Levin | floor(phase_limited) | 2.38e-18 | +8.0e-17 |
| 16pi | 9 | right | [0.733633, 0.734154] | 9 | direct | floor | 2.40e-15 | -6.8e-18 |
| 16pi | 12 | left | [0.708005, 0.713580] | 7 | Levin | floor(phase_limited) | 2.38e-18 | +8.0e-17 |
| 16pi | 12 | right | [0.733633, 0.734154] | 9 | direct | floor(phase_limited) | 2.40e-15 | -6.8e-18 |
| 24pi | 6 | left | [0.700212, 0.711327] | 6 | Levin | residual | 4.81e-18 | -6.5e-17 |
| 24pi | 6 | right | [0.735887, 0.740013] | 6 | Levin | residual | 1.30e-16 | -6.6e-17 |
| 24pi | 9 | left | [0.709938, 0.711327] | 9 | direct | nested_pair | 7.56e-10 | -9.6e-17 |
| 24pi | 9 | right | [0.735887, 0.736403] | 9 | direct | floor | 2.91e-15 | +2.8e-17 |
| 24pi | 12 | left | [0.710980, 0.711327] | 11 | direct | floor(phase_limited) | 1.78e-15 | +3.2e-17 |
| 24pi | 12 | right | [0.735887, 0.736403] | 9 | direct | floor(phase_limited) | 2.91e-15 | +2.8e-17 |
| 32pi | 6 | left | [0.698342, 0.709427] | 6 | Levin | floor | 2.42e-18 | +2.0e-18 |
| 32pi | 6 | right | [0.737786, 0.745981] | 5 | Levin | floor(phase_limited) | 2.23e-18 | +6.7e-17 |
| 32pi | 9 | left | [0.698342, 0.709427] | 6 | Levin | floor(phase_limited) | 2.42e-18 | +2.0e-18 |
| 32pi | 9 | right | [0.737786, 0.745981] | 5 | Levin | floor(phase_limited) | 2.23e-18 | +6.7e-17 |
| 32pi | 12 | left | [0.698342, 0.709427] | 6 | Levin | floor(phase_limited) | 2.42e-18 | +2.0e-18 |
| 32pi | 12 | right | [0.737786, 0.745981] | 5 | Levin | floor(phase_limited) | 2.23e-18 | +6.7e-17 |
| 48pi | 6 | left | [0.695205, 0.706240] | 6 | Levin | floor | 2.10e-18 | +3.2e-17 |
| 48pi | 6 | right | [0.740973, 0.745021] | 6 | Levin | residual | 1.44e-16 | +7.2e-17 |
| 48pi | 9 | left | [0.695205, 0.706240] | 6 | Levin | floor(phase_limited) | 2.10e-18 | +3.2e-17 |
| 48pi | 9 | right | [0.740973, 0.741479] | 9 | direct | floor | 4.05e-15 | -1.4e-17 |
| 48pi | 12 | left | [0.695205, 0.706240] | 6 | Levin | floor(phase_limited) | 2.10e-18 | +3.2e-17 |
| 48pi | 12 | right | [0.740973, 0.741479] | 9 | direct | floor(phase_limited) | 4.05e-15 | -1.4e-17 |
| 64pi | 6 | left | [0.692561, 0.703554] | 6 | Levin | residual | 1.14e-16 | +5.6e-17 |
| 64pi | 6 | right | [0.743660, 0.759681] | 4 | Levin | floor(phase_limited) | 2.95e-18 | +6.9e-17 |
| 64pi | 9 | left | [0.698057, 0.703554] | 7 | Levin | floor(phase_limited) | 1.36e-18 | -5.7e-17 |
| 64pi | 9 | right | [0.743660, 0.759681] | 4 | Levin | floor(phase_limited) | 2.95e-18 | +6.9e-17 |
| 64pi | 12 | left | [0.698057, 0.703554] | 7 | Levin | floor(phase_limited) | 1.36e-18 | -5.7e-17 |
| 64pi | 12 | right | [0.743660, 0.759681] | 4 | Levin | floor(phase_limited) | 2.95e-18 | +6.9e-17 |
| 128pi | 6 | left | [0.684384, 0.695248] | 6 | Levin | residual | 1.44e-16 | +7.2e-17 |
| 128pi | 6 | right | [0.751966, 0.759717] | 5 | Levin | floor(phase_limited) | 1.43e-18 | -5.9e-17 |
| 128pi | 9 | left | [0.693890, 0.695248] | 9 | Levin | residual | 2.21e-17 | -1.9e-19 |
| 128pi | 9 | right | [0.751966, 0.759717] | 5 | Levin | floor(phase_limited) | 1.43e-18 | -5.9e-17 |
| 128pi | 12 | left | [0.695078, 0.695248] | 12 | direct | floor | 1.86e-15 | +4.0e-17 |
| 128pi | 12 | right | [0.751966, 0.759717] | 5 | Levin | floor(phase_limited) | 1.43e-18 | -5.9e-17 |
| 512pi | 6 | left | [0.646048, 0.666888] | 5 | Levin | floor(phase_limited) | 2.57e-18 | +3.7e-17 |
| 512pi | 6 | right | [0.780325, 0.787190] | 5 | Levin | floor(phase_limited) | 9.96e-19 | +3.3e-17 |
| 512pi | 9 | left | [0.646048, 0.666888] | 5 | Levin | floor(phase_limited) | 2.57e-18 | +3.7e-17 |
| 512pi | 9 | right | [0.780325, 0.787190] | 5 | Levin | floor(phase_limited) | 9.96e-19 | +3.3e-17 |
| 512pi | 12 | left | [0.646048, 0.666888] | 5 | Levin | floor(phase_limited) | 2.57e-18 | +3.7e-17 |
| 512pi | 12 | right | [0.780325, 0.787190] | 5 | Levin | floor(phase_limited) | 9.96e-19 | +3.3e-17 |

### The same runs with an exact theta_mod_2pi supplied (`q12_c1e6w0_thetamod.json`)

| Phi_w | d_max | abserr raw / mod | true raw / mod | window declared raw / mod | outside declared raw / mod |
|---|---|---|---|---|---|
| 2pi | 6 | 1.672e-11 / 1.672e-11 | +1.68e-11 / +1.68e-11 | 5.416e-15 / 5.416e-15 | 1.672e-11 / 1.672e-11 |
| 2pi | 9 | 4.552e-14 / 4.550e-14 | +1.66e-16 / +1.75e-16 | 5.338e-15 / 5.338e-15 | 4.018e-14 / 4.016e-14 |
| 2pi | 12 | 8.570e-11 / 8.245e-11 | -9.34e-16 / -9.14e-16 | 5.328e-15 / 5.328e-15 | 8.569e-11 / 8.244e-11 |
| 8pi | 6 | 4.637e-14 / 4.635e-14 | -2.26e-14 / -2.26e-14 | 2.138e-14 / 2.137e-14 | 2.499e-14 / 2.498e-14 |
| 8pi | 9 | 9.837e-10 / 9.837e-10 | +5.10e-16 / +4.98e-16 | 2.128e-14 / 2.127e-14 | 9.837e-10 / 9.837e-10 |
| 8pi | 12 | 5.163e-10 / 5.163e-10 | +5.23e-16 / +5.14e-16 | 2.126e-14 / 2.125e-14 | 5.163e-10 / 5.163e-10 |

### Decomposition, c1e6w0 (`q3_c1e6w0.json`)

Levin leaves: the step-(4) residual V_P - V_L - V_R split into resolution / node / endpoint-theta / other (each the parent's component minus the halves'); 'ends' lists which of P, L, R has a computed end node off its span end (offset in units of 1e-16). CC (direct) leaves: the nested pair split into CC_31's and CC_16's errors, each resolution + node + argument + other; and the summed nested pair, floor and true error after 1, 2, 3 further bisections.

| config | span | dtheta_near | theta' | kind | declared basis | residual / nested | floor | split | theta_mod_2pi: residual | true error of P (V_P - exact) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2pi_d6 | [0.7088109, 0.7200619] | 6.28 | -3.5e+03 | Levin d6 | Levin | 1.67e-11 | 7.49e-18 | res +1.7e-11, node -9.8e-18, end-theta -2.3e-27, other -1.8e-19; ends P(+0.00/-1.11) R(+0.00/-1.11) | 1.67e-11 | +1.7e-11 |
| 2pi_d6 | [0.7271517, 0.7314150] | 6.28 | 3.5e+03 | Levin d6 | Levin | 1.40e-15 | 5.98e-18 | res +1.3e-15, node +1.0e-16, end-theta +9.8e-20, other -7.4e-19; ends P(+1.11/+0.00) L=CC R=CC | 1.40e-15 | +1.4e-15 |
| 2pi_d6 | [0.7314150, 0.7356782] | 30.5 | 7.8e+03 | Levin d6 | Levin | 1.68e-16 | 2.92e-18 | res -5.3e-19, node +1.7e-16, end-theta +2.0e-19, other +3.6e-20; ends L(+0.00/+1.11) R(-1.11/+0.00) | 1.68e-16 | +8.9e-19 |
| 2pi_d6 | [0.0000000, 0.3600309] | 6.61e+04 | -3.6e+05 | Levin d1 | Levin | 7.26e-17 | 1.27e-16 | res -4.5e-22, node -1.9e-17, end-theta -5.4e-17, other -1.5e-21; ends R(+0.28/+0.00) | 1.88e-17 | -1.5e-17 |
| 2pi_d9 | [0.6961536, 0.6975600] | 339 | -2.6e+04 | Levin d9 | Levin | 2.77e-16 | 7.94e-19 | res +1.2e-21, node +2.8e-16, end-theta +1.2e-18, other -1.7e-19; ends P(+1.11/+0.00) L(-1.11/+0.00) R=CC | 2.75e-16 | +6.3e-17 |
| 2pi_d9 | [0.7003727, 0.7017791] | 238 | -2.2e+04 | Levin d9 | Levin | 1.50e-16 | 9.37e-19 | res +4.7e-24, node -1.5e-16, end-theta +2.5e-18, other -2.4e-20; ends L=CC R=CC | 1.52e-16 | +4.9e-19 |
| 2pi_d9 | [0.6905281, 0.6919345] | 502 | -3.2e+04 | Levin d9 | Levin | 1.49e-16 | 6.63e-19 | res +1.9e-35, node +1.5e-16, end-theta +3.4e-19, other +1.2e-19; ends P(+0.00/-1.11) R(+0.00/+1.11) | 1.49e-16 | +7.4e-17 |
| 2pi_d9 | [0.7172491, 0.7186555] | 12.3 | -5e+03 | CC d9 | direct | 3.31e-15 | 3.72e-15 | CC31 err +1.7e-17 (res +6e-31, node +1.7e-17, arg -1.4e-19); CC16 err +3.3e-15 (res +3.3e-15); refine +1: nested 1.6e-17, floor 3.5e-15, true +6.0e-17; +2: nested 1.6e-17, floor 3.4e-15, true -6.4e-17; +3: nested 1.2e-17, floor 3.4e-15, true +1.3e-17 | -- | +1.7e-17 |
| 2pi_d9 | [0.7186555, 0.7200619] | 6.28 | -3.5e+03 | CC d9 | direct | 5.01e-16 | 2.91e-15 | CC31 err +3.8e-17 (res -7e-33, node +3.8e-17, arg +3.3e-20); CC16 err +5.4e-16 (res +5.0e-16); refine +1: nested 4.8e-18, floor 2.7e-15, true -6.7e-17; +2: nested 1.1e-17, floor 2.6e-15, true +7.7e-17; +3: nested 8.1e-18, floor 2.5e-15, true +1.1e-17 | -- | +3.8e-17 |
| 2pi_d9 | [0.7292833, 0.7303491] | 16.1 | 5.7e+03 | CC d8 | direct | 8.78e-17 | 3.10e-15 | CC31 err -1.5e-17 (res +8e-34, node -1.5e-17, arg -2.5e-19); CC16 err +7.3e-17 (res +1.0e-16); refine +1: nested 2.6e-17, floor 3.0e-15, true -4.7e-17; +2: nested 1.2e-17, floor 2.9e-15, true +1.1e-16; +3: nested 3.7e-18, floor 2.9e-15, true -5.1e-17 | -- | -1.5e-17 |
| 2pi_d9 | [0.7271517, 0.7282175] | 6.28 | 3.5e+03 | CC d8 | direct | 1.51e-17 | 2.11e-15 | CC31 err -8.2e-18 (res -6e-37, node -8.2e-18, arg -2.6e-20); CC16 err +6.9e-18 (res -5.5e-19); refine +1: nested 1.2e-17, floor 2.0e-15, true +7.7e-17; +2: nested 4.9e-18, floor 1.9e-15, true -2.6e-18; +3: nested 1.4e-18, floor 1.9e-15, true -5.4e-18 | -- | -8.2e-18 |
| 2pi_d9 | [0.0000000, 0.3600309] | 6.61e+04 | -3.6e+05 | Levin d1 | Levin | 7.26e-17 | 1.27e-16 | res -4.5e-22, node -1.9e-17, end-theta -5.4e-17, other -1.5e-21; ends R(+0.28/+0.00) | 1.88e-17 | -1.5e-17 |
| 2pi_d12 | [0.0000000, 0.3600309] | 6.61e+04 | -3.6e+05 | Levin d1 | Levin | 7.26e-17 | 1.27e-16 | res -4.5e-22, node -1.9e-17, end-theta -5.4e-17, other -1.5e-21; ends R(+0.28/+0.00) | 1.88e-17 | -1.5e-17 |
| 2pi_d12 | [0.7144364, 0.7172491] | 20.2 | -6.4e+03 | Levin d8 | Levin | 2.98e-18 | 3.21e-18 | res -4.8e-21, node +2.2e-18, end-theta +5.6e-19, other +2.1e-19; ends L=CC R=CC | 2.44e-18 | +1.3e-18 |
| 2pi_d12 | [0.7356782, 0.7442047] | 72.9 | 1.2e+04 | Levin d5 | Levin | 8.54e-19 | 2.54e-18 | res +1.1e-18, node -1.3e-19, end-theta -4.2e-33, other -9.1e-20; ends all exact | 8.61e-19 | +1.5e-18 |
| 2pi_d12 | [0.6413051, 0.6414809] | 3.37e+03 | -8.2e+04 | CC d12 | direct | 4.17e-12 | 4.56e-15 | CC31 err +2.5e-17 (res +1e-25, node +2.7e-17, arg -2.3e-18); CC16 err +4.2e-12 (res +4.2e-12); refine +1: nested 5.6e-17, floor 4.6e-15, true +9.0e-17; +2: nested 5.2e-17, floor 4.6e-15, true -2.3e-16; +3: nested 2.0e-17, floor 4.5e-15, true +1.3e-16 | -- | +2.5e-17 |
| 2pi_d12 | [0.6416567, 0.6418325] | 3.34e+03 | -8.2e+04 | CC d12 | direct | 3.68e-12 | 4.54e-15 | CC31 err +2.0e-17 (res -9e-26, node +2.1e-17, arg -8.8e-19); CC16 err -3.7e-12 (res -3.7e-12); refine +1: nested 7.2e-17, floor 4.5e-15, true -1.2e-18; +2: nested 5.7e-17, floor 4.5e-15, true +2.3e-17; +3: nested 2.5e-17, floor 4.5e-15, true -2.5e-17 | -- | +2.0e-17 |
| 2pi_d12 | [0.6425357, 0.6427115] | 3.27e+03 | -8.1e+04 | CC d12 | direct | 3.29e-12 | 4.51e-15 | CC31 err -2.5e-17 (res +7e-26, node -2.6e-17, arg +1.8e-18); CC16 err +3.3e-12 (res +3.3e-12); refine +1: nested 5.8e-17, floor 4.5e-15, true +8.3e-18; +2: nested 5.0e-17, floor 4.5e-15, true -3.2e-17; +3: nested 2.5e-17, floor 4.5e-15, true +3.6e-17 | -- | -2.5e-17 |
| 2pi_d12 | [0.7186555, 0.7200619] | 6.28 | -3.5e+03 | CC d9 | direct | 5.01e-16 | 2.91e-15 | CC31 err +3.8e-17 (res -7e-33, node +3.8e-17, arg +3.3e-20); CC16 err +5.4e-16 (res +5.0e-16); refine +1: nested 4.8e-18, floor 2.7e-15, true -6.7e-17; +2: nested 1.1e-17, floor 2.6e-15, true +7.7e-17; +3: nested 8.1e-18, floor 2.5e-15, true +1.1e-17 | -- | +3.8e-17 |
| 2pi_d12 | [0.7271517, 0.7282175] | 6.28 | 3.5e+03 | CC d8 | direct | 1.51e-17 | 2.11e-15 | CC31 err -8.2e-18 (res -6e-37, node -8.2e-18, arg -2.6e-20); CC16 err +6.9e-18 (res -5.5e-19); refine +1: nested 1.2e-17, floor 2.0e-15, true +7.7e-17; +2: nested 4.9e-18, floor 1.9e-15, true -2.6e-18; +3: nested 1.4e-18, floor 1.9e-15, true -5.4e-18 | -- | -8.2e-18 |
| 8pi_d9 | [0.6871286, 0.6885280] | 615 | -3.5e+04 | Levin d9 | Levin | 1.49e-16 | 6.03e-19 | res +1.5e-36, node +1.5e-16, end-theta -2.7e-18, other +7.9e-20; ends P(+1.11/+0.00) L(-1.11/+0.00) | 1.52e-16 | +7.4e-17 |
| 8pi_d9 | [0.6857291, 0.6871286] | 665 | -3.6e+04 | Levin d9 | Levin | 1.49e-16 | 5.82e-19 | res +1.0e-36, node +1.5e-16, end-theta -2.7e-18, other -1.4e-19; ends P(+0.00/-1.11) R(+0.00/+1.11) | 1.52e-16 | +7.5e-17 |
| 8pi_d9 | [0.5849689, 0.5863684] | 9.42e+03 | -1.4e+05 | Levin d9 | Levin | 1.32e-16 | 1.92e-19 | res +2.9e-48, node +1.3e-16, end-theta +1.9e-18, other +5.0e-21; ends P(+0.00/-1.11) R(+0.00/+1.11) | 1.30e-16 | +6.5e-17 |
| 8pi_d9 | [0.7095197, 0.7109192] | 80.5 | -1.3e+04 | CC d9 | direct | 5.63e-10 | 7.99e-15 | CC31 err -7.9e-17 (res +5e-21, node -7.8e-17, arg -3.2e-19); CC16 err -5.6e-10 (res -5.6e-10); refine +1: nested 7.0e-14, floor 7.8e-15, true -5.5e-18; +2: nested 6.3e-17, floor 7.7e-15, true +1.2e-17; +3: nested 3.3e-17, floor 7.6e-15, true +1.1e-17 | -- | -7.9e-17 |
| 8pi_d9 | [0.7109192, 0.7123186] | 63.7 | -1.1e+04 | CC d9 | direct | 3.15e-10 | 7.23e-15 | CC31 err -1.3e-16 (res +2e-22, node -1.3e-16, arg -6.6e-19); CC16 err -3.2e-10 (res -3.2e-10); refine +1: nested 7.9e-15, floor 7.0e-15, true -4.0e-17; +2: nested 6.7e-17, floor 6.9e-15, true +4.4e-17; +3: nested 2.4e-17, floor 6.9e-15, true +5.0e-17 | -- | -1.3e-16 |
| 8pi_d9 | [0.7123186, 0.7137181] | 48.9 | -9.9e+03 | CC d9 | direct | 5.40e-11 | 6.46e-15 | CC31 err +1.2e-17 (res -6e-24, node +1.2e-17, arg -1.3e-19); CC16 err +5.4e-11 (res +5.4e-11); refine +1: nested 1.8e-15, floor 6.3e-15, true +8.2e-17; +2: nested 4.2e-17, floor 6.1e-15, true +9.6e-17; +3: nested 2.3e-17, floor 6.1e-15, true -1.3e-16 | -- | +1.2e-17 |
| 8pi_d9 | [0.7151175, 0.7165170] | 25.1 | -7.1e+03 | CC d9 | direct | 3.08e-13 | 4.91e-15 | CC31 err +1.7e-17 (res +2e-27, node +1.6e-17, arg +2.9e-19); CC16 err -3.1e-13 (res -3.1e-13); refine +1: nested 2.6e-17, floor 4.7e-15, true +3.2e-17; +2: nested 2.2e-17, floor 4.6e-15, true +6.0e-17; +3: nested 1.1e-17, floor 4.5e-15, true -4.8e-18 | -- | +1.7e-17 |
| 8pi_d9 | [0.7306966, 0.7312226] | 25.1 | 7.1e+03 | CC d9 | direct | 2.57e-17 | 1.74e-15 | CC31 err +3.8e-17 (res -2e-42, node +3.8e-17, arg +1.7e-20); CC16 err +6.4e-17 (res +3.4e-20); refine +1: nested 1.2e-17, floor 1.7e-15, true +6.6e-17; +2: nested 4.7e-18, floor 1.7e-15, true +3.6e-17; +3: nested 1.2e-18, floor 1.7e-15, true -1.4e-17 | -- | +3.8e-17 |
| 8pi_d9 | [0.0000000, 0.3582585] | 6.67e+04 | -3.7e+05 | Levin d1 | Levin | 1.69e-21 | 1.26e-16 | res +1.2e-21, node -2.9e-17, end-theta +2.9e-17, other +4.9e-22; ends R(-0.28/+0.00) | 2.91e-17 | -7.9e-18 |
| 8pi_d12 | [0.5851439, 0.5853188] | 9.56e+03 | -1.4e+05 | Levin d12 | Levin | 1.52e-16 | 1.19e-19 | res -3.7e-28, node +1.5e-16, end-theta +4.2e-18, other -1.7e-20; ends P(+0.00/-1.11) L=CC R=CC | 1.48e-16 | +6.1e-17 |
| 8pi_d12 | [0.5767472, 0.5769221] | 1.08e+04 | -1.5e+05 | Levin d12 | Levin | 1.47e-16 | 1.12e-19 | res +2.6e-27, node -1.5e-16, end-theta +4.5e-18, other +3.0e-20; ends P(+0.00/-1.11) L=CC R=CC | 1.51e-16 | -6.6e-17 |
| 8pi_d12 | [0.5853188, 0.5854937] | 9.54e+03 | -1.4e+05 | Levin d12 | Levin | 1.39e-16 | 1.19e-19 | res -3.5e-28, node +1.4e-16, end-theta -5.4e-18, other +5.9e-21; ends P(+1.11/+0.00) L=CC R=CC | 1.45e-16 | +6.1e-17 |
| 8pi_d12 | [0.6187306, 0.6189055] | 5.48e+03 | -1e+05 | CC d12 | direct | 9.92e-11 | 5.26e-15 | CC31 err -4.2e-17 (res +2e-22, node -4.9e-17, arg +7.4e-18); CC16 err +9.9e-11 (res +9.9e-11); refine +1: nested 6.5e-16, floor 5.3e-15, true -4.9e-17; +2: nested 1.3e-16, floor 5.3e-15, true +5.5e-17; +3: nested 5.2e-17, floor 5.3e-15, true -1.1e-16 | -- | -4.2e-17 |
| 8pi_d12 | [0.6189055, 0.6190805] | 5.46e+03 | -1e+05 | CC d12 | direct | 8.77e-11 | 5.26e-15 | CC31 err -5.7e-17 (res +2e-22, node -5.1e-17, arg -5.9e-18); CC16 err +8.8e-11 (res +8.8e-11); refine +1: nested 2.1e-15, floor 5.3e-15, true -4.3e-17; +2: nested 1.3e-16, floor 5.3e-15, true +6.2e-17; +3: nested 5.2e-17, floor 5.2e-15, true -6.2e-17 | -- | -5.7e-17 |
| 8pi_d12 | [0.6185557, 0.6187306] | 5.5e+03 | -1e+05 | CC d12 | direct | 8.47e-11 | 5.27e-15 | CC31 err +9.4e-17 (res +2e-22, node +9.1e-17, arg +2.9e-18); CC16 err +8.5e-11 (res +8.5e-11); refine +1: nested 2.8e-15, floor 5.3e-15, true +1.2e-16; +2: nested 9.9e-17, floor 5.3e-15, true -1.1e-16; +3: nested 5.3e-17, floor 5.3e-15, true +2.3e-16 | -- | +9.4e-17 |
| 8pi_d12 | [0.7158173, 0.7165170] | 25.1 | -7.1e+03 | CC d10 | direct | 4.74e-18 | 2.25e-15 | CC31 err +1.1e-17 (res -8e-38, node +1.1e-17, arg +3.5e-20); CC16 err +5.9e-18 (res +1.2e-18); refine +1: nested 5.8e-18, floor 2.2e-15, true -6.8e-17; +2: nested 5.3e-18, floor 2.2e-15, true +4.6e-17; +3: nested 4.4e-18, floor 2.2e-15, true +5.6e-18 | -- | +1.1e-17 |
| 8pi_d12 | [0.7306966, 0.7312226] | 25.1 | 7.1e+03 | CC d9 | direct | 2.57e-17 | 1.74e-15 | CC31 err +3.8e-17 (res -2e-42, node +3.8e-17, arg +1.7e-20); CC16 err +6.4e-17 (res +3.4e-20); refine +1: nested 1.2e-17, floor 1.7e-15, true +6.6e-17; +2: nested 4.7e-18, floor 1.7e-15, true +3.6e-17; +3: nested 1.2e-18, floor 1.7e-15, true -1.4e-17 | -- | +3.8e-17 |
| 8pi_d12 | [0.0000000, 0.3582585] | 6.67e+04 | -3.7e+05 | Levin d1 | Levin | 1.69e-21 | 1.26e-16 | res +1.2e-21, node -2.9e-17, end-theta +2.9e-17, other +4.9e-22; ends R(-0.28/+0.00) | 2.91e-17 | -7.9e-18 |
| 128pi_d9 | [0.6558683, 0.6572262] | 2.2e+03 | -6.6e+04 | Levin d9 | Levin | 1.40e-16 | 3.39e-19 | res -1.4e-42, node -1.4e-16, end-theta +3.3e-18, other +1.0e-21; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.44e-16 | -8.2e-19 |
| 128pi_d9 | [0.6680894, 0.6694473] | 1.47e+03 | -5.4e+04 | Levin d9 | Levin | 1.39e-16 | 4.06e-19 | res +6.2e-40, node +1.4e-16, end-theta -1.4e-18, other +8.8e-20; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.40e-16 | +1.4e-19 |
| 128pi_d9 | [0.6341418, 0.6354997] | 3.88e+03 | -8.8e+04 | Levin d9 | Levin | 1.35e-16 | 2.67e-19 | res -8.4e-44, node +1.4e-16, end-theta -1.1e-18, other +4.5e-20; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.36e-16 | +3.1e-18 |
| 128pi_d9 | [0.6938896, 0.6952475] | 402 | -2.8e+04 | Levin d9 | Levin | 2.21e-17 | 7.30e-19 | res +1.0e-35, node -2.2e-17, end-theta -1.1e-19, other -2.6e-21; ends L(+0.00/-1.11) R(+1.11/+0.00) | 2.20e-17 | -1.9e-19 |
| 128pi_d9 | [0.7519661, 0.7597171] | 402 | 2.8e+04 | Levin d5 | Levin | 1.08e-19 | 1.43e-18 | res +1.3e-24, node -9.1e-20, end-theta +3.4e-33, other -1.4e-20; ends P(+0.00/+1.11) R(+0.00/+1.11) | 1.08e-19 | -5.9e-17 |
| 128pi_d9 | [0.0000000, 0.3476238] | 7.07e+04 | -3.8e+05 | Levin d1 | Levin | 1.06e-21 | 1.19e-16 | res -1.0e-21, node +2.1e-26, end-theta -2.7e-33, other -5.5e-23; ends all exact | 1.06e-21 | -1.6e-17 |

History of every Levin leaf made outside the window (all configs pooled): median and max of the step-(4) residual, the eq. (151) floor, and the true errors of V_P and V_L + V_R, by depth; and the share of leaves whose residual exceeds 10x the floor.

| config | depth | leaves | median residual | max residual | median floor | median abs(V_P err) | median abs(V_L+V_R err) | residual > 10 floor |
|---|---|---|---|---|---|---|---|---|
| 2pi_d6 | 0 | 1 | 3.1e-06 | 3.1e-06 | 2.1e-14 | 7.6e-06 | 4.6e-06 | 1/1 |
| 2pi_d6 | 1 | 2 | 1.4e-06 | 2.7e-06 | 1.6e-15 | 2.3e-06 | 9.0e-07 | 1/2 |
| 2pi_d6 | 2 | 3 | 1.5e-06 | 2.3e-06 | 7.0e-16 | 1.8e-06 | 3.1e-07 | 2/3 |
| 2pi_d6 | 3 | 4 | 1.6e-07 | 9.5e-07 | 1.1e-16 | 1.6e-07 | 7.7e-09 | 2/4 |
| 2pi_d6 | 4 | 6 | 2.0e-16 | 1.0e-07 | 1.5e-17 | 1.0e-16 | 1.0e-16 | 4/6 |
| 2pi_d6 | 5 | 10 | 4.8e-17 | 1.4e-08 | 5.9e-18 | 3.4e-17 | 7.2e-17 | 6/10 |
| 2pi_d6 | 6 | 16 | 1.1e-20 | 1.3e-10 | 2.7e-18 | 1.4e-17 | 3.5e-17 | 6/16 |
| 2pi_d6 | 7 | 2 | 1.4e-12 | 2.9e-12 | 4.7e-18 | 1.4e-12 | 6.9e-16 | 1/2 |
| 2pi_d6 | 8 | 2 | 7.8e-16 | 1.4e-15 | 4.4e-18 | 6.9e-16 | 9.8e-17 | 2/2 |
| 2pi_d9 | 0 | 1 | 3.1e-06 | 3.1e-06 | 2.1e-14 | 7.6e-06 | 4.6e-06 | 1/1 |
| 2pi_d9 | 1 | 2 | 1.4e-06 | 2.7e-06 | 1.6e-15 | 2.3e-06 | 9.0e-07 | 1/2 |
| 2pi_d9 | 2 | 3 | 1.5e-06 | 2.3e-06 | 7.0e-16 | 1.8e-06 | 3.1e-07 | 2/3 |
| 2pi_d9 | 3 | 4 | 1.6e-07 | 9.5e-07 | 1.1e-16 | 1.6e-07 | 7.7e-09 | 2/4 |
| 2pi_d9 | 4 | 6 | 2.0e-16 | 1.0e-07 | 1.5e-17 | 1.0e-16 | 1.0e-16 | 4/6 |
| 2pi_d9 | 5 | 10 | 4.8e-17 | 1.4e-08 | 5.9e-18 | 3.4e-17 | 7.2e-17 | 6/10 |
| 2pi_d9 | 6 | 16 | 1.1e-20 | 1.3e-10 | 2.7e-18 | 1.4e-17 | 3.5e-17 | 6/16 |
| 2pi_d9 | 7 | 12 | 9.8e-17 | 2.9e-12 | 1.2e-18 | 4.8e-17 | 4.3e-17 | 10/12 |
| 2pi_d9 | 8 | 19 | 1.2e-16 | 1.4e-15 | 7.7e-19 | 1.5e-18 | 7.5e-17 | 18/19 |
| 2pi_d9 | 9 | 34 | 1.0e-16 | 2.8e-16 | 6.0e-19 | 5.0e-17 | 6.4e-17 | 32/34 |
| 2pi_d12 | 0 | 1 | 3.1e-06 | 3.1e-06 | 2.1e-14 | 7.6e-06 | 4.6e-06 | 1/1 |
| 2pi_d12 | 1 | 2 | 1.4e-06 | 2.7e-06 | 1.6e-15 | 2.3e-06 | 9.0e-07 | 1/2 |
| 2pi_d12 | 2 | 3 | 1.5e-06 | 2.3e-06 | 7.0e-16 | 1.8e-06 | 3.1e-07 | 2/3 |
| 2pi_d12 | 3 | 4 | 1.6e-07 | 9.5e-07 | 1.1e-16 | 1.6e-07 | 7.7e-09 | 2/4 |
| 2pi_d12 | 4 | 6 | 2.0e-16 | 1.0e-07 | 1.5e-17 | 1.0e-16 | 1.0e-16 | 4/6 |
| 2pi_d12 | 5 | 10 | 4.8e-17 | 1.4e-08 | 5.9e-18 | 3.4e-17 | 7.2e-17 | 6/10 |
| 2pi_d12 | 6 | 16 | 1.1e-20 | 1.3e-10 | 2.7e-18 | 1.4e-17 | 3.5e-17 | 6/16 |
| 2pi_d12 | 7 | 12 | 9.8e-17 | 2.9e-12 | 1.2e-18 | 4.8e-17 | 4.3e-17 | 10/12 |
| 2pi_d12 | 8 | 19 | 1.2e-16 | 1.4e-15 | 7.7e-19 | 1.5e-18 | 7.5e-17 | 18/19 |
| 2pi_d12 | 9 | 34 | 1.0e-16 | 2.8e-16 | 6.0e-19 | 5.0e-17 | 6.4e-17 | 32/34 |
| 2pi_d12 | 10 | 47 | 1.0e-16 | 1.4e-16 | 3.0e-19 | 3.7e-17 | 5.4e-17 | 46/47 |
| 2pi_d12 | 11 | 64 | 5.1e-17 | 1.0e-16 | 2.6e-19 | 4.3e-17 | 2.5e-17 | 62/64 |
| 8pi_d9 | 0 | 1 | 3.0e-07 | 3.0e-07 | 1.0e-14 | 5.3e-07 | 2.3e-07 | 1/1 |
| 8pi_d9 | 1 | 2 | 8.4e-08 | 1.7e-07 | 8.5e-16 | 1.1e-07 | 3.0e-08 | 1/2 |
| 8pi_d9 | 2 | 3 | 5.3e-08 | 1.1e-07 | 3.5e-16 | 6.1e-08 | 7.9e-09 | 2/3 |
| 8pi_d9 | 3 | 4 | 3.8e-09 | 2.7e-08 | 6.4e-17 | 4.0e-09 | 1.6e-10 | 2/4 |
| 8pi_d9 | 4 | 8 | 5.2e-17 | 2.5e-09 | 1.3e-17 | 5.4e-17 | 5.5e-17 | 4/8 |
| 8pi_d9 | 5 | 8 | 1.0e-16 | 4.6e-11 | 3.0e-18 | 2.7e-17 | 5.2e-17 | 7/8 |
| 8pi_d9 | 6 | 14 | 1.1e-16 | 6.3e-13 | 1.1e-18 | 2.3e-17 | 6.0e-17 | 14/14 |
| 8pi_d9 | 7 | 28 | 1.0e-16 | 3.0e-16 | 5.3e-19 | 2.9e-17 | 5.7e-17 | 28/28 |
| 8pi_d9 | 8 | 56 | 5.6e-20 | 1.8e-16 | 3.2e-19 | 2.3e-17 | 5.9e-17 | 19/56 |
| 8pi_d9 | 9 | 34 | 1.1e-16 | 1.8e-16 | 2.4e-19 | 5.6e-17 | 5.4e-17 | 34/34 |
| 8pi_d9 | 10 | 8 | 1.6e-16 | 1.8e-16 | 7.4e-19 | 4.4e-17 | 8.7e-17 | 8/8 |
| 8pi_d9 | 11 | 8 | 7.0e-17 | 1.1e-16 | 5.8e-19 | 7.3e-17 | 4.5e-17 | 8/8 |
| 8pi_d12 | 0 | 1 | 3.0e-07 | 3.0e-07 | 1.0e-14 | 5.3e-07 | 2.3e-07 | 1/1 |
| 8pi_d12 | 1 | 2 | 8.4e-08 | 1.7e-07 | 8.5e-16 | 1.1e-07 | 3.0e-08 | 1/2 |
| 8pi_d12 | 2 | 3 | 5.3e-08 | 1.1e-07 | 3.5e-16 | 6.1e-08 | 7.9e-09 | 2/3 |
| 8pi_d12 | 3 | 4 | 3.8e-09 | 2.7e-08 | 6.4e-17 | 4.0e-09 | 1.6e-10 | 2/4 |
| 8pi_d12 | 4 | 8 | 5.2e-17 | 2.5e-09 | 1.3e-17 | 5.4e-17 | 5.5e-17 | 4/8 |
| 8pi_d12 | 5 | 8 | 1.0e-16 | 4.6e-11 | 3.0e-18 | 2.7e-17 | 5.2e-17 | 7/8 |
| 8pi_d12 | 6 | 14 | 1.1e-16 | 6.3e-13 | 1.1e-18 | 2.3e-17 | 6.0e-17 | 14/14 |
| 8pi_d12 | 7 | 28 | 1.0e-16 | 3.0e-16 | 5.3e-19 | 2.9e-17 | 5.7e-17 | 28/28 |
| 8pi_d12 | 8 | 56 | 5.6e-20 | 1.8e-16 | 3.2e-19 | 2.3e-17 | 5.9e-17 | 19/56 |
| 8pi_d12 | 9 | 34 | 1.1e-16 | 1.8e-16 | 2.4e-19 | 5.6e-17 | 5.4e-17 | 34/34 |
| 8pi_d12 | 10 | 56 | 5.6e-17 | 1.8e-16 | 1.7e-19 | 4.4e-18 | 6.1e-17 | 38/56 |
| 8pi_d12 | 11 | 44 | 8.2e-17 | 1.5e-16 | 1.4e-19 | 5.7e-17 | 4.9e-17 | 44/44 |
| 8pi_d12 | 12 | 64 | 4.5e-17 | 1.5e-16 | 1.2e-19 | 6.7e-18 | 4.9e-17 | 63/64 |
| 128pi_d9 | 1 | 1 | 6.4e-10 | 6.4e-10 | 2.6e-15 | 7.3e-10 | 8.9e-11 | 1/1 |
| 128pi_d9 | 2 | 3 | 2.3e-11 | 8.5e-11 | 2.7e-16 | 2.4e-11 | 6.8e-13 | 2/3 |
| 128pi_d9 | 3 | 4 | 3.4e-13 | 4.3e-12 | 5.3e-17 | 3.4e-13 | 3.0e-15 | 2/4 |
| 128pi_d9 | 4 | 4 | 2.9e-15 | 6.4e-14 | 1.7e-17 | 3.1e-15 | 1.1e-16 | 2/4 |
| 128pi_d9 | 5 | 4 | 1.8e-16 | 2.2e-16 | 6.3e-18 | 9.5e-17 | 7.6e-17 | 4/4 |
| 128pi_d9 | 6 | 8 | 1.2e-16 | 1.8e-16 | 2.5e-18 | 2.0e-18 | 8.9e-17 | 5/8 |
| 128pi_d9 | 7 | 14 | 6.1e-18 | 1.7e-16 | 1.2e-18 | 5.8e-17 | 6.0e-17 | 6/14 |
| 128pi_d9 | 8 | 16 | 9.6e-17 | 1.5e-16 | 6.8e-19 | 3.1e-17 | 7.2e-17 | 10/16 |
| 128pi_d9 | 9 | 24 | 1.2e-16 | 1.5e-16 | 4.7e-19 | 5.0e-17 | 6.1e-17 | 24/24 |
| 128pi_d9 | 10 | 48 | 9.0e-20 | 1.4e-16 | 3.7e-19 | 2.5e-17 | 6.8e-17 | 18/48 |

### Inside the window, c1e6w0

Whole window: the sums over its leaves. Then sampled depth-0 pieces, summed over the leaves inside each at the two depths; then three single leaves with the fine rule's true error split in mpmath.

| Phi_w | d_max | leaves | nested pair | C1 | C2 | C3 | declared | true (signed sum) | abs true (sum) |
|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | 192 | 1.88e-18 | 1.82e-17 | 1.73e-18 | 5.40e-15 | 5.42e-15 | +1.8e-18 | 1.01e-16 |
| 2pi | 9 | 1536 | 2.69e-19 | 1.82e-17 | 1.73e-18 | 5.32e-15 | 5.34e-15 | -2.2e-19 | 1.32e-16 |
| 8pi | 6 | 576 | 4.91e-18 | 3.66e-17 | 1.38e-17 | 2.13e-14 | 2.14e-14 | +8.3e-19 | 5.42e-16 |
| 8pi | 9 | 4608 | 9.79e-19 | 3.66e-17 | 1.38e-17 | 2.12e-14 | 2.13e-14 | -6.6e-20 | 5.87e-16 |
| 32pi | 6 | 2112 | 1.82e-17 | 7.34e-17 | 1.11e-16 | 8.50e-14 | 8.51e-14 | +1.6e-17 | 2.80e-15 |
| 32pi | 9 | 16896 | 5.88e-18 | 7.34e-17 | 1.11e-16 | 8.48e-14 | 8.50e-14 | -8.4e-19 | 1.81e-15 |
| 128pi | 6 | 8256 | 8.89e-17 | 1.47e-16 | 8.85e-16 | 3.39e-13 | 3.40e-13 | -1.6e-17 | 9.98e-15 |

| Phi_w | d_max | piece | dtheta at ends | leaves | nested pair | C1 | C2 | C3 | declared | true (signed sum) | abs true (sum) | max abs true per leaf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | 0 | 6.28 / 2.09 | 64 | 3.2e-19 | 3.9e-18 | 6.9e-19 | 1.76e-15 | 1.76e-15 | -1.4e-18 | 2.5e-17 | 2.7e-18 |
| 2pi | 6 | 1 | 2.09 / 4.19 | 64 | 1.5e-18 | 1.3e-17 | 6.4e-19 | 2.74e-15 | 2.76e-15 | +3.1e-18 | 3.9e-17 | 7.1e-18 |
| 2pi | 6 | 2 | 4.19 / 6.28 | 64 | 6.1e-20 | 1.7e-18 | 4.0e-19 | 8.95e-16 | 8.97e-16 | +7.7e-20 | 3.6e-17 | 1.2e-18 |
| 2pi | 9 | 0 | 6.28 / 2.09 | 512 | 7.7e-20 | 3.9e-18 | 6.9e-19 | 1.75e-15 | 1.76e-15 | +1.8e-19 | 2.7e-17 | 3.4e-19 |
| 2pi | 9 | 1 | 2.09 / 4.19 | 512 | 1.6e-19 | 1.3e-17 | 6.4e-19 | 2.67e-15 | 2.69e-15 | -4.9e-19 | 1.0e-16 | 1.1e-18 |
| 2pi | 9 | 2 | 4.19 / 6.28 | 512 | 3.1e-20 | 1.7e-18 | 4.0e-19 | 8.93e-16 | 8.95e-16 | +8.3e-20 | 4.6e-18 | 1.5e-19 |
| 8pi | 6 | 0 | 25.1 / 19.4 | 64 | 2.2e-19 | 2.2e-18 | 2.2e-18 | 2.38e-15 | 2.38e-15 | +2.8e-18 | 1.8e-17 | 3.7e-18 |
| 8pi | 6 | 2 | 13.6 / 7.85 | 64 | 1.2e-18 | 3.2e-18 | 1.5e-18 | 2.40e-15 | 2.40e-15 | -2.9e-18 | 1.3e-16 | 4.2e-18 |
| 8pi | 6 | 4 | 2.09 / 4.19 | 64 | 1.5e-18 | 1.3e-17 | 6.4e-19 | 2.74e-15 | 2.76e-15 | +3.1e-18 | 3.9e-17 | 7.1e-18 |
| 8pi | 6 | 8 | 19.9 / 25.1 | 64 | 2.9e-19 | 2.0e-18 | 2.1e-18 | 2.26e-15 | 2.26e-15 | +5.8e-19 | 7.8e-17 | 3.5e-18 |
| 8pi | 9 | 0 | 25.1 / 19.4 | 512 | 1.2e-19 | 2.2e-18 | 2.2e-18 | 2.38e-15 | 2.38e-15 | +3.4e-19 | 3.4e-17 | 4.7e-19 |
| 8pi | 9 | 2 | 13.6 / 7.85 | 512 | 9.3e-20 | 3.2e-18 | 1.5e-18 | 2.39e-15 | 2.40e-15 | +3.5e-19 | 8.8e-17 | 5.2e-19 |
| 8pi | 9 | 4 | 2.09 / 4.19 | 512 | 1.6e-19 | 1.3e-17 | 6.4e-19 | 2.67e-15 | 2.69e-15 | -4.9e-19 | 1.0e-16 | 1.1e-18 |
| 8pi | 9 | 8 | 19.9 / 25.1 | 512 | 8.9e-20 | 2.0e-18 | 2.1e-18 | 2.26e-15 | 2.26e-15 | -8.8e-20 | 4.2e-17 | 4.3e-19 |
| 32pi | 6 | 0 | 101 / 94.4 | 64 | 4.0e-19 | 1.1e-18 | 4.9e-18 | 2.48e-15 | 2.48e-15 | +1.1e-19 | 6.8e-17 | 3.8e-18 |
| 32pi | 6 | 8 | 51.3 / 45.2 | 64 | 5.9e-19 | 1.6e-18 | 3.5e-18 | 2.51e-15 | 2.52e-15 | +3.8e-18 | 1.3e-16 | 4.0e-18 |
| 32pi | 6 | 16 | 2.09 / 4.19 | 64 | 1.5e-18 | 1.3e-17 | 6.4e-19 | 2.74e-15 | 2.76e-15 | +3.1e-18 | 3.9e-17 | 7.1e-18 |
| 32pi | 6 | 32 | 94.5 / 101 | 64 | 4.0e-19 | 1.2e-18 | 5.0e-18 | 2.66e-15 | 2.66e-15 | -2.6e-18 | 1.2e-16 | 4.0e-18 |
| 32pi | 9 | 0 | 101 / 94.4 | 512 | 1.9e-19 | 1.1e-18 | 4.9e-18 | 2.48e-15 | 2.48e-15 | -4.1e-21 | 4.8e-17 | 4.8e-19 |
| 32pi | 9 | 8 | 51.3 / 45.2 | 512 | 2.3e-19 | 1.6e-18 | 3.5e-18 | 2.51e-15 | 2.52e-15 | -5.0e-19 | 1.6e-17 | 5.0e-19 |
| 32pi | 9 | 16 | 2.09 / 4.19 | 512 | 1.6e-19 | 1.3e-17 | 6.4e-19 | 2.67e-15 | 2.69e-15 | -4.9e-19 | 1.0e-16 | 1.1e-18 |
| 32pi | 9 | 32 | 94.5 / 101 | 512 | 2.1e-19 | 1.2e-18 | 5.0e-18 | 2.66e-15 | 2.66e-15 | +3.7e-19 | 1.5e-16 | 5.0e-19 |
| 128pi | 6 | 0 | 402 / 396 | 64 | 6.9e-19 | 5.4e-19 | 9.7e-18 | 2.39e-15 | 2.40e-15 | +2.8e-18 | 1.1e-16 | 3.8e-18 |
| 128pi | 6 | 32 | 202 / 196 | 64 | 8.0e-19 | 7.8e-19 | 7.0e-18 | 2.46e-15 | 2.47e-15 | -3.6e-18 | 6.1e-17 | 3.9e-18 |
| 128pi | 6 | 64 | 2.09 / 4.19 | 64 | 1.5e-18 | 1.3e-17 | 6.4e-19 | 2.74e-15 | 2.76e-15 | +3.1e-18 | 3.9e-17 | 7.1e-18 |
| 128pi | 6 | 128 | 396 / 402 | 64 | 1.0e-18 | 6.0e-19 | 1.1e-17 | 2.87e-15 | 2.88e-15 | +7.1e-20 | 1.5e-16 | 4.3e-18 |

| Phi_w | d_max | leaf | dtheta mid | nested pair | C3 | true | resolution | node | argument | other |
|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | [0.72356745, 0.72364465] | 2.79e-07 | 6.8e-21 | 1.4e-18 | -3.6e-21 | -2e-55 | -5.7e-21 | -1.4e-25 | +2.2e-21 |
| 2pi | 6 | [0.72078761, 0.72081102] | 3.94 | 2.0e-21 | 2.8e-17 | +4.2e-21 | -8e-55 | +2.1e-21 | +2.1e-21 | +8.5e-23 |
| 2pi | 6 | [0.72006189, 0.72008530] | 6.24 | 1.5e-20 | 3.5e-17 | -2.1e-18 | -4e-55 | -2.1e-18 | -8.1e-22 | -1.0e-21 |
| 2pi | 9 | [0.72360605, 0.72361570] | 8.32e-06 | 0.0e+00 | 4.8e-20 | +6.7e-22 | +0e+00 | -7.1e-23 | -3.6e-29 | +7.5e-22 |
| 2pi | 9 | [0.72080809, 0.72081102] | 3.91 | 5.3e-23 | 3.4e-18 | +6.5e-23 | -7e-55 | -1.4e-23 | +1.7e-23 | +6.1e-23 |
| 2pi | 9 | [0.72006189, 0.72006482] | 6.28 | 0.0e+00 | 4.3e-18 | +2.8e-19 | -3e-55 | +2.8e-19 | +2.5e-23 | -1.5e-22 |
| 8pi | 6 | [0.72356745, 0.72364465] | 2.79e-07 | 6.8e-21 | 1.4e-18 | -3.6e-21 | -2e-55 | -5.7e-21 | -1.4e-25 | +2.2e-21 |
| 8pi | 6 | [0.71899655, 0.71901615] | 10.6 | 3.4e-21 | 3.8e-17 | -3.5e-18 | +4e-55 | -3.5e-18 | -2.2e-21 | -2.1e-23 |
| 8pi | 6 | [0.71651698, 0.71653050] | 25.1 | 6.8e-21 | 3.9e-17 | +2.5e-18 | +1e-55 | +2.5e-18 | -2.2e-21 | -1.2e-21 |
| 8pi | 9 | [0.72360605, 0.72361570] | 8.32e-06 | 0.0e+00 | 4.8e-20 | +6.7e-22 | +0e+00 | -7.1e-23 | -3.6e-29 | +7.5e-22 |
| 8pi | 9 | [0.71901370, 0.71901615] | 10.5 | 1.1e-22 | 4.7e-18 | -1.7e-22 | -9e-56 | -1.5e-23 | -2.2e-22 | +6.3e-23 |
| 8pi | 9 | [0.71651698, 0.71651867] | 25.1 | 1.1e-22 | 4.9e-18 | +3.2e-19 | +3e-55 | +3.2e-19 | -1.1e-22 | -5.0e-23 |
| 32pi | 6 | [0.72356745, 0.72364465] | 2.79e-07 | 6.8e-21 | 1.4e-18 | -3.6e-21 | -2e-55 | -5.7e-21 | -1.4e-25 | +2.2e-21 |
| 32pi | 6 | [0.71377993, 0.71378973] | 48.2 | 2.1e-21 | 3.9e-17 | -3.7e-18 | -3e-55 | -3.7e-18 | +6.7e-21 | -1.0e-22 |
| 32pi | 6 | [0.70942717, 0.70943405] | 100 | 1.1e-20 | 3.9e-17 | +7.4e-21 | -5e-55 | +2.5e-21 | +5.1e-21 | -1.5e-22 |
| 32pi | 9 | [0.72360605, 0.72361570] | 8.32e-06 | 0.0e+00 | 4.8e-20 | +6.7e-22 | +0e+00 | -7.1e-23 | -3.6e-29 | +7.5e-22 |
| 32pi | 9 | [0.71378850, 0.71378973] | 48.2 | 2.6e-22 | 4.9e-18 | +5.0e-23 | +2e-55 | -1.8e-23 | +9.6e-23 | -2.8e-23 |
| 32pi | 9 | [0.70942717, 0.70942803] | 101 | 5.8e-22 | 4.9e-18 | +4.4e-22 | +6e-55 | +5.6e-24 | +4.9e-22 | -5.2e-23 |
| 128pi | 6 | [0.72356745, 0.72364465] | 2.79e-07 | 6.8e-21 | 1.4e-18 | -3.6e-21 | -2e-55 | -5.7e-21 | -1.4e-25 | +2.2e-21 |
| 128pi | 6 | [0.70365339, 0.70365828] | 199 | 5.5e-21 | 3.9e-17 | -8.9e-21 | -6e-55 | -9.9e-21 | +9.1e-22 | +9.1e-23 |
| 128pi | 6 | [0.69524754, 0.69525099] | 402 | 9.1e-21 | 3.8e-17 | +2.6e-18 | +3e-54 | +2.6e-18 | -1.1e-20 | -3.0e-23 |

## c1e8w0: {'c': 100000000.0, 'x_s': 0.723606797749979, 'omega': 0.0}

Inputs: `q12_c1e8w0.json` (2026-10-29T19:51:05.603591+00:00), `q12_c1e8w0_128.json` (2026-10-29T19:51:53.913167+00:00)

### Per run

| Phi_w | d_max | abserr | true error | request | window declared | outside declared | outside true (sum) | outside declared by component |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | 6.48e-08 | +6.3e-08 | 1.40e-15 | 5.39e-15 | 6.48e-08 | +6.3e-08 | residual 6.5e-08, floor(phase_limited) 2.4e-16, floor 1.1e-18 |
| 2pi | 9 | 1.35e-11 | +1.4e-11 | 1.41e-15 | 5.31e-15 | 1.35e-11 | +1.4e-11 | residual 1.4e-11, floor(phase_limited) 2.5e-16, floor 1.5e-18 |
| 2pi | 12 | 8.81e-11 | -5.7e-16 | 1.41e-15 | 5.30e-15 | 8.81e-11 | -5.7e-16 | nested_pair 8.8e-11, floor 4.5e-14, floor(phase_limited) 6.6e-15, residual 1.4e-16 |
| 8pi | 6 | 1.65e-09 | +1.8e-09 | 1.41e-15 | 2.13e-14 | 1.65e-09 | +1.8e-09 | residual 1.6e-09, floor(phase_limited) 2.0e-16, floor 5.8e-18 |
| 8pi | 9 | 4.04e-14 | -1.7e-14 | 1.41e-15 | 2.12e-14 | 1.92e-14 | -1.7e-14 | residual 1.9e-14, floor(phase_limited) 2.1e-16 |
| 8pi | 12 | 6.50e-11 | +8.4e-16 | 1.41e-15 | 2.12e-14 | 6.50e-11 | +8.4e-16 | nested_pair 6.5e-11, floor 3.1e-14, residual 7.1e-16, floor(phase_limited) 2.2e-16 |
| 32pi | 6 | 1.40e-11 | +1.4e-11 | 1.41e-15 | 8.49e-14 | 1.39e-11 | +1.4e-11 | residual 1.4e-11, floor(phase_limited) 2.5e-16, floor 2.7e-18 |
| 32pi | 9 | 8.65e-14 | -3.0e-16 | 1.41e-15 | 8.48e-14 | 1.71e-15 | -3.0e-16 | residual 1.5e-15, floor(phase_limited) 2.5e-16 |
| 32pi | 12 | 2.20e-11 | -8.7e-16 | 1.41e-15 | 8.48e-14 | 2.19e-11 | -8.7e-16 | nested_pair 2.2e-11, floor 4.2e-15, residual 1.0e-15, floor(phase_limited) 2.5e-16 |
| 128pi | 6 | 3.80e-13 | +3.7e-14 | 1.41e-15 | 3.39e-13 | 4.06e-14 | +3.7e-14 | residual 4.0e-14, floor(phase_limited) 1.5e-16, floor 2.6e-17 |
| 128pi | 9 | 3.50e-13 | +1.0e-15 | 1.41e-15 | 3.39e-13 | 1.06e-14 | +1.0e-15 | residual 1.0e-14, floor(phase_limited) 1.9e-16 |

### Outside leaves carrying >= 1% of the outside declared error, grouped by component and type

| Phi_w | d_max | component | type | leaves | depth | dtheta_near (rad) | abs theta' near | declared (sum) | abs true (sum) | max abs true | true (signed sum) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | residual | Levin | 12 | 6-6 | 6.28-3.16e+05 | 3.5e+04-7.9e+06 | 6.48e-08 | 6.64e-08 | 6.47e-08 | +6.3e-08 |
| 2pi | 9 | residual | Levin | 16 | 9-9 | 6.28-3.05e+05 | 3.5e+04-7.8e+06 | 1.35e-11 | 1.35e-11 | 1.35e-11 | +1.4e-11 |
| 2pi | 12 | nested_pair | direct | 18 | 12-12 | 14.1-299 | 5.3e+04-2.4e+05 | 8.80e-11 | 1.05e-15 | 1.13e-16 | -3.1e-16 |
| 8pi | 6 | residual | Levin | 10 | 6-6 | 25.1-1.61e+05 | 7.1e+04-5.7e+06 | 1.65e-09 | 1.76e-09 | 1.74e-09 | +1.8e-09 |
| 8pi | 9 | residual | Levin | 20 | 9-9 | 25.1-7.54e+04 | 7.1e+04-3.9e+06 | 1.90e-14 | 1.78e-14 | 1.74e-14 | -1.7e-14 |
| 8pi | 9 | floor(phase_limited) | Levin | 33 | 1-8 | 159-1.47e+07 | 1.8e+05-5.4e+07 | 2.14e-16 | 1.32e-15 | 1.38e-16 | +3.8e-16 |
| 8pi | 12 | nested_pair | direct | 8 | 12-12 | 25.1-148 | 7.1e+04-1.7e+05 | 6.50e-11 | 4.34e-16 | 9.52e-17 | -2.1e-16 |
| 32pi | 6 | residual | Levin | 2 | 6-6 | 101-101 | 1.4e+05-1.4e+05 | 1.39e-11 | 1.43e-11 | 1.42e-11 | +1.4e-11 |
| 32pi | 9 | residual | Levin | 10 | 9-9 | 101-6.38e+03 | 1.4e+05-1.1e+06 | 1.46e-15 | 3.34e-16 | 7.07e-17 | +2.1e-16 |
| 32pi | 9 | floor(phase_limited) | Levin | 20 | 1-8 | 311-6.57e+06 | 2.5e+05-3.6e+07 | 2.52e-16 | 6.41e-16 | 1.01e-16 | -5.1e-16 |
| 32pi | 12 | nested_pair | direct | 15 | 12-12 | 110-294 | 1.5e+05-2.4e+05 | 2.19e-11 | 5.20e-16 | 9.13e-17 | +2.6e-16 |
| 128pi | 6 | residual | Levin | 36 | 6-6 | 402-3.47e+06 | 2.8e+05-2.6e+07 | 4.05e-14 | 3.77e-14 | 3.74e-14 | +3.7e-14 |
| 128pi | 9 | residual | Levin | 110 | 9-9 | 402-3.69e+06 | 2.8e+05-2.7e+07 | 1.04e-14 | 3.60e-15 | 7.64e-17 | +9.6e-16 |
| 128pi | 9 | floor(phase_limited) | Levin | 93 | 1-8 | 402-6.6e+06 | 2.8e+05-3.6e+07 | 1.94e-16 | 5.04e-15 | 1.34e-16 | +4.7e-17 |

### The leaves adjacent to the window

| Phi_w | d_max | side | span | depth | type | component | declared | true |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | left | [0.711951, 0.723252] | 6 | Levin | residual | 6.32e-08 | +6.5e-08 |
| 2pi | 6 | right | [0.723961, 0.728274] | 6 | Levin | residual | 1.57e-09 | -1.7e-09 |
| 2pi | 9 | left | [0.721840, 0.723252] | 9 | Levin | residual | 1.35e-11 | +1.4e-11 |
| 2pi | 9 | right | [0.723961, 0.724500] | 9 | Levin | residual | 3.78e-15 | +3.7e-15 |
| 2pi | 12 | left | [0.723076, 0.723252] | 12 | direct | floor | 3.95e-15 | +9.0e-17 |
| 2pi | 12 | right | [0.723961, 0.724096] | 11 | direct | floor(phase_limited) | 2.79e-15 | -2.3e-17 |
| 8pi | 6 | left | [0.711603, 0.722898] | 6 | Levin | residual | 1.62e-09 | +1.7e-09 |
| 8pi | 6 | right | [0.724316, 0.728623] | 6 | Levin | residual | 2.53e-11 | +2.5e-11 |
| 8pi | 9 | left | [0.721486, 0.722898] | 9 | Levin | residual | 1.74e-14 | -1.7e-14 |
| 8pi | 9 | right | [0.724316, 0.724854] | 9 | Levin | residual | 1.26e-16 | -6.2e-17 |
| 8pi | 12 | left | [0.722721, 0.722898] | 12 | direct | nested_pair | 4.62e-12 | +5.9e-17 |
| 8pi | 12 | right | [0.724316, 0.724383] | 12 | direct | floor | 2.21e-15 | -2.5e-17 |
| 32pi | 6 | left | [0.710905, 0.722189] | 6 | Levin | residual | 1.39e-11 | +1.4e-11 |
| 32pi | 6 | right | [0.725025, 0.729321] | 6 | Levin | residual | 6.20e-14 | +6.2e-14 |
| 32pi | 9 | left | [0.720778, 0.722189] | 9 | Levin | residual | 1.67e-16 | +2.3e-18 |
| 32pi | 9 | right | [0.725025, 0.725562] | 9 | Levin | residual | 1.27e-16 | -6.4e-17 |
| 32pi | 12 | left | [0.722013, 0.722189] | 12 | Levin | residual | 2.93e-17 | -7.0e-19 |
| 32pi | 12 | right | [0.725025, 0.725092] | 12 | direct | floor | 4.23e-15 | -5.8e-17 |
| 128pi | 6 | left | [0.709509, 0.720771] | 6 | Levin | residual | 3.71e-14 | +3.7e-14 |
| 128pi | 6 | right | [0.726443, 0.730717] | 6 | Levin | residual | 1.95e-17 | -4.4e-17 |
| 128pi | 9 | left | [0.719363, 0.720771] | 9 | Levin | residual | 1.24e-16 | +6.2e-17 |
| 128pi | 9 | right | [0.726443, 0.728580] | 7 | Levin | floor(phase_limited) | 3.62e-19 | -6.3e-17 |

### Decomposition, c1e8w0 (`q3_c1e8w0.json`)

Levin leaves: the step-(4) residual V_P - V_L - V_R split into resolution / node / endpoint-theta / other (each the parent's component minus the halves'); 'ends' lists which of P, L, R has a computed end node off its span end (offset in units of 1e-16). CC (direct) leaves: the nested pair split into CC_31's and CC_16's errors, each resolution + node + argument + other; and the summed nested pair, floor and true error after 1, 2, 3 further bisections.

| config | span | dtheta_near | theta' | kind | declared basis | residual / nested | floor | split | theta_mod_2pi: residual | true error of P (V_P - exact) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2pi_d6 | [0.7119515, 0.7232523] | 6.28 | -3.5e+04 | Levin d6 | Levin | 6.32e-08 | 2.81e-17 | res +6.3e-08, node +1.3e-16, end-theta -4.0e-21, other -1.3e-21; ends P(+0.00/-1.11) R(+0.00/+1.11) | 6.32e-08 | +6.5e-08 |
| 2pi_d6 | [0.7239613, 0.7282744] | 6.28 | 3.5e+04 | Levin d6 | Levin | 1.57e-09 | 4.43e-18 | res -1.6e-09, node -4.4e-18, end-theta -1.9e-24, other +6.7e-21; ends P(+1.11/+0.00) L(+1.11/+0.00) | 1.57e-09 | -1.7e-09 |
| 2pi_d6 | [0.6554474, 0.6667482] | 1.62e+05 | -5.7e+06 | Levin d6 | Levin | 1.45e-16 | 1.27e-18 | res -1.5e-31, node -1.4e-16, end-theta -2.7e-18, other -8.2e-23; ends P(+1.11/+0.00) L(-1.11/+0.00) | 1.42e-16 | -7.1e-17 |
| 2pi_d6 | [0.0000000, 0.3616262] | 6.55e+06 | -3.6e+07 | Levin d1 | Levin | 6.62e-24 | 1.28e-16 | res -2.0e-25, node +1.4e-17, end-theta -1.4e-17, other +6.5e-24; ends R(-0.28/+0.00) | 1.41e-17 | -2.9e-17 |
| 32pi_d9 | [0.7207783, 0.7221888] | 101 | -1.4e+05 | Levin d9 | Levin | 1.67e-16 | 2.91e-19 | res +2.3e-18, node +1.6e-16, end-theta +2.3e-20, other +3.5e-21; ends L(+0.00/+1.11) R(-1.11/+0.00) | 1.67e-16 | +2.3e-18 |
| 32pi_d9 | [0.7165467, 0.7179573] | 1.6e+03 | -5.6e+05 | Levin d9 | Levin | 1.62e-16 | 1.82e-19 | res +4.1e-27, node -1.6e-16, end-theta -1.1e-19, other -4.8e-22; ends L(+0.00/+1.11) R(-1.11/+0.00) | 1.61e-16 | -9.5e-21 |
| 32pi_d9 | [0.7255618, 0.7260989] | 191 | 2e+05 | Levin d9 | Levin | 1.60e-16 | 1.44e-19 | res +4.7e-25, node +1.6e-16, end-theta -1.1e-19, other -9.9e-21; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.60e-16 | +2.4e-20 |
| 32pi_d9 | [0.7250248, 0.7255618] | 101 | 1.4e+05 | Levin d9 | Levin | 1.27e-16 | 1.83e-19 | res -4.3e-23, node -1.3e-16, end-theta -2.5e-20, other -4.2e-21; ends P(-1.11/+0.00) L(+1.11/+0.00) | 1.27e-16 | -6.4e-17 |
| 32pi_d9 | [0.0000000, 0.3610944] | 6.57e+06 | -3.6e+07 | Levin d1 | Levin | 9.93e-24 | 1.27e-16 | res -4.3e-26, node -1.9e-30, end-theta +0.0e+00, other +1.0e-23; ends all exact | 9.93e-24 | -4.7e-17 |
| 32pi_d12 | [0.7149599, 0.7151362] | 3.59e+03 | -8.5e+05 | Levin d12 | Levin | 1.57e-16 | 3.51e-20 | res -2.6e-43, node -1.6e-16, end-theta +3.4e-19, other -5.6e-22; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.58e-16 | +3.1e-19 |
| 32pi_d12 | [0.7123152, 0.7124915] | 6.18e+03 | -1.1e+06 | Levin d12 | Levin | 1.56e-16 | 2.97e-20 | res +5.7e-45, node -1.6e-16, end-theta -2.7e-19, other -1.5e-22; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.56e-16 | -3.3e-20 |
| 32pi_d12 | [0.7200730, 0.7202494] | 564 | -3.4e+05 | Levin d12 | Levin | 1.39e-16 | 6.97e-20 | res +1.2e-35, node +1.4e-16, end-theta +6.7e-21, other -1.8e-20; ends L(+0.00/-1.11) R(+1.11/+0.00) | 1.39e-16 | -1.0e-21 |
| 32pi_d12 | [0.7259646, 0.7260318] | 278 | 2.4e+05 | CC d12 | direct | 6.74e-12 | 6.92e-15 | CC31 err -7.2e-17 (res -1e-24, node -7.1e-17, arg -2.9e-19); CC16 err -6.7e-12 (res -6.7e-12); refine +1: nested 1.8e-16, floor 6.9e-15, true -1.1e-16; +2: nested 2.0e-17, floor 6.8e-15, true -5.3e-17; +3: nested 1.1e-17, floor 6.8e-15, true -7.7e-17 | -- | -7.2e-17 |
| 32pi_d12 | [0.7258975, 0.7259646] | 262 | 2.3e+05 | CC d12 | direct | 4.96e-12 | 6.73e-15 | CC31 err +6.8e-17 (res +6e-25, node +6.8e-17, arg +1.5e-19); CC16 err +5.0e-12 (res +5.0e-12); refine +1: nested 1.6e-16, floor 6.7e-15, true +1.1e-16; +2: nested 1.7e-17, floor 6.7e-15, true +6.9e-17; +3: nested 1.1e-17, floor 6.6e-15, true +8.2e-17 | -- | +6.8e-17 |
| 32pi_d12 | [0.7260318, 0.7260989] | 294 | 2.4e+05 | CC d12 | direct | 4.41e-12 | 7.11e-15 | CC31 err +2.7e-17 (res +3e-24, node +2.7e-17, arg +1.3e-20); CC16 err +4.4e-12 (res +4.4e-12); refine +1: nested 3.9e-16, floor 7.1e-15, true +3.6e-17; +2: nested 2.6e-17, floor 7.0e-15, true +1.5e-17; +3: nested 1.2e-17, floor 7.0e-15, true +2.7e-17 | -- | +2.7e-17 |
| 32pi_d12 | [0.7220125, 0.7221888] | 101 | -1.4e+05 | Levin d12 | Levin | 2.93e-17 | 1.48e-19 | res -1.5e-26, node -2.9e-17, end-theta +9.8e-20, other -3.7e-20; ends L=CC R=CC | 2.93e-17 | -7.0e-19 |
| 32pi_d12 | [0.7250248, 0.7250919] | 101 | 1.4e+05 | CC d12 | direct | 1.84e-15 | 4.23e-15 | CC31 err -5.8e-17 (res -3e-31, node -5.8e-17, arg -3.6e-20); CC16 err +1.8e-15 (res +1.8e-15); refine +1: nested 2.8e-17, floor 4.2e-15, true +4.4e-17; +2: nested 1.7e-17, floor 4.2e-15, true +5.8e-17; +3: nested 4.3e-18, floor 4.1e-15, true +4.4e-17 | -- | -5.8e-17 |
| 32pi_d12 | [0.0000000, 0.3610944] | 6.57e+06 | -3.6e+07 | Levin d1 | Levin | 9.93e-24 | 1.27e-16 | res -4.3e-26, node -1.9e-30, end-theta +0.0e+00, other +1.0e-23; ends all exact | 9.93e-24 | -4.7e-17 |

History of every Levin leaf made outside the window (all configs pooled): median and max of the step-(4) residual, the eq. (151) floor, and the true errors of V_P and V_L + V_R, by depth; and the share of leaves whose residual exceeds 10x the floor.

| config | depth | leaves | median residual | max residual | median floor | median abs(V_P err) | median abs(V_L+V_R err) | residual > 10 floor |
|---|---|---|---|---|---|---|---|---|
| 2pi_d6 | 0 | 1 | 8.2e-08 | 8.2e-08 | 2.1e-13 | 1.3e-06 | 1.2e-06 | 1/1 |
| 2pi_d6 | 1 | 2 | 7.2e-08 | 1.4e-07 | 1.6e-14 | 6.1e-07 | 5.4e-07 | 1/2 |
| 2pi_d6 | 2 | 3 | 1.7e-07 | 2.3e-07 | 6.9e-15 | 1.1e-06 | 8.6e-07 | 2/3 |
| 2pi_d6 | 3 | 4 | 1.3e-07 | 3.0e-07 | 8.8e-16 | 4.3e-07 | 2.8e-07 | 2/4 |
| 2pi_d6 | 4 | 4 | 1.5e-07 | 3.1e-07 | 2.3e-16 | 2.8e-07 | 1.3e-07 | 3/4 |
| 2pi_d6 | 5 | 6 | 1.6e-16 | 2.7e-07 | 5.7e-18 | 7.4e-17 | 1.3e-16 | 6/6 |
| 2pi_d6 | 6 | 12 | 1.2e-16 | 1.4e-07 | 1.8e-18 | 5.8e-17 | 9.0e-17 | 10/12 |
| 2pi_d6 | 7 | 4 | 1.6e-16 | 2.8e-08 | 1.5e-18 | 8.1e-17 | 8.0e-17 | 3/4 |
| 2pi_d6 | 8 | 6 | 6.4e-17 | 1.6e-09 | 5.8e-19 | 4.0e-17 | 8.1e-17 | 4/6 |
| 32pi_d9 | 0 | 1 | 3.4e-09 | 3.4e-09 | 5.2e-14 | 1.5e-08 | 1.2e-08 | 1/1 |
| 32pi_d9 | 1 | 2 | 2.1e-09 | 4.2e-09 | 3.9e-15 | 5.9e-09 | 3.8e-09 | 1/2 |
| 32pi_d9 | 2 | 3 | 4.0e-09 | 4.3e-09 | 1.7e-15 | 7.6e-09 | 3.6e-09 | 2/3 |
| 32pi_d9 | 3 | 4 | 1.3e-09 | 3.5e-09 | 2.4e-16 | 1.8e-09 | 5.6e-10 | 2/4 |
| 32pi_d9 | 4 | 4 | 4.6e-10 | 1.8e-09 | 6.4e-17 | 5.6e-10 | 9.4e-11 | 2/4 |
| 32pi_d9 | 5 | 6 | 7.5e-17 | 5.3e-10 | 6.7e-18 | 3.6e-17 | 3.9e-17 | 4/6 |
| 32pi_d9 | 6 | 8 | 1.1e-16 | 7.2e-11 | 2.7e-18 | 2.0e-17 | 1.2e-16 | 5/8 |
| 32pi_d9 | 7 | 10 | 8.3e-23 | 3.7e-12 | 1.3e-18 | 9.6e-17 | 9.9e-17 | 3/10 |
| 32pi_d9 | 8 | 6 | 1.4e-16 | 6.2e-14 | 5.9e-19 | 7.0e-17 | 6.8e-17 | 5/6 |
| 32pi_d9 | 9 | 10 | 1.5e-16 | 1.7e-16 | 1.9e-19 | 3.3e-17 | 7.0e-17 | 9/10 |
| 32pi_d9 | 10 | 2 | 6.4e-17 | 1.3e-16 | 2.1e-19 | 3.2e-17 | 3.2e-17 | 1/2 |
| 32pi_d9 | 11 | 2 | 1.4e-16 | 1.6e-16 | 1.6e-19 | 3.2e-17 | 1.1e-16 | 2/2 |
| 32pi_d12 | 0 | 1 | 3.4e-09 | 3.4e-09 | 5.2e-14 | 1.5e-08 | 1.2e-08 | 1/1 |
| 32pi_d12 | 1 | 2 | 2.1e-09 | 4.2e-09 | 3.9e-15 | 5.9e-09 | 3.8e-09 | 1/2 |
| 32pi_d12 | 2 | 3 | 4.0e-09 | 4.3e-09 | 1.7e-15 | 7.6e-09 | 3.6e-09 | 2/3 |
| 32pi_d12 | 3 | 4 | 1.3e-09 | 3.5e-09 | 2.4e-16 | 1.8e-09 | 5.6e-10 | 2/4 |
| 32pi_d12 | 4 | 4 | 4.6e-10 | 1.8e-09 | 6.4e-17 | 5.6e-10 | 9.4e-11 | 2/4 |
| 32pi_d12 | 5 | 6 | 7.5e-17 | 5.3e-10 | 6.7e-18 | 3.6e-17 | 3.9e-17 | 4/6 |
| 32pi_d12 | 6 | 8 | 1.1e-16 | 7.2e-11 | 2.7e-18 | 2.0e-17 | 1.2e-16 | 5/8 |
| 32pi_d12 | 7 | 10 | 8.3e-23 | 3.7e-12 | 1.3e-18 | 9.6e-17 | 9.9e-17 | 3/10 |
| 32pi_d12 | 8 | 6 | 1.4e-16 | 6.2e-14 | 5.9e-19 | 7.0e-17 | 6.8e-17 | 5/6 |
| 32pi_d12 | 9 | 10 | 1.5e-16 | 1.7e-16 | 1.9e-19 | 3.3e-17 | 7.0e-17 | 9/10 |
| 32pi_d12 | 10 | 18 | 1.5e-20 | 1.3e-16 | 8.6e-20 | 7.0e-17 | 8.0e-17 | 5/18 |
| 32pi_d12 | 11 | 12 | 1.2e-16 | 1.6e-16 | 7.6e-20 | 5.5e-17 | 5.9e-17 | 11/12 |
| 32pi_d12 | 12 | 24 | 1.9e-17 | 1.7e-16 | 6.1e-20 | 2.6e-17 | 6.3e-17 | 13/24 |
| 32pi_d12 | 13 | 8 | 9.9e-17 | 2.4e-16 | 1.1e-19 | 7.2e-17 | 3.5e-17 | 8/8 |

## c1e8w1: {'c': 100000000.0, 'x_s': 0.723606797749979, 'omega': 1.0}

Inputs: `q12_c1e8w1.json` (2026-10-29T19:52:11.089565+00:00), `q12_c1e8w1_128.json` (2026-10-29T19:53:00.385814+00:00)

### Per run

| Phi_w | d_max | abserr | true error | request | window declared | outside declared | outside true (sum) | outside declared by component |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | 6.52e-08 | +7.3e-08 | 1.13e-14 | 5.39e-15 | 6.52e-08 | +7.3e-08 | residual 6.5e-08, floor(phase_limited) 2.3e-16, floor 8.1e-18 |
| 2pi | 9 | 6.74e-12 | +6.7e-12 | 1.13e-14 | 5.31e-15 | 6.73e-12 | +6.7e-12 | residual 6.7e-12, floor(phase_limited) 2.4e-16, floor 5.8e-19 |
| 2pi | 12 | 4.63e-11 | +2.8e-16 | 1.13e-14 | 5.30e-15 | 4.62e-11 | +2.8e-16 | nested_pair 4.6e-11, floor 3.6e-14, floor(phase_limited) 1.1e-14, residual 1.6e-16 |
| 8pi | 6 | 1.07e-09 | +1.2e-09 | 1.13e-14 | 2.13e-14 | 1.07e-09 | +1.2e-09 | residual 1.1e-09, floor(phase_limited) 2.4e-16, floor 7.7e-19 |
| 8pi | 9 | 3.10e-14 | -9.5e-15 | 1.13e-14 | 2.12e-14 | 9.80e-15 | -9.5e-15 | residual 9.6e-15, floor(phase_limited) 2.4e-16, floor 2.3e-19 |
| 8pi | 12 | 4.07e-11 | -4.9e-17 | 1.13e-14 | 2.12e-14 | 4.07e-11 | -5.0e-17 | nested_pair 4.1e-11, floor 4.4e-14, residual 4.6e-16, floor(phase_limited) 2.4e-16 |
| 32pi | 6 | 6.18e-12 | +6.3e-12 | 1.13e-14 | 8.49e-14 | 6.09e-12 | +6.3e-12 | residual 6.1e-12, floor(phase_limited) 2.1e-16, floor 7.7e-18 |
| 32pi | 9 | 8.51e-14 | -1.0e-15 | 1.13e-14 | 8.48e-14 | 2.58e-16 | -1.0e-15 | floor(phase_limited) 2.4e-16, residual 1.8e-17, floor 2.1e-19 |
| 32pi | 12 | 8.56e-14 | -1.1e-15 | 1.13e-14 | 8.48e-14 | 8.04e-16 | -1.1e-15 | residual 5.6e-16, floor(phase_limited) 2.4e-16, floor 8.5e-20 |
| 128pi | 6 | 3.53e-13 | +1.3e-14 | 1.13e-14 | 3.39e-13 | 1.36e-14 | +1.3e-14 | residual 1.3e-14, floor(phase_limited) 2.4e-16 |
| 128pi | 9 | 3.40e-13 | +8.2e-16 | 1.13e-14 | 3.39e-13 | 5.51e-16 | +8.2e-16 | residual 3.1e-16, floor(phase_limited) 2.5e-16 |

### Outside leaves carrying >= 1% of the outside declared error, grouped by component and type

| Phi_w | d_max | component | type | leaves | depth | dtheta_near (rad) | abs theta' near | declared (sum) | abs true (sum) | max abs true | true (signed sum) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | residual | Levin | 5 | 6-6 | 6.28-3.16e+05 | 3.5e+04-7.9e+06 | 6.52e-08 | 7.33e-08 | 7.20e-08 | +7.3e-08 |
| 2pi | 9 | residual | Levin | 3 | 9-9 | 6.28-39.9 | 3.5e+04-8.9e+04 | 6.73e-12 | 6.67e-12 | 6.67e-12 | +6.7e-12 |
| 2pi | 12 | nested_pair | direct | 3 | 12-12 | 14.1-39.1 | 5.3e+04-8.8e+04 | 4.62e-11 | 5.10e-17 | 2.59e-17 | +3.6e-17 |
| 8pi | 6 | residual | Levin | 3 | 6-6 | 25.1-7.21e+03 | 7.1e+04-1.2e+06 | 1.07e-09 | 1.19e-09 | 1.15e-09 | +1.2e-09 |
| 8pi | 9 | residual | Levin | 3 | 9-9 | 25.1-77.8 | 7.1e+04-1.2e+05 | 9.56e-15 | 9.55e-15 | 9.54e-15 | -9.5e-15 |
| 8pi | 9 | floor(phase_limited) | Levin | 23 | 1-8 | 159-6.56e+06 | 1.8e+05-3.6e+07 | 2.42e-16 | 7.36e-16 | 9.52e-17 | +7.5e-20 |
| 8pi | 12 | nested_pair | direct | 5 | 12-12 | 25.1-148 | 7.1e+04-1.7e+05 | 4.06e-11 | 1.12e-16 | 6.95e-17 | -2.8e-17 |
| 32pi | 6 | residual | Levin | 17 | 6-6 | 101-1.46e+06 | 1.4e+05-1.7e+07 | 6.09e-12 | 6.32e-12 | 6.28e-12 | +6.3e-12 |
| 32pi | 9 | floor(phase_limited) | Levin | 72 | 1-8 | 101-6.57e+06 | 1.4e+05-3.6e+07 | 2.40e-16 | 3.50e-15 | 1.20e-16 | -1.0e-15 |
| 32pi | 9 | residual | Levin | 1 | 9-9 | 101-101 | 1.4e+05-1.4e+05 | 1.79e-17 | 6.75e-18 | 6.75e-18 | -6.8e-18 |
| 32pi | 12 | residual | Levin | 7 | 12-12 | 101-306 | 1.4e+05-2.5e+05 | 5.64e-16 | 3.04e-16 | 8.22e-17 | -4.7e-17 |
| 32pi | 12 | floor(phase_limited) | Levin | 73 | 1-9 | 101-6.57e+06 | 1.4e+05-3.6e+07 | 2.40e-16 | 3.50e-15 | 1.20e-16 | -1.0e-15 |
| 128pi | 6 | residual | Levin | 8 | 6-6 | 402-5.36e+04 | 2.8e+05-3.3e+06 | 1.33e-14 | 1.30e-14 | 1.29e-14 | +1.3e-14 |
| 128pi | 6 | floor(phase_limited) | Levin | 11 | 1-5 | 6.48e+03-6.6e+06 | 1.1e+06-3.6e+07 | 2.40e-16 | 3.96e-16 | 1.21e-16 | +2.1e-16 |
| 128pi | 9 | residual | Levin | 2 | 9-9 | 1.95e+04-2.23e+04 | 2e+06-2.1e+06 | 3.05e-16 | 1.53e-16 | 7.67e-17 | +1.5e-16 |
| 128pi | 9 | floor(phase_limited) | Levin | 33 | 1-8 | 402-6.6e+06 | 2.8e+05-3.6e+07 | 2.46e-16 | 1.24e-15 | 1.21e-16 | +6.6e-16 |

### The leaves adjacent to the window

| Phi_w | d_max | side | span | depth | type | component | declared | true |
|---|---|---|---|---|---|---|---|---|
| 2pi | 6 | left | [0.711951, 0.723252] | 6 | Levin | residual | 6.37e-08 | +7.2e-08 |
| 2pi | 6 | right | [0.723961, 0.728274] | 6 | Levin | residual | 1.45e-09 | +1.3e-09 |
| 2pi | 9 | left | [0.721840, 0.723252] | 9 | Levin | residual | 6.73e-12 | +6.7e-12 |
| 2pi | 9 | right | [0.723961, 0.724500] | 9 | Levin | residual | 6.66e-15 | +6.6e-15 |
| 2pi | 12 | left | [0.723076, 0.723252] | 12 | direct | floor | 3.95e-15 | -2.0e-17 |
| 2pi | 12 | right | [0.723961, 0.724096] | 11 | direct | floor(phase_limited) | 2.79e-15 | -2.8e-19 |
| 8pi | 6 | left | [0.711603, 0.722898] | 6 | Levin | residual | 1.04e-09 | +1.2e-09 |
| 8pi | 6 | right | [0.724316, 0.728623] | 6 | Levin | residual | 3.14e-11 | +3.2e-11 |
| 8pi | 9 | left | [0.721486, 0.722898] | 9 | Levin | residual | 9.56e-15 | -9.5e-15 |
| 8pi | 9 | right | [0.724316, 0.724854] | 9 | Levin | residual | 1.57e-18 | +1.1e-18 |
| 8pi | 12 | left | [0.722721, 0.722898] | 12 | direct | nested_pair | 4.24e-12 | -6.9e-17 |
| 8pi | 12 | right | [0.724316, 0.724383] | 12 | direct | floor | 2.21e-15 | -2.9e-17 |
| 32pi | 6 | left | [0.710905, 0.722189] | 6 | Levin | residual | 6.05e-12 | +6.3e-12 |
| 32pi | 6 | right | [0.725025, 0.729321] | 6 | Levin | residual | 4.48e-14 | +4.5e-14 |
| 32pi | 9 | left | [0.720778, 0.722189] | 9 | Levin | residual | 1.79e-17 | -6.8e-18 |
| 32pi | 9 | right | [0.725025, 0.726099] | 8 | Levin | floor(phase_limited) | 2.46e-19 | +4.4e-20 |
| 32pi | 12 | left | [0.722013, 0.722189] | 12 | Levin | residual | 4.42e-17 | -1.1e-17 |
| 32pi | 12 | right | [0.725025, 0.726099] | 8 | Levin | floor(phase_limited) | 2.46e-19 | +4.4e-20 |
| 128pi | 6 | left | [0.709509, 0.720771] | 6 | Levin | residual | 1.28e-14 | +1.3e-14 |
| 128pi | 6 | right | [0.726443, 0.730717] | 6 | Levin | residual | 3.11e-17 | +1.9e-17 |
| 128pi | 9 | left | [0.717955, 0.720771] | 8 | Levin | floor(phase_limited) | 5.19e-19 | +6.3e-17 |
| 128pi | 9 | right | [0.726443, 0.728580] | 7 | Levin | floor(phase_limited) | 3.62e-19 | -1.2e-17 |

### Decomposition, c1e8w1 (`q3_c1e8w1.json`)

Levin leaves: the step-(4) residual V_P - V_L - V_R split into resolution / node / endpoint-theta / other (each the parent's component minus the halves'); 'ends' lists which of P, L, R has a computed end node off its span end (offset in units of 1e-16). CC (direct) leaves: the nested pair split into CC_31's and CC_16's errors, each resolution + node + argument + other; and the summed nested pair, floor and true error after 1, 2, 3 further bisections.

| config | span | dtheta_near | theta' | kind | declared basis | residual / nested | floor | split | theta_mod_2pi: residual | true error of P (V_P - exact) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2pi_d12 | [0.7218397, 0.7220163] | 126 | -1.6e+05 | Levin d12 | Levin | 6.95e-17 | 1.33e-19 | res +3.0e-25, node +6.9e-17, end-theta +1.5e-19, other -7.1e-21; ends L=CC R=CC | 6.93e-17 | -6.6e-19 |
| 2pi_d12 | [0.7220163, 0.7221928] | 100 | -1.4e+05 | Levin d12 | Levin | 4.68e-17 | 1.48e-19 | res -1.5e-26, node -4.7e-17, end-theta -1.0e-20, other +9.5e-21; ends L=CC R=CC | 4.67e-17 | +1.5e-18 |
| 2pi_d12 | [0.7221928, 0.7223694] | 76.6 | -1.2e+05 | Levin d12 | Levin | 3.54e-17 | 1.68e-19 | res -2.4e-29, node -3.5e-17, end-theta -6.6e-20, other -2.4e-21; ends L=CC R=CC | 3.54e-17 | +1.7e-18 |
| 2pi_d12 | [0.7225460, 0.7227226] | 39.1 | -8.8e+04 | CC d12 | direct | 4.16e-11 | 7.88e-15 | CC31 err +2.6e-17 (res +9e-23, node +2.6e-17, arg -4.4e-20); CC16 err -4.2e-11 (res -4.2e-11); refine +1: nested 1.1e-15, floor 7.5e-15, true +2.6e-17; +2: nested 4.3e-17, floor 7.4e-15, true -1.6e-17; +3: nested 6.1e-17, floor 7.3e-15, true -5.1e-18 | -- | +2.6e-17 |
| 2pi_d12 | [0.7227226, 0.7228991] | 25 | -7.1e+04 | CC d12 | direct | 4.39e-12 | 6.57e-15 | CC31 err -7.5e-18 (res -3e-25, node -7.5e-18, arg +6.9e-21); CC16 err +4.4e-12 (res +4.4e-12); refine +1: nested 1.8e-16, floor 6.2e-15, true +1.7e-17; +2: nested 4.6e-17, floor 6.1e-15, true -2.1e-17; +3: nested 4.1e-17, floor 6.0e-15, true +1.2e-17 | -- | -7.5e-18 |
| 2pi_d12 | [0.7228991, 0.7230757] | 14.1 | -5.3e+04 | CC d12 | direct | 1.72e-13 | 5.26e-15 | CC31 err +1.8e-17 (res +2e-27, node +1.8e-17, arg -2.7e-20); CC16 err +1.7e-13 (res +1.7e-13); refine +1: nested 2.2e-17, floor 4.9e-15, true +1.5e-17; +2: nested 3.0e-17, floor 4.8e-15, true -5.4e-17; +3: nested 2.9e-17, floor 4.7e-15, true +1.1e-17 | -- | +1.8e-17 |
| 2pi_d12 | [0.7230757, 0.7232523] | 6.28 | -3.5e+04 | CC d12 | direct | 1.62e-15 | 3.95e-15 | CC31 err -2.0e-17 (res +4e-30, node -2.0e-17, arg +1.5e-20); CC16 err +1.6e-15 (res +1.6e-15); refine +1: nested 3.6e-17, floor 3.6e-15, true +5.4e-17; +2: nested 1.0e-17, floor 3.5e-15, true +4.0e-17; +3: nested 1.5e-17, floor 3.4e-15, true -6.8e-18 | -- | -2.0e-17 |
| 2pi_d12 | [0.7239613, 0.7240961] | 6.28 | 3.5e+04 | CC d11 | direct | 6.13e-17 | 2.79e-15 | CC31 err -2.8e-19 (res -3e-35, node -2.7e-19, arg -1.4e-20); CC16 err +6.1e-17 (res +2.9e-17); refine +1: nested 3.9e-17, floor 2.6e-15, true +8.4e-17; +2: nested 2.2e-19, floor 2.5e-15, true +3.8e-17; +3: nested 1.9e-18, floor 2.4e-15, true +9.1e-18 | -- | -2.8e-19 |
| 2pi_d12 | [0.0000000, 0.3616261] | 6.55e+06 | -3.6e+07 | Levin d1 | Levin | 1.65e-24 | 1.28e-16 | res -1.0e-25, node -2.3e-17, end-theta +2.3e-17, other +2.1e-24; ends R(+0.28/+0.00) | 2.28e-17 | -3.3e-17 |
| 32pi_d12 | [0.7209546, 0.7211309] | 306 | -2.5e+05 | Levin d12 | Levin | 1.64e-16 | 9.01e-20 | res +3.9e-33, node -1.6e-16, end-theta +2.2e-20, other -1.0e-20; ends P(+0.00/+1.11) R(+0.00/-1.11) | 1.64e-16 | -8.2e-17 |
| 32pi_d12 | [0.7211309, 0.7213072] | 264 | -2.3e+05 | Levin d12 | Levin | 1.64e-16 | 9.60e-20 | res +7.8e-33, node -1.6e-16, end-theta +2.2e-20, other +1.0e-21; ends P(-1.11/+0.00) L(+1.11/+0.00) | 1.64e-16 | -8.2e-17 |
| 32pi_d12 | [0.7214836, 0.7216599] | 190 | -1.9e+05 | Levin d12 | Levin | 1.30e-16 | 1.11e-19 | res +1.4e-22, node +1.3e-16, end-theta +3.2e-19, other -6.9e-21; ends P(+0.00/+1.11) L=CC R=CC | 1.30e-16 | +6.4e-17 |
| 32pi_d12 | [0.7220125, 0.7221888] | 101 | -1.4e+05 | Levin d12 | Levin | 4.42e-17 | 1.48e-19 | res -1.7e-26, node -4.4e-17, end-theta +6.4e-21, other +2.6e-20; ends P(+0.00/+1.11) L=CC R=CC | 4.42e-17 | -1.1e-17 |
| 32pi_d12 | [0.7250248, 0.7260989] | 101 | 1.4e+05 | Levin d8 | Levin | 3.22e-20 | 2.46e-19 | res +1.5e-19, node -1.8e-19, end-theta -2.5e-33, other -3.9e-21; ends all exact | 3.13e-20 | +4.4e-20 |
| 32pi_d12 | [0.0000000, 0.3610944] | 6.57e+06 | -3.6e+07 | Levin d1 | Levin | 2.15e-17 | 1.27e-16 | res -1.5e-25, node +5.8e-18, end-theta +1.6e-17, other -1.1e-23; ends R(-0.28/+0.00) | 5.80e-18 | -3.5e-17 |

History of every Levin leaf made outside the window (all configs pooled): median and max of the step-(4) residual, the eq. (151) floor, and the true errors of V_P and V_L + V_R, by depth; and the share of leaves whose residual exceeds 10x the floor.

| config | depth | leaves | median residual | max residual | median floor | median abs(V_P err) | median abs(V_L+V_R err) | residual > 10 floor |
|---|---|---|---|---|---|---|---|---|
| 2pi_d12 | 0 | 1 | 1.6e-08 | 1.6e-08 | 2.1e-13 | 5.3e-07 | 5.2e-07 | 1/1 |
| 2pi_d12 | 1 | 2 | 1.6e-08 | 3.2e-08 | 1.6e-14 | 2.6e-07 | 2.4e-07 | 1/2 |
| 2pi_d12 | 2 | 3 | 4.0e-08 | 5.8e-08 | 6.9e-15 | 4.8e-07 | 4.3e-07 | 2/3 |
| 2pi_d12 | 3 | 6 | 5.4e-18 | 9.6e-08 | 2.5e-17 | 2.9e-17 | 2.9e-17 | 2/6 |
| 2pi_d12 | 4 | 4 | 5.6e-08 | 1.3e-07 | 2.3e-16 | 1.6e-07 | 9.9e-08 | 3/4 |
| 2pi_d12 | 5 | 6 | 1.2e-16 | 1.4e-07 | 5.7e-18 | 6.0e-17 | 6.0e-17 | 5/6 |
| 2pi_d12 | 6 | 10 | 5.3e-18 | 1.1e-07 | 1.5e-18 | 4.3e-17 | 5.2e-17 | 4/10 |
| 2pi_d12 | 7 | 12 | 7.3e-23 | 3.7e-08 | 8.3e-19 | 2.3e-17 | 3.5e-17 | 3/12 |
| 2pi_d12 | 8 | 6 | 4.3e-20 | 1.5e-09 | 6.9e-19 | 5.9e-17 | 5.9e-17 | 2/6 |
| 2pi_d12 | 9 | 4 | 3.4e-12 | 1.8e-10 | 6.2e-19 | 3.3e-12 | 3.0e-14 | 2/4 |
| 2pi_d12 | 10 | 4 | 3.0e-14 | 1.8e-12 | 4.5e-19 | 3.0e-14 | 4.7e-17 | 2/4 |
| 2pi_d12 | 11 | 7 | 1.2e-18 | 6.7e-15 | 2.1e-19 | 1.8e-19 | 1.0e-18 | 2/7 |
| 2pi_d12 | 12 | 7 | 2.5e-17 | 6.9e-17 | 1.9e-19 | 1.3e-18 | 2.7e-17 | 7/7 |
| 32pi_d12 | 0 | 1 | 6.2e-10 | 6.2e-10 | 5.2e-14 | 3.1e-09 | 2.5e-09 | 1/1 |
| 32pi_d12 | 1 | 2 | 4.0e-10 | 8.1e-10 | 3.9e-15 | 1.2e-09 | 8.2e-10 | 1/2 |
| 32pi_d12 | 2 | 3 | 8.0e-10 | 8.4e-10 | 1.7e-15 | 1.6e-09 | 8.5e-10 | 2/3 |
| 32pi_d12 | 3 | 4 | 2.8e-10 | 7.3e-10 | 2.4e-16 | 4.2e-10 | 1.5e-10 | 2/4 |
| 32pi_d12 | 4 | 6 | 1.5e-16 | 4.2e-10 | 1.2e-17 | 7.5e-17 | 7.8e-17 | 4/6 |
| 32pi_d12 | 5 | 12 | 1.1e-16 | 1.5e-10 | 3.5e-18 | 8.6e-18 | 9.7e-17 | 11/12 |
| 32pi_d12 | 6 | 24 | 1.2e-16 | 2.5e-11 | 1.4e-18 | 4.7e-17 | 6.2e-17 | 24/24 |
| 32pi_d12 | 7 | 48 | 1.5e-22 | 1.9e-12 | 6.2e-19 | 3.7e-17 | 6.1e-17 | 20/48 |
| 32pi_d12 | 8 | 40 | 2.6e-23 | 4.5e-14 | 3.1e-19 | 4.8e-17 | 4.1e-17 | 2/40 |
| 32pi_d12 | 9 | 4 | 8.9e-18 | 1.8e-16 | 3.2e-19 | 3.9e-17 | 5.6e-18 | 2/4 |
| 32pi_d12 | 10 | 4 | 1.1e-17 | 1.6e-16 | 1.9e-19 | 7.8e-20 | 5.7e-18 | 2/4 |
| 32pi_d12 | 11 | 4 | 3.9e-19 | 1.3e-16 | 1.3e-19 | 4.7e-17 | 8.2e-17 | 1/4 |
| 32pi_d12 | 12 | 8 | 4.6e-17 | 1.6e-16 | 1.1e-19 | 3.7e-17 | 5.0e-17 | 6/8 |

## Levin residual above the floor against end-node tiling (`endnodes.json`)

Every Levin leaf made outside the window whose own value is resolved (abs(V_P err) < 1e-15). 'Tile': the computed end nodes of the parent and its halves coincide (left half's bottom = parent's, right half's top = parent's, left's top = right's bottom).

| file | class | leaves | max residual | parent's end nodes off a, b |
|---|---|---|---|---|
| q3_c1e6w0.json | residual<=10floor | computed end nodes do not tile | halves CC | 4 | 5.4e-18 | 1 |
| q3_c1e6w0.json | residual<=10floor | computed end nodes do not tile | halves Levin | 17 | 2.0e-16 | 6 |
| q3_c1e6w0.json | residual<=10floor | computed end nodes tile | halves CC | 4 | 1.9e-17 | 4 |
| q3_c1e6w0.json | residual<=10floor | computed end nodes tile | halves Levin | 120 | 1.2e-18 | 110 |
| q3_c1e6w0.json | residual>10floor | computed end nodes do not tile | halves CC | 95 | 2.8e-16 | 27 |
| q3_c1e6w0.json | residual>10floor | computed end nodes do not tile | halves Levin | 301 | 3.0e-16 | 181 |
| q3_c1e6w0.json | residual>10floor | computed end nodes tile | halves CC | 90 | 1.6e-16 | 89 |
| q3_c1e6w0.json | residual>10floor | computed end nodes tile | halves Levin | 1 | 2.2e-16 | 0 |
| q3_c1e8w0.json | residual<=10floor | computed end nodes do not tile | halves Levin | 4 | 7.5e-17 | 1 |
| q3_c1e8w0.json | residual<=10floor | computed end nodes tile | halves Levin | 51 | 1.0e-18 | 37 |
| q3_c1e8w0.json | residual>10floor | computed end nodes do not tile | halves CC | 8 | 2.4e-16 | 5 |
| q3_c1e8w0.json | residual>10floor | computed end nodes do not tile | halves Levin | 60 | 2.0e-16 | 28 |
| q3_c1e8w0.json | residual>10floor | computed end nodes tile | halves CC | 4 | 4.5e-17 | 1 |
| q3_c1e8w0.json | residual>10floor | computed end nodes tile | halves Levin | 1 | 1.7e-16 | 1 |
| q3_c1e8w1.json | residual<=10floor | computed end nodes do not tile | halves Levin | 11 | 6.8e-17 | 5 |
| q3_c1e8w1.json | residual<=10floor | computed end nodes tile | halves Levin | 107 | 1.6e-18 | 91 |
| q3_c1e8w1.json | residual>10floor | computed end nodes do not tile | halves CC | 7 | 1.3e-16 | 3 |
| q3_c1e8w1.json | residual>10floor | computed end nodes do not tile | halves Levin | 64 | 2.0e-16 | 30 |
| q3_c1e8w1.json | residual>10floor | computed end nodes tile | halves CC | 5 | 2.5e-17 | 0 |
| q3_c1e8w1.json | residual>10floor | computed end nodes tile | halves Levin | 1 | 1.8e-16 | 0 |

## LiouvilleGreen seed 0, test_JJJ, Levin call 15

x_span [-0.04917942646450173, 27.631021115928547], kwargs {'atol': 1e-14, 'rtol': 1e-10, 'chebyshev_order': 12, 'depth_max': 20, 'build_p_sample': False, 'notify_interval': 300, 'notify_label': 'phase3', 'emit_diagnostics': False}.

| d_max | wall s | abserr | request | window leaves | window declared | window by component | outside leaves | outside declared | outside by component |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 3 | 4.56e-12 | 2.59e-12 | 1088 | 4.56e-12 | floor 4.6e-12 | 21 | 8.36e-16 | floor(phase_limited) 8.2e-16, floor 1.4e-17 |
| 8 | 12 | 4.55e-12 | 2.59e-12 | 4352 | 4.55e-12 | floor 4.6e-12 | 21 | 8.36e-16 | floor(phase_limited) 8.4e-16 |
| 10 | 52 | 4.55e-12 | 2.59e-12 | 17408 | 4.55e-12 | floor 4.5e-12 | 21 | 8.36e-16 | floor(phase_limited) 8.4e-16 |
| 12 | 188 | 4.55e-12 | 2.59e-12 | 69632 | 4.55e-12 | floor 4.5e-12 | 21 | 8.36e-16 | floor(phase_limited) 8.4e-16 |

Window floor term by term (sums over the window leaves):

| d_max | C1 | C2 | C3 | D (declared theta_abserr) | max rel. diff. of C1+C2+C3+D from phase_err |
|---|---|---|---|---|---|
| 6 | 7.81e-16 | 6.11e-17 | 1.81e-14 | 4.54e-12 | 0.0e+00 |
| 8 | 7.81e-16 | 6.11e-17 | 1.80e-14 | 4.53e-12 | 0.0e+00 |

Outside leaves carrying the declared error (top 5 per depth):

| d_max | span | depth | type | component | dtheta_near | theta' near | abserr | floor | declared |
|---|---|---|---|---|---|---|---|---|---|
| 6 | [8.96099, 10.39715] | 4 | Levin | floor(phase_limited) | 7.65e+03 | 7.65e+03 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 6 | [10.39715, 11.83331] | 4 | Levin | floor(phase_limited) | 3.22e+04 | 3.22e+04 | 4.2e-22 | 5.8e-17 | 5.8e-17 |
| 6 | [23.32255, 24.75871] | 4 | Levin | floor(phase_limited) | 1.32e+10 | 1.32e+10 | 4.0e-28 | 5.8e-17 | 5.8e-17 |
| 6 | [13.26946, 14.70562] | 4 | Levin | floor(phase_limited) | 5.69e+05 | 5.69e+05 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 6 | [14.70562, 16.14177] | 4 | Levin | floor(phase_limited) | 2.39e+06 | 2.39e+06 | 1.3e-23 | 5.8e-17 | 5.8e-17 |
| 8 | [8.96099, 10.39715] | 4 | Levin | floor(phase_limited) | 7.65e+03 | 7.65e+03 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 8 | [10.39715, 11.83331] | 4 | Levin | floor(phase_limited) | 3.22e+04 | 3.22e+04 | 4.2e-22 | 5.8e-17 | 5.8e-17 |
| 8 | [23.32255, 24.75871] | 4 | Levin | floor(phase_limited) | 1.32e+10 | 1.32e+10 | 4.0e-28 | 5.8e-17 | 5.8e-17 |
| 8 | [13.26946, 14.70562] | 4 | Levin | floor(phase_limited) | 5.69e+05 | 5.69e+05 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 8 | [14.70562, 16.14177] | 4 | Levin | floor(phase_limited) | 2.39e+06 | 2.39e+06 | 1.3e-23 | 5.8e-17 | 5.8e-17 |
| 10 | [8.96099, 10.39715] | 4 | Levin | floor(phase_limited) | 7.65e+03 | 7.65e+03 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 10 | [10.39715, 11.83331] | 4 | Levin | floor(phase_limited) | 3.22e+04 | 3.22e+04 | 4.2e-22 | 5.8e-17 | 5.8e-17 |
| 10 | [23.32255, 24.75871] | 4 | Levin | floor(phase_limited) | 1.32e+10 | 1.32e+10 | 4.0e-28 | 5.8e-17 | 5.8e-17 |
| 10 | [13.26946, 14.70562] | 4 | Levin | floor(phase_limited) | 5.69e+05 | 5.69e+05 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 10 | [14.70562, 16.14177] | 4 | Levin | floor(phase_limited) | 2.39e+06 | 2.39e+06 | 1.3e-23 | 5.8e-17 | 5.8e-17 |
| 12 | [8.96099, 10.39715] | 4 | Levin | floor(phase_limited) | 7.65e+03 | 7.65e+03 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 12 | [10.39715, 11.83331] | 4 | Levin | floor(phase_limited) | 3.22e+04 | 3.22e+04 | 4.2e-22 | 5.8e-17 | 5.8e-17 |
| 12 | [23.32255, 24.75871] | 4 | Levin | floor(phase_limited) | 1.32e+10 | 1.32e+10 | 4.0e-28 | 5.8e-17 | 5.8e-17 |
| 12 | [13.26946, 14.70562] | 4 | Levin | floor(phase_limited) | 5.69e+05 | 5.69e+05 | 0.0e+00 | 5.8e-17 | 5.8e-17 |
| 12 | [14.70562, 16.14177] | 4 | Levin | floor(phase_limited) | 2.39e+06 | 2.39e+06 | 1.3e-23 | 5.8e-17 | 5.8e-17 |

Gauss-Legendre references on sampled leaves (fsum; the integrand from the basis's own eval_basis):

| d_max | span | type | depth | declared | nested / residual | floor | GL 64-32 | GL 128-64 | GL 256-128 | value - GL256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | [0.160865, 0.190871] | window | 6 | 5.6e-14 | 0.0e+00 | 5.6e-14 | +0.0e+00 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| 6 | [4.651584, 4.652526] | window | 6 | 2.2e-17 | 3.4e-20 | 2.2e-17 | +1.5e-19 | -5.4e-20 | +2.7e-20 | +2.0e-20 |
| 6 | [3.962983, 3.964805] | window | 6 | 2.5e-17 | 2.7e-20 | 2.5e-17 | +0.0e+00 | +0.0e+00 | -2.7e-20 | +0.0e+00 |
| 6 | [8.960994, 10.397150] | Levin | 4 | 5.8e-17 | 0.0e+00 | 5.8e-17 | -1.8e-02 | +2.0e-02 | -6.5e-03 | -2.5e-04 |
| 6 | [10.397150, 11.833306] | Levin | 4 | 5.8e-17 | 4.2e-22 | 5.8e-17 | -3.1e-02 | +4.3e-02 | -1.6e-02 | +4.8e-03 |
| 6 | [23.322553, 24.758709] | Levin | 4 | 5.8e-17 | 4.0e-28 | 5.8e-17 | +2.3e-02 | -1.2e-02 | -7.7e-03 | +2.3e-03 |
| 8 | [0.175868, 0.183370] | window | 8 | 1.4e-14 | 0.0e+00 | 1.4e-14 | +0.0e+00 | +0.0e+00 | -1.1e-19 | +1.1e-19 |
| 8 | [4.652291, 4.652526] | window | 8 | 5.5e-18 | 0.0e+00 | 5.5e-18 | +6.8e-21 | +2.4e-20 | -1.4e-20 | -3.4e-20 |
| 8 | [3.962983, 3.963439] | window | 8 | 6.2e-18 | 0.0e+00 | 6.2e-18 | +0.0e+00 | +0.0e+00 | -6.8e-21 | +0.0e+00 |
| 8 | [8.960994, 10.397150] | Levin | 4 | 5.8e-17 | 0.0e+00 | 5.8e-17 | -1.8e-02 | +2.0e-02 | -6.5e-03 | -2.5e-04 |
| 8 | [10.397150, 11.833306] | Levin | 4 | 5.8e-17 | 4.2e-22 | 5.8e-17 | -3.1e-02 | +4.3e-02 | -1.6e-02 | +4.8e-03 |
| 8 | [23.322553, 24.758709] | Levin | 4 | 5.8e-17 | 4.0e-28 | 5.8e-17 | +2.3e-02 | -1.2e-02 | -7.7e-03 | +2.3e-03 |
