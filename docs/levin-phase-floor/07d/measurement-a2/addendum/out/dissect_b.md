# levin-phase-floor 07d Part A' addendum: item B, R:a:W2-0-12977-5 region by region
# written 2026-11-24T11:05:21Z on mps026298.phys.susx.ac.uk
# script dissect_b.py sha256 71cfb3fde5f1c92930d66372c86e2152e2fc01736c60d336b9ae1fc82e19a817
# tree: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3 HEAD c2c57a36cbad3780986baceb807dcb447975b498 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/replays/tree/R_a_W2-0-12977-5.json sha256 d04da1a386df21f0b5d26f40b5150e6142e4b2f7fae641fc6274250790cb3d6a
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/replays/V93T8/R_a_W2-0-12977-5.json sha256 9a74d64f033ede76810420492b33d83ecc545727ded3a90fa8121b3a949ae580
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/replays/V93T10/R_a_W2-0-12977-5.json sha256 67dc0eb1e78c3ed3531dc0a9f6f2ff0ac1b0f99ef7f7beab56b03bf3d1a5dc59
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/replays/V93T12/R_a_W2-0-12977-5.json sha256 d2714eeebaebb0251b44930c699973690cc518d6848f56ac33c795f80626a6de

call R:a:W2-0-12977-5; ref -1.615200725239328779714563e-24; common regions 15

| code | value | abserr | |v - ref| | ratio | regions | differing | sum true err (diff) | sum total_err (diff) |
|---|---|---|---|---|---|---|---|---|
| tree | -1.615201e-24 | 9.24e-32 | 4.48e-33 | 0.0485 | 36 | 21 | -1.13e-35 | 7.58e-32 |
| V93T8 | -1.615201e-24 | 8.33e-32 | 4.48e-33 | 0.0538 | 32 | 17 | -1.41e-35 | 6.67e-32 |
| V93T10 | -1.615201e-24 | 9.03e-32 | 4.49e-33 | 0.0498 | 30 | 15 | -2e-37 | 7.37e-32 |
| V93T12 | -1.615201e-24 | 1.28e-31 | 3.58e-31 | 2.79 | 21 | 6 | -3.62e-31 | 1.11e-31 |

### tree: differing regions

| start | end | depth | type | class | V/pi | total_err | true err | decl/true | same span, other value under |
|---|---|---|---|---|---|---|---|---|---|
| 0x1.cfb190be8907ap+3 | 0x1.d34db960fd311p+3 | 7 | Levin | Levin | 1.53e+03 | 7.6e-38 | -2.62e-36 | 0.029 | ['V93T10', 'V93T12', 'V93T8'] |
| 0x1.d34db960fd311p+3 | 0x1.d51bcdb23745cp+3 | 8 | Levin | Levin | 701 | 3.19e-38 | -8.58e-38 | 0.372 | ['V93T10', 'V93T12', 'V93T8'] |
| 0x1.d51bcdb23745cp+3 | 0x1.d6e9e203715a8p+3 | 8 | Levin | Levin | 662 | 1.57e-38 | 1.62e-36 | 0.00972 | ['V93T10', 'V93T12', 'V93T8'] |
| 0x1.3744465540140p+4 | 0x1.39125aa67a28cp+4 | 7 | Levin | Levin | 10.7 | 4.77e-33 | -3.27e-36 | 1.46e+03 | ['V93T10', 'V93T8'] |
| 0x1.39125aa67a28cp+4 | 0x1.3ae06ef7b43d8p+4 | 7 | Levin | Levin | 9.52 | 6.72e-33 | -2.38e-37 | 2.82e+04 | ['V93T10', 'V93T8'] |
| 0x1.3ae06ef7b43d8p+4 | 0x1.3b53f40c02c2bp+4 | 9 | direct | CC, V < 8 pi | 2.22 | 1.04e-35 | 2e-37 | 52.2 | [] |
| 0x1.3b53f40c02c2bp+4 | 0x1.3bc779205147ep+4 | 9 | direct | CC, V < 8 pi | 2.16 | 9.6e-36 | 8.32e-37 | 11.5 | [] |
| 0x1.3bc779205147ep+4 | 0x1.3c3afe349fcd1p+4 | 9 | direct | CC, V < 8 pi | 2.1 | 9.44e-37 | 8.76e-37 | 1.08 | [] |
| 0x1.3c3afe349fcd1p+4 | 0x1.3cae8348ee524p+4 | 9 | direct | CC, V < 8 pi | 2.04 | 2.89e-36 | 5.87e-37 | 4.92 | [] |
| 0x1.3cae8348ee524p+4 | 0x1.3e7c979a2866fp+4 | 7 | Levin | Levin | 7.6 | 1.26e-33 | 1.04e-36 | 1.21e+03 | [] |
| 0x1.3e7c979a2866fp+4 | 0x1.404aabeb627bap+4 | 7 | Levin | Levin | 6.79 | 2e-33 | 5.61e-37 | 3.57e+03 | ['V93T10', 'V93T8'] |
| 0x1.404aabeb627bap+4 | 0x1.4218c03c9c906p+4 | 7 | Levin | Levin | 6.06 | 1.54e-33 | 1.32e-36 | 1.17e+03 | ['V93T10', 'V93T8'] |
| 0x1.4218c03c9c906p+4 | 0x1.43e6d48dd6a52p+4 | 7 | direct | CC, V < 8 pi | 5.42 | 1.18e-33 | -1.03e-36 | 1.15e+03 | ['V93T10', 'V93T8'] |
| 0x1.43e6d48dd6a52p+4 | 0x1.445a59a2252a5p+4 | 9 | direct | CC, V < 8 pi | 1.26 | 3.83e-36 | 2.31e-36 | 1.66 | [] |
| 0x1.445a59a2252a5p+4 | 0x1.44cddeb673af8p+4 | 9 | direct | CC, V < 8 pi | 1.23 | 1.28e-35 | 2.94e-39 | 4.34e+03 | [] |
| 0x1.44cddeb673af8p+4 | 0x1.454163cac234bp+4 | 9 | direct | CC, V < 8 pi | 1.19 | 7.71e-36 | -8.54e-37 | 9.03 | [] |
| 0x1.454163cac234bp+4 | 0x1.45b4e8df10b9ep+4 | 9 | direct | CC, V < 8 pi | 1.16 | 5.49e-36 | -2.52e-36 | 2.18 | [] |
| 0x1.45b4e8df10b9ep+4 | 0x1.4951118184e35p+4 | 6 | Levin | Levin | 8.18 | 4.13e-32 | 4.1e-36 | 1.01e+04 | ['V93T8'] |
| 0x1.4951118184e35p+4 | 0x1.4ced3a23f90ccp+4 | 6 | Levin | Levin | 6.53 | 9.83e-33 | -6.59e-36 | 1.49e+03 | ['V93T8'] |
| 0x1.4ced3a23f90ccp+4 | 0x1.508962c66d364p+4 | 6 | direct | CC, V < 8 pi | 5.21 | 6.5e-33 | -5.91e-36 | 1.1e+03 | ['V93T8'] |
| 0x1.508962c66d364p+4 | 0x1.54258b68e15fbp+4 | 6 | direct | CC, V < 8 pi | 4.16 | 6.3e-34 | -1.58e-36 | 398 | ['V93T8'] |

### V93T8: differing regions

| start | end | depth | type | class | V/pi | total_err | true err | decl/true | same span, other value under |
|---|---|---|---|---|---|---|---|---|---|
| 0x1.cfb190be8907ap+3 | 0x1.d34db960fd311p+3 | 7 | Levin | Levin | 1.53e+03 | 7.6e-38 | -2.68e-36 | 0.0284 | ['tree'] |
| 0x1.d34db960fd311p+3 | 0x1.d51bcdb23745cp+3 | 8 | Levin | Levin | 701 | 3.19e-38 | -6.23e-38 | 0.512 | ['tree'] |
| 0x1.d51bcdb23745cp+3 | 0x1.d6e9e203715a8p+3 | 8 | Levin | Levin | 662 | 1.57e-38 | 1.63e-36 | 0.00962 | ['tree'] |
| 0x1.3744465540140p+4 | 0x1.39125aa67a28cp+4 | 7 | Levin | Levin | 10.7 | 4.77e-33 | -4.12e-36 | 1.16e+03 | ['tree'] |
| 0x1.39125aa67a28cp+4 | 0x1.3ae06ef7b43d8p+4 | 7 | Levin | Levin | 9.52 | 6.72e-33 | 9.01e-37 | 7.46e+03 | ['tree'] |
| 0x1.3ae06ef7b43d8p+4 | 0x1.3bc779205147ep+4 | 8 | direct | CC, V < 8 pi | 4.37 | 8.06e-36 | -2.49e-36 | 3.24 | [] |
| 0x1.3bc779205147ep+4 | 0x1.3cae8348ee524p+4 | 8 | direct | CC, V < 8 pi | 4.13 | 6.31e-36 | -1.93e-36 | 3.27 | [] |
| 0x1.3cae8348ee524p+4 | 0x1.3d958d718b5cap+4 | 8 | direct | CC, V < 8 pi | 3.91 | 1.19e-35 | -1.66e-36 | 7.18 | [] |
| 0x1.3d958d718b5cap+4 | 0x1.3e7c979a2866fp+4 | 8 | direct | CC, V < 8 pi | 3.69 | 1.43e-35 | 6.02e-37 | 23.7 | [] |
| 0x1.3e7c979a2866fp+4 | 0x1.404aabeb627bap+4 | 7 | direct | CC, V < 8 pi | 6.79 | 4.47e-34 | -9.4e-38 | 4.75e+03 | ['tree'] |
| 0x1.404aabeb627bap+4 | 0x1.4218c03c9c906p+4 | 7 | direct | CC, V < 8 pi | 6.06 | 9.38e-34 | 7.26e-37 | 1.29e+03 | ['tree'] |
| 0x1.4218c03c9c906p+4 | 0x1.43e6d48dd6a52p+4 | 7 | direct | CC, V < 8 pi | 5.42 | 2.07e-34 | -3.71e-37 | 557 | ['tree'] |
| 0x1.43e6d48dd6a52p+4 | 0x1.45b4e8df10b9ep+4 | 7 | direct | CC, V < 8 pi | 4.84 | 1.08e-34 | -4.35e-37 | 249 | [] |
| 0x1.45b4e8df10b9ep+4 | 0x1.4951118184e35p+4 | 6 | Levin | Levin | 8.18 | 4.13e-32 | 3.45e-37 | 1.2e+05 | ['tree'] |
| 0x1.4951118184e35p+4 | 0x1.4ced3a23f90ccp+4 | 6 | direct | CC, V < 8 pi | 6.53 | 4.18e-33 | -1.77e-36 | 2.37e+03 | ['tree'] |
| 0x1.4ced3a23f90ccp+4 | 0x1.508962c66d364p+4 | 6 | direct | CC, V < 8 pi | 5.21 | 4.2e-33 | 4.06e-37 | 1.04e+04 | ['tree'] |
| 0x1.508962c66d364p+4 | 0x1.54258b68e15fbp+4 | 6 | direct | CC, V < 8 pi | 4.16 | 3.73e-33 | -3.08e-36 | 1.21e+03 | ['tree'] |

### V93T10: differing regions

| start | end | depth | type | class | V/pi | total_err | true err | decl/true | same span, other value under |
|---|---|---|---|---|---|---|---|---|---|
| 0x1.cfb190be8907ap+3 | 0x1.d34db960fd311p+3 | 7 | Levin | Levin | 1.53e+03 | 7.6e-38 | -2.68e-36 | 0.0284 | ['tree'] |
| 0x1.d34db960fd311p+3 | 0x1.d51bcdb23745cp+3 | 8 | Levin | Levin | 701 | 3.19e-38 | -6.23e-38 | 0.512 | ['tree'] |
| 0x1.d51bcdb23745cp+3 | 0x1.d6e9e203715a8p+3 | 8 | Levin | Levin | 662 | 1.57e-38 | 1.63e-36 | 0.00962 | ['tree'] |
| 0x1.3744465540140p+4 | 0x1.39125aa67a28cp+4 | 7 | Levin | Levin | 10.7 | 4.77e-33 | -4.12e-36 | 1.16e+03 | ['tree'] |
| 0x1.39125aa67a28cp+4 | 0x1.3ae06ef7b43d8p+4 | 7 | direct | CC, V in [8, 10) pi | 9.52 | 3.93e-34 | 9.01e-37 | 436 | ['tree'] |
| 0x1.3ae06ef7b43d8p+4 | 0x1.3bc779205147ep+4 | 8 | direct | CC, V < 8 pi | 4.37 | 8.06e-36 | -2.49e-36 | 3.24 | [] |
| 0x1.3bc779205147ep+4 | 0x1.3cae8348ee524p+4 | 8 | direct | CC, V < 8 pi | 4.13 | 6.31e-36 | -1.93e-36 | 3.27 | [] |
| 0x1.3cae8348ee524p+4 | 0x1.3d958d718b5cap+4 | 8 | direct | CC, V < 8 pi | 3.91 | 1.19e-35 | -1.66e-36 | 7.18 | [] |
| 0x1.3d958d718b5cap+4 | 0x1.3e7c979a2866fp+4 | 8 | direct | CC, V < 8 pi | 3.69 | 1.43e-35 | 6.02e-37 | 23.7 | [] |
| 0x1.3e7c979a2866fp+4 | 0x1.404aabeb627bap+4 | 7 | direct | CC, V < 8 pi | 6.79 | 4.47e-34 | -9.4e-38 | 4.75e+03 | ['tree'] |
| 0x1.404aabeb627bap+4 | 0x1.4218c03c9c906p+4 | 7 | direct | CC, V < 8 pi | 6.06 | 9.38e-34 | 7.26e-37 | 1.29e+03 | ['tree'] |
| 0x1.4218c03c9c906p+4 | 0x1.43e6d48dd6a52p+4 | 7 | direct | CC, V < 8 pi | 5.42 | 2.07e-34 | -3.71e-37 | 557 | ['tree'] |
| 0x1.43e6d48dd6a52p+4 | 0x1.45b4e8df10b9ep+4 | 7 | direct | CC, V < 8 pi | 4.84 | 1.08e-34 | -4.35e-37 | 249 | [] |
| 0x1.45b4e8df10b9ep+4 | 0x1.4ced3a23f90ccp+4 | 5 | Levin | Levin | 14.7 | 3.27e-32 | 1.14e-35 | 2.86e+03 | [] |
| 0x1.4ced3a23f90ccp+4 | 0x1.54258b68e15fbp+4 | 5 | direct | CC, V in [8, 10) pi | 9.37 | 3.41e-32 | -1.65e-36 | 2.07e+04 | [] |

### V93T12: differing regions

| start | end | depth | type | class | V/pi | total_err | true err | decl/true | same span, other value under |
|---|---|---|---|---|---|---|---|---|---|
| 0x1.cfb190be8907ap+3 | 0x1.d34db960fd311p+3 | 7 | Levin | Levin | 1.53e+03 | 7.6e-38 | -2.68e-36 | 0.0284 | ['tree'] |
| 0x1.d34db960fd311p+3 | 0x1.d51bcdb23745cp+3 | 8 | Levin | Levin | 701 | 3.19e-38 | -6.23e-38 | 0.512 | ['tree'] |
| 0x1.d51bcdb23745cp+3 | 0x1.d6e9e203715a8p+3 | 8 | Levin | Levin | 662 | 1.57e-38 | 1.63e-36 | 0.00962 | ['tree'] |
| 0x1.3744465540140p+4 | 0x1.45b4e8df10b9ep+4 | 4 | Levin | Levin | 59.4 | 4.46e-32 | -3.62e-31 | 0.123 | [] |
| 0x1.45b4e8df10b9ep+4 | 0x1.4ced3a23f90ccp+4 | 5 | Levin | Levin | 14.7 | 3.27e-32 | 1.14e-35 | 2.86e+03 | [] |
| 0x1.4ced3a23f90ccp+4 | 0x1.54258b68e15fbp+4 | 5 | direct | CC, V in [8, 10) pi | 9.37 | 3.41e-32 | -1.65e-36 | 2.07e+04 | [] |
