# Part 1 of the references (run levin-phase-floor-06g-references-20261101T175231): results

Run `done`, exit 0, 60 of 60 units, 17:52:31Z–18:40:42Z (driver pid 5223, sh pid 5219). Every
command's output is in `out/cmd/30`–`35`; the readings in `out/readings_06i13/`, their logs in
`out/reading_logs/06i13/`, the command line's in `out/cli/`.

## The captures (`out/cmd/30`, `31`; `out/capture.json` `62b91d27…`, `out/check_captures.json` `ed6c6fb4…`)

- **102 of 102 captured and matching the g census bit for bit** on x1_capture.py's tuple (value and
  abserr by hex, num_regions, max_depth, converged, label): Z3's 30, the 60 drawn, V8's 12. Capture
  wall 1,180 s, six census workers.
- **102 of 102 reproduce the g census's region arrays bit for bit** (every key of the call's region
  record, arrays by dtype, shape and bytes), and every capture worker ran the census's harness
  byte for byte (`scripts_sha256` equal to the census record's).
- Every worker reported `ray_initialized` false.

## Slicing and 06i's inputs (`out/cmd/32`; `out/inputs.json` `d6d2a978…`)

102 captures sliced into `state/inputs/<sid>.pkl`, each blob's SHA-256 equal to the capture record.
06i's 13 inputs on this host equal 06i's `inputs_SHA256SUMS.txt`, 13 of 13.

## 06i's 13 groups (`out/cmd/33`; `out/compare13.json` `34931fc1…`)

**13 of 13 equal**, both the criterion as written (route, span; Route A signs, coefficients and
orders; Route B signs and each active factor's phase spline knots, coefficients, degree, leading
table nodes and breaks, by bytes) and the integrand as the tool reads it (f_sin, f_cos at 65 points;
Route A K_exact, C at 40 digits, u_breaks, the residual at the 65 points; Route B theta,
theta_mod_2pi, theta_deriv at the 65 points and production_edges). Descriptive only: on the seven
Route A samples the 06i inputs' groups have no `K_lo` (pickled before 06h) and the g captures' do;
theta and theta_mod_2pi at the 65 points evaluated equal. So the 13 keep 06i's references.

## Routes (`out/cmd/34`; `out/routes.json` `93023673…`)

All 102 on the expected route (analytic blocks A, groups B); no `production_edges` raised.

## The tool against 06i's readings (`out/cmd/35`; `out/tool_vs_06i.json` `9902e9be…`)

**13 of 13 agree** under the fixed rule (every reading within s_tool + s_06i + m/100). 44 of the 52
readings are bit-identical to 06i's. The eight that are not: FCC res 1 and res 2 on Route A samples
9, 22, 25, 27, each pair offset by the same amount at both resolutions:
D/m = 2.79e-6 (s09), 1.20e-5 (s22), 4.01e-5 (s25), 2.46e-5 (s27); D over the allowance at most
0.0033; panel counts, K_exact, C and u_breaks equal 06i's. Every QAWO reading and every Route B
reading is bit-identical. The section 8 statistic (reported, not used): sample 8 is 5.14 for both
the tool and 06i, as 06i found; the others below 1.
**The command line** on z25 and z13: each of its four readings bit-identical to `one_reading.py`'s.
