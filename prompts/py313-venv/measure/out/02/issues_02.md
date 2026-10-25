# py313-venv 02 — issues (board §3/§4 form)

**Open after prompt 02: none.** The one issue below was opened by the first dispatch and is already
**closed**, on the laptop, by `14ada2f`. The board's §4 holds it, so it adds no row to
`docs/OPEN_ISSUES.md`. The resume opened nothing. The first dispatch's file is kept as written at
`state/issues_first_dispatch_as_written.md`.

### [02-the-token-mode-setting-comes-after-import-ray-so-every-ray-start-fails] — CLOSED

**Opened 2026-10-25 by py313-venv 02's first dispatch (R7, Mac Studio, at `aac52bc`). Closed
2026-10-25 on the laptop by `14ada2f`, and shown closed on the Mac Studio by the resume at
`c9026af`.**

**What it was.** `d833ee7` set `RAY_AUTH_MODE=token` in `config/ray_local.py` at import. But
Ray 2.59.0 caches the auth mode in its C++ `RayConfig` when `ray` is imported, and every Ray start
here imports `ray` first:
- `RunDrivers/quadsource_atol_sweep.py:478/480` and `:552/554`;
- `main.py:12`, before `:84`.

So the driver cached `disabled`, while the GCS server it started inherited the environment and ran
in token mode. Ray's own default path refreshes the cache (`_enable_token_auth()` →
`Config.initialize("")`); setting the variable did not.

**Measured by the first dispatch.** R7's first `reference` (`py313-venv-02-reference-20261025T163202`)
ended `failed`/1 in 23 s:
- the driver logged `InvalidAuthToken` 20 times (22 lines match `InvalidAuthToken|AuthenticationError`
  across its three logs), and the GCS server logged `Missing authorization header`;
- then `RuntimeError: Failed to start GCS … RAY_AUTH_MODE is currently 'disabled' (not 'token')`;
- no store was created.

Without starting Ray, `out/auth_mode_probe_02.txt` showed the cached mode by import order: 0
(disabled) with `ray` first, and 1 (token) with `config.ray_local` first.

**Why the tests missed it:** `RunDrivers/tests/test_ray_auth_mode.py` checked that the import came
before `ray.init`, not before `import ray`.

**The fix, `14ada2f`.**
- `config/ray_local.py` now calls `ray._raylet.Config.initialize("")` after setting the variable,
  when `ray` is already imported.
- `TestRaySeesTheMode` (3 tests) reads Ray's own mode in a fresh interpreter for both import orders.
  RunDrivers is now 127.

**Shown on the Mac Studio by the resume (`c9026af`):**
- RunDrivers is 127 OK.
- R7's four launches all passed `ray.init`.
- The `gcs_server`'s environment held `RAY_AUTH_MODE=token`.
- None of R7's twelve logs, nor any reader log, holds `InvalidAuthToken`, `AuthenticationError` or
  `Token authentication is enabled`.
- Both stores are `39809dca…`.

**Also for the board (a note, not an issue).** The prompt's step-8 `ps -E -p <driver pid>` check
could not show a variable that the process sets in itself after `exec`. macOS `ps -E` reads the
exec-time environment (`out/ps_env_probe_02.txt`). The children (`gcs_server`, `raylet`) carry the
variable. §7 amended the check to name the `gcs_server`, and the resume made it so.
