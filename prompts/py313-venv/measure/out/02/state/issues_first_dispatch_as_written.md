# py313-venv 02 — issues opened (board §3 form)

### [02-the-token-mode-setting-comes-after-import-ray-so-every-ray-start-fails]

**Opened 2026-10-25 by py313-venv 02 (R7, Mac Studio).** `d833ee7` sets `RAY_AUTH_MODE=token` in
`config/ray_local.py` at import. Ray 2.59.0 caches the auth mode in its C++ `RayConfig` when `ray` is
imported. Every Ray start here imports `ray` first: `RunDrivers/quadsource_atol_sweep.py:478/480` and
`:552/554`, and `main.py:12` before `:84`. So the driver caches `disabled`. The GCS server inherits the
environment and runs in token mode. Ray's own default path refreshes the cache
(`_enable_token_auth()` → `Config.initialize("")`); setting the variable does not.

**Measured:** R7 `reference` (`py313-venv-02-reference-20261025T163202`) ended `failed`/1 in 23 s.
- 20× `InvalidAuthToken` from the driver, and the GCS server's `Missing authorization header`.
- Then `RuntimeError: Failed to start GCS … RAY_AUTH_MODE is currently 'disabled' (not 'token')`.
- No store was created.
- Without starting Ray, `out/auth_mode_probe_02.txt` shows the cached mode by import order: 0
  (disabled) with `ray` first, 1 (token) with `config.ray_local` first.

**Impact:** every Ray start of this repository on Ray 2.59.0: builds, and probably the readers (not
checked). Nothing is stored wrongly; the start fails.
**Why the tests missed it:** `RunDrivers/tests/test_ray_auth_mode.py` checks the import comes before
`ray.init`, not before `import ray`.
**Next step:** the user's choice — fix the import order with a test that catches it, or revert
`d833ee7` — then 02 is re-dispatched from step 7.

### Also for the board (not a new issue)
The prompt's step-8 `ps -E -p <driver pid>` check cannot show a variable the process sets in itself
after exec (macOS `ps -E` reads the exec-time environment; `out/ps_env_probe_02.txt`). The children
(`gcs_server`, `raylet`) carry it. The check should name them instead.
