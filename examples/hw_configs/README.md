# Sample `--hw-config` files

Starting points for `main.py --hw-config <path>` (see `schemas/config.py`'s
`HWConfig` for the full field list). Each is a real, valid config verified
against this project's own calibration/convergence logic, not just
schema-valid JSON:

| File | Use it to see... |
|---|---|
| `cim_v1_128x128.json` | The default config `main.py` uses when `--hw-config` is omitted -- a reasonable starting template. Uncalibrated (`adc_bits=8` doesn't match any `tools/calibration.py` reference) and converges immediately. |
| `calibrated_neurosim_v1_5.json` | `crossbar_rows`/`crossbar_cols`/`adc_bits` exactly match `tools.calibration.KNOWN_REFERENCES[0]` (NeuroSim V1.5's published config) -- the dashboard's calibration section will show "calibrated" with a real citation instead of the uncalibrated warning. |
| `high_ir_drop_never_converges.json` | `wire_resistance_ohm_per_um=5.0` pushes `ir_drop_error_pct` past `verifier_tool`'s 5% convergence bound -- the run will hit HITL after `MAX_RETRY_LIMIT` (nodes/evaluator.py) retries instead of converging, useful for exercising/demoing that flow. |
