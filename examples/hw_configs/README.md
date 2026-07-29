# Sample `--hw-config` files

Starting points for `main.py --hw-config <path>` (see `schemas/config.py`'s
`HWConfig` for the full field list). Each is a real, valid config verified
against this project's own calibration/convergence logic, not just
schema-valid JSON:

| File | Use it to see... |
|---|---|
| `default_cim_v1_128x128.json` | The default config `main.py` uses when `--hw-config` is omitted -- a reasonable starting template. Uncalibrated (`adc_bits=8` doesn't match any `tools/calibration.py` reference) and converges immediately. |
| `high_ir_drop_never_converges.json` | `wire_resistance_ohm_per_um=5.0` pushes `ir_drop_error_pct` past `verifier_tool`'s 5% convergence bound -- the run will hit HITL after `MAX_RETRY_LIMIT` (nodes/evaluator.py) retries instead of converging, useful for exercising/demoing that flow. |

## Calibrated configs (real literature/silicon data)

Each `calibrated_*.json` below has `crossbar_rows`/`crossbar_cols`/`adc_bits`
set to *exactly* match one entry in `tools.calibration.KNOWN_REFERENCES` --
pass one of these to `--hw-config` and the run starts already corrected
against real (mostly silicon-measured) data instead of the uncalibrated
default (factor 1.0). No hand-editing of `crossbar_rows`/`cols`/`adc_bits`
needed; the other fields (`num_tiles`, `dac_bits`, `noc_topology`,
`wire_resistance_ohm_per_um`, etc.) are filled with the same reasonable
defaults `default_cim_v1_128x128.json` uses, since `KNOWN_REFERENCES` doesn't
constrain them. Session start prints `[calibration] exact match` to confirm.

| File | Reference (`tools/calibration.py`) | Array | ADC | Tech |
|---|---|---|---|---|
| `calibrated_neurosim_v1_5.json` | NeuroSim V1.5 (simulated) | 128x128 | 7-bit | ReRAM |
| `calibrated_correll_2025_reram_256x64.json` | Correll et al., IEEE JSSC 2025 | 256x64 | 8-bit | ReRAM |
| `calibrated_daram_2021_reram_64x32.json` | Chen, Chen & Gu, ISSCC 2021 (DARAM) | 64x32 | 5-bit | ReRAM |
| `calibrated_yu_2020_sram_128x128_1bit.json` | Yu et al., CICC 2020, 1-bit ADC mode | 128x128 | 1-bit | SRAM |
| `calibrated_yu_2020_sram_128x128_5bit.json` | Yu et al., CICC 2020, 5-bit ADC mode (same chip as above) | 128x128 | 5-bit | SRAM |
| `calibrated_dong_2020_sram_64x64.json` | Dong, Sinangil et al., ISSCC 2020 | 64x64 | 4-bit | SRAM |
| `calibrated_deaville_2022_mram_256x512.json` | Deaville, Zhang & Verma, VLSI Symp. 2022 | 256x512 | 6-bit | MRAM |
| `calibrated_korea_univ_2022_sram_256x80.json` | Korea Univ. (arXiv:2211.16008 / Lee, Kim & Park, IEEE JSSC 2024) | 256x80 | 4-bit | SRAM |

Have a specific target chip's array size / ADC bit resolution instead of one
of these? `--allow-approximate-calibration` will scale-law-transfer the
*nearest* reference's correction factor to any custom `--hw-config` instead
of leaving it uncalibrated -- see the README's CLI options table.
