# btc-cent-forecast

BTC price forecasting agent for make.fun bounty. Pure Python 3.13 stdlib, no dependencies.

## Commands

```sh
# Setup
python3 --version  # requires Python 3.13+

# Run a case
mkdir -p input output
cp p00.in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
cat output/p00.out

# Test all practice cases
python3 test_runner.py
```

## Conventions

- Python 3.13, standard library only (no pip installs)
- Deterministic: same input produces identical output every run
- Input env vars: `MAKEFUN_INPUT_DIR` (default `/input`), `MAKEFUN_OUTPUT_DIR` (default `/output`)
- Input: one `.in` JSON file per case in input dir
- Output: `<case_id>.out` JSON with `{"forecasts": [{"case": "...", "price": N}]}`
- Prices clamped to [0.00001, 9999999.99999], rounded to 5 decimals

## Boundaries

- **NEVER** install external packages (numpy, scipy, pandas, etc.)
- **NEVER** make network calls — sandbox has no network
- **NEVER** use random seeds — output must be deterministic
- **NEVER** exceed 8 seconds, 1 CPU, 1024 MB per case

## Dependencies

None. Python 3.13 stdlib only: `json`, `math`, `os`, `statistics`.

## Config

- `MAKEFUN_INPUT_DIR` — input directory (default: `/input`)
- `MAKEFUN_OUTPUT_DIR` — output directory (default: `/output`)

## Error Handling

- Missing input dir: exits with error
- Invalid JSON or missing markets: exits with error
- Non-finite prices: skipped silently
- Any exception: writes persistence forecast (last price) as fallback

## Troubleshooting

1. **"No .in files found"** — ensure `MAKEFUN_INPUT_DIR` contains exactly one `.in` file
2. **Output differs between runs** — model is deterministic; check input file hasn't changed
3. **Ties with persistence** — gate threshold not met for this case's deviation level; this is expected behavior
4. **Memory/time exceeded** — model uses O(n) memory and O(n) time; verify 4320-point input