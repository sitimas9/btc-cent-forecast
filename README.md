# btc-cent-forecast

BTC price forecasting agent for make.fun's ["Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours"](https://make.fun/bounty/btc-cent-forecast-01) bounty.

## Model

Anchor-reversion model with gated activation. Uses three lookback windows (8h, 12h, 24h) to detect mean-reverting deviations from geometric mean. A soft gate (theta=0.75 sigma) suppresses the reversion signal during mild fluctuations. Includes a heavily-shrunk long-horizon drift term for trending regimes.

**Parameters:**
- Anchors: 480min (phi=0.35), 720min (phi=0.15), 1440min (phi=0.32)
- Gate sigma: 0.75
- Drift kappa: 0.35
- Deterministic. Stdlib only. No external dependencies.

## Run

```sh
mkdir -p input output
cp p00.in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
cat output/p00.out
```

## License

MIT