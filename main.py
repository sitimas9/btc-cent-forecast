#!/usr/bin/env python3
"""
btc-cent-forecast -- BTC/USDC 4-hour-ahead point forecaster.

Pure Python 3 standard library only.  Deterministic.  No network.

Model
-----
The forecast is produced in log space and is the sum of two economically
grounded components, both estimated only from the supplied price history
(72 h of 1-minute closes):

1. Gated multi-scale mean reversion (the main edge).
   Bitcoin 4 h returns are close to unpredictable, but large *deviations from
   the local mean* overshoot and revert (liquidity cascades / overreaction).
   Empirically (walk-forward, variance-ratio 0.54 at the 4 h horizon) the
   forward 4 h return is negatively correlated with the price's distance from
   its 8 h / 12 h / 24 h geometric mean -- but ONLY for large deviations.
   Mild deviations drift on.  So each anchor contributes

       -phi_L * excess(z_L, gate)      z_L = log(P_t / geom_mean_L)
       excess(z, c) = z - sign(z) * min(|z|, c)

   which is exactly 0 (=> plain persistence) until |z_L| exceeds the gate,
   then reverts with slope phi_L.  This is what the naive models miss and it
   is where persistence is beaten.

2. Heavily damped long-horizon drift (a Holt-style trend term).
   A small fraction of the 72 h drift is carried forward.  It is nearly
   neutral in choppy windows and pays off in trending ones.

Constants below were selected by walk-forward evaluation over every
non-overlapping and overlapping 4 h block derivable from the practice data
(six evaluation regimes); see AGENTS.md.  Every regime improves on the
persistence baseline.

I/O contract
------------
Input : $MAKEFUN_INPUT_DIR/<id>.in   (default /input)
        {"at", "case", "horizon_minutes", "markets":[{"id","name","prices":[...]}]}
Output: $MAKEFUN_OUTPUT_DIR/<id>.out (default /output)
        {"forecasts":[{"case":"<id>","price":<number>}]}
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Tuned model constants (see AGENTS.md for the walk-forward evidence)
# ---------------------------------------------------------------------------
ANCHORS = ((480, 0.35), (720, 0.15), (1440, 0.32))  # (lookback minutes, phi_L)
GATE_SIGMA = 0.75        # reversion only kicks in beyond this many sigma of z
DRIFT_KAPPA = 0.35       # fraction of the long-horizon drift carried forward
DRIFT_REF_ANCHOR = 4320  # long-horizon drift measured over the full 72 h
SIGMA_WINDOW = 2880      # window used to scale z into sigma units
SIGMA_STEP = 60          # sampling stride for that scale estimate
SIGMA_FLOOR = 0.5        # never trust a scale below this * random-walk scale
SIGMA_RET_WINDOW = 1440  # window for the 1-minute vol used in the RW scale
SIGMA_RET_STEP = 10      # sampling stride for the 1-minute vol
MAX_LOG_MOVE = 0.03      # hard cap on the predicted log move (+/- 3 %)
DEFAULT_HORIZON = 240

MIN_PRICE = 0.0
MAX_PRICE = 10_000_000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sanitize(raw):
    """Keep only finite, strictly positive floats."""
    out = []
    if not isinstance(raw, (list, tuple)):
        return out
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > MIN_PRICE:
            out.append(f)
    return out


def _prefix_logs(prices):
    logs = [math.log(p) for p in prices]
    pref = [0.0] * (len(logs) + 1)
    for i, v in enumerate(logs):
        pref[i + 1] = pref[i] + v
    return logs, pref


def _scale_estimate(logs, pref, n):
    """Volatility scale of a 24 h deviation, in log units.

    Two independent estimates, whichever is larger:
      * std-dev of the 24 h deviation statistic itself over the recent past
        (stationary, directly measures the quantity being gated), and
      * a random-walk scale sigma_1m * sqrt(1440 / 3) implied by the recent
        1-minute volatility (insurance against an under-estimated scale,
        which would make the gate too tight and over-trade).
    """
    # --- std of z_1440 over the recent window
    sd_roll = None
    start = max(1500, n - SIGMA_WINDOW)
    samples = []
    s = start
    while s < n:
        z = logs[s] - (pref[s + 1] - pref[s + 1 - 1440]) / 1440.0
        samples.append(z)
        s += SIGMA_STEP
    if len(samples) > 10:
        m = sum(samples) / len(samples)
        var = sum((x - m) ** 2 for x in samples) / (len(samples) - 1)
        if var > 0.0:
            sd_roll = math.sqrt(var)

    # --- random-walk scale from 1-minute returns
    sd_rw = None
    rets = []
    i = max(1, n - SIGMA_RET_WINDOW)
    while i < n:
        rets.append(logs[i] - logs[i - 1])
        i += SIGMA_RET_STEP
    if len(rets) > 20:
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        if var > 0.0:
            sd_rw = math.sqrt(var) * math.sqrt(1440.0 / 3.0)

    sd = sd_roll if sd_roll else None
    if sd_rw is not None:
        floor = SIGMA_FLOOR * sd_rw
        sd = floor if sd is None else max(sd, floor)

    if sd is None:  # last resort: spread of the 24 h deviation over all data
        fallback = [logs[k] - (pref[k + 1] - pref[max(0, k - 1439)]) / min(1440, k + 1)
                    for k in range(len(logs))]
        if len(fallback) > 2:
            m = sum(fallback) / len(fallback)
            var = sum((x - m) ** 2 for x in fallback) / (len(fallback) - 1)
            sd = math.sqrt(var) if var > 0.0 else 0.0
        else:
            sd = 0.0
    return sd


def _reversion(price, logs, pref, n, horizon, sigma):
    """Gated multi-scale mean-reversion component, in log-return units."""
    move = 0.0
    if sigma is None or sigma <= 0.0 or n < 2:
        return 0.0
    gate = GATE_SIGMA * sigma
    for look, phi in ANCHORS:
        if n < look:
            continue
        mean = (pref[n] - pref[n - look]) / look  # mean log over last `look` pts
        z = logs[n - 1] - mean
        excess = z - math.copysign(min(abs(z), gate), z)
        move += -phi * excess
    return move


def _drift(logs, n, horizon):
    """Heavily damped long-horizon drift component, in log-return units."""
    if n <= DRIFT_REF_ANCHOR or DRIFT_REF_ANCHOR <= 0:
        return 0.0
    run = logs[n - 1] - logs[n - 1 - DRIFT_REF_ANCHOR]
    return DRIFT_KAPPA * run * (float(horizon) / DRIFT_REF_ANCHOR)


def forecast_market(prices, horizon):
    """Return a single point forecast of the price `horizon` minutes ahead."""
    prices = _sanitize(prices)
    n = len(prices)
    if n == 0:
        return None
    last = prices[-1]
    if n < 2:
        return last

    logs, pref = _prefix_logs(prices)
    sigma = _scale_estimate(logs, pref, n)
    move = _reversion(last, logs, pref, n, horizon, sigma) + _drift(logs, n, horizon)

    move = max(-MAX_LOG_MOVE, min(MAX_LOG_MOVE, move))
    return last * math.exp(move)


def forecast_case(data):
    """Average the per-market forecasts of one case (single market in practice)."""
    horizon = data.get("horizon_minutes", DEFAULT_HORIZON)
    try:
        horizon = float(horizon)
    except (TypeError, ValueError):
        horizon = DEFAULT_HORIZON
    if not math.isfinite(horizon) or horizon <= 0:
        horizon = DEFAULT_HORIZON

    markets = data.get("markets")
    if not isinstance(markets, list) or not markets:
        # tolerate a bare {"prices": [...]}
        markets = [data] if isinstance(data.get("prices"), list) else []

    preds = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        prices = m.get("prices")
        p = forecast_market(prices, horizon) if prices is not None else None
        if p is not None and math.isfinite(p):
            preds.append(p)
    if not preds:
        return None
    return sum(preds) / len(preds)


def make_price(value):
    """Clamp + round into the contract's legal range (5 decimals)."""
    if value is None or not math.isfinite(value):
        return None
    lo = MIN_PRICE
    hi = MAX_PRICE
    v = min(max(float(value), lo), hi)
    v = round(v, 5)
    if v <= lo:
        v = round(lo + 1e-5, 5)
    if v >= hi:
        v = round(hi - 1e-5, 5)
    return v


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _case_id(stem, data):
    cid = data.get("case") if isinstance(data, dict) else None
    return cid if isinstance(cid, str) and cid else stem


def process_file(in_path, out_dir):
    stem = os.path.basename(in_path)
    if stem.endswith(".in"):
        stem = stem[:-3]
    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    case_id = _case_id(stem, data)
    try:
        price = forecast_case(data)
    except Exception as exc:  # never fail to produce a syntactically valid file
        print("WARN: model error on %s: %s" % (stem, exc), file=sys.stderr)
        price = None
    if price is None:  # last-ditch fallback: persistence
        m = (data.get("markets") or [{}])[0] if isinstance(data, dict) else {}
        ps = _sanitize(m.get("prices") if isinstance(m, dict) else None)
        price = ps[-1] if ps else 1.0
    price = make_price(price)

    out_path = os.path.join(out_dir, stem + ".out")
    payload = {"forecasts": [{"case": case_id, "price": price}]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.write("\n")
    return case_id, price, out_path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    in_dir = argv[0] if len(argv) > 0 else os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = argv[1] if len(argv) > 1 else os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(in_dir))
    except OSError as exc:
        print("ERROR: cannot list %s: %s" % (in_dir, exc), file=sys.stderr)
        return 2

    cases = [n for n in names if n.endswith(".in")]
    if not cases:
        print("ERROR: no .in files in %s" % in_dir, file=sys.stderr)
        return 3

    for name in cases:
        case_id, price, out_path = process_file(os.path.join(in_dir, name), out_dir)
        print("%s -> %s  price=%s" % (case_id, out_path, price))
    return 0


if __name__ == "__main__":
    sys.exit(main())
