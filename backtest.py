#!/usr/bin/env python3
"""
Historical backtest - walk-forward validation across multiple market regimes.

Why this exists: the paper-trading trial validates execution (fees, slippage,
stop-losses) honestly, but it only ever runs during WHATEVER market regime is
happening right now - a few weeks is one regime, not a real test. Research on
algorithmic trading strategies consistently finds that backtested performance
on a single period is a poor predictor of live results (a well-known study of
888 published strategies found backtest Sharpe ratios explain under 3% of the
variance in real returns), and industry practice is to require validation
across multiple market cycles (bull/bear/chop) before trusting a strategy.

This script replays the EXACT SAME screen()/finalize() scoring engine used in
production, hour by hour, against real historical price data spanning several
months - so it can answer "does this scoring approach have genuine edge across
different market conditions, or did it just get lucky in one regime". It does
NOT modify data/state.json (production state) - it builds a fully separate
state under backtest_data/, and is meant to be run manually, on demand:

    python3 backtest.py --months 6

Known, deliberate limitations (so the results are read honestly):
  - No hype/sentiment layer: historical CoinGecko sentiment for arbitrary past
    hours isn't available, so hype_notes is always {} during backtest (the
    "why" text will say "ei leidnud värsket kajastust" for every trade). This
    means the backtest validates the momentum/trend/alpha/order-book math, NOT
    the hype-bonus layer - live paper trading is still the only way to
    evaluate that part.
  - No order-book replay either (same reason - no historical order-book data
    available) - book_imbalance feature defaults to neutral (0.5) throughout.
  - Fear & Greed feature defaults to neutral too (no historical FNG series
    fetched here) - this backtest points engine.DATA_DIR at its own isolated
    folder specifically so it never reads today's real market_regime.json
    and mistakenly applies a live reading to a simulated past hour.
  - The funding-arb, grid, and regime-gated short sleeves stay entirely
    INACTIVE here - funding-arb needs real funding-rate history (not
    fetched), grid needs 15m candles (only hourly are fetched for this
    tool), and the short sleeve's regime gate requires a real Fear&Greed
    reading (see above, always None here). Only the momentum long
    portfolio (state["portfolio"]) produces any trades in this backtest.
  - Crypto.com's public candlestick API caps each call at 300 candles, so
    reaching back several months requires many paginated calls per instrument
    (see fetch_full_history). A 6-month backtest over ~15 instruments takes on
    the order of 10-20 minutes - this is a diagnostic tool you run when you
    want a regime-robustness check, not something wired into the hourly job.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import engine
import broker as broker_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BT_DIR = os.path.join(BASE_DIR, "backtest_data")
CANDLE_URL = "https://api.crypto.com/exchange/v1/public/get-candlestick"

# Kept smaller than the full 45-symbol watchlist by default - each extra
# instrument multiplies the number of paginated history fetches needed.
# Override with --instruments to backtest a different/larger set.
DEFAULT_INSTRUMENTS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "AVAXUSD",
    "LINKUSD", "PEPEUSD", "WIFUSD", "BONKUSD", "SHIBUSD",
]

REGIME_WINDOW_HOURS = 30 * 24   # 30-day rolling window for regime classification
REGIME_BULL_PCT = 15.0          # BTC up >15% over the window -> "bull"
REGIME_BEAR_PCT = -15.0         # BTC down >15% over the window -> "bear"


def http_get_json(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _real_instrument_name(symbol):
    """Same USD->USD_pair resolution fetch_and_run.py uses, simplified: try
    BASE_USD then BASE_USDT (backtest doesn't need -PERP)."""
    base = symbol[:-3] if symbol.endswith("USD") else symbol
    for real in (f"{base}_USD", f"{base}_USDT"):
        try:
            raw = http_get_json(f"{CANDLE_URL}?instrument_name={real}&timeframe=1h&count=2")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            continue
        if (raw.get("result") or {}).get("data"):
            return real
    return None


def fetch_full_history(real_name, months, sleep_s=0.15):
    """Paginate backwards through Crypto.com's 300-candle-per-call cap to
    build `months` worth of hourly candles (oldest first)."""
    target_hours = months * 30 * 24
    all_candles = {}
    end_ts = None
    calls = 0
    while len(all_candles) < target_hours:
        url = f"{CANDLE_URL}?instrument_name={real_name}&timeframe=1h&count=300"
        if end_ts is not None:
            url += f"&end_ts={end_ts}"
        try:
            raw = http_get_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            print(f"    WARN: {real_name} fetch failed ({e}), stopping pagination here", file=sys.stderr)
            break
        data = (raw.get("result") or {}).get("data") or []
        if not data:
            break
        new_count = 0
        for d in data:
            if d["t"] not in all_candles:
                # API returns OHLCV as strings - normalize to float once here
                all_candles[d["t"]] = {"t": d["t"], "o": float(d["o"]), "h": float(d["h"]),
                                       "l": float(d["l"]), "c": float(d["c"]), "v": float(d["v"])}
                new_count += 1
        if new_count == 0:
            break  # hit the start of available history, no new candles
        oldest_ts = min(d["t"] for d in data)
        end_ts = oldest_ts - 1
        calls += 1
        time.sleep(sleep_s)
        if calls > 60:  # safety valve against runaway pagination
            break
    ordered = sorted(all_candles.values(), key=lambda d: d["t"])
    return ordered[-target_hours:] if len(ordered) > target_hours else ordered


def classify_regime(btc_closes_by_ts, ts):
    """Bull/bear/chop label for one hour, based on BTC's own trailing
    REGIME_WINDOW_HOURS % change - the same idea as the live engine's alpha
    benchmark, just used here to slice backtest results by market condition."""
    sorted_ts = [t for t in btc_closes_by_ts if t <= ts]
    if len(sorted_ts) < REGIME_WINDOW_HOURS:
        return "chop"  # not enough trailing history yet - default, low-confidence label
    sorted_ts.sort()
    window = sorted_ts[-REGIME_WINDOW_HOURS:]
    start_price = btc_closes_by_ts[window[0]]
    end_price = btc_closes_by_ts[window[-1]]
    if start_price <= 0:
        return "chop"
    pct = (end_price - start_price) / start_price * 100
    if pct > REGIME_BULL_PCT:
        return "bull"
    if pct < REGIME_BEAR_PCT:
        return "bear"
    return "chop"


def run_backtest(months, instruments):
    os.makedirs(BT_DIR, exist_ok=True)
    state_path = os.path.join(BT_DIR, "state.json")
    if os.path.exists(state_path):
        os.remove(state_path)  # always start from a clean slate

    print(f"Laen {months} kuu ajaloolisi tunniküünlaid {len(instruments)} instrumendi kohta "
          f"(Crypto.com lubab 300 küünalt/kutse, seega mitu kutset instrumendi kohta - see võtab aega)...")

    histories = {}   # instrument -> {ts_ms: candle_dict}
    for sym in instruments:
        real = _real_instrument_name(sym)
        if not real:
            print(f"  {sym}: ei leidnud toimivat instrumenti, jäetakse vahele")
            continue
        candles = fetch_full_history(real, months)
        if len(candles) < REGIME_WINDOW_HOURS:
            print(f"  {sym} ({real}): ainult {len(candles)} tunniküünalt saadaval, jäetakse vahele "
                  f"(vaja vähemalt {REGIME_WINDOW_HOURS} režiimi-klassifikatsiooniks)")
            continue
        histories[sym] = {c["t"]: c for c in candles}
        print(f"  {sym} ({real}): {len(candles)} tunniküünalt "
              f"({datetime.utcfromtimestamp(candles[0]['t']/1000).strftime('%Y-%m-%d')} -> "
              f"{datetime.utcfromtimestamp(candles[-1]['t']/1000).strftime('%Y-%m-%d')})")

    if "BTCUSD" not in histories:
        print("BTCUSD ajalugu puudub - regiimi-klassifikatsioon ja alpha-arvutus vajavad seda. Katkestan.")
        return

    btc_closes_by_ts = {t: float(c["c"]) for t, c in histories["BTCUSD"].items()}
    all_ts = sorted(set.intersection(*[set(h.keys()) for h in histories.values()]))
    print(f"\nÜhine ajatelg: {len(all_ts)} tundi kõigi instrumentide peale.\n")

    # Rolling 24h volume (sum of the last 24 hourly candles' notional) per
    # symbol, keyed by ts - production's volume_value is a genuine 24h
    # figure (Crypto.com ticker's "vv" field); using a single hour's candle
    # volume here would be ~24x too small and would wrongly trip
    # MIN_CANDIDATE_VOLUME_USD for almost everything.
    volume_24h_by_symbol = {}
    for sym, hist in histories.items():
        sorted_ts = sorted(hist.keys())
        notionals = [hist[t]["v"] * hist[t]["c"] for t in sorted_ts]
        rolling = {}
        window_sum = 0.0
        window = []
        for t, notional in zip(sorted_ts, notionals):
            window.append(notional)
            window_sum += notional
            if len(window) > 24:
                window_sum -= window.pop(0)
            rolling[t] = window_sum
        volume_24h_by_symbol[sym] = rolling

    tickers_path = os.path.join(BT_DIR, "tickers.json")
    cands_path = os.path.join(BT_DIR, "cands.json")
    hype_path = os.path.join(BT_DIR, "hype.json")
    prices_path = os.path.join(BT_DIR, "prices.json")
    summary_path = os.path.join(BT_DIR, "summary.txt")
    with open(hype_path, "w") as f:
        json.dump({}, f)  # always empty - see module docstring

    # Redirect the engine module's file targets at this isolated backtest
    # folder - must happen once, before the loop, and BEFORE any engine call
    # that reads DATA_DIR-relative paths (e.g. market_regime.json), so a
    # simulated 2026-01 hour never accidentally sees today's real FNG value.
    engine.DATA_DIR = BT_DIR
    engine.STATE_PATH = state_path
    engine.DASHBOARD_PATH = os.path.join(BT_DIR, "dashboard.html")
    engine.INDEX_PATH = os.path.join(BT_DIR, "index.html")

    # engine.py uses now_ts() everywhere for elapsed-time logic - position
    # age, 24h/7d follow-up resolution, kill-switch's daily window, alert
    # cooldowns. Left at the real wall clock, an 8-month backtest that
    # finishes in a few minutes would never let any of that fire (a
    # position can't turn 24h old when the whole run takes 4 minutes of
    # real time) - confirmed in the wild as 0 trades ever resolving via the
    # natural 24h check and the model training 0 steps on every backtest
    # run. Overriding now_ts() to the current simulated hour, updated each
    # iteration below, makes all of that logic operate on simulated time
    # consistently, exactly like a real hourly cron would experience it.
    def _simulated_now():
        return _simulated_now.ts_seconds
    _simulated_now.ts_seconds = all_ts[0] / 1000.0 if all_ts else 0.0
    engine.now_ts = _simulated_now

    t0 = time.time()
    for i, ts in enumerate(all_ts):
        _simulated_now.ts_seconds = ts / 1000.0
        tickers = []
        for sym, hist in histories.items():
            c = hist.get(ts)
            if not c:
                continue
            past_ts = ts - 24 * 3600 * 1000
            past_c = hist.get(past_ts)
            if not past_c:
                # fall back to the earliest available candle for this symbol
                # as a rough baseline rather than skipping the symbol entirely
                earliest = min(hist.keys())
                past_c = hist[earliest]
            change = (c["c"] - past_c["c"]) / past_c["c"] if past_c["c"] else 0.0
            vol_24h = volume_24h_by_symbol[sym].get(ts, c["v"] * c["c"])
            tickers.append({"instrument_name": sym, "last": c["c"], "change": change,
                            "volume_value": vol_24h})
        engine.save_json(tickers_path, tickers)
        engine.screen(tickers_path, cands_path)

        current_prices = {t["instrument_name"]: t["last"] for t in tickers}
        engine.save_json(prices_path, current_prices)
        # finalize() calls broker_mod.get_broker() internally - this returns
        # PaperBroker as long as TRADING_MODE isn't set to "live" in the
        # environment, which is the default and is never touched here.
        engine.finalize(cands_path, hype_path, prices_path, summary_path)

        if (i + 1) % 200 == 0 or i == len(all_ts) - 1:
            elapsed = time.time() - t0
            pct = (i + 1) / len(all_ts) * 100
            print(f"  ... {i+1}/{len(all_ts)} tundi töödeldud ({pct:.0f}%, {elapsed:.0f}s)")

    # ---- report ----
    st = engine.load_state()
    closed = st["portfolio"]["closed_trades"]
    print(f"\n{'='*70}\nBACKTEST TULEMUS ({months} kuud, {len(histories)} instrumenti, "
          f"{len(all_ts)} simuleeritud tundi)\n{'='*70}")

    by_regime = {"bull": [], "bear": [], "chop": []}
    for t in closed:
        regime = classify_regime(btc_closes_by_ts, t["entry_ts"] * 1000)
        by_regime[regime].append(t)

    def report(label, trades):
        if not trades:
            print(f"\n{label}: 0 kauplust")
            return
        pf = engine.profit_factor(trades)
        exp = engine.expectancy_pct(trades)
        wins = sum(1 for t in trades if t["pnl_usd"] > 0)
        print(f"\n{label}: {len(trades)} kauplust, {wins}/{len(trades)} võitu "
              f"({wins/len(trades)*100:.0f}%)")
        print(f"  Profit factor: {pf:.2f}" if pf is not None else "  Profit factor: – (kaotusi pole)")
        print(f"  Oodatav väärtus/kauplus: {exp:+.2f}%" if exp is not None else "  Oodatav väärtus: –")

    report("KOKKU", closed)
    for regime in ("bull", "bear", "chop"):
        report(f"Režiim: {regime.upper()}", by_regime[regime])

    print(f"\nMudel treenis {st['model']['n_updates']} sammu selle backtesti jooksul.")
    print(f"Täisandmed: {state_path}")
    print("\nMEELDETULETUS: hype/orderiraamatu kihte selles backtestis ei simuleeritud "
          "(vt selle faili docstring) - see valideerib momentum/trendi/alpha matemaatikat, "
          "mitte kogu live-süsteemi tervikuna.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=6, help="Mitu kuud tagasi minna (vaikimisi 6 - piisav, et näha vähemalt ühte režiimivahetust)")
    ap.add_argument("--instruments", type=str, default=None,
                    help="Komaeraldatud sümbolite list (vaikimisi: väiksem valik majors+meme)")
    args = ap.parse_args()
    instruments = args.instruments.split(",") if args.instruments else DEFAULT_INSTRUMENTS
    run_backtest(args.months, instruments)
