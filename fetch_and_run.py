#!/usr/bin/env python3
"""
Standalone orchestrator for GitHub Actions (no Claude/MCP tools available here -
just plain Python + the public internet). This replaces the "agent does the
fetching" steps from the Cowork version with direct HTTP calls, and reuses the
exact same engine.py scoring logic (screen -> finalize) unchanged.

What's different from the Cowork version:
  - Market data comes straight from Crypto.com's public REST API (no auth
    needed for tickers).
  - The hype/sentiment check is NOT a live WebSearch (GitHub Actions can't
    call Claude's WebSearch tool) - instead it queries CoinGecko's free public
    API (no key required) for each Stage-1 candidate: real community
    sentiment-vote percentages, plus CoinGecko's own curated `public_notice`/
    `additional_notices` fields (their scam/delisting/caution flags). This is
    a genuine, structured signal rather than an approximation, just sourced
    differently than the Cowork path's news-search summaries. Symbols with no
    known CoinGecko listing (or on any API hiccup) fall back to "not found",
    same graceful degradation engine.py already handles.
  - Notification goes out via Telegram instead of a Cowork chat message.
  - State/dashboard files are committed back to the repo by the GitHub Actions
    workflow (see .github/workflows/hourly.yml), not by this script.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

import engine  # same file used by the Cowork version - unchanged

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")

CRYPTO_COM_TICKER_URL = "https://api.crypto.com/exchange/v1/public/get-tickers"
CRYPTO_COM_CANDLE_URL = "https://api.crypto.com/exchange/v1/public/get-candlestick"
CRYPTO_COM_BOOK_URL = "https://api.crypto.com/exchange/v1/public/get-book"
CRYPTO_COM_VALUATIONS_URL = "https://api.crypto.com/exchange/v1/public/get-valuations"
FNG_URL = "https://api.alternative.me/fng/?limit=1"
BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
ONCHAIN_HISTORY_KEEP = 168   # ~7 days of hourly readings for the trailing baseline
COINGECKO_COIN_URL = "https://api.coingecko.com/api/v3/coins/{id}"

CANDLE_TIMEFRAME = "15m"
CANDLE_COUNT = 96   # 24h of 15m candles - engine uses these for volatility
                    # (96 return samples) and trend (24 points over 6h)
                    # instead of the 6..24 sparse hourly snapshots

# watchlist symbol -> CoinGecko coin id, resolved once via CoinGecko's /search
# endpoint (picking the highest-market-cap-rank exact ticker match for each
# symbol) so the hourly run never needs to do that lookup itself - CoinGecko's
# free tier rate-limits hard (a handful of req/min) and a live search per
# candidate on top of the detail call would burn through that budget fast.
# Extend this manually if watchlist.json grows with a new symbol.
SYMBOL_TO_COINGECKO_ID = {
    "BTCUSD": "bitcoin", "ETHUSD": "ethereum", "SOLUSD": "solana", "XRPUSD": "ripple",
    "ADAUSD": "cardano", "DOGEUSD": "dogecoin", "AVAXUSD": "avalanche-2", "LINKUSD": "chainlink",
    "DOTUSD": "polkadot", "LTCUSD": "litecoin", "BCHUSD": "bitcoin-cash", "ATOMUSD": "cosmos",
    "NEARUSD": "near", "ARBUSD": "arbitrum", "OPUSD": "optimism", "SUIUSD": "sui",
    "APTUSD": "aptos", "INJUSD": "injective-protocol", "RUNEUSD": "thorchain",
    "HBARUSD": "hedera-hashgraph", "ICPUSD": "internet-computer", "FILUSD": "filecoin",
    "RENDERUSD": "render-token", "TIAUSD": "celestia", "TAOUSD": "bittensor",
    "ONDOUSD": "ondo-finance", "AAVEUSD": "aave", "UNIUSD": "uniswap",
    "PEPEUSD": "pepe", "WIFUSD": "dogwifcoin", "BONKUSD": "bonk", "SHIBUSD": "shiba-inu",
    "FLOKIUSD": "floki", "GOATUSD": "goatseus-maximus", "PNUTUSD": "peanut-the-squirrel",
    "MOODENGUSD": "moo-deng", "FARTCOINUSD": "fartcoin", "PENGUUSD": "pudgy-penguins",
    "TRUMPUSD": "official-trump", "MELANIAUSD": "melania-meme", "POPCATUSD": "popcat",
    "TURBOUSD": "turbo", "ARKMUSD": "arkham", "VIRTUALUSD": "virtual-protocol", "AIXBTUSD": "aixbt",
}

HYPE_SLEEP_SECONDS = 2.5  # CoinGecko's free tier rate-limits hard; only called for
                          # up to ~10 Stage-1 candidates per run, so this stays well
                          # inside an hourly job's time budget

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def http_get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _candidate_instrument_names(symbol):
    """watchlist.json stores plain symbols like 'BTCUSD' (no separator), but
    Crypto.com's real instrument names never look like that. Spot pairs are
    'BASE_QUOTE' (e.g. BTC_USD or, for some coins, only BTC_USDT exists, not
    BTC_USD). Perpetual futures are 'BASEQUOTE-PERP' (e.g. BTCUSD-PERP, no
    underscore). This was a real, confirmed bug: EVERY fetch was silently
    returning nothing because 'BTCUSD' itself isn't a valid instrument_name
    on Crypto.com at all - try the real formats in order and use whichever
    one the API actually recognizes."""
    if symbol.endswith("USD"):
        base = symbol[:-3]
        return [f"{base}_USD", f"{base}_USDT", f"{symbol}-PERP"]
    return [symbol]


def fetch_ticker(symbol):
    """Call Crypto.com's public get-tickers endpoint for a watchlist symbol
    (e.g. 'BTCUSD'). Returns a dict shaped like engine.py expects (keyed by
    the ORIGINAL watchlist symbol, regardless of which real Crypto.com
    instrument name actually had data), or None if none of the candidate
    formats resolved to anything."""
    tried = []
    for real_name in _candidate_instrument_names(symbol):
        tried.append(real_name)
        url = f"{CRYPTO_COM_TICKER_URL}?instrument_name={real_name}"
        try:
            raw = http_get_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            continue
        data = (raw.get("result") or {}).get("data") or []
        if not data:
            continue
        d = data[0]
        try:
            if d.get("a") is None or d.get("c") is None:
                continue
            return {
                "instrument_name": symbol,  # keep the watchlist's own key, not Crypto.com's real name
                "real_instrument": real_name,  # the exchange's actual name - LiveBroker orders need this
                "last": d.get("a"),
                "change": d.get("c"),
                "volume_value": d.get("vv"),
            }
        except Exception:
            continue
    print(f"WARN: no working Crypto.com instrument found for {symbol} (tried {tried})", file=sys.stderr)
    return None


def fetch_all_tickers(instruments):
    out = []
    for inst in instruments:
        t = fetch_ticker(inst)
        if t and t["last"] is not None and t["change"] is not None:
            out.append(t)
        time.sleep(0.15)  # be polite to the public API, no need to hammer it
    return out


def fetch_candles(real_name):
    """24h of 15m OHLCV candles for one real exchange instrument. Returns a
    list of {t,o,h,l,c,v} dicts (oldest first) or None on any failure -
    engine.py falls back to hourly-snapshot math when candles are missing."""
    url = (f"{CRYPTO_COM_CANDLE_URL}?instrument_name={real_name}"
           f"&timeframe={CANDLE_TIMEFRAME}&count={CANDLE_COUNT}")
    try:
        raw = http_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    data = (raw.get("result") or {}).get("data") or []
    out = []
    for d in data:
        try:
            out.append({"t": d["t"], "o": float(d["o"]), "h": float(d["h"]),
                        "l": float(d["l"]), "c": float(d["c"]), "v": float(d["v"])})
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


def fetch_book_note(real_name, depth=10):
    """Order book snapshot -> two numbers the engine can use: notional
    imbalance of the top levels (-1 = all sell pressure, +1 = all buy
    pressure) and the bid/ask spread in %. Thin/wide books are where market
    orders get hurt - the engine penalizes them before opening anything."""
    url = f"{CRYPTO_COM_BOOK_URL}?instrument_name={real_name}&depth={depth}"
    try:
        raw = http_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    data = (raw.get("result") or {}).get("data") or []
    if not data:
        return None
    d = data[0]
    try:
        bids = [(float(p), float(q)) for p, q, *_ in (d.get("bids") or [])]
        asks = [(float(p), float(q)) for p, q, *_ in (d.get("asks") or [])]
        if not bids or not asks:
            return None
        bid_notional = sum(p * q for p, q in bids)
        ask_notional = sum(p * q for p, q in asks)
        total = bid_notional + ask_notional
        if total <= 0:
            return None
        imbalance = (bid_notional - ask_notional) / total
        mid = (bids[0][0] + asks[0][0]) / 2
        spread_pct = (asks[0][0] - bids[0][0]) / mid * 100 if mid > 0 else None
        return {"imbalance": round(imbalance, 3), "spread_pct": round(spread_pct, 4)}
    except (ValueError, TypeError, IndexError):
        return None


def fetch_market_regime():
    """Crypto Fear & Greed index (free, one number 0-100). Market-wide mood:
    momentum behaves differently in panic vs greed phases, so the model gets
    this as a feature and learns what it is worth. Returns dict or {}."""
    try:
        raw = http_get_json(FNG_URL)
        d = (raw.get("data") or [{}])[0]
        return {"value": int(d["value"]), "classification": d.get("value_classification", "")}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError, KeyError, TypeError, IndexError):
        return {}


def fetch_funding_rate_annualized(symbol, hours=24):
    """Recent funding rate for a watchlist symbol's perpetual (BTCUSD ->
    BTCUSD-PERP), annualized. Crypto.com's funding_hist valuation returns an
    hourly rate; averaging the last `hours` readings and scaling by 24*365
    smooths out noise from any single settlement. Returns None if the perp
    doesn't exist or the call fails - funding-arb just skips that symbol,
    same graceful-degradation pattern as the hype-check."""
    perp_name = f"{symbol}-PERP"
    url = f"{CRYPTO_COM_VALUATIONS_URL}?instrument_name={perp_name}&valuation_type=funding_hist&count={hours}"
    try:
        raw = http_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    data = (raw.get("result") or {}).get("data") or []
    if not data:
        return None
    try:
        rates = [float(d["v"]) for d in data]
    except (KeyError, ValueError, TypeError):
        return None
    if not rates:
        return None
    avg_hourly = sum(rates) / len(rates)
    return avg_hourly * 24 * 365 * 100  # -> annualized %


def fetch_onchain_activity():
    """BTC on-chain transaction volume, free and keyless (blockchain.info's
    public /stats endpoint) - the honest limitation here is that GENUINE
    labeled whale/exchange-netflow data (Nansen, Glassnode, Arkham) is paid
    and per-chain, so a free, no-key signal across all 45 watchlist tokens
    (many different blockchains) isn't realistic to build. This is BTC-only,
    used as a market-wide activity signal (like Fear & Greed) rather than a
    per-token feature - an unusually large surge in real BTC network value
    moving on-chain is a genuine, if blunt, proxy for large players active
    in the market right now. Returns the raw USD figure, or None on failure."""
    try:
        raw = http_get_json(BLOCKCHAIN_STATS_URL)
        return float(raw["estimated_transaction_volume_usd"])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError, KeyError, TypeError):
        return None


def onchain_activity_ratio(current_value, data_dir):
    """Ratio of the current on-chain volume reading to its trailing 7-day
    hourly average - >1 means unusually high value moving on-chain right
    now. Persists its own small rolling history file (independent of
    engine.py's state.json) since this is a raw external reading, not a
    trading outcome."""
    hist_path = os.path.join(data_dir, "onchain_history.json")
    hist = engine.load_json(hist_path, [])
    ratio = None
    if hist:
        baseline = sum(hist) / len(hist)
        if baseline > 0:
            ratio = current_value / baseline
    hist.append(current_value)
    hist = hist[-ONCHAIN_HISTORY_KEEP:]
    engine.save_json(hist_path, hist)
    return ratio


def fetch_hype_note(symbol):
    """Real hype/sentiment check for one Stage-1 candidate, sourced from
    CoinGecko's free public API instead of a live web search (not available
    here). Returns the same shape engine.hype_adjustment() expects:
    {"found": bool, "sentiment": "positive"/"neutral"/"negative"/"warning",
    "summary": str}. Falls back to {"found": False} on anything unexpected -
    unmapped symbol, timeout, rate limit, malformed response - so a flaky
    API call degrades to "no fresh info" (engine.py's existing, already-
    tested behavior) rather than crashing the whole run."""
    coingecko_id = SYMBOL_TO_COINGECKO_ID.get(symbol)
    if not coingecko_id:
        return {"found": False}

    url = (COINGECKO_COIN_URL.format(id=coingecko_id) +
           "?localization=false&tickers=false&market_data=false"
           "&community_data=true&developer_data=false")
    try:
        d = http_get_json(url, timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"WARN: CoinGecko lookup failed for {symbol} ({coingecko_id}): {e}", file=sys.stderr)
        return {"found": False}

    rank = d.get("market_cap_rank")
    rank_txt = f"turu positsioon #{rank}" if rank else "turu positsioon reastamata (väga väike/uus token)"

    notices = [n for n in ([d.get("public_notice")] + list(d.get("additional_notices") or [])) if n]
    if notices:
        text = " ".join(str(n) for n in notices)[:250]
        return {"found": True, "sentiment": "warning", "summary": f"CoinGecko hoiatus: {text}"}

    up = d.get("sentiment_votes_up_percentage")
    down = d.get("sentiment_votes_down_percentage")
    if up is None:
        return {"found": False}

    if up >= 65:
        sentiment = "positive"
    elif up <= 35:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    summary = f"CoinGecko kogukonna hääletus: {up:.0f}% positiivne / {(down or 0):.0f}% negatiivne, {rank_txt}."
    return {"found": True, "sentiment": sentiment, "summary": summary}


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARN: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, skipping send. "
              "Message would have been:\n" + text, file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"ERROR sending Telegram message: {e}", file=sys.stderr)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    wl_raw = engine.load_json(WATCHLIST_PATH, {"majors": [], "meme_trend": []})
    instruments = wl_raw.get("majors", []) + wl_raw.get("meme_trend", [])

    tickers = fetch_all_tickers(instruments)
    tickers_path = os.path.join(DATA_DIR, "tickers_latest.json")
    engine.save_json(tickers_path, tickers)
    print(f"Fetched {len(tickers)}/{len(instruments)} tickers from Crypto.com.")

    # watchlist symbol -> real exchange instrument name, refreshed each run.
    # LiveBroker reads this file to place orders under the name the exchange
    # actually recognizes (BTC_USD / BTC_USDT / ...-PERP).
    instrument_map = {t["instrument_name"]: t["real_instrument"]
                      for t in tickers if t.get("real_instrument")}
    engine.save_json(os.path.join(DATA_DIR, "instrument_map.json"), instrument_map)

    # 15m OHLCV candles for every watched instrument - real intra-hour data
    # for volatility/trend instead of one snapshot per hour.
    candles = {}
    for symbol, real_name in instrument_map.items():
        cd = fetch_candles(real_name)
        if cd:
            candles[symbol] = cd
        time.sleep(0.12)
    candles_path = os.path.join(DATA_DIR, "candles_latest.json")
    engine.save_json(candles_path, candles)
    print(f"Fetched candles for {len(candles)}/{len(instrument_map)} instruments.")

    # Market-wide mood (Fear & Greed) - one call per run.
    regime = fetch_market_regime()
    engine.save_json(os.path.join(DATA_DIR, "market_regime.json"), regime)
    if regime:
        print(f"Turu meeleolu: {regime.get('classification')} ({regime.get('value')}/100)")

    # On-chain activity (BTC-only, free - see fetch_onchain_activity docstring
    # for why this can't cover the whole watchlist).
    onchain_value = fetch_onchain_activity()
    onchain_ratio = onchain_activity_ratio(onchain_value, DATA_DIR) if onchain_value is not None else None
    engine.save_json(os.path.join(DATA_DIR, "onchain_latest.json"),
                     {"value_usd": onchain_value, "ratio_vs_7d_avg": onchain_ratio})
    if onchain_ratio is not None:
        print(f"On-chain aktiivsus (BTC): {onchain_ratio:.2f}x 7-päeva keskmisest")

    # Funding rates for the market-neutral funding-arb sleeve - one call per
    # instrument's perpetual (independent of Stage-1 candidates, since this
    # strategy doesn't care about momentum at all).
    funding_rates = {}
    for symbol in instrument_map:
        apr = fetch_funding_rate_annualized(symbol)
        if apr is not None:
            funding_rates[symbol] = apr
        time.sleep(0.12)
    funding_path = os.path.join(DATA_DIR, "funding_rates.json")
    engine.save_json(funding_path, funding_rates)
    print(f"Funding-määrad: {len(funding_rates)}/{len(instrument_map)} instrumendile leiti perpetual.")

    candidates_path = os.path.join(DATA_DIR, "candidates.json")
    engine.screen(tickers_path, candidates_path, candles_path=candles_path)

    # Real hype/sentiment check for just the Stage-1 candidates (not all 45
    # watched instruments) via CoinGecko - see fetch_hype_note() docstring
    # for why this replaces the Cowork version's live WebSearch step.
    candidates = engine.load_json(candidates_path, [])
    hype_notes = {}
    for c in candidates:
        inst = c["instrument"]
        hype_notes[inst] = fetch_hype_note(inst)
        time.sleep(HYPE_SLEEP_SECONDS)
    hype_notes_path = os.path.join(DATA_DIR, "hype_notes.json")
    engine.save_json(hype_notes_path, hype_notes)
    n_found = sum(1 for n in hype_notes.values() if n.get("found"))
    print(f"Hype-kontroll: {n_found}/{len(hype_notes)} kandidaadi kohta leidus CoinGecko andmeid.")

    # Order book snapshot for each Stage-1 candidate (thin-book guard +
    # buy/sell pressure feature for the model).
    book_notes = {}
    for c in candidates:
        inst = c["instrument"]
        real = instrument_map.get(inst)
        if not real:
            continue
        note = fetch_book_note(real)
        if note:
            book_notes[inst] = note
        time.sleep(0.12)
    book_notes_path = os.path.join(DATA_DIR, "book_notes.json")
    engine.save_json(book_notes_path, book_notes)

    # Fresh prices for everything the finalize step manages: pending 24h/7d
    # followups AND open portfolio positions (stop-loss checks need a price
    # every run, not just when a followup is due). Reuse this run's ticker
    # fetch where possible; only re-fetch what wasn't in the watchlist pull.
    state = engine.load_state()
    needed = ({rec["instrument"] for rec in state["pending_followups"]}
              | {p["instrument"] for p in state["portfolio"]["open_positions"]}
              | {p["instrument"] for p in state.get("funding_arb", {}).get("open_positions", [])}
              | {p["instrument"] for p in state.get("grid", {}).get("open_positions", [])}
              | {p["instrument"] for p in state.get("short_reversal", {}).get("open_positions", [])})
    current_prices = {t["instrument_name"]: float(t["last"])
                      for t in tickers
                      if t["instrument_name"] in needed and t["last"] is not None}
    for inst in needed - set(current_prices):
        t = fetch_ticker(inst)
        if t and t["last"] is not None:
            current_prices[inst] = float(t["last"])
    current_prices_path = os.path.join(DATA_DIR, "current_prices.json")
    engine.save_json(current_prices_path, current_prices)

    summary_path = os.path.join(DATA_DIR, "summary_latest.txt")
    engine.finalize(candidates_path, hype_notes_path, current_prices_path, summary_path,
                    candles_path=candles_path, book_path=book_notes_path,
                    funding_path=funding_path)

    with open(summary_path) as f:
        summary = f.read()
    send_telegram(summary)


if __name__ == "__main__":
    main()
