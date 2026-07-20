#!/usr/bin/env python3
"""
Standalone orchestrator for GitHub Actions (no Claude/MCP tools available here -
just plain Python + the public internet). This replaces the "agent does the
fetching" steps from the Cowork version with direct HTTP calls, and reuses the
exact same engine.py scoring logic (screen -> finalize) unchanged.

What's different from the Cowork version:
  - Market data comes straight from Crypto.com's public REST API (no auth
    needed for tickers).
  - There is NO live web-search hype check here (GitHub Actions can't call
    Claude's WebSearch tool). Scoring is momentum + trend + liquidity/scam
    heuristics only. hype_notes.json is always empty - engine.py already
    handles that gracefully (just skips the hype bonus/explanation bit).
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

    candidates_path = os.path.join(DATA_DIR, "candidates.json")
    engine.screen(tickers_path, candidates_path)

    # No live hype-check here (no WebSearch available in GitHub Actions) -
    # always pass an empty notes file, engine.py handles that gracefully.
    hype_notes_path = os.path.join(DATA_DIR, "hype_notes.json")
    engine.save_json(hype_notes_path, {})

    # Resolve any due 24h/7d followups using fresh prices for those instruments.
    state = engine.load_state()
    due_instruments = {rec["instrument"] for rec in state["pending_followups"]}
    current_prices = {}
    for inst in due_instruments:
        t = fetch_ticker(inst)
        if t and t["last"] is not None:
            current_prices[inst] = float(t["last"])
    current_prices_path = os.path.join(DATA_DIR, "current_prices.json")
    engine.save_json(current_prices_path, current_prices)

    summary_path = os.path.join(DATA_DIR, "summary_latest.txt")
    engine.finalize(candidates_path, hype_notes_path, current_prices_path, summary_path)

    with open(summary_path) as f:
        summary = f.read()
    send_telegram(summary)


if __name__ == "__main__":
    main()
