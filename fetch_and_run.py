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
COINGECKO_COIN_URL = "https://api.coingecko.com/api/v3/coins/{id}"

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

    candidates_path = os.path.join(DATA_DIR, "candidates.json")
    engine.screen(tickers_path, candidates_path)

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

    # Fresh prices for everything the finalize step manages: pending 24h/7d
    # followups AND open portfolio positions (stop-loss checks need a price
    # every run, not just when a followup is due). Reuse this run's ticker
    # fetch where possible; only re-fetch what wasn't in the watchlist pull.
    state = engine.load_state()
    needed = ({rec["instrument"] for rec in state["pending_followups"]}
              | {p["instrument"] for p in state["portfolio"]["open_positions"]})
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
    engine.finalize(candidates_path, hype_notes_path, current_prices_path, summary_path)

    with open(summary_path) as f:
        summary = f.read()
    send_telegram(summary)


if __name__ == "__main__":
    main()
